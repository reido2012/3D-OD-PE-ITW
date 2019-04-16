import tensorflow as tf
import click
import glob
import subprocess
import cv2
import os.path as osp
from potential_features import FEATURES_LIST, KEYS
from tensorflow.python.framework import errors
import numpy as np
from itertools import chain
from nets import nets_factory, resnet_v1

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 50
NUM_CPU_CORES = 8
IMAGE_SIZE = 224
TRIPLET_LOSS_MARGIN = 1
REG_CONSTANT = 1e-3
MODEL_DIR = ""
PATH_TO_RD_META = ""
STARTING_LR = 1e-4
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/synth_domain_tfrecords_all_negs/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]

print("ghfyguhijo;klkjhfgxdfhjkl")

record_iterator_train1 = tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "imagenet_train.tfrecords")
record_iterator_train2 = tf.python_io.tf_record_iterator(path= TFRECORDS_DIR + "pascal_train.tfrecords")
record_iterator_train3 = tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "imagenet_val.tfrecords")
print("Here")
ALL_ITERATORS = chain(record_iterator_train1, record_iterator_train2, record_iterator_train3)
print("Here 2")
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
EVAL_ITERATOR = tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "pascal_val.tfrecords")
NETWORK_NAME = 'resnet_v1_50'
PRETRAINED_MODEL_DIR = "/home/omarreid/selerio/final_year_project/models/test_one"
RESNET_V1_CHECKPOINT_DIR = "/home/omarreid/selerio/datasets/pre_trained_weights/resnet_v1_50.ckpt"
DATASET_DIR = osp.expanduser('/home/omarreid/selerio/datasets/PASCAL3D+_release1.1')
OBJ_DIR = DATASET_DIR + "/OBJ/"


def synth_domain_cnn_model_fn(features, labels, mode):
    rgb_descriptors, positive_depth_images, negative_depth_images = features

    with tf.variable_scope('synth_domain'):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
            network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)

            positive_depth_descriptors, endpoints = network_fn(positive_depth_images, reuse=tf.AUTO_REUSE)
            negative_depth_descriptors, endpoints = network_fn(negative_depth_images, reuse=tf.AUTO_REUSE)

    variables_to_restore = slim.get_variables_to_restore(include=['synth_domain'])

    if tf.gfile.IsDirectory(MODEL_DIR):
        checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
        variables_to_restore = [v for v in variables_to_restore if 'resnet_v1_50/' in v.name]
        tf.train.init_from_checkpoint(checkpoint_path,
                                      {v.name.split(':')[0]: v.name.split(':')[0] for v in variables_to_restore})
    else:
        checkpoint_path = RESNET_V1_CHECKPOINT_DIR
        variables_to_restore = [v for v in variables_to_restore if
                                'resnet_v1_50/' in v.name and 'real_domain/' not in v.name and 'synth_domain/' in v.name]
        tf.train.init_from_checkpoint(checkpoint_path,
                                      {v.name.split(':')[0].replace('synth_domain/', '', 1): v.name.split(':')[0] for v
                                       in variables_to_restore})

    loss = similarity_loss(rgb_descriptors, positive_depth_descriptors, negative_depth_descriptors)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=STARTING_LR,
            global_step=global_step,
            decay_steps=28000,
            decay_rate=0.1,
            staircase=True,
            name="learning_rate"
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step
        )

        tf.summary.image('pos_depth', positive_depth_images)
        tf.summary.image('neg_depth', negative_depth_images)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def similarity_loss(rgb_descriptor, pos_descriptor, neg_descriptor):
    s_pos = tf.reduce_sum(tf.square(rgb_descriptor - pos_descriptor), 1)
    s_neg = tf.reduce_sum(tf.square(rgb_descriptor - neg_descriptor), 1)

    return descriptor_loss(s_pos, s_neg) + REG_CONSTANT * tf.losses.get_regularization_loss()


def descriptor_loss(s_pos, s_neg):
    loss = tf.maximum(0.0, s_pos - s_neg + TRIPLET_LOSS_MARGIN)
    loss = tf.reduce_mean(loss)
    return loss


def tfrecord_parser(serialized_example):
    """
        Parses a single tf.Example into image and label tensors.
    """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features=FEATURES_LIST[7]
    )

    object_class = tf.cast(features['object_class'], tf.string)
    data_id = tf.cast(features['data_id'], tf.string)
    cad_index = tf.cast(features['cad_index'], tf.string)

    rgb_descriptor = tf.cast(features['rgb_descriptor'], tf.float32)

    key = np.random.choice(KEYS[7:])
    negative_depth_image = convert_string_to_image(features[key], standardize=True)
    pos_depth_image = convert_string_to_image(features['positive_depth_image'], standardize=True)

    return (rgb_descriptor, pos_depth_image, negative_depth_image, cad_index, data_id), object_class


def convert_string_to_image(image_string, standardize=True):
    """
    Converts image string extracted from TFRecord to an image

    :param image_string: String that represents an image
    :param standardize: If the image should be standardized or not
    :return: The image represented by the string
    """
    greyscale_size = tf.constant(50176)
    greyscale_channel = tf.constant(1)

    image = tf.decode_raw(image_string, tf.uint8)
    image = tf.to_float(image)

    # Image is not in correct shape so
    shape_pred = tf.cast(tf.equal(tf.size(image), greyscale_size), tf.bool)
    image_shape = tf.cond(shape_pred, lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 1]),
                          lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3]))


    input_image = tf.reshape(image, image_shape)

    channel_pred = tf.cast(tf.equal(tf.shape(input_image)[2], greyscale_channel), tf.bool)
    input_image = tf.cond(channel_pred, lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
    input_image = tf.reshape(input_image, (224, 224, 3))

    if standardize:
        input_image = tf.image.per_image_standardization(input_image)

    return input_image


# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _render_py_function(features, label):
    rgb_descriptor, pos_depth_image, negative_depth_image, cad_index, data_id = features
    # rot_x, rot_y, rot_z,, bbox_dims
    # x_dim, y_dim, z_dim = bbox_dims
    object_class = label

    all_model_paths = list(glob.glob(OBJ_DIR + "*/*.obj"))  # All classes, all objs

    random_model_obj_path = np.random.choice(all_model_paths)
    while object_class in random_model_obj_path and cad_index in random_model_obj_path:
        random_model_obj_path = np.random.choice(all_model_paths)

    random_cad_index = random_model_obj_path.split("/")[-1][:-4]

    depth_path = "/home/omarreid/selerio/datasets/random_render/0" + "/" + data_id + "_" + str(random_cad_index) + "_0001.png"

    command = "blender -noaudio --background --python ./blender_render.py -- --specific_viewpoint=True " \
              "--cad_index=" + random_cad_index + " --obj_id=" + data_id + " --radians=True " \
                                                                           "--viewpoint=" + str(
        0) + "," + str(
        90) + "," + str(
        0) + " --bbox=" + str(
        1) + "," + str(
        1) + "," + str(
        1) + " --output_folder /home/omarreid/selerio/datasets/random_render/0" + " "

    full_command = command + random_model_obj_path

    try:
        subprocess.run(full_command.split(), check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        raise e

    print("Command: " + full_command)
    negative_depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR)
    negative_depth_image = cv2.cvtColor(negative_depth_image, cv2.COLOR_BGR2RGB)

    return (rgb_descriptor, pos_depth_image, negative_depth_image), label


def _resize_function(features, label):
    rgb_descriptor, pos_depth_image, negative_depth_image = features
    negative_depth_image = tf.image.resize_images(negative_depth_image, [224, 224, 3])
    return (rgb_descriptor, pos_depth_image, negative_depth_image), label


def dataset_base(dataset, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=NUM_CPU_CORES)  # Parallelize data transformation
    dataset = dataset.map(lambda features, label: tuple(tf.py_func(_render_py_function, [np.array([features]), label], [tf.float32, tf.float32, tf.float32, tf.string, tf.string, tf.string])))
    dataset = dataset.map(_resize_function)
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset.prefetch(buffer_size=6)


def train_input_fn():
    """
    Builds an input pipeline that yields batches of feature and label pairs
    """
    dataset = tf.data.TFRecordDataset(TRAINING_TFRECORDS)
    dataset = dataset_base(dataset)
    dataset = dataset.repeat(count=10)  # Train for count epochs

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def eval_input_fn():
    """
    Builds an input pipeline that yields batches of feature and label pairs for evaluation
    """
    dataset = tf.data.TFRecordDataset(EVAL_TFRECORDS)
    dataset = dataset_base(dataset, shuffle=False)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/synth_models/test",
              help='Path to model to evaluate')
def main(model_dir):
    # Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    # To handle all of our TF Records


    global MODEL_DIR
    MODEL_DIR = model_dir

    with tf.device("/device:GPU:0"):
        # Create the Estimator
        synth_domain_cnn = tf.estimator.Estimator(
            model_fn=synth_domain_cnn_model_fn,
            model_dir=model_dir
        )
        print("Post Model Definition")
        tensors_to_log = {"loss": "loss", "learning_rate": "learning_rate", }
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        train_spec = tf.estimator.TrainSpec(input_fn=magic_input_fn, hooks=[logging_hook])
        print("Post Train Spec")
        eval_spec = tf.estimator.EvalSpec(input_fn=magic_input_eval_fn, hooks=[logging_hook])

        tf.estimator.train_and_evaluate(synth_domain_cnn, train_spec, eval_spec)


def magic_input_fn():
    all_features = []
    all_labels = []

    iterator = ALL_ITERATORS
    print("Inside Magic Input FN")
    print(len(list(iterator)))

    for string_record in iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        features = example.features.feature

        rgb_descriptor = features['rgb_descriptor'].float_list.value
        object_class = features['object_class'].bytes_list.value[0].decode("utf-8")
        data_id = features['data_id'].bytes_list.value[0].decode("utf-8")
        cad_index = features['cad_index'].bytes_list.value[0].decode("utf-8")

        img_string = example.features.feature['positive_depth_image'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        pos_depth_image = img_1d.reshape((224, 224, 3))

        print(f"RGB Descriptor: {rgb_descriptor}")
        print(f"Object Class: {object_class}")
        print(f"Data ID: {data_id}")
        print(f"CAD Index: {cad_index}")

        all_model_paths = list(glob.glob(OBJ_DIR + "*/*.obj"))  # All classes, all objs

        pos_obj = OBJ_DIR + str(object_class) + "/" + str(cad_index) + ".obj"

        print(f"Pos Obj: {pos_obj}")

        random_model_obj_path = np.random.choice(all_model_paths)
        while pos_obj == random_model_obj_path:
            random_model_obj_path = np.random.choice(all_model_paths)

        random_cad_index = random_model_obj_path.split("/")[-1][:-4]

        print(f"Random Obj Model: {random_model_obj_path}")
        print(f"Random Cad Index: {random_cad_index}")

        depth_path = "/home/omarreid/selerio/datasets/random_render/0" + "/" + data_id + "_" + str(
            random_cad_index) + "_0001.png"

        command = "blender -noaudio --background --python ./blender_render.py -- --specific_viewpoint=True " \
                  "--cad_index=" + random_cad_index + " --obj_id=" + data_id + " --radians=True " \
                                                                               "--viewpoint=" + str(
            0) + "," + str(
            90) + "," + str(
            0) + " --bbox=" + str(
            1) + "," + str(
            1) + "," + str(
            1) + " --output_folder /home/omarreid/selerio/datasets/random_render/0" + " "

        full_command = command + random_model_obj_path

        try:
            subprocess.run(full_command.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            raise e

        print("Command: " + full_command)
        negative_depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR)
        negative_depth_image = cv2.cvtColor(negative_depth_image, cv2.COLOR_BGR2RGB)

        single_feature = (rgb_descriptor, pos_depth_image, negative_depth_image)
        single_label = object_class

        all_features.append(single_feature)
        all_labels.append(single_label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    dataset = dataset.repeat(count=10)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def magic_input_eval_fn():
    all_features = []
    all_labels = []

    iterator = EVAL_ITERATOR

    print("Inside Magic Input FN")

    for string_record in iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        features = example.features.feature

        rgb_descriptor = features['rgb_descriptor'].float_list.value
        object_class = features['object_class'].bytes_list.value[0].decode("utf-8")
        data_id = features['data_id'].bytes_list.value[0].decode("utf-8")
        cad_index = features['cad_index'].bytes_list.value[0].decode("utf-8")

        img_string = example.features.feature['positive_depth_image'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        pos_depth_image = img_1d.reshape((224, 224, 3))

        print(f"RGB Descriptor: {rgb_descriptor}")
        print(f"Object Class: {object_class}")
        print(f"Data ID: {data_id}")
        print(f"CAD Index: {cad_index}")

        all_model_paths = list(glob.glob(OBJ_DIR + "*/*.obj"))  # All classes, all objs

        pos_obj = OBJ_DIR + str(object_class) + "/" + str(cad_index) + ".obj"

        print(f"Pos Obj: {pos_obj}")

        random_model_obj_path = np.random.choice(all_model_paths)
        while pos_obj == random_model_obj_path:
            random_model_obj_path = np.random.choice(all_model_paths)

        random_cad_index = random_model_obj_path.split("/")[-1][:-4]

        print(f"Random Obj Model: {random_model_obj_path}")
        print(f"Random Cad Index: {random_cad_index}")

        depth_path = "/home/omarreid/selerio/datasets/random_render/0" + "/" + data_id + "_" + str(
            random_cad_index) + "_0001.png"

        command = "blender -noaudio --background --python ./blender_render.py -- --specific_viewpoint=True " \
                  "--cad_index=" + random_cad_index + " --obj_id=" + data_id + " --radians=True " \
                                                                               "--viewpoint=" + str(
            0) + "," + str(
            90) + "," + str(
            0) + " --bbox=" + str(
            1) + "," + str(
            1) + "," + str(
            1) + " --output_folder /home/omarreid/selerio/datasets/random_render/0" + " "

        full_command = command + random_model_obj_path

        try:
            subprocess.run(full_command.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            raise e

        print("Command: " + full_command)
        negative_depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR)
        negative_depth_image = cv2.cvtColor(negative_depth_image, cv2.COLOR_BGR2RGB)

        single_feature = (rgb_descriptor, pos_depth_image, negative_depth_image)
        single_label = object_class

        all_features.append(single_feature)
        all_labels.append(single_label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    dataset = dataset.repeat(count=10)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


if __name__ == "__main__":
    main()
