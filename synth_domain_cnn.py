import tensorflow as tf
import click
import glob
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
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/synth_domain_tfrecords/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",  TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
NETWORK_NAME = 'resnet_v1_50'
PRETRAINED_MODEL_DIR = "/home/omarreid/selerio/final_year_project/models/test_one"
RESNET_V1_CHECKPOINT_DIR = "/home/omarreid/selerio/datasets/pre_trained_weights/resnet_v1_50.ckpt"


def synth_domain_cnn_model_fn(features, labels, mode):
    rgb_images, positive_depth_images, negative_depth_images = features


    rgb_descriptors = get_pretrained_resnet_descriptors(rgb_images, is_training=True)

    # Get variables to store for real domain CNN
    real_domain_variables_to_restore = slim.get_variables_to_restore(exclude=['synth_domain'])
    checkpoint_path = tf.train.latest_checkpoint(PRETRAINED_MODEL_DIR)
    tf.train.init_from_checkpoint(checkpoint_path, {v.name.split(':')[0]: v for v in real_domain_variables_to_restore})

    with tf.variable_scope('synth_domain'):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
            network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)

            positive_depth_descriptors, endpoints = network_fn(positive_depth_images, reuse=tf.AUTO_REUSE)
            negative_depth_descriptors, endpoints = network_fn(negative_depth_images, reuse=tf.AUTO_REUSE)

    variables_to_restore = slim.get_variables_to_restore(include=['synth_domain'], exclude=['real_domain'])
    print("New Synth Variables")
    print(variables_to_restore)

    if tf.gfile.IsDirectory(MODEL_DIR):
        checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    else:
        checkpoint_path = RESNET_V1_CHECKPOINT_DIR
        variables_to_restore = [v for v in variables_to_restore if 'resnet_v1_50/' in v.name and 'real_domain/' not in v.name and 'synth_domain/' in v.name]

    tf.train.init_from_checkpoint(checkpoint_path, {v.name.split(':')[0].replace('synth_domain/', '', 1): v.name.split(':')[0] for v in variables_to_restore})

    loss = similarity_loss(rgb_descriptors, positive_depth_descriptors, negative_depth_descriptors)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=STARTING_LR,
            global_step=global_step,
            decay_steps=23206,
            decay_rate=0.1,
            staircase=True,
            name="learning_rate"
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step
        )

        tf.summary.image('RGB', rgb_images)
        tf.summary.image('Pos_Depth', positive_depth_images)
        tf.summary.image('Neg_Depth', negative_depth_images)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_pretrained_resnet_descriptors(depth_image, is_training):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
        network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=is_training)
        image_descriptors, endpoints = network_fn(depth_image)
        return image_descriptors


def similarity_loss(rgb_descriptor, pos_descriptor, neg_descriptor):
    s_pos = tf.reduce_sum(tf.square(rgb_descriptor - pos_descriptor), 1)
    s_neg = tf.reduce_sum(tf.square(rgb_descriptor - neg_descriptor), 1)

    return descriptor_loss(s_pos, s_neg) + REG_CONSTANT * tf.losses.get_regularization_loss()


def descriptor_loss(s_pos, s_neg):
    loss = tf.maximum(0.0, TRIPLET_LOSS_MARGIN + s_pos - s_neg)
    loss = tf.reduce_mean(loss)
    return loss


def tfrecord_parser(serialized_example):
    """
        Parses a single tf.Example into image and label tensors.
    """

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'rgb_image': tf.FixedLenFeature([], tf.string),
            'pos_depth': tf.FixedLenFeature([], tf.string),
            'object_class': tf.FixedLenFeature([], tf.string),
            'object_index': tf.FixedLenFeature([], tf.string),
            'data_id': tf.FixedLenFeature([], tf.string),
            'cad_index': tf.FixedLenFeature([], tf.string),
        }
    )

    # rgb_image = convert_string_to_image(features['rgb_image'])
    rgb_image = tf.decode_raw(features['rgb_image'], tf.uint8)
    rgb_image = tf.to_float(rgb_image)
    rgb_image = tf.reshape(rgb_image, (IMAGE_SIZE, IMAGE_SIZE, 3))

    pos_depth_image = tf.decode_raw(features['pos_depth'], tf.uint8)
    pos_depth_image = tf.to_float(pos_depth_image)
    pos_depth_image = tf.reshape(pos_depth_image, (IMAGE_SIZE, IMAGE_SIZE, 3))

    data_id = features['data_id']
    obj_id = features['object_index']
    object_class = features['object_class']
    cad_index = features['cad_index']

    # Create Path to Folder Containing Negative Example Depth Image
    all_depths = "/home/omarreid/selerio/datasets/synth_renderings/" + data_id + "/" + obj_id + "_[!" + cad_index + "]*_0001.png"
    depth_paths = tf.train.match_filenames_once(all_depths)

    random_index = tf.random_uniform([1], 0, tf.size(depth_paths), dtype=tf.int32)
    random_index = tf.squeeze(random_index, 0)

    random_index = tf.Print(random_index, [random_index], message="Random Index: ")
    depth_paths = tf.Print(depth_paths, [depth_paths], message="Depth Paths: ")

    negative_depth_image_raw = tf.read_file(depth_paths[random_index])
    negative_depth_image = tf.image.decode_png(negative_depth_image_raw, channels=3)
    negative_depth_image = tf.cast(negative_depth_image, tf.float32)
    negative_depth_image = tf.reshape(negative_depth_image, (IMAGE_SIZE, IMAGE_SIZE, 3))

    return (rgb_image, pos_depth_image, negative_depth_image), object_class


def convert_string_to_image(image_string):
    """
    Converts image string extracted from TFRecord to an image

    :param image_string: String that represents an image
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

    return input_image


def dataset_base(dataset, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=NUM_CPU_CORES)  # Parallelize data transformation
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset.prefetch(buffer_size=2)


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
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/synth_models/model_one",
              help='Path to model to evaluate')
def main(model_dir):
    # Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    # To handle all of our TF Records
    global MODEL_DIR
    MODEL_DIR = model_dir
    with tf.device("/device:GPU:0"):
        # Create the Estimator
        real_domain_cnn = tf.estimator.Estimator(
            model_fn=synth_domain_cnn_model_fn,
            model_dir=model_dir
        )

        tensors_to_log = {"loss": "loss", "learning_rate": "learning_rate", }
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[logging_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

        tf.estimator.train_and_evaluate(real_domain_cnn, train_spec, eval_spec)

        # acc_pi_6, med_error = run_eval(model_dir)
        # logging.debug("ACC PI/6: " + acc_pi_6 + " | Med Error: " + str(med_error) + " | Epochs Elapsed: " + str(40))


if __name__ == "__main__":
    main()
