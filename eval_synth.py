import tensorflow as tf
import click
import os
import cv2
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from nets import nets_factory, resnet_v1

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
BATCH_SIZE = 50
NUM_CPU_CORES = 8
IMAGE_SIZE = 224
MODEL_DIR = ""
NETWORK_NAME = 'resnet_v1_50'
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/synth_domain_tfrecords_new/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]


@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/synth_models/model_two",
              help='Path to model to evaluate')
@click.option('--tfrecords_file', default=EVAL_TFRECORDS, help='Path to TFRecords file to evaluate model on', type=str)
def main(model_dir, tfrecords_file):
    global MODEL_DIR
    MODEL_DIR = model_dir
    visualize_embeddings(tfrecords_file)


def visualize_embeddings(tfrecords_file):

    with tf.device('/GPU:0'):
        synth_domain_cnn = tf.estimator.Estimator(
            model_fn=synth_domain_cnn_model_fn_predict,
            model_dir=MODEL_DIR
        )

        all_model_predictions = synth_domain_cnn.predict(input_fn=lambda: predict_input_fn(tfrecords_file),
                                                         yield_single_examples=False)

        # # TODO: Try displaying only positive depth embeddings
        pos_embeddings = np.zeros((BATCH_SIZE, 2048))
        pos_depth_images = np.zeros((BATCH_SIZE, 224, 224, 3))
        # neg_embeddings = np.zeros((BATCH_SIZE, 2048))
        # rgb_embeddings = np.zeros((BATCH_SIZE, 2048))

        for counter, prediction in enumerate(all_model_predictions):
            pos_embeddings[counter] = prediction["positive_depth_embeddings"]
            pos_depth_images[counter] = prediction["positive_depth_images"]
        #     neg_embeddings[i] = prediction['negative_depth_embeddings']
        #     rgb_embeddings[i] = prediction['rgb_embeddings']

        create_sprite(pos_depth_images, "pos_depth_sprite.png")
        tf.logging.info("Positive Embeddings shape: {}".format(pos_embeddings.shape))

        # Visualize test embeddings
        pos_embedding_var = tf.Variable(pos_embeddings, name='pos_depth')

        eval_dir = os.path.join(MODEL_DIR, "eval")
        summary_writer = tf.summary.FileWriter(eval_dir)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = pos_embedding_var.name

        embedding.sprite.image_path = 'pos_depth_sprite.png'
        embedding.sprite.single_image_dim.extend([224, 224])

        # Specify where you find the metadata
        # Save the metadata file needed for Tensorboard projector
        # metadata_filename = "pos_depth.tsv"
        # with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        #     for i in range(BATCH_SIZE):
        #         c = labels[i]
        #         f.write('{}\n'.format(c))
        # embedding.metadata_path = metadata_filename

        # Say that you want to visualise the embeddings
        projector.visualize_embeddings(summary_writer, config)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(pos_embedding_var.initializer)
            saver.save(sess, os.path.join(eval_dir, "pos_embeddings.ckpt"))


def create_sprite(images, filename):
    eval_dir = os.path.join(MODEL_DIR, "eval")
    print("Eval Dir: ")
    print(eval_dir)
    sprite = images_to_sprite(images)
    sprite_filepath = os.path.join(eval_dir, filename)
    cv2.imwrite(sprite_filepath, sprite)

    return sprite_filepath


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def synth_domain_cnn_model_fn_predict(features, labels, mode):
    object_images, rgb_descriptors, positive_depth_images, negative_depth_images = features

    with tf.variable_scope('synth_domain'):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
            network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)

            positive_depth_descriptors, endpoints = network_fn(positive_depth_images, reuse=tf.AUTO_REUSE)
            negative_depth_descriptors, endpoints = network_fn(negative_depth_images, reuse=tf.AUTO_REUSE)

    variables_to_restore = slim.get_variables_to_restore(include=['synth_domain'])

    checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    variables_to_restore = [v for v in variables_to_restore if 'resnet_v1_50/' in v.name]
    tf.train.init_from_checkpoint(checkpoint_path,
                                  {v.name.split(':')[0]: v.name.split(':')[0] for v in variables_to_restore})

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "positive_depth_embeddings": positive_depth_descriptors,
        "negative_depth_embeddings": negative_depth_descriptors,
        "rgb_embeddings": rgb_descriptors,
        "positive_depth_images": positive_depth_images,
        "negative_depth_images": negative_depth_images,
        "object_images": object_images
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def predict_input_fn(tfrecords_file):
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset_base(dataset, shuffle=False)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def dataset_base(dataset, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=NUM_CPU_CORES)  # Parallelize data transformation
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset.prefetch(buffer_size=2)


def tfrecord_parser(serialized_example):
    """
        Parses a single tf.Example into image and label tensors.
    """

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'positive_depth_image': tf.FixedLenFeature([], tf.string),
            'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
            'object_image':  tf.FixedLenFeature([], tf.string),
            'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
            'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
            'object_class': tf.FixedLenFeature([], tf.string)
        }
    )

    negative_depth_image = convert_string_to_image(features["neg/depth/img/0"])
    pos_depth_image = convert_string_to_image(features['positive_depth_image'])
    object_image = convert_string_to_image(features['object_image'])

    object_class = features['object_class']
    rgb_descriptor = tf.cast(features['rgb_descriptor'], tf.float32)

    return (object_image, rgb_descriptor, pos_depth_image, negative_depth_image), object_class


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


if __name__ == "__main__":
    main()
