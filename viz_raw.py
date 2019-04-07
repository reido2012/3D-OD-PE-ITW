"""Train the model"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import model.mnist_dataset as mnist_dataset
from model.utils import Params
from model.input_fn import test_input_fn
from model.model_fn import model_fn
from nets import nets_factory, resnet_v1

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
BATCH_SIZE = 50
NUM_CPU_CORES = 8
IMAGE_SIZE = 224
MODEL_DIR = "/home/omarreid/selerio/final_year_project/synth_models/model_two"
NETWORK_NAME = 'resnet_v1_50'
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/synth_domain_tfrecords_all_negs/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]

def main():
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=MODEL_DIR)

    estimator = tf.estimator.Estimator(model_fn, config=config)

    # EMBEDDINGS VISUALIZATION

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions = estimator.predict(input_fn=lambda: predict_input_fn(EVAL_TFRECORDS))

    pos_embeddings = np.zeros((BATCH_SIZE, 2048))
    pos_depth_images = np.zeros((BATCH_SIZE, 224, 224, 3))

    for counter, prediction in enumerate(predictions):
        if counter == 50:
            break

        pos_emb = np.reshape(prediction["positive_depth_embeddings"].squeeze(), (1, 2048))
        pos_embeddings[counter] = pos_emb
        pos_depth_images[counter] = prediction["positive_depth_images"]

    tf.logging.info("Embeddings shape: {}".format(pos_embeddings.shape))

    # Visualize test embeddings
    embedding_var = tf.Variable(pos_embeddings, name='pos_embedding')

    eval_dir = os.path.join(MODEL_DIR, "eval")
    summary_writer = tf.summary.FileWriter(eval_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the sprite (we will create this later)
    # Copy the embedding sprite image to the eval directory
    # shutil.copy2(args.sprite_filename, eval_dir)
    embedding.sprite.image_path = 'pos_depth_sprite.png'
    embedding.sprite.single_image_dim.extend([224, 224])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "pos_depth.ckpt"))


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

    dataset = dataset.map(map_func=tfrecord_parser,
                          num_parallel_calls=NUM_CPU_CORES)  # Parallelize data transformation
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
            'object_image': tf.FixedLenFeature([], tf.string),
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


if __name__ == '__main__':
    main()
