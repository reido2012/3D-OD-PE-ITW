import tensorflow as tf
import json
import numpy as np
import glob
import pathlib
import sqlite3
from nets import nets_factory, resnet_v1

NETWORK_NAME = 'resnet_v1_50'
MODEL_DIR = ""
slim = tf.contrib.slim

FULL_POSE_TFRECORD = "/home/omarreid/selerio/datasets/full_pose_space.tfrecords"


def main(model_dir):
    global MODEL_DIR
    MODEL_DIR = model_dir

    with tf.device("/device:GPU:0"):
        print(f"Model Dir: {model_dir}")
        store_to_db(model_dir)


def store_to_db(model_dir):

    synth_domain_cnn = tf.estimator.Estimator(
        model_fn=synth_domain_cnn_model_fn_predict,
        model_dir=model_dir
    )
    conn = sqlite3.connect('full_pose_2.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE full_pose_space
                 (viewpoint text, image_path text, object_class text, cad_index text, depth_embedding text)''')

    all_model_predictions = synth_domain_cnn.predict(input_fn=lambda: predict_input_fn(FULL_POSE_TFRECORD),
                                                     yield_single_examples=True)

    for counter, prediction in enumerate(all_model_predictions):
        depth_emb = json.dumps(list(prediction["depth_embeddings"].squeeze().astype(str)))
        cad_index = prediction["cad_index"].decode("utf-8")
        object_class = prediction["object_class"].decode("utf-8")
        image_path = prediction["image_path"].decode("utf-8")
        rot_x = prediction["rot_x"].decode("utf-8")
        rot_y = prediction["rot_y"].decode("utf-8")
        rot_z = prediction["rot_z"].decode("utf-8")
        viewpoint = json.dumps((rot_x, rot_y, rot_z))

        c.execute("INSERT INTO full_pose_space VALUES (?, ?, ?, ?, ?)",
                  (viewpoint, image_path, object_class, cad_index, depth_emb))

    c.close()


def predict_input_fn(tfrecords_file):
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset_base(dataset, shuffle=False)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def dataset_base(dataset, shuffle=False):
    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=8)  # Parallelize data transformation
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=50)
    return dataset.prefetch(buffer_size=2)


def tfrecord_parser(serialized_example):
    """
        Parses a single tf.Example into image and label tensors.
    """

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'depth_image': tf.FixedLenFeature([], tf.string),
            'object_class': tf.FixedLenFeature([], tf.string),
            'cad_index': tf.FixedLenFeature([], tf.string),
            'image_path': tf.FixedLenFeature([], tf.string),
            'rot_x': tf.FixedLenFeature([], tf.string),
            'rot_y': tf.FixedLenFeature([], tf.string),
            'rot_z': tf.FixedLenFeature([], tf.string)

        }
    )

    object_class = features['object_class']
    cad_index = features['cad_index']
    image_path = features['image_path']
    rot_x = features['rot_x']
    rot_y = features['rot_y']
    rot_z = features['rot_z']

    depth_image = convert_string_to_image(features['depth_image'], standardize=False)

    return (depth_image, object_class, image_path, cad_index, rot_x, rot_y, rot_z), object_class


def synth_domain_cnn_model_fn_predict(features, labels, mode):
    depth_images, object_class, image_path, cad_index, rot_x, rot_y, rot_z = features

    with tf.variable_scope('synth_domain'):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
            network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)
            depth_descriptors, endpoints = network_fn(depth_images, reuse=tf.AUTO_REUSE)

    variables_to_restore = slim.get_variables_to_restore(include=['synth_domain'])

    checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    variables_to_restore = [v for v in variables_to_restore if 'resnet_v1_50/' in v.name]
    tf.train.init_from_checkpoint(checkpoint_path,
                                  {v.name.split(':')[0]: v.name.split(':')[0] for v in variables_to_restore})

    predictions = {
        "depth_embeddings": depth_descriptors,
        "image_path": image_path,
        "object_class": object_class,
        "cad_index": cad_index,
        "rot_x": rot_x,
        "rot_y": rot_y,
        "rot_z": rot_z
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


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
    image_shape = tf.cond(shape_pred, lambda: tf.stack([224, 224, 1]),
                          lambda: tf.stack([224, 224, 3]))

    input_image = tf.reshape(image, image_shape)

    channel_pred = tf.cast(tf.equal(tf.shape(input_image)[2], greyscale_channel), tf.bool)
    input_image = tf.cond(channel_pred, lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
    input_image = tf.reshape(input_image, (224, 224, 3))

    if standardize:
        input_image = tf.image.per_image_standardization(input_image)

    return input_image


if __name__ == '__main__':
    main("/home/omarreid/selerio/final_year_project/synth_models/model_one")
