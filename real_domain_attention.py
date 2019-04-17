# Imports
import logging
import math
import tensorflow as tf
import numpy as np
import click
import cv2
import os
import scipy.stats as st
from scipy import ndimage

from nets import nets_factory, resnet_v1
from real_domain_cnn import pose_loss, train_input_fn

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

TFRECORDS_DIR = "/home/omarreid/selerio/datasets/real_domain_tfrecords/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
RESNET_V1_CHECKPOINT_DIR = "/home/omarreid/selerio/datasets/pre_trained_weights/resnet_v1_50.ckpt"
NETWORK_NAME = 'resnet_v1_50'
BATCH_SIZE = 50
NUM_CPU_CORES = 8
IMAGE_SIZE = 224  # To match ResNet dimensions
GREYSCALE_SIZE = tf.constant(50176)
GREYSCALE_CHANNEL = tf.constant(1)
# META PARAMETERS
ALPHA = 1
BETA = math.exp(-5)
GAMMA = math.exp(-3)
TRIPLET_LOSS_MARGIN = 1
HUBER_LOSS_DELTA = 0.01
STARTING_LR = 0.0001
ABS_THRESH = 1.5
MODEL_DIR = ""

_SUPPORTED_ATTENTION_TYPES = [
    'use_l2_normalized_feature', 'use_default_input_feature'
]

_SUPPORTED_ATTENTION_NONLINEARITY = ['softplus']

# The variable scope for the attention portion of the model.
_ATTENTION_VARIABLE_SCOPE = 'attention_block'


def real_domain_attention_cnn_model_fn(features, labels, mode):
    """
    Real Domain CNN from 3D Object Detection and Pose Estimation paper
    """
    features = tf.identity(features, name="input")  # Used when converting to unity
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # with tf.variable_scope('base_resnet'):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
        network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=is_training)
        _, end_points = network_fn(features)

    # image_descriptors = tf.identity(image_descriptors, name="image_descriptors")
    feature_map = end_points['resnet_v1_50/block4']

    with slim.arg_scope([slim.batch_norm], is_training=True):
        _, attention_prob, _, end_points = attention_subnetwork(feature_map, end_points,
                                                                attention_type=
                                                                _SUPPORTED_ATTENTION_TYPES[
                                                                    0],
                                                                kernel=1,
                                                                reuse=True)

    attention = tf.reshape(attention_prob, [-1])
    feature_map = tf.reshape(feature_map, [-1, 2048])

    # Use attention score to select feature vectors.
    indices = tf.reshape(tf.where(attention >= ABS_THRESH), [-1])
    selected_features = tf.gather(feature_map, indices)

    # Add a dense layer to get the 19 neuron linear output layer
    logits = tf.layers.dense(selected_features, 19)
    logits = tf.squeeze(logits, name='2d_predictions')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "2d_prediction": logits,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(MODEL_DIR):
        tf.logging.info(
            'Ignoring RESNET50 CKPT because a checkpoint already exists in %s'
            % MODEL_DIR)

    variables_to_restore = slim.get_variables_to_restore()

    if tf.gfile.IsDirectory(MODEL_DIR):
        checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    else:
        checkpoint_path = RESNET_V1_CHECKPOINT_DIR
        variables_to_restore = [v for v in variables_to_restore if 'resnet_v1_50/' in v.name]

    tf.train.init_from_checkpoint(checkpoint_path, {v.name.split(':')[0]: v for v in variables_to_restore})

    # create a pose_loss function so that we can get the loss
    loss = pose_loss(labels, logits)

    # Configure the Training Op (for TRAIN mode)
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

        tf.summary.image('input_images', features)
        tf.summary.histogram('vc_points_predictions', logits)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops={})


def attention_subnetwork(feature_map, end_points, attention_type=_SUPPORTED_ATTENTION_TYPES[0], kernel=1, reuse=False):
    """Constructs the part of the model performing attention.
    Args:
      feature_map: A tensor of size [batch, height, width, channels]. Usually it
        corresponds to the output feature map of a fully-convolutional network.
      end_points: Set of activations of the network constructed so far.
      attention_type: Type of the attention structure.
      kernel: Convolutional kernel to use in attention layers (eg, [3, 3]).
      reuse: Whether or not the layer and its variables should be reused.
    Returns:
      prelogits: A tensor of size [batch, 1, 1, channels].
      attention_prob: Attention score after the non-linearity.
      attention_score: Attention score before the non-linearity.
      end_points: Updated set of activations, for external use.
    Raises:
      ValueError: If unknown attention_type is provided.
    """
    with tf.variable_scope(_ATTENTION_VARIABLE_SCOPE, values=[feature_map, end_points], reuse=reuse):
        if attention_type not in _SUPPORTED_ATTENTION_TYPES:
            raise ValueError('Unknown attention_type.')
        if attention_type == 'use_l2_normalized_feature':
            attention_feature_map = tf.nn.l2_normalize(
                feature_map, 3, name='l2_normalize')
        elif attention_type == 'use_default_input_feature':
            attention_feature_map = feature_map

        end_points['attention_feature_map'] = attention_feature_map

        attention_outputs = perform_attention(attention_feature_map, feature_map, 'softplus', kernel)
        prelogits, attention_prob, attention_score = attention_outputs
        end_points['prelogits'] = prelogits
        end_points['attention_prob'] = attention_prob
        end_points['attention_score'] = attention_score

    return prelogits, attention_prob, attention_score, end_points


def perform_attention(attention_feature_map, feature_map, attention_nonlinear, kernel=1):
    """Helper function to construct the attention part of the model.
    Computes attention score map and aggregates the input feature map based on
    the attention score map.
    Args:
      attention_feature_map: Potentially normalized feature map that will
        be aggregated with attention score map.
      feature_map: Unnormalized feature map that will be used to compute
        attention score map.
      attention_nonlinear: Type of non-linearity that will be applied to
        attention value.
      kernel: Convolutional kernel to use in attention layers (eg: 1, [3, 3]).
    Returns:
      attention_feat: Aggregated feature vector.
      attention_prob: Attention score map after the non-linearity.
      attention_score: Attention score map before the non-linearity.
    Raises:
      ValueError: If unknown attention non-linearity type is provided.
    """
    with tf.variable_scope('attention', values=[attention_feature_map, feature_map]):
        with tf.variable_scope('compute', values=[feature_map]):
            activation_fn_conv1 = tf.nn.relu
            feature_map_conv1 = slim.conv2d(feature_map, 512, kernel, rate=1, activation_fn=activation_fn_conv1,
                                            scope='conv1')

            attention_score = slim.conv2d(feature_map_conv1, 1, kernel, rate=1, activation_fn=None, normalizer_fn=None,
                                          scope='conv2')

        # Set activation of conv2 layer of attention model.
        with tf.variable_scope(
                'merge', values=[attention_feature_map, attention_score]):
            if attention_nonlinear not in _SUPPORTED_ATTENTION_NONLINEARITY:
                raise ValueError('Unknown attention non-linearity.')
            if attention_nonlinear == 'softplus':
                with tf.variable_scope('softplus_attention', values=[attention_feature_map, attention_score]):
                    attention_prob = tf.nn.softplus(attention_score)
                    attention_feat = tf.reduce_mean(tf.multiply(attention_feature_map, attention_prob), [1, 2])

            attention_feat = tf.expand_dims(tf.expand_dims(attention_feat, 1), 2)

    return attention_feat, attention_prob, attention_score


@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/models/attention",
              help='Path to model to evaluate')
def main(model_dir):
    # Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    # To handle all of our TF Records
    global MODEL_DIR
    MODEL_DIR = model_dir
    with tf.device("/device:GPU:0"):
        # Create the Estimator
        real_domain_cnn = tf.estimator.Estimator(
            model_fn=real_domain_attention_cnn_model_fn,
            model_dir=model_dir
        )

        tensors_to_log = {"logits": "2d_predictions", "learning_rate": "learning_rate", }
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        real_domain_cnn.train(input_fn=train_input_fn, hooks=[logging_hook])
