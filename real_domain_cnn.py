# Imports
import logging
import math
import tensorflow as tf
import numpy as np
import click
import scipy.stats as st

# from eval_metrics import run_eval
# from tf.keras.applications.resnet50 import ResNet50, preprocess_input

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

TFRECORDS_DIR = "/home/omarreid/selerio/datasets/real_domain_tfrecords/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
BATCH_SIZE = 30
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


def real_domain_cnn_model_fn(features, labels, mode):
    """
    Real Domain CNN from 3D Object Detection and Pose Estimation paper
    """
    # Features are images
    # Training End to End - So weights start from scratch
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if is_training:
        # Want to train all the layers
        for layer in base_model.layers[:]:
            layer.trainable = True

    tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)

    features = tf.identity(features, name="input")  # Used when converting to unity

    features = tf.keras.applications.resnet50.preprocess_input(features)
    image_descriptors = base_model(features)
    image_descriptors = tf.identity(image_descriptors, name="image_descriptors")
    # image_descriptors = tf.layers.dropout(image_descriptors, rate=0.5, training=is_training)

    # Add a dense layer to get the 19 neuron linear output layer
    logits = tf.layers.dense(image_descriptors, 19, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    logits = tf.squeeze(logits, name='2d_predictions')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "2d_prediction": logits,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # create a pose_loss function so that we can ge tthe loss
    loss = pose_loss(labels, logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=STARTING_LR,
            global_step=global_step,
            decay_steps=50000,
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
        eval_metric_ops = {
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions,
                                          eval_metric_ops=eval_metric_ops)


def pose_loss(labels, logits):
    return projection_loss(labels[:, :16], logits[:, :16]) + dimension_loss(labels[:, 16:], logits[:,
                                                                                            16:]) + BETA * tf.losses.get_regularization_loss()


def projection_loss(bbox_labels, logits_bbox):
    return tf.losses.huber_loss(bbox_labels, logits_bbox, delta=HUBER_LOSS_DELTA)


def dimension_loss(dimension_labels, dimension_logits):
    return tf.losses.huber_loss(dimension_labels, dimension_logits, delta=HUBER_LOSS_DELTA)


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


def tfrecord_parser(serialized_example):
    """
        Parses a single tf.Example into image and label tensors.
        """

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'object_image': tf.FixedLenFeature([], tf.string),
            'output_vector': tf.FixedLenFeature([19], tf.float32),
        }
    )

    # Convert Scalar String to uint8
    input_image = tf.decode_raw(features['object_image'], tf.uint8)
    input_image = tf.to_float(input_image)

    # Image is not in correct shape so fix it
    shape_pred = tf.cast(tf.equal(tf.size(input_image), GREYSCALE_SIZE), tf.bool)
    image_shape = tf.cond(shape_pred, lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 1]),
                          lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3]))

    input_image = tf.reshape(input_image, image_shape)

    channel_pred = tf.cast(tf.equal(tf.shape(input_image)[2], GREYSCALE_CHANNEL), tf.bool)
    input_image = tf.cond(channel_pred, lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
    input_image = tf.reshape(input_image, (224, 224, 3))

    output_vector = tf.cast(features['output_vector'], tf.float32)

    # blur_predicate = tf.math.greater(tf.random_uniform([1], 0, 1, dtype=tf.float32),  0.5)
    # input_image = tf.cond(blur_predicate, lambda: blur_image(input_image), lambda: input_image)

    return input_image, output_vector


def blur_image(input_image):
    random_size, random_sigma = get_random_gauss_atr()
    return conv(input_image, random_size, random_sigma)


def conv(input_image, filter_size, sigma, padding='SAME'):
    # Get the number of channels in the input
    c_i = input_image.get_shape().as_list()[3]
    # Convolution for a given input and kernel
    kernel = make_gauss_var('gauss_weight', filter_size, sigma, c_i)
    output = tf.nn.depthwise_conv2d(input_image, kernel, [1, 1, 1, 1], padding=padding)
    return output


def get_random_gauss_atr():
    random_size = tf.random.uniform([1], 3, 21, dtype=tf.int64)
    random_sigma = tf.random.uniform([1], 0, 4, dtype=tf.float32)
    return random_size, random_sigma


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def make_gauss_var(name, size, sigma, c_i):
    kernel = gauss_kernel(size, sigma, c_i)
    var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
    return var


@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/models/model_two",
              help='Path to model to evaluate')
def main(model_dir):
    # Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    # To handle all of our TF Records
    with tf.device("/device:GPU:0"):
        # Create the Estimator
        real_domain_cnn = tf.estimator.Estimator(
            model_fn=real_domain_cnn_model_fn,
            model_dir=model_dir
        )

        tensors_to_log = {"logits": "2d_predictions", "learning_rate": "learning_rate", }
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[logging_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=600, start_delay_secs=300, throttle_secs=1000)

        tf.estimator.train_and_evaluate(real_domain_cnn, train_spec, eval_spec)

        # acc_pi_6, med_error = run_eval(model_dir)
        # logging.debug("ACC PI/6: " + acc_pi_6 + " | Med Error: " + str(med_error) + " | Epochs Elapsed: " + str(40))


if __name__ == "__main__":
    main()
