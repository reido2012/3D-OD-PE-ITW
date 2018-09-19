# Imports
import math
import tensorflow as tf
import numpy as np
from nets import nets_factory
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)


TFRECORDS_DIR = "/notebooks/selerio/new_tfrecords/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",  TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
OVERFIT_TEST_TFRECORDS = "/notebooks/selerio/overfit_check.tfrecords"
EVAL_TEST_TFRECORDS =  "/notebooks/selerio/eval_check.tfrecords"
BATCH_SIZE = 50
NUM_CPU_CORES = 8
IMAGE_SIZE = 224 # To match ResNet dimensions 
RESNET_V1_CHECKPOINT_DIR = "/notebooks/selerio/pre_trained_weights/resnet_v1_50.ckpt"
GREYSCALE_SIZE = tf.constant(50176)
GREYSCALE_CHANNEL = tf.constant(1)
#META PARAMETERS 
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
    #with tf.device('/gpu:0'):
    # Use Feature Extractor to extract the image descriptors from the images
    #Could use mobilenet for a smaller model
    network_name = 'resnet_v1_50'
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # Retrieve the function that returns logits and endpoints - ResNet was pretrained on ImageNet
    network_fn = nets_factory.get_network_fn(network_name, num_classes=None, is_training=True)

    # Retrieve the model scope from network factory
    model_scope = nets_factory.arg_scopes_map[network_name]
    
    # Retrieve the (pre) logits and network endpoints (for extracting activations)
    # Note: endpoints is a dictionary with endpoints[name] = tf.Tensor

    features = tf.identity(features, name="input") # Used when converting to unity
  
    image_descriptors, endpoints = network_fn(features)
#     image_descriptors = tf.layers.batch_normalization(image_descriptors, training=is_training)
    
    # Find the checkpoint file
    checkpoint_path = RESNET_V1_CHECKPOINT_DIR

    if tf.gfile.IsDirectory(checkpoint_path):
        print("Found Directory")
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    # Load pre-trained weights into the model
    variables_to_restore = slim.get_variables_to_restore()
    restore_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

   # Start the session and load the pre-trained weights
    sess = tf.Session()
    restore_fn(sess)
    
#     image_descriptors = tf.layers.dropout(image_descriptors, rate=0.1, training=is_training)
    #Add a dense layer to get the 19 neuron linear output layer
    logits = tf.layers.dense(image_descriptors, 19,  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
#     logits = tf.Print(logits, [logits], "Pre Squeeeze Logits")
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
            decay_steps=26206, #15000
            decay_rate=0.1,
            staircase=True,
            name="learning_rate"
        )
    
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step
        )

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
def pose_loss(labels, logits):
#     logits = tf.Print(logits, [logits], "Logits")
    return projection_loss(labels[:, :16], logits[:, :16]) +  dimension_loss(labels[:, 16:], logits[:, 16:]) + BETA * tf.losses.get_regularization_loss()

def projection_loss(bbox_labels, logits_bbox):
    return tf.losses.huber_loss(bbox_labels, logits_bbox, delta=HUBER_LOSS_DELTA)

def dimension_loss(dimension_labels, dimension_logits):
    return tf.losses.huber_loss(dimension_labels, dimension_logits, delta=HUBER_LOSS_DELTA)

def dataset_base(dataset, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=26206)
    
    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=NUM_CPU_CORES) #Parallelize data transformation
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset.prefetch(buffer_size=4)

def train_input_fn():
    """
    Builds an input pipeline that yields batches of feature and label pairs
    """
    dataset = tf.data.TFRecordDataset(TRAINING_TFRECORDS).repeat(count=10) #Train for count epochs
    dataset = dataset_base(dataset)
    
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def predict_input_fn():
    dataset = tf.data.TFRecordDataset(EVAL_TEST_TFRECORDS).repeat(count=1)
    dataset = dataset_base(dataset, shuffle=False)
    
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def eval_input_fn():
    """
    Builds an input pipeline that yields batches of feature and label pairs for evaluation 
    """
    dataset = tf.data.TFRecordDataset(EVAL_TFRECORDS).repeat(count=1)
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
#             'blur': tf.FixedLenFeature([], tf.int64)
            #'data_id': tf.FixedLenFeature([], tf.string),
            #'object_counter': tf.FixedLenFeature([], tf.int64)            
        }
    )

    # Convert Scalar String to uint8
    input_image = tf.decode_raw(features['object_image'], tf.uint8)
    input_image = tf.to_float(input_image)
    
    #Image is not in correct shape so 
    shape_pred = tf.cast(tf.equal(tf.size(input_image), GREYSCALE_SIZE), tf.bool)
    image_shape = tf.cond(shape_pred, lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 1]), 
                          lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3]))
    
    input_image = tf.reshape(input_image, image_shape)
    
    channel_pred = tf.cast(tf.equal(tf.shape(input_image)[2], GREYSCALE_CHANNEL), tf.bool)
    input_image = tf.cond(channel_pred, lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
    input_image = tf.reshape(input_image, (224, 224, 3))
    output_vector = tf.cast(features['output_vector'], tf.float32)

    return input_image, output_vector
@click.command()
@click.option('--model_dir', default="/notebooks/selerio/pose_estimation_models/the_one_three", help='Path to model to evaluate')       
def main(model_dir):
    #Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    #To handle all of our TF Records
    
    # Create the Estimator
    real_domain_cnn = tf.estimator.Estimator(
        model_fn=real_domain_cnn_model_fn, 
        model_dir=model_dir
    )
    
    #"""
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"logits": "2d_predictions", "learning_rate": "learning_rate",}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    real_domain_cnn.train(
      input_fn=train_input_fn,
      hooks=[logging_hook]
    )
   #""" 

#     eval_results = real_domain_cnn.evaluate(input_fn=eval_input_fn)
#     print(eval_results)
    

#     predictions = real_domain_cnn.predict(input_fn=predict_input_fn)
#     print("Predictions")
#     print(list(predictions))



if __name__ == "__main__":
    tmain()