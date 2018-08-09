# Imports
import math
import tensorflow as tf
from nets import nets_factory
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)


TFRECORDS_DIR = "/notebooks/selerio/pascal3d_tfrecords/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords"]
BATCH_SIZE = 50
NUM_CPU_CORES = 8
IMAGE_SIZE = 224 # To match ResNet dimensions 
RESNET_V2_CHECKPOINT_DIR = "/notebooks/selerio/pre_trained_weights/resnet_v2_50.ckpt"

#META PARAMETERS 
ALPHA = 1
BETA = tf.exp(-5)
GAMMA = tf.exp(-3)
TRIPLET_LOSS_MARGIN = 1
HUBER_LOSS_DELTA = 0.01
STARTING_LR = 0.0001
"""
class RealDomainCNN:
    
    def __init__(self, batch_size, num_cpu_cores):
        self.num_cpu_cores = num_cpu_cores
        self.batch_size = batch_size
        self.image_dimensions = 224
"""

def real_domain_cnn_model_fn(features, labels, mode):
    """
    Real Domain CNN from 3D Object Detection and Pose Estimation paper
    """

    # Use Feature Extractor to extract the image descriptors from the images
    network_name = 'resnet_v2_50'
    
    # Retrieve the function that returns logits and endpoints - ResNet was pretrained on ImageNet
    network_fn = nets_factory.get_network_fn(network_name, num_classes=None, is_training=False)
    
    # Retrieve the model scope from network factory
    model_scope = nets_factory.arg_scopes_map[network_name]
    
    # Retrieve the (pre) logits and network endpoints (for extracting activations)
    # Note: endpoints is a dictionary with endpoints[name] = tf.Tensor
    image_descriptors, endpoints = network_fn(features)

    #Potentially apply dropout

    # Find the checkpoint file
    checkpoint_path = RESNET_V2_CHECKPOINT_DIR
       
    if tf.gfile.IsDirectory(checkpoint_path):
        print("Found Directory")
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    # Load pre-trained weights into the model
    variables_to_restore = slim.get_variables_to_restore()
    restore_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

    # Start the session and load the pre-trained weights
    sess = tf.Session()
    restore_fn(sess)
    

    #Add a dense layer to get the 19 neuron linear output layer
    logits = tf.layers.dense(image_descriptors, 19)
    logits = tf.squeeze(logits, name='2d_predictions')
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "logits": logits,
    }
        
    # create a pose_loss function so that we can ge tthe loss
    #loss = pose_loss()
    
    #Testing so use huber for now
    loss = tf.losses.huber_loss(labels, logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        
        learning_rate = tf.train.exponential_decay(
            learning_rate=STARTING_LR, 
            global_step=global_step, 
            decay_steps=100, 
            decay_rate=0.001
        )
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step
        )
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    

def pose_loss(labels, logits):
    return projection_loss(labels[:, :16], logits[:, :16]) + ALPHA * dimension_loss(labels[:, 16:], logits:, [16:]) + BETA * regularization

def projection_loss(bbox_labels, logits_bbox)
    return tf.losses.huber_loss(bbox_labels, logits_bbox, delta=HUBER_LOSS_DELTA)

def dimension_loss(dimension_labels, dimension_logits):
    return tf.losses.huber_loss(dimension_labels, dimension_logits, delta=HUBER_LOSS_DELTA) / 16

def train_input_fn():
    """
    Builds an input pipeline that yields batches of feature and label pairs
    """
    dataset = tf.data.TFRecordDataset(TRAINING_TFRECORDS).repeat()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=NUM_CPU_CORES) #Parallelize data transformation
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=2)
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
            'output_vector': tf.FixedLenFeature([19], tf.float32)
        }
    )

    # Convert Scalar String to Image
    input_image = tf.decode_raw(features['object_image'], tf.uint8)
    print(input_image)
    # Reshape Image
    input_image = tf.to_float(input_image)
    image_shape = tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3])
    input_image = tf.reshape(input_image, image_shape)
    #turn back into floats
#    print(features.items())
 #   print(features['output_vector'])
    output_vector = tf.cast(features['output_vector'], tf.float32)

    #Add Blur 
    #blurred_image = utils.blur_image(input_image)

    return input_image, output_vector


def main(unused_argv):
    #Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    #To handle all of our TF Records
    
    # Create the Estimator
    real_domain_cnn = tf.estimator.Estimator(
        model_fn=real_domain_cnn_model_fn, 
        model_dir="/notebooks/selerio/real_domain_convnet"
    )
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"logits": "2d_predictions"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    real_domain_cnn.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook]
    )
    


if __name__ == "__main__":
    tf.app.run()