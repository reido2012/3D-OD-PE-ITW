import tensorflow as tf

TFRECORDS_DIR = "/notebooks/selerio/new_tfrecords/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords", TFRECORDS_DIR + "imagenet_val.tfrecords"]
OVERFIT_TEST_TFRECORDS = "/notebooks/selerio/overfit_check.tfrecords"
EVAL_TEST_TFRECORDS =  "/notebooks/selerio/eval_check.tfrecords"
BATCH_SIZE = 50
SHUFFLE_BUFFER_SIZE = 18000
TRAIN_EPOCHS = 100
PRE_FETCH_BUFFER_SIZE = 4
NUM_CPU_CORES = 8
IMAGE_SIZE = 224 # To match ResNet dimensions 
GREYSCALE_SIZE = tf.constant(50176)
GREYSCALE_CHANNEL = tf.constant(1)
    
def dataset_base(dataset, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=NUM_CPU_CORES) #Parallelize data transformation
    dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset.prefetch(buffer_size=PRE_FETCH_BUFFER_SIZE)


def tfrecord_parser(serialized_example):
    """
    #Parses a single tf.Example into image and label tensors.
    """

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'object_image': tf.FixedLenFeature([], tf.string),
            'output_vector': tf.FixedLenFeature([19], tf.float32),
            'data_id': tf.FixedLenFeature([], tf.string),
            'object_index': tf.FixedLenFeature([], tf.int64)            
        }
    )

    # Convert Scalar String to uint8
    input_image = tf.decode_raw(features['object_image'], tf.uint8)
    input_image = tf.to_float(input_image)
     
    data_id = features['data_id']
    
    #Image is not in correct shape so 
    shape_pred = tf.cast(tf.equal(tf.size(input_image), GREYSCALE_SIZE), tf.bool)
    image_shape = tf.cond(shape_pred, lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 1]), 
                          lambda: tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3]))

    input_image = tf.reshape(input_image, image_shape)

    channel_pred = tf.cast(tf.equal(tf.shape(input_image)[2], GREYSCALE_CHANNEL), tf.bool)
    input_image = tf.cond(channel_pred, lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
    input_image = tf.reshape(input_image, (224, 224, 3))

    output_vector = tf.cast(features['output_vector'], tf.float32)
    
    model_input = {
        "img": input_image, 
        "object_index": tf.cast(features['object_index'], tf.int32),
        "data_id": data_id,
        "ground_truth_output": output_vector
    }
    
    return model_input, output_vector

def train_input_fn(tfrecords):
    """
    Builds an input pipeline that yields batches of feature and label pairs
    """
    dataset = tf.data.TFRecordDataset(tfrecords).repeat(count=TRAIN_EPOCHS) #Train for 300 epochs
    dataset = dataset_base(dataset)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def predict_input_fn(tfrecords):
    dataset = tf.data.TFRecordDataset(tfrecords).repeat(count=1)
    dataset = dataset_base(dataset, shuffle=False)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
        
def eval_input_fn(tfrecords):
    """
    Builds an input pipeline that yields batches of feature and label pairs for evaluation 
    """
    dataset = tf.data.TFRecordDataset(tfrecords).repeat(count=1)
    dataset = dataset_base(dataset, shuffle=False)
    
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


"""
class RealDomainCNN:
    def __init__(self):
        self.tfrecords_dir = "/notebooks/selerio/pascal3d_tfrecords/"
        self.training_tfrecords = [self.tfrecords_dir + "imagenet_train.tfrecords", self.tfrecords_dir + "pascal_train.tfrecords"]
        self.eval_tfrecords = [self.tfrecords_dir + "pascal_val.tfrecords", self.tfrecords_dir + "imagenet_val.tfrecords"]
        self.predict_test_tfrecords = "/notebooks/selerio/test.tfrecords"
        self.overfit_test_tfrecords = "/notebooks/selerio/overfit_test.tfrecords"
        self.eval_test_tfrecords =  "/notebooks/selerio/eval_.tfrecords"
        self.varied_test_tfrecords =  "/notebooks/selerio/varied_test.tfrecords"
        self.batch_size = 50
        self.num_cpu_cores = 8
        self.image_size = 224 # To match ResNet dimensions 
        self.resnet_v2_checkpoint_dir = "/notebooks/selerio/pre_trained_weights/resnet_v2_50.ckpt"
        self.greyscale_size = tf.constant(50176)
        self.greyscale_channel = tf.constant(1)
        
        #META PARAMETERS 
        self.alpha = 1
        self.beta = math.exp(-5)
        self.gamma = math.exp(-3)
        self.triplet_loss_margin = 1
        self.huber_loss_delta = 0.0035
        self.starting_lr = 0.0008
        
   def real_domain_cnn_model_fn(self, features, labels, mode):
        #Real Domain CNN from 3D Object Detection and Pose Estimation paper
    
        #with tf.device('/gpu:0'):
        # Use Feature Extractor to extract the image descriptors from the images
        #Could use mobilenet for a smaller model
        network_name = 'resnet_v2_50'

        # Retrieve the function that returns logits and endpoints - ResNet was pretrained on ImageNet
        network_fn = nets_factory.get_network_fn(network_name, num_classes=None, is_training=False)

        # Retrieve the model scope from network factory
        model_scope = nets_factory.arg_scopes_map[network_name]

        # Retrieve the (pre) logits and network endpoints (for extracting activations)
        # Note: endpoints is a dictionary with endpoints[name] = tf.Tensor
        image_descriptors, endpoints = network_fn(features)

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
        logits = tf.layers.dense(image_descriptors, 19,  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0025))
        logits = tf.squeeze(logits, name='2d_predictions')

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "2d_prediction": logits,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # create a pose_loss function so that we can ge tthe loss
        loss = self.pose_loss(labels, logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()

            learning_rate = tf.train.exponential_decay(
                learning_rate=self.starting_lr, 
                global_step=global_step, 
                decay_steps= 20000, 
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

    def pose_loss(self, labels, logits):
        return self.projection_loss(labels[:, :16], logits[:, :16]) + ALPHA * self.dimension_loss(labels[:, 16:], logits[:, 16:]) + BETA * tf.losses.get_regularization_loss()

    def projection_loss(self, bbox_labels, logits_bbox):
        return tf.losses.huber_loss(bbox_labels, logits_bbox, delta=HUBER_LOSS_DELTA)

    def dimension_loss(self, dimension_labels, dimension_logits):
        return tf.losses.huber_loss(dimension_labels, dimension_logits, delta=HUBER_LOSS_DELTA)

    def dataset_base(self, dataset, shuffle=True):
        if shuffle:
            dataset = dataset.shuffle(buffer_size=18000)

        dataset = dataset.map(map_func=tfrecord_parser, num_parallel_calls=self.num_cpu_cores) #Parallelize data transformation
        dataset.apply(tf.contrib.data.ignore_errors())
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset.prefetch(buffer_size=4)
        
    def train_input_fn(self):
        #Builds an input pipeline that yields batches of feature and label pairs
        
        dataset = tf.data.TFRecordDataset(self.training_tfrecords).repeat(count=100) #Train for 300 epochs
        dataset = dataset_base(dataset)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
   
    def predict_input_fn(self):
        dataset = tf.data.TFRecordDataset(self.eval_test_tfrecords).repeat(count=1)
        dataset = dataset_base(dataset, shuffle=False)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
        
    def eval_input_fn(self):
        #Builds an input pipeline that yields batches of feature and label pairs for evaluation 
        
        dataset = tf.data.TFRecordDataset(self.eval_tfrecords).repeat(count=1)
        dataset = dataset_base(dataset)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
   
    def tfrecord_parser(self, serialized_example):
        #Parses a single tf.Example into image and label tensors.

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'object_image': tf.FixedLenFeature([], tf.string),
                'output_vector': tf.FixedLenFeature([19], tf.float32)
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
        # data_id = features['data_id']
        # obj_index = features['object_counter']
        # labels = (output_vector, data_id, obj_index)
        return input_image, output_vector
        
    def get_estimator_model(self, model_path):
        return tf.estimator.Estimator(
            model_fn=self.real_domain_cnn_model_fn, 
            model_dir=model_path
        )
        
    def train_model(self, estimator_model):
        tensors_to_log = {"logits": "2d_predictions", "learning_rate": "learning_rate",}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        estimator_model.train(
          input_fn=self.train_input_fn,
          hooks=[logging_hook]
        )
        
    def evaluate_model(self, estimator_model):
        return estimator_model.evaluate(input_fn=eval_input_fn)
       
"""