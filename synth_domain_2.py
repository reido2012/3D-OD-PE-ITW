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
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/synth_domain_tfrecords_new/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
NETWORK_NAME = 'resnet_v1_50'
PRETRAINED_MODEL_DIR = "/home/omarreid/selerio/final_year_project/models/test_one"
RESNET_V1_CHECKPOINT_DIR = "/home/omarreid/selerio/datasets/pre_trained_weights/resnet_v1_50.ckpt"


class SynthDomainCNN:

    def __init__(self, learning_rate, batch_size):
        # self.x_data = x_data
        # self.y_data = y_data
        # self.synth_dataset = synth_ds .
        self.learning_rate = learning_rate
        self.batch_size = batch_size


        # Initialize training dataset
        self.initialize_dataset()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    def initialize_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices(TRAINING_TFRECORDS)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size * 5))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(self.batch_size * 5)
        self.dataset = self.dataset.make_one_shot_iterator()
        self.dataset = self.dataset.get_next()

    # Initialize session
    def set_session(self, sess):
        self.sess = sess

    def build_model(self):
        self.positive_depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='positive_depth_images')
        self.negative_depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='negative_depth_images')
        self.rgb_descriptors = tf.placeholder(tf.float32, [None, 2048], name='rgb_descriptors')
        # Define placeholder for learning rate
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')

        with tf.variable_scope('synth_domain'):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
                network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)

                self.positive_depth_descriptors, endpoints = network_fn(self.positive_depth_images, reuse=tf.AUTO_REUSE)
                self.negative_depth_descriptors, endpoints = network_fn(self.negative_depth_images, reuse=tf.AUTO_REUSE)

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

        self.loss = self.similarity_loss(self.rgb_descriptors, self.positive_depth_descriptors, self.negative_depth_descriptors)

        learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=23206,
            decay_rate=0.1,
            staircase=True,
            name="learning_rate"
        )

        self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            loss=self.loss,
            global_step=self.global_step
        )

        tf.summary.scalar("Loss", self.loss)
        tf.summary.image('pos_depth', self.positive_depth_images)
        tf.summary.image('neg_depth', self.negative_depth_images)
        self.merged_summaries = tf.summary.merge_all()

    def similarity_loss(self, rgb_descriptor, pos_descriptor, neg_descriptor):
        s_pos = tf.reduce_sum(tf.square(rgb_descriptor - pos_descriptor), 1)
        s_neg = tf.reduce_sum(tf.square(rgb_descriptor - neg_descriptor), 1)

        return self.descriptor_loss(s_pos, s_neg) + REG_CONSTANT * tf.losses.get_regularization_loss()

    def descriptor_loss(self, s_pos, s_neg):
        loss = tf.maximum(0.0, TRIPLET_LOSS_MARGIN + s_pos - s_neg)
        loss = tf.reduce_mean(loss)
        return loss

        # Train model

    def train(self):

        # Define summary writer for saving log files
        self.writer = tf.summary.FileWriter('./Model/logs/', graph=tf.get_default_graph())

        # Iterate through 20000 training steps
        while not self.sess.should_stop():

            # Update globale step
            step = tf.train.global_step(self.sess, self.global_step)

            # Retrieve batch from data loader
            (rgb_descriptor, pos_depth_image, negative_depth_image), _ = self.sess.run(self.dataset)

            # Run optimization operation for current mini-batch
            fd = {self.positive_depth_images: pos_depth_image, self.negative_depth_images: negative_depth_image, self.rgb_descriptors: rgb_descriptor, self.learning_rt: self.learning_rate}
            self.sess.run(self.optim, feed_dict=fd)

            # Save summary every 100 steps
            if step % 100 == 0:
                summary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.writer.add_summary(summary, step)
                self.writer.flush()

            # Display progress every 1000 steps
            if step % 100 == 0:
                loss = self.sess.run(self.loss, feed_dict=fd)
                print("Step %d:  %.10f" % (step, loss))


def tfrecord_parser(serialized_example):
    """
        Parses a single tf.Example into image and label tensors.
    """

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'positive_depth_image': tf.FixedLenFeature([], tf.string),
            'rgb_descriptor': tf.FixedLenFeature([], tf.float32),
            'negative_depth_images': tf.FixedLenFeature([], tf.string),
            'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
            'object_class': tf.FixedLenFeature([], tf.string)
        }
    )

    pos_depth_image = convert_string_to_image(features['positive_depth_image'])

    # Get random depth image
    num_neg_depth_imgs = tf.cast(features['num_negative_depth_images'], tf.int64)
    random_idx = tf.random_uniform([], 0, num_neg_depth_imgs, dtype=tf.int64)
    neg_depth_key = "img/neg/depth/" + random_idx
    negative_depth_image = convert_string_to_image(features[neg_depth_key])
    object_class = features['object_class']
    rgb_descriptor = tf.cast(features['rgb_descriptor'], tf.float32)

    return (rgb_descriptor, pos_depth_image, negative_depth_image), object_class


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


def get_train_iterator():
    """
    Builds an input pipeline that yields batches of feature and label pairs
    """
    dataset = tf.data.TFRecordDataset(TRAINING_TFRECORDS)
    dataset = dataset_base(dataset)
    dataset = dataset.repeat(count=10)  # Train for count epochs

    iterator = dataset.make_one_shot_iterator()
    # features, labels = iterator.get_next()
    # return features, labels
    return iterator


@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/synth_models/model_test",
              help='Path to model to evaluate')
def main(model_dir):
    # Create your own input function - https://www.tensorflow.org/guide/custom_estimators
    # To handle all of our TF Records
    global MODEL_DIR
    MODEL_DIR = model_dir
    with tf.device("/device:GPU:0"):
        # record_iterator = tf.python_io.tf_record_iterator(path=TRAINING_TFRECORDS)


        # Create artificial data
        # x_data = np.pi / 2 * np.random.normal(size=[100 * 10000, 1])
        # y_data = np.sin(x_data)

        # Specify initial learning rate
        learning_rate = STARTING_LR

        # Initialize model
        model = SynthDomainCNN(learning_rate, BATCH_SIZE)

        # Specify number of training steps
        training_steps = 20000

        # Initialize TensorFlow monitored training session
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=model_dir,
                hooks=[tf.train.StopAtStepHook(last_step=training_steps)],
                save_summaries_steps=None,
                save_checkpoint_steps=500) as sess:
            # Initialize model session
            model.set_session(sess)

            # Train model
            model.train()

        # train_iterator = get_train_iterator()
        #
        #
        # next_example, next_label = train_iterator.get_next()
        #
        # loss = synth_domain_cnn_model_fn(next_example, next_label)
        #
        # global_step = tf.train.get_global_step()
        #
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate=STARTING_LR,
        #     global_step=global_step,
        #     decay_steps=23206,
        #     decay_rate=0.1,
        #     staircase=True,
        #     name="learning_rate"
        # )
        #
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #
        # training_op = optimizer.minimize(
        #     loss=loss,
        #     global_step=global_step
        # )
        #
        # tensors_to_log = {"loss": "loss", "learning_rate": "learning_rate", }
        # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
        #
        # with tf.train.MonitoredTrainingSession(checkpoint_dir=MODEL_DIR, hooks=[logging_hook],
        #                                        save_checkpoint_steps=100) as sess:
        #     while not sess.should_stop():
        #         sess.run(training_op)


if __name__ == "__main__":
    main()
