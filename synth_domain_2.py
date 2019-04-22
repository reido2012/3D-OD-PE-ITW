import tensorflow as tf
import click
import glob
import cv2
import subprocess
import numpy as np
import os.path as osp
from nets import nets_factory, resnet_v1
from itertools import chain

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
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords",
                      TFRECORDS_DIR + "imagenet_val.tfrecords"]

EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
EVAL_ITERATOR = tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "pascal_val.tfrecords")
DATASET_DIR = osp.expanduser('/home/omarreid/selerio/datasets/PASCAL3D+_release1.1')
OBJ_DIR = DATASET_DIR + "/OBJ/"
IMAGE_SHAPE = tf.TensorShape([None, 224, 224, 3])
NETWORK_NAME = 'resnet_v1_50'
PRETRAINED_MODEL_DIR = "/home/omarreid/selerio/final_year_project/models/test_one"
RESNET_V1_CHECKPOINT_DIR = "/home/omarreid/selerio/datasets/pre_trained_weights/resnet_v1_50.ckpt"


class SynthDomainCNN:

    def __init__(self, learning_rate, batch_size):

        self.positive_depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.negative_depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.rgb_descriptors = tf.placeholder(tf.float32, [None, 2048])
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.all_iterators = list(chain(tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "imagenet_train.tfrecords"),
                                   tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "pascal_train.tfrecords"),
                                   tf.python_io.tf_record_iterator(path=TFRECORDS_DIR + "imagenet_val.tfrecords")))

        # Initialize training dataset
        self.initialize_dataset()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    def create_element(self, string_record):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        features = example.features.feature

        rgb_descriptor = np.array(features['rgb_descriptor'].float_list.value)
        rgb_descriptor = rgb_descriptor.astype(np.float32)

        object_class = features['object_class'].bytes_list.value[0].decode("utf-8")

        data_id = features['data_id'].bytes_list.value[0].decode("utf-8")

        cad_index = features['cad_index'].bytes_list.value[0].decode("utf-8")

        img_string = example.features.feature['positive_depth_image'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        pos_depth_image = img_1d.reshape((224, 224, 3))
        pos_depth_image = pos_depth_image.astype(np.float32)

        print(f"RGB Descriptor: {rgb_descriptor}")
        print(f"RGB Descriptor Shape: {rgb_descriptor.shape}")

        print(f"Object Class: {object_class}")
        print(f"Data ID: {data_id}")
        print(f"CAD Index: {cad_index}")

        all_model_paths = list(glob.glob(OBJ_DIR + "*/*.obj"))  # All classes, all objs

        pos_obj = OBJ_DIR + str(object_class) + "/" + str(cad_index) + ".obj"

        print(f"Pos Obj: {pos_obj}")

        random_model_obj_path = np.random.choice(all_model_paths)
        while pos_obj == random_model_obj_path:
            random_model_obj_path = np.random.choice(all_model_paths)

        random_cad_index = random_model_obj_path.split("/")[-1][:-4]

        print(f"Random Obj Model: {random_model_obj_path}")
        print(f"Random Cad Index: {random_cad_index}")

        depth_path = "/home/omarreid/selerio/datasets/random_render/0" + "/" + data_id + "_" + str(
            random_cad_index) + "_0001.png"

        command = "blender -noaudio --background --python ./blender_render.py -- --specific_viewpoint=True " \
                  "--cad_index=" + random_cad_index + " --obj_id=" + data_id + " --radians=True " \
                                                                               "--viewpoint=" + str(
            0) + "," + str(
            90) + "," + str(
            0) + " --bbox=" + str(
            1) + "," + str(
            1) + "," + str(
            1) + " --output_folder /home/omarreid/selerio/datasets/random_render/0" + " "

        full_command = command + random_model_obj_path

        try:
            subprocess.run(full_command.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            raise e

        print("Command: " + full_command)
        negative_depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR)
        negative_depth_image = cv2.cvtColor(negative_depth_image, cv2.COLOR_BGR2RGB)
        negative_depth_image = negative_depth_image.astype(np.float32)

        single_feature = (rgb_descriptor, pos_depth_image, negative_depth_image)
        single_label = str(object_class)

        return single_feature, single_label

    def synth_dataset_generator(self):
        for string_record in self.all_iterators:
            single_feature, single_label = self.create_element(string_record)
            print("Single Feature")
            print(single_feature)

            print("Single Label")
            print(single_label)

            yield single_feature, single_label

    def initialize_dataset(self):

        self.dataset = tf.data.Dataset.from_generator(
            self.synth_dataset_generator,
            output_types=((tf.float32, tf.float32, tf.float32), tf.string),
            output_shapes=((tf.TensorShape([None, 2048]), IMAGE_SHAPE, IMAGE_SHAPE), tf.TensorShape([]))
        )
        print("Dataset Output Shapes")
        print(self.dataset.output_shapes)
        self.dataset = self.dataset.apply(tf.contrib.data.ignore_errors())
        self.dataset = self.dataset.batch(50)
        self.dataset = self.dataset.repeat(count=10)  # Train for count epochs
        self.dataset = self.dataset.make_one_shot_iterator()

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
                                          {v.name.split(':')[0].replace('synth_domain/', '', 1): v.name.split(':')[0]
                                           for v
                                           in variables_to_restore})

        self.loss = self.similarity_loss(self.rgb_descriptors, self.positive_depth_descriptors,
                                         self.negative_depth_descriptors)

        learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=28000,
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

    def train(self, model_dir):

        # Define summary writer for saving log files
        self.writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())

        # Iterate through 20000 training steps
        while not self.sess.should_stop():

            # Update globale step
            step = tf.train.global_step(self.sess, self.global_step)

            # Retrieve batch from data loader
            (rgb_descriptor, pos_depth_image, negative_depth_image), _ = self.dataset.get_next()

            # Run optimization operation for current mini-batch
            fd = {self.positive_depth_images: pos_depth_image, self.negative_depth_images: negative_depth_image,
                  self.rgb_descriptors: rgb_descriptor, self.learning_rt: self.learning_rate}
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


@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/synth_models/test_2",
              help='Path to model to evaluate')
def main(model_dir):
    global MODEL_DIR
    MODEL_DIR = model_dir
    with tf.device("/device:GPU:0"):
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
                config=tf.ConfigProto(allow_soft_placement=True),
                save_summaries_steps=None,
                save_checkpoint_steps=500) as sess:
            # Initialize model session
            model.set_session(sess)

            # Train model
            model.train(model_dir)


if __name__ == "__main__":
    main()
