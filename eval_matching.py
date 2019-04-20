import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np
import json
import sqlite3
import cv2
from eval_metrics import get_ground_truth_rotation_matrix, get_single_examples_from_batch
from model_dataset_utils import predict_input_fn
from sklearn.neighbors import KDTree
from nets import nets_factory, resnet_v1
from math import degrees

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.WARN)

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
MODEL_DIR = ""


# @click.command()
# @click.option('--image_path', help='Path to image to retrieve model for')
def main():
    global MODEL_DIR
    model_dir = "/home/omarreid/selerio/final_year_project/models/diss_test_1"
    MODEL_DIR = model_dir
    start_eval(model_dir)


def start_eval(model_path, visualize=True):
    real_domain_cnn = tf.estimator.Estimator(
        model_fn=real_domain_cnn_model_fn_predict,
        model_dir=model_path
    )

    real_domain_predictions = real_domain_cnn.predict(input_fn=lambda: predict_input_fn(EVAL_TFRECORDS),
                                                      yield_single_examples=False)
    # If yielding single examples uncomment the line below - do this if you have a tensor/batch error (slower)
    all_model_predictions = get_single_examples_from_batch(real_domain_predictions)
    # all_model_predictions = real_domain_predictions

    correct = 0
    num_predictions = len(list(all_model_predictions))

    for counter, model_prediction in enumerate(all_model_predictions):
        if counter == 5:
            visualize = False

        # model_output = model_prediction["2d_prediction"]
        image = np.uint8(model_prediction["original_img"])
        data_id = model_prediction["data_id"].decode('utf-8')
        cad_index = model_prediction["cad_index"].decode('utf-8')
        object_class = model_prediction["object_class"].decode('utf-8')
        object_index = model_prediction["object_index"]
        # ground_truth_output = model_prediction["output_vector"]
        rgb_embedding = np.array(model_prediction["image_descriptor"])

        ground_truth_rotation_matrix, focal, viewpoint_obj = get_ground_truth_rotation_matrix(data_id, object_index)
        rot_x, rot_y, rot_z = rot_to_interval(ground_truth_rotation_matrix, 30)

        full_pose_embeddings, embeddings_info = get_synth_embeddings_at_viewpoint((rot_x, rot_y, rot_z))

        print("Number of Embeddings")
        print(len(full_pose_embeddings))

        closest_embedding = match_embeddings(rgb_embedding, full_pose_embeddings, closest_neighbours=3)

        closest_embedding_info = get_embedding_info(closest_embedding, embeddings_info)
        depth_image_path = closest_embedding_info['depth_image_path']
        depth_image_path = depth_image_path.replace("/./", "/")

        print(f"Original CAD Index: {cad_index}")
        print(f"Original Object Class: {object_class}")
        print("*" * 40)
        synth_cad_index = closest_embedding_info['cad_index']
        synth_obj_class = closest_embedding_info['object_class']
        print(f"CAD Index: {synth_cad_index}")
        print(f"Object Class: {synth_obj_class}")
        print(f"Depth Image Path: {depth_image_path}")

        if visualize:
            fig = plt.figure(figsize=(15, 15))
            ax = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)

            # Depth Image
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_COLOR)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

            ax.imshow(image)
            ax2.imshow(depth_image)

            plt.savefig("./{}_eval.jpg".format(counter))

        if synth_cad_index == object_index and synth_obj_class == object_class:
            correct += 1

        return

    top_1_accuracy = correct / float(num_predictions)

    print(f"Top 1 Accuracy: {top_1_accuracy}")


def rot_to_interval(ground_truth_rotation_matrix, interval=30):
    rot_x, rot_y, rot_z = mat2euler(ground_truth_rotation_matrix)[::-1]
    rot_x = degrees(rot_x)
    rot_y = degrees(rot_y)
    rot_z = degrees(rot_z)
    rot_x = get_closest_interval(rot_x, interval)
    rot_y = get_closest_interval(rot_y, interval)
    rot_z = get_closest_interval(rot_z, interval)

    return rot_x, rot_y, rot_z


def get_closest_interval(angle, interval):
    return round(angle / float(interval)) * interval


def get_synth_embeddings_at_viewpoint(viewpoint):
    db_path = "/home/omarreid/selerio/final_year_project/full_pose_2.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    view = (json.dumps(viewpoint),)
    print("View Point")
    print(view)

    cursor.execute('SELECT depth_embedding, image_path, object_class, cad_index FROM full_pose_space WHERE viewpoint=?',
                   view)

    all_info = cursor.fetchall()
    print("All Info")
    print(all_info)
    all_embeddings, embedding_info = reformat_info(all_info)

    return all_embeddings, embedding_info


def match_embeddings(rgb_embedding, full_pose_embeddings, closest_neighbours=3):
    tree = KDTree(full_pose_embeddings, leaf_size=40, metric="euclidean")
    dist, ind = tree.query(rgb_embedding, k=closest_neighbours)
    print(f"Indices: {ind}")
    print(f"Distance: {dist}")
    closest_index = ind[0]
    return full_pose_embeddings[closest_index]


def get_embedding_info(embedding, embeddings_info):
    embedding = tuple(embedding.astype(str))
    return embeddings_info[embedding]


def reformat_info(all_info):
    reformatted_embeddings = []
    embedding_info = {}
    for depth_embedding, image_path, object_class, cad_index in all_info:
        embedding_string = json.loads(depth_embedding)
        embedding = np.array(embedding_string, dtype=np.float)
        reformatted_embeddings.append(embedding)

        emb_key = tuple(embedding_string.astype(str))
        embedding_info[emb_key] = {"depth_image_path": image_path, "object_class": object_class,
                                   "cad_index": cad_index}

    return np.array(reformatted_embeddings), embedding_info


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix '''

    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = 1e-6
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if cy > cy_thresh:  # cos(y) not close to zero, standard form
        z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13, cy)  # atan2(sin(y), cy)
        x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21, r22)
        y = math.atan2(r13, cy)  # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def real_domain_cnn_model_fn_predict(features, labels, mode):
    """
    Real Domain CNN from 3D Object Detection and Pose Estimation paper
    """

    input_images = tf.identity(features['img'], name="input")  # Used when converting to unity

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
        network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)
        image_descriptors, endpoints = network_fn(input_images)

    image_descriptors = tf.identity(image_descriptors, name="image_descriptors")

    # Add a dense layer to get the 19 neuron linear output layer
    logits = tf.layers.dense(image_descriptors, 19,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    logits = tf.squeeze(logits, name='2d_predictions')

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(MODEL_DIR):
        tf.logging.info(
            'Ignoring RESNET50 CKPT because a checkpoint already exists in %s'
            % MODEL_DIR)

    checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    variables_to_restore = slim.get_variables_to_restore()
    tf.train.init_from_checkpoint(checkpoint_path, {v.name.split(':')[0]: v for v in variables_to_restore})

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "2d_prediction": logits,
        "image_descriptor": image_descriptors,
        "data_id": features['data_id'],
        "cad_index": features['cad_index'],
        "object_class": features['object_class'],
        "object_index": features['object_index'],
        "output_vector": features['ground_truth_output'],
        "original_img": features['normal_img']
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    main()
