import tensorflow as tf
import json
import numpy as np
import glob
import scipy.misc
import pathlib
from nets import nets_factory, resnet_v1
NETWORK_NAME = 'resnet_v1_50'
MODEL_DIR = ""
slim = tf.contrib.slim


def main():
    with tf.device("/device:GPU:0"):
        writer = tf.python_io.TFRecordWriter("/home/omarreid/selerio/datasets/full_pose_space.tfrecords")

        for image_path in glob.glob("/home/omarreid/selerio/datasets/full_pose_space/./*/*/*_0001.png"):
            depth_image = scipy.misc.imread(image_path, mode='RGB')
            depth_image = scipy.misc.imresize(depth_image, (224, 224, 3))
            object_class = image_path.split("/")[-3]
            cad_index = image_path.split("/")[-2]
            rotation_info_str = image_path.split("/")[-1][:-9]
            rot_x, rot_y, rot_z = np.array(rotation_info_str.split("_"))[1, 3, 5]
            print("Image Path: ")
            print(image_path)

            print(f"Object Class: {object_class}")
            print(f"Cad Index: {cad_index}")
            print(f"Rotation: {(rot_x, rot_y, rot_z)}")
            return
            write_record(writer, depth_image, object_class, cad_index, rot_x, rot_y, rot_z)

        writer.close()


def write_record(record_writer, depth_image, object_class, cad_index, rot_x, rot_y, rot_z):

    depth_img_raw = depth_image.tostring()
    feature = {
        'depth_image': bytes_feature(depth_img_raw),
        'object_class': bytes_feature(object_class.encode('utf-8')),
        'cad_index': bytes_feature(cad_index.encode('utf-8')),
        'rot_x': floats_feature(rot_x),
        'rot_y': floats_feature(rot_y),
        'rot_z': floats_feature(rot_z)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    record_writer.write(example.SerializeToString())


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
