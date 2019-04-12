import tensorflow as tf
import json
import numpy as np
from nets import nets_factory, resnet_v1
NETWORK_NAME = 'resnet_v1_50'
MODEL_DIR = ""
slim = tf.contrib.slim


def main(json_file_name, model_dir):
    global MODEL_DIR
    MODEL_DIR = model_dir

    full_pose_space_db = dict({})

    with tf.device("/device:GPU:0"):
        synth_domain_cnn = tf.estimator.Estimator(
            model_fn=synth_domain_cnn_model_fn_predict,
            model_dir=model_dir
        )

        filename_dataset = tf.data.Dataset.list_files("/home/omarreid/selerio/datasets/full_pose_space/*/*/*_0001.png")
        all_model_predictions = synth_domain_cnn.predict(input_fn=lambda: predict_input_fn(filename_dataset))

        for counter, prediction in enumerate(all_model_predictions):
            depth_emb = tuple(prediction["depth_embeddings"].squeeze())
            object_class = prediction["object_class"]
            cad_index = prediction["cad_index"]
            rot_x = prediction["rot_x"]
            rot_y = prediction["rot_y"]
            rot_z = prediction["rot_z"]
            depth_image_path = prediction["depth_image_path"]

            descriptor_info = {
                "cad_index": cad_index,
                "object_class": object_class,
                "depth_image_path": depth_image_path
            }

            viewpoint = (rot_x, rot_y, rot_z)

            if viewpoint in full_pose_space_db:
                full_pose_space_db[viewpoint][depth_emb] = descriptor_info
            else:
                full_pose_space_db[viewpoint] = {depth_emb: descriptor_info}

    with open(json_file_name + '.json', 'w') as fp:
        json.dump(full_pose_space_db, fp, indent=4)


def record_maker(depth_image_path):
    depth_image = tf.image.decode_png(tf.read_file(depth_image_path))

    object_class = tf.string_split([depth_image_path], "/").values[-3]
    cad_index = tf.string_split([depth_image_path], "/").values[-2]
    rotation_info_str = tf.string_split([depth_image_path], "/").values[-1][:-9]
    rot_x, rot_y, rot_z = np.array(tf.string_split([rotation_info_str], "_").values)[1, 3, 5]
    return (depth_image, object_class, cad_index),  (rot_x, rot_y, rot_z, depth_image_path)


def predict_input_fn(path_ds):
    dataset = path_ds.map(lambda x: record_maker(x))
    iterator = dataset.make_one_shot_iterator()
    features, orientation = iterator.get_next()
    return features, orientation


def synth_domain_cnn_model_fn_predict(features, labels, mode):
    depth_images, object_class, cad_index = features
    rot_x, rot_y, rot_z, depth_image_path = labels

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
        # Generate predictions (for PREDICT and EVAL mode)
        "depth_embeddings": depth_descriptors,
        "object_class": object_class,
        "cad_index": cad_index,
        "rot_x": rot_x,
        "rot_y": rot_y,
        "rot_z": rot_z,
        "depth_image_path": depth_image_path
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    main("full_pose_descriptors", "/home/omarreid/selerio/final_year_project/synth_models/model_three")
