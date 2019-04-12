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
            depth_image_path = prediction["depth_image_paths"]

            object_class = depth_image_path.split("/")[-3]
            cad_index = depth_image_path.split("/")[-2]
            rotation_info_str = depth_image_path.split("/")[-1][:-9]
            rot_x, rot_y, rot_z = np.array(rotation_info_str.split("_"))[1, 3, 5]

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
    depth_image = convert_string_to_image(tf.read_file(depth_image_path), standardize=False)
    print("Depth Image Path")
    depth_image_path = tf.convert_to_tensor(depth_image_path, dtype=tf.string)
    print(depth_image_path)
    print(depth_image_path.shape)

    return depth_image, depth_image_path


def predict_input_fn(path_ds):
    dataset = path_ds.map(lambda x: record_maker(x))
    dataset = dataset.batch(batch_size=50)
    iterator = dataset.make_one_shot_iterator()
    features, image_paths = iterator.get_next()
    return features, image_paths


def synth_domain_cnn_model_fn_predict(features, labels, mode):
    depth_images = features
    depth_image_paths = labels

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
        "depth_image_paths": depth_image_paths
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def convert_string_to_image(image_string, standardize=True):
    """
    Converts image string extracted from TFRecord to an image

    :param image_string: String that represents an image
    :param standardize: If the image should be standardized or not
    :return: The image represented by the string
    """
    greyscale_size = tf.constant(50176)
    greyscale_channel = tf.constant(1)

    image = tf.decode_raw(image_string, tf.uint8)
    image = tf.to_float(image)

    # Image is not in correct shape so
    shape_pred = tf.cast(tf.equal(tf.size(image), greyscale_size), tf.bool)
    image_shape = tf.cond(shape_pred, lambda: tf.stack([224, 224, 1]),
                          lambda: tf.stack([224, 224, 3]))

    print("Within Convert String")
    input_image = tf.reshape(image, image_shape)

    channel_pred = tf.cast(tf.equal(tf.shape(input_image)[2], greyscale_channel), tf.bool)
    input_image = tf.cond(channel_pred, lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
    input_image = tf.reshape(input_image, (224, 224, 3))

    if standardize:
        input_image = tf.image.per_image_standardization(input_image)

    return input_image


if __name__ == '__main__':
    main("full_pose_descriptors", "/home/omarreid/selerio/final_year_project/synth_models/model_three")
