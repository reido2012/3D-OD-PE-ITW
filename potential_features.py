import tensorflow as tf

FEATURES_3 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_4 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_5 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/4': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_6 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/4': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/5': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_7 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/4': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/5': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/6': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_8 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/4': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/5': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/6': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/7': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_9 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/4': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/5': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/6': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/7': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/8': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

FEATURES_10 = {
    'positive_depth_image': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/0': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/1': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/2': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/3': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/4': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/5': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/6': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/7': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/8': tf.FixedLenFeature([], tf.string),
    'neg/depth/img/9': tf.FixedLenFeature([], tf.string),
    'rgb_descriptor': tf.FixedLenFeature([2048], tf.float32),
    'num_negative_depth_images': tf.FixedLenFeature([], tf.int64),
    'object_class': tf.FixedLenFeature([], tf.string)
}

KEYS = ["neg/depth/img/9", "neg/depth/img/8", "neg/depth/img/7", "neg/depth/img/6", "neg/depth/img/5",
        "neg/depth/img/4", "neg/depth/img/3", "neg/depth/img/2", "neg/depth/img/1", "neg/depth/img/0"]

FEATURES_LIST = [FEATURES_10, FEATURES_9, FEATURES_8, FEATURES_7, FEATURES_6, FEATURES_5, FEATURES_4, FEATURES_3]
