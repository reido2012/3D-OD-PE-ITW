import tensorflow as tf

TFRECORDS_DIR = "/notebooks/selerio/"
TRAINING_TFRECORDS = [TFRECORDS_DIR + "imagenet_train.tfrecords", TFRECORDS_DIR + "pascal_train.tfrecords"]
EVAL_TFRECORDS = [TFRECORDS_DIR + "pascal_val.tfrecords"]
BATCH_SIZE = 50
SHUFFLE_BUFFER_SIZE = 18000
PRE_FETCH_BUFFER_SIZE = 4
TRAIN_EPOCHS = 100
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
            'object_class': tf.FixedLenFeature([], tf.string),
            'cad_index': tf.FixedLenFeature([], tf.string),
            'object_index': tf.FixedLenFeature([], tf.int64),
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
    normal_img = input_image
    input_image = tf.image.per_image_standardization(input_image)
    output_vector = tf.cast(features['output_vector'], tf.float32)

    model_input = {
        "data_id": data_id,
        "object_index": features['object_index'],
        "img": input_image,
        "normal_img": normal_img,
        "cad_index": features['cad_index'],
        "object_class": features['object_class'],
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
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset_base(dataset, shuffle=False)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def eval_input_fn(tfrecords):
    """
    Builds an input pipeline that yields batches of feature and label pairs for evaluation 
    """
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset_base(dataset, shuffle=False)
    
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
