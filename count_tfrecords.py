import pascal3d
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

reconstructed_records = []
record_iterator_train1 = list(tf.python_io.tf_record_iterator(path="/notebooks/selerio/pascal3d_tfrecords/imagenet_train.tfrecords"))
record_iterator_train2 = list(tf.python_io.tf_record_iterator(path="/notebooks/selerio/pascal3d_tfrecords/pascal_train.tfrecords"))
record_iterator_train3 = list(tf.python_io.tf_record_iterator(path="/notebooks/selerio/pascal3d_tfrecords/imagenet_val.tfrecords"))
all_iterators=record_iterator_train1+record_iterator_train2+record_iterator_train3
print(len(all_iterators))
"""
for string_record in all_iterators:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)    
    
    output_vector = example.features.feature['output_vector'].float_list.value
    
    img_string = (example.features.feature['object_image']
                                  .bytes_list
                                  .value[0])

    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((224, 224, -1))
    
    reconstructed_records.append((reconstructed_img, output_vector))
    
print(len(reconstructed_records))
"""