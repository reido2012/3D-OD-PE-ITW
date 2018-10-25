# IMPORTS
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
# freeze_graph "screenshots" the graph
from tensorflow.python.tools import freeze_graph

# optimize_for_inference lib optimizes this frozen graph
from tensorflow.python.tools import optimize_for_inference_lib

# os and os.path are used to create the output file where we save our frozen graphs
import os
import os.path as path
RESNET_V1_CHECKPOINT_DIR = "/notebooks/selerio/pre_trained_weights/resnet_v1_50.ckpt"
# EXPORT GAPH FOR UNITY
def export_model(input_node_names, output_node_name):
    # creates the 'out' folder where our frozen graphs will be saved
    base_path = "/notebooks/selerio/frozen_graphs/coreml/"
    if not path.exists(base_path):
        os.mkdir(base_path)

    # an arbitrary name for our graph
    GRAPH_NAME = 'pose_estimator'
    input_graph_path = '/notebooks/selerio/pose_estimation_models/the_one_three/graph.pbtxt'
    
#     # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state('/notebooks/selerio/pose_estimation_models/the_one_three/')
    input_checkpoint = checkpoint.model_checkpoint_path
    with tf.device('/cpu:0'):
        #Create the chkp from the meta
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta' )
            saver.restore(sess, input_checkpoint)


#              # GRAPH SAVING - '.bytes'
            freeze_graph.freeze_graph(input_graph_path, None, False,
                                      input_checkpoint, output_node_name,
                                      "save/restore_all", "save/Const:0",
                                      base_path + 'frozen_' + GRAPH_NAME + '.bytes', True, "", variable_names_blacklist="global_step")

            # GRAPH OPTIMIZING
            input_graph_def = tf.GraphDef()
            with tf.gfile.Open(base_path + 'frozen_' + GRAPH_NAME + '.bytes', "rb") as f:
                input_graph_def.ParseFromString(f.read())

            output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                    input_graph_def, input_node_names, [output_node_name],
                    tf.float32.as_datatype_enum)

            with tf.gfile.FastGFile(base_path + 'optimized_' + GRAPH_NAME + '.bytes', "wb") as f:
                f.write(output_graph_def.SerializeToString())

            print("graph saved!")

            print("graph saved!")

            
            
def export_model_2(input_node_names, output_node_name):
    import tensorflow as tf
    from nets import nets_factory
    from tensorflow.python.framework import graph_util
    import sys
    slim = tf.contrib.slim
    
    checkpoint = tf.train.get_checkpoint_state('/notebooks/selerio/pose_estimation_models/the_one_four/')
    input_checkpoint = checkpoint.model_checkpoint_path
    
    with tf.Graph().as_default() as graph:

        images = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32, name='image_placeholder')
        
        network_name = 'resnet_v1_50'
    
        # Retrieve the function that returns logits and endpoints - ResNet was pretrained on ImageNet
        network_fn = nets_factory.get_network_fn(network_name, num_classes=None, is_training=False)

        # Retrieve the model scope from network factory
        model_scope = nets_factory.arg_scopes_map[network_name]
        image_descriptors, endpoints = network_fn(images)
        
#         checkpoint_path = RESNET_V1_CHECKPOINT_DIR
#         checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

#         # Load pre-trained weights into the model
#         variables_to_restore = slim.get_variables_to_restore()
#         restore_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
    
        image_descriptors = tf.layers.dropout(image_descriptors, rate=0.1, training=False)
        #Add a dense layer to get the 19 neuron linear output layer
        logits = tf.layers.dense(image_descriptors, 19,  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
        logits = tf.squeeze(logits, name='2d_predictions')

        #Setup graph def
        input_graph_def = graph.as_graph_def()
        output_graph_name = "/notebooks/selerio/frozen_graphs/coreml/pose_estimator.pb"

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta' )
            saver.restore(sess, input_checkpoint)
#             restore_fn(sess)

            #Exporting the graph
            print ("Exporting graph...")
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_name.split(","))

            with tf.gfile.GFile(output_graph_name, "wb") as f:
                f.write(output_graph_def.SerializeToString())
    
export_model_2(["input"], '2d_predictions')