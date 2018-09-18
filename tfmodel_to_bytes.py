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

# EXPORT GAPH FOR UNITY
def export_model(input_node_names, output_node_name):
    # creates the 'out' folder where our frozen graphs will be saved
    base_path = "/notebooks/selerio/frozen_graphs/1.4.0/"
    if not path.exists(base_path):
        os.mkdir(base_path)

    # an arbitrary name for our graph
    GRAPH_NAME = 'pose_estimator'
    input_graph_path = '/notebooks/selerio/model_1.4.0/graph.pbtxt'
    
#     # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state('/notebooks/selerio/model_1.4.0/')
    input_checkpoint = checkpoint.model_checkpoint_path
    with tf.device('/cpu:0'):
        #Create the chkp from the meta
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta' )
            saver.restore(sess, input_checkpoint)


            # GRAPH SAVING - '.bytes'
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

# IN MAIN WITHIN TF SESSION SCOPE
export_model(["input"], '2d_predictions')