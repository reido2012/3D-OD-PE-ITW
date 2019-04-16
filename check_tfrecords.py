
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pascal3d
from PIL import Image
import glob
import os.path as osp
import numpy as np
import skimage.io as io
import tensorflow as tf
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/synth_domain_tfrecords_all_negs/"
# RECORD_TO_CHECK = TFRECORDS_DIR + "imagenet_train.tfrecords"
RECORD_TO_CHECK = TFRECORDS_DIR + "pascal_val.tfrecords"
# OVERFIT_TEST_TFRECORDS = "/notebooks/selerio/overfit_check.tfrecords"
# EVAL_TEST_TFRECORDS = "/home/omarreid/selerio/datasets/real_domain_tfrecords/imagenet_val.tfrecords"
DATASET_DIR = osp.expanduser('/home/omarreid/selerio/datasets/PASCAL3D+_release1.1')
OBJ_DIR = DATASET_DIR + "/OBJ/"

def main():
    reconstructed_records = []
    record_iterator = tf.python_io.tf_record_iterator(path=RECORD_TO_CHECK)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        features = example.features.feature

        rgb_descriptor = features['rgb_descriptor'].float_list.value
        object_class = features['object_class'].bytes_list.value[0].decode("utf-8")
        data_id = features['data_id'].bytes_list.value[0].decode("utf-8")
        cad_index = features['cad_index'].bytes_list.value[0].decode("utf-8")

        img_string = example.features.feature['positive_depth_image'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        pos_depth_image = img_1d.reshape((224, 224, 3))

        print(f"RGB Descriptor: {rgb_descriptor}")
        print(f"Object Class: {object_class}")
        print(f"Data ID: {data_id}")
        print(f"CAD Index: {cad_index}")

        all_model_paths = list(glob.glob(OBJ_DIR + "*/*.obj"))  # All classes, all objs

        random_model_obj_path = np.random.choice(all_model_paths)
        pos_obj = OBJ_DIR + str(object_class) + "/" + str(cad_index) + ".obj"

        print(f"Pos Obj: {pos_obj}")

        while pos_obj == random_model_obj_path:
            random_model_obj_path = np.random.choice(all_model_paths)

        random_cad_index = random_model_obj_path.split("/")[-1][:-4]

        print(f"Random Obj Model: {random_model_obj_path}")
        print(f"Random Cad Index: {random_cad_index}")

        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.imshow(pos_depth_image)

        plt.savefig("checkkkkkkk.png")

        return

    for counter , (image, output_vector) in enumerate(reconstructed_records):
        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1,2,2)
        ax.imshow(image)
        ax2.imshow(image)
        current_color = 'b'
        
        #output_vector_predicted = overfit_test_model_output_vectors[counter]
        print(np.array(output_vector))
        virtual_control_points = np.array(output_vector[:16]).reshape(8,2) * 224
        print("Real Labels:")
        print(virtual_control_points)
        #virtual_control_points_pred = np.array(output_vector_predicted[:16]).reshape(8,2) * 224
        #print("Predicted Labels:")
        #print(virtual_control_points_pred)
        #ax.scatter(virtual_control_points[:, 0], virtual_control_points[:, 1], c='r')
        #ax2.scatter(virtual_control_points_pred[:, 0], virtual_control_points_pred[:, 1])
        #line_boxes(ax2, virtual_control_points_pred, 'b')
        #line_boxes(ax, virtual_control_points, 'r')

        #plt.savefig("./{}_check.jpg".format(counter))
        print("Done")



def line_boxes(axis, virtual_control_points_pred, current_color):
    # Connect Points to Visualize Cube
    axis.plot(virtual_control_points_pred[:2,0], virtual_control_points_pred[:2,1],current_color)
    axis.plot(virtual_control_points_pred[2:4,0], virtual_control_points_pred[2:4,1],current_color)
    axis.plot(virtual_control_points_pred[4:6,0], virtual_control_points_pred[4:6,1],current_color)
    axis.plot(virtual_control_points_pred[6:8,0], virtual_control_points_pred[6:8,1],current_color)
    #ALONG
    axis.plot([virtual_control_points_pred[1,0],virtual_control_points_pred[5,0]],
             [virtual_control_points_pred[1,1],virtual_control_points_pred[5,1]], current_color)
    axis.plot([virtual_control_points_pred[2,0],virtual_control_points_pred[6,0]],
             [virtual_control_points_pred[2,1],virtual_control_points_pred[6,1]], 'y')
    axis.plot([virtual_control_points_pred[3,0],virtual_control_points_pred[7,0]],
             [virtual_control_points_pred[3,1],virtual_control_points_pred[7,1]], 'y')
    axis.plot([virtual_control_points_pred[0,0],virtual_control_points_pred[4,0]],
             [virtual_control_points_pred[0,1],virtual_control_points_pred[4,1]],current_color)

    #VERTICAL
    axis.plot([virtual_control_points_pred[2,0],virtual_control_points_pred[0,0]],
             [virtual_control_points_pred[2,1],virtual_control_points_pred[0,1]],current_color)
    axis.plot([virtual_control_points_pred[3,0],virtual_control_points_pred[1,0]],
             [virtual_control_points_pred[3,1],virtual_control_points_pred[1,1]],current_color)
    axis.plot([virtual_control_points_pred[5,0],virtual_control_points_pred[7,0]],
             [virtual_control_points_pred[5,1],virtual_control_points_pred[7,1]],current_color)
    axis.plot([virtual_control_points_pred[4,0],virtual_control_points_pred[6,0]],
             [virtual_control_points_pred[4,1],virtual_control_points_pred[6,1]], current_color)


if __name__ == "__main__":
    main()
