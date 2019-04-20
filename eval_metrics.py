import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pascal3d
import math
import skimage.io as io
import tensorflow as tf
import numpy as np
import click

from tqdm import tqdm
from numpy import linalg as LA
from scipy.linalg import logm
from nets import nets_factory, resnet_v1
from pascal3d import utils
from itertools import product
from model_dataset_utils import dataset_base
from check_tfrecords import line_boxes

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

UNIT_CUBE = np.array(list(product([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])), dtype=np.float32)
data_type = 'all'
DATASET = pascal3d.dataset.Pascal3DDataset(data_type, dataset_path="/home/omarreid/selerio/datasets/PASCAL3D+_release1.1", generate=True)
TFRECORDS_DIR = "/home/omarreid/selerio/datasets/real_domain_tfrecords/"
EVAL_TFRECORDS = TFRECORDS_DIR + "pascal_val.tfrecords"
NETWORK_NAME = 'resnet_v1_50'
MODEL_DIR = ""

@click.command()
@click.option('--model_dir', default="/home/omarreid/selerio/final_year_project/models/test_one", help='Path to model to evaluate')
@click.option('--tfrecords_file', default=EVAL_TFRECORDS, help='Path to TFRecords file to evaluate model on', type=str)
@click.option('--generate_imgs', default=False, help='If true will plot model results 10 images and save them ')
def main(model_dir, tfrecords_file, generate_imgs):
    global MODEL_DIR
    MODEL_DIR = model_dir
    tfrecords_file=str(tfrecords_file)
    get_viewpoint_errors(model_dir, real_domain_cnn_model_fn_predict, tfrecords_file, generate_imgs)


def run_eval(model_dir):
    return get_viewpoint_errors(model_dir, real_domain_cnn_model_fn_predict, EVAL_TFRECORDS, False)


def predict_input_fn( tfrecords_file):
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset_base(dataset, shuffle=False)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def get_viewpoint_errors(model_dir, model_fn, tfrecords_file, generate_imgs):
    """
    The percentage of all viewpoint differences smaller than pi over 6 (30 degrees)
    """
    with tf.device('/cpu:0'):
        real_domain_cnn = tf.estimator.Estimator(
            model_fn=model_fn, 
            model_dir=model_dir
        )

        records_within_limit_radians = 0
        all_differences = []
        
        #yield single examples = False useful if model_fn returns some tensors whose first dimension is not equal to the batch size.
        all_model_predictions = real_domain_cnn.predict(input_fn=lambda : predict_input_fn(tfrecords_file), yield_single_examples=False)
        #If yielding single examples uncomment the line below - do this if you have a tensor/batch error (slower)
        all_model_predictions = get_single_examples_from_batch(all_model_predictions)
        
        for counter, model_prediction in enumerate(all_model_predictions):
            model_output = model_prediction["2d_prediction"]
            image = np.uint8(model_prediction["original_img"])
            #img_1d = np.fromstring(model_prediction["original_img"], dtype=np.uint8)
            #image = img_1d.reshape((224, 224, -1))
            data_id = model_prediction["data_id"].decode('utf-8')
            object_index = model_prediction["object_index"]
            ground_truth_output = model_prediction["output_vector"]
            print("Data ID: ")
            print(data_id)
            
            ground_truth_rotation_matrix, focal, viewpoint_obj = get_ground_truth_rotation_matrix(data_id, object_index)
            pred_rotation_matrix, reprojected_virtual_control_points = get_predicted_3d_pose(model_output, focal, ground_truth_output)
             
            #print("Ground Truth Rotation Matrix")
            #print(ground_truth_rotation_matrix)
            #print("Predicted Rotation Matrix")
            #print(pred_rotation_matrix)
            
            difference = viewpoint_prediction_difference(ground_truth_rotation_matrix, pred_rotation_matrix)
            
            if generate_imgs:
                fig = plt.figure(figsize=(15,15))
                ax = plt.subplot(1, 3, 1)
                ax2 = plt.subplot(1, 3, 2)
                ax3 = plt.subplot(1, 3, 3)
                
                ax.imshow(image)
                ax2.imshow(image)
                ax3.imshow(image)

                virtual_control_points_pred = np.array(model_output[:16]).reshape(8,2)*224
                reprojected_virtual_control_points = reprojected_virtual_control_points*224
                virtual_control_points = np.array(ground_truth_output[:16]).reshape(8,2)*224

                line_boxes(ax, virtual_control_points, 'r')
                line_boxes(ax2, virtual_control_points_pred, 'b')
                line_boxes(ax3, reprojected_virtual_control_points, 'g')
                plt.tight_layout()

                plt.savefig("./{}_eval.jpg".format(counter))



            all_differences.append(difference)
            
            print("Difference in Degrees")
            print(math.degrees(difference))

            #If difference is in radians otherwise 30 degrees
            if difference < math.pi/6:
                print("Within Limit")
                print("Difference in Degrees")
                print(math.degrees(difference))
                
                records_within_limit_radians +=1

            if counter % 500 == 0:
                print("")
                print("")
                print("************************************************")
                print("************************************************")
                print("Current ACC PI/6")
                print((float(records_within_limit_radians)/len(all_differences)) * 100)
                print("************************************************")
                print("************************************************")
                print("")
                print("")

            
            print("************************************************")
            
            if generate_imgs:
                if counter == 10:
                    break
           
    acc_pi_over_6 = (float(records_within_limit_radians)/len(all_differences)) * 100
    median_error = np.median(np.array(all_differences))
    print("Final")
    print("Acc Pi Over 6 : % of image < 30 degrees")
    print(str(acc_pi_over_6) + "%")
    print("Median Error in Degres ")
    print(str(math.degrees(median_error)))

    return str(acc_pi_over_6), math.degrees(median_error)


def get_single_examples_from_batch(all_model_predictions):
    single_examples = []
    for output_batch in all_model_predictions:
        data_ids = output_batch['data_id']
        object_classes = output_batch['object_class']
        cad_indices = output_batch['cad_index']
        object_indices = output_batch['object_index']
        output_vectors = output_batch['output_vector']
        image_descriptors = output_batch['image_descriptor']
        predictions_2d = output_batch['2d_prediction']
        # images = output_batch['img']
        original_images = output_batch['original_img']
        number_in_batch = min(len(data_ids), len(object_indices), len(output_vectors), len(predictions_2d), len(original_images))
        
        if type(predictions_2d[0]) is not np.ndarray:
            single_examples.append({
                "data_id": data_ids[0],
                "object_index": object_indices[0],
                "cad_index": cad_indices[0],
                "output_vector": output_vectors[0],
                "image_descriptor": image_descriptors[0],
                "object_class": object_classes[0],
                # "img": images[0],
                "original_img": original_images[0], 
                "2d_prediction": predictions_2d     
            })
        else:       
            for index in range(number_in_batch):
                single_examples.append({
                    "data_id": data_ids[index],
                    "object_index": object_indices[index],
                    "output_vector": output_vectors[index],
                    "object_class": object_classes[index],
                    "image_descriptor": image_descriptors[index],
                    # "img": images[index],
                    "cad_index": cad_indices[index],
                    "original_img": original_images[index], 
                    "2d_prediction": predictions_2d[index]     
                })

    return single_examples

def get_predicted_3d_pose(output_vector, focal, ground_truth_output):
    """
     Uses values from ouput of real domain cnn to get the 3D pose
    """
    predicted_dims = np.array(output_vector[16:])
    virtual_control_points_2d = np.array(output_vector[:16]).reshape(8,2) #These points are normalized
    #print("GT VC")
    #print(ground_truth_output[:16].reshape(8,2)*224)
    #print("PRED VC")
    #print(virtual_control_points_2d)
    #print("GT DIMS")
    #print(ground_truth_output[16:])
    #print("PRED DIMS")
    #print(predicted_dims)

    scaled_unit_cube = compute_scaled_unit_cube(predicted_dims).astype(np.float32)
    
    # Camera internals
    skew = 0
    aspect_ratio = 1
    focal_length = 1 #Should be 1
    image_center = (0.5, 0.5)
    dist_coeffs = np.zeros(4) # Assuming no lens distortion
    camera_matrix = np.array([[focal_length, 0, image_center[0]], [0, focal_length, image_center[1]], [0, 0, 1]])
    
    virtual_control_points_2d[:, 1] = 1 - virtual_control_points_2d[:, 1] 
    N, M = virtual_control_points_2d.shape
    imagePoints = np.ascontiguousarray(virtual_control_points_2d[:,:2]).reshape((N,1,2))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(scaled_unit_cube, imagePoints, camera_matrix, dist_coeffs, rvec=(1,1,1), tvec=(1,1,1), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    success, rotation_vector, translation_vector, _= cv2.solvePnPRansac(scaled_unit_cube, imagePoints, camera_matrix, distCoeffs=dist_coeffs, reprojectionError=0.5, iterationsCount=10000, rvec=rotation_vector, tvec=translation_vector, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    #computes projections of 3D points to the image plane 
    projected_cube = cv2.projectPoints(scaled_unit_cube, rotation_vector, translation_vector,camera_matrix, None, aspectRatio=aspect_ratio)[0]
    reprojected_vc_points = projected_cube.reshape(8, 2)

    #print("Reprojected VC Points")
    #print(reprojected_vc_points)
    
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]

    return rotation_matrix, reprojected_vc_points

def get_ground_truth_rotation_matrix(data_id, object_index):
    data = DATASET.get_data(0, data_id=data_id)
    objects = data['objects']
    
    # Need to find out which object index the object is lol
    #Can assume the objs are in tfrecord order so we need to store how many times we've seen a specific id (to work during training)
    #cls = objects[object_index][0]
    obj = objects[object_index][1]
    distance =obj['viewpoint']['distance']
    azimuth = obj['viewpoint']['azimuth']
    elevation = obj['viewpoint']['elevation']
    focal  = obj['viewpoint']['focal']
    
    #if mirrored:
        #Flip azimuth angle
       # azimuth = 2*math.pi - azimuth
        
    R, R_rot = utils.get_transformation_matrix(azimuth, elevation, distance)
    
    return R_rot, focal, obj['viewpoint']

def viewpoint_prediction_difference(r_ground_truth, r_predicted):
    r_gt_transpose = r_ground_truth.transpose()
    r_mul = np.matmul(r_gt_transpose, r_predicted)
    return LA.norm(logm(r_mul), 'fro' )/np.sqrt(2)

def compute_scaled_unit_cube(dimensions):
    scalar_x, scalar_y, scalar_z = dimensions
    scale_matrix = np.array([[scalar_x, 0, 0, 0], [0, scalar_y, 0, 0], [0, 0, scalar_z , 0], [0, 0, 0, 1]])
    return apply_transformation(UNIT_CUBE, scale_matrix)

def apply_transformation(cube, transformation_matrix):
    """
    Copied from dataset.py
    """
    transformed_cube = []
    for vector in cube:
        new_vector = np.dot(transformation_matrix, np.append(vector,[1]))
        transformed_cube.append(new_vector[:3])

    return np.array(transformed_cube)


def real_domain_cnn_model_fn_predict(features, labels, mode):
    """
    Real Domain CNN from 3D Object Detection and Pose Estimation paper
    """

    # Training End to End - So weights start from scratch
    input_images = tf.identity(features['img'], name="input")  # Used when converting to unity

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # Retrieve the function that returns logits and endpoints - ResNet was pre trained on ImageNet
        network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=None, is_training=True)
        image_descriptors, endpoints = network_fn(input_images)

    image_descriptors = tf.identity(image_descriptors, name="image_descriptors")

    # Add a dense layer to get the 19 neuron linear output layer
    logits = tf.layers.dense(image_descriptors, 19, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
    logits = tf.squeeze(logits, name='2d_predictions')

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(MODEL_DIR):
        tf.logging.info(
            'Ignoring RESNET50 CKPT because a checkpoint already exists in %s'
            % MODEL_DIR)

    checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
    variables_to_restore = slim.get_variables_to_restore()
    tf.train.init_from_checkpoint(checkpoint_path,  {v.name.split(':')[0]: v for v in variables_to_restore})

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "2d_prediction": logits,
        "data_id":features['data_id'],
        "image_descriptor": image_descriptors,
        "object_index": features['object_index'],
        "output_vector": features['ground_truth_output'],
        "object_class": features['object_class'],
        "img": features['img'],
        "original_img": features['normal_img']
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == "__main__":
    main()
