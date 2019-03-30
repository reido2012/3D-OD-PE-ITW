#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import math
import os
import os.path as osp
import shlex
import subprocess
import warnings
import glob
import cv2
import matplotlib
import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import scipy.io
import scipy.misc
import skimage.color
import sklearn.model_selection
import tqdm

from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from itertools import product
from nets import nets_factory, resnet_v1
from pascal3d import utils

slim = tf.contrib.slim

NETWORK_NAME = 'resnet_v1_50'
UNIT_CUBE = np.array(list(product([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])))
DATASET_DIR = osp.expanduser('/home/omarreid/selerio/datasets/PASCAL3D+_release1.1')
IMAGENET_IMAGESET_DIR = DATASET_DIR + "/Image_sets/"
PASCAL_IMAGESET_DIR = DATASET_DIR + "/PASCAL/VOCdevkit/VOC2012/ImageSets/Main/"
OBJ_DIR = DATASET_DIR + "/OBJ/"
MODEL_DIR = "/home/omarreid/selerio/final_year_project/models/test_one"
BATCH_SIZE = 50
SHUFFLE_BUFFER_SIZE = 18000
NUM_CPU_CORES = 8
PRE_FETCH_BUFFER_SIZE = 4
IMAGE_SIZE = 224 # To match ResNet dimensions
GREYSCALE_SIZE = tf.constant(50176)
GREYSCALE_CHANNEL = tf.constant(1)


class Pascal3DAnnotation(object):

    def __init__(self, ann_file):
        ann_data = scipy.io.loadmat(ann_file)

        self.img_filename = ann_data['record']['filename'][0][0][0]
        self.database = ann_data['record']['database']

        if self.database == 'ImageNet':
            self.segmented = True
        else:
            self.segmented = ann_data['record']['segmented'][0][0][0]

        self.objects = []
        for obj in ann_data['record']['objects'][0][0][0]:
            if not obj['viewpoint']:
                continue
            elif 'distance' not in obj['viewpoint'].dtype.names:
                continue
            elif obj['viewpoint']['distance'][0][0][0][0] == 0:
                continue

            cad_index = obj['cad_index'][0][0] - 1
            bbox = obj['bbox'][0]
            anchors = obj['anchors']

            viewpoint = obj['viewpoint']
            azimuth = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
            elevation = viewpoint['elevation'][0][0][0][0] * math.pi / 180
            distance = viewpoint['distance'][0][0][0][0]
            focal = viewpoint['focal'][0][0][0][0]
            theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
            principal = np.array([viewpoint['px'][0][0][0][0],
                                  viewpoint['py'][0][0][0][0]])
            viewport = viewpoint['viewport'][0][0][0][0]

            truncated = obj['truncated'][0][0]
            occluded = obj['occluded'][0][0]

            skip = False
            if truncated or occluded:
                skip = True

            self.objects.append({
                'cad_index': cad_index,
                'bbox': bbox,
                'anchors': anchors,
                'skip': skip,
                'viewpoint': {
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': distance,
                    'focal': focal,
                    'theta': theta,
                    'principal': principal,
                    'viewport': viewport,
                },
            })


class Pascal3DDataset(object):
    voc2012_class_names = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ]

    class_names = [
        'background',
        'aeroplane',
        'bicycle',
        'boat',
        'bottle',
        'bus',
        'car',
        'chair',
        'diningtable',
        'motorbike',
        'sofa',
        'train',
        'tvmonitor',
    ]

    def __init__(self, data_type, generate=True,
                 dataset_path="/home/omarreid/selerio/datasets/pascal3d/PASCAL3D+_release1.1"):
        assert data_type in ('train', 'val', 'all')
        self.dataset_dir = osp.expanduser(dataset_path)
        # get all data ids
        if generate:
            print('Generating index for annotations...')
            data_ids = []
            print(self.class_names[1:])
            for counter, cls in enumerate(self.class_names[1:]):
                pascal_cls_ann_dir = osp.join(self.dataset_dir, 'Annotations/{}_pascal'.format(cls))
                imagenet_class_ann_dir = osp.join(self.dataset_dir, 'Annotations/{}_imagenet'.format(cls))
                all_annotation_dirs = glob.glob(pascal_cls_ann_dir) + glob.glob(imagenet_class_ann_dir)
                for annotation_directory in all_annotation_dirs:
                    for ann_file in glob.glob(annotation_directory + "/*.mat"):
                        ann = Pascal3DAnnotation(ann_file)
                        if not ann.segmented:
                            continue
                        # print("Ann File Path: {}".format(ann_file))
                        data_id = ann_file.split("/")[-1][:-4]
                        # print("Data Id: {}".format(data_id))
                        data_ids.append(data_id)

            print(len(set(data_ids)))
            print('Done.')
            data_ids = list(set(data_ids))
            # split data to train and val
            if data_type != 'all':
                ids_train, ids_val = sklearn.model_selection.train_test_split(
                    data_ids, test_size=0.25, random_state=1234)
                if data_type == 'train':
                    self.data_ids = ids_train
                else:
                    self.data_ids = ids_val
            else:
                self.data_ids = data_ids

    def __len__(self):
        return len(self.data_ids)

    def get_data(self, i, data_id=None):

        if not data_id:
            data_id = self.data_ids[i]

        data = {
            'img': None,
            'objects': [],
            'class_cads': {},
            'label_cls': None,
        }

        for class_name in self.class_names[1:]:
            ann_file = osp.join(self.dataset_dir, 'Annotations/{}_pascal/{}.mat'.format(class_name, data_id))
            if not osp.exists(ann_file):
                ann_file = osp.join(self.dataset_dir, 'Annotations/{}_imagenet/{}.mat'.format(class_name, data_id))
                if not osp.exists(ann_file):
                    continue

            ann = Pascal3DAnnotation(ann_file)
            # When creating the TF.Records we don't care about label cls
            # and ann.database != "ImageNet" and data_id is None
            if data['label_cls'] is None and ann.database != "ImageNet" and data_id is None:
                label_cls_file = osp.join(
                    self.dataset_dir,
                    'PASCAL/VOCdevkit/VOC2012/SegmentationClass/{}.png'.format(data_id))

                if not osp.exists(label_cls_file):
                    label_cls_file = osp.join(
                        self.dataset_dir,
                        'PASCAL/VOCdevkit/VOC2012/SegmentationClass/{}.jpg'.format(data_id))

                label_cls = PIL.Image.open(label_cls_file)
                label_cls = np.array(label_cls)
                label_cls[label_cls == 255] = 0  # set boundary as background
                # convert label from voc2012 to pascal3D
                for voc2012_id, cls in enumerate(self.voc2012_class_names):
                    cls = cls.replace('/', '')
                    if cls in self.class_names:
                        pascal3d_id = self.class_names.index(cls)
                        label_cls[label_cls == voc2012_id] = pascal3d_id
                    else:
                        # set background class id
                        label_cls[label_cls == voc2012_id] = 0
                data['label_cls'] = label_cls

            if class_name not in data['class_cads']:
                cad_file = osp.join(
                    self.dataset_dir,
                    'CAD/{}.mat'.format(class_name))
                cad = scipy.io.loadmat(cad_file)[class_name][0]
                data['class_cads'][class_name] = cad

            if data['img'] is None:
                needle = 'Images/{}_pascal'
                if ann.database == "ImageNet":
                    needle = 'Images/{}_imagenet'

                img_file = osp.join(
                    self.dataset_dir,
                    needle.format(class_name),
                    ann.img_filename)

                data['img'] = scipy.misc.imread(img_file)

            for obj in ann.objects:
                obj['cad_basename'] = osp.join(
                    self.dataset_dir,
                    'CAD/{}/{:02}'.format(class_name, obj['cad_index'] + 1))
                data['objects'].append((class_name, obj))

        return data

    def show_annotation(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        label_cls = data['label_cls']

        ax1 = plt.subplot(121)
        plt.axis('off')

        ax2 = plt.subplot(122)
        plt.axis('off')
        label_viz = skimage.color.label2rgb(label_cls, bg_label=0)
        ax2.imshow(label_viz)

        for cls, obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

            if not obj['anchors']:
                continue
            anchors = obj['anchors'][0][0]
            for name in anchors.dtype.names:
                anchor = anchors[name]
                if anchor['status'] != 1:
                    continue
                x, y = anchor['location'][0][0][0]
                cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
        ax1.imshow(img)

        plt.tight_layout()
        plt.show()

    def show_cad(self, i, camframe=False):
        if camframe:
            return self.show_cad_camframe(i)

        data = self.get_data(i)

        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

        for cls, obj in objects:
            # show image
            ax1 = plt.subplot(1, 2, 1)
            plt.axis('off')
            ax1.imshow(img)

            ax2 = plt.subplot(1, 2, 2, projection='3d')

            cad_index = obj['cad_index']
            cad = class_cads[cls]

            # show camera model
            height, width = img.shape[:2]
            x = utils.get_camera_polygon(
                height=height,
                width=width,
                theta=obj['viewpoint']['theta'],
                focal=obj['viewpoint']['focal'],
                principal=obj['viewpoint']['principal'],
                viewport=obj['viewpoint']['viewport'],
            )
            R, R_rot = utils.get_transformation_matrix(
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            x = np.hstack((x, np.ones((len(x), 1), dtype=np.float64)))
            x = np.dot(np.linalg.inv(R)[:3, :4], x.T).T
            x0, x1, x2, x3, x4 = x
            verts = [
                [x0, x1, x2],
                [x0, x2, x3],
                [x0, x3, x4],
                [x0, x4, x1],
                [x1, x2, x3, x4],
            ]
            ax2.add_collection3d(
                Poly3DCollection([verts[0]], facecolors='r', linewidths=1))
            ax2.add_collection3d(
                Poly3DCollection(verts[1:], facecolors='w',
                                 linewidths=1, alpha=0.5))
            x, y, z = zip(*x)
            ax2.plot(x, y, z)  # to show the camera model in the range

            max_x = max(x)
            max_y = max(y)
            max_z = max(z)
            min_x = min(x)
            min_y = min(y)
            min_z = min(z)

            # display the cad model
            vertices_3d = cad[cad_index]['vertices']
            x, y, z = zip(*vertices_3d)
            ax2.plot(x, y, z, color='b')

            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
            max_z = max(max_z, max(z))
            min_x = min(min_x, min(x))
            min_y = min(min_y, min(y))
            min_z = min(min_z, min(z))

            # align bounding box
            max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.5
            mid_x = (max_x + min_x) * 0.5
            mid_y = (max_y + min_y) * 0.5
            mid_z = (max_z + min_z) * 0.5
            ax2.set_xlim(mid_x - max_range, mid_x + max_range)
            ax2.set_ylim(mid_y - max_range, mid_y + max_range)
            ax2.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.tight_layout()
            plt.show()

    def show_cad_camframe(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img)
        plt.axis('off')

        ax2 = plt.subplot(1, 2, 2, projection='3d')
        ax2.plot([0], [0], [0], marker='o')

        max_x = min_x = 0
        max_y = min_y = 0
        max_z = min_z = 0
        for cls, obj in objects:
            cad_index = obj['cad_index']
            cad = class_cads[cls]

            vertices_3d = cad[cad_index]['vertices']

            vertices_3d_camframe = utils.transform_to_camera_frame(
                vertices_3d,
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )

            # XXX: Not sure this is correct...
            delta = (obj['viewpoint']['principal'] /
                     obj['viewpoint']['viewport'])
            vertices_3d_camframe[:, 0] += delta[0] * 10
            vertices_3d_camframe[:, 1] -= delta[1] * 10

            x, y, z = zip(*vertices_3d_camframe)
            ax2.plot(x, y, z)

            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
            max_z = max(max_z, max(z))
            min_x = min(min_x, min(x))
            min_y = min(min_y, min(y))
            min_z = min(min_z, min(z))

        # align bounding box
        max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.5
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        mid_z = (max_z + min_z) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def show_cad_overlay(self, i, data_id):

        if data_id:
            data = self.get_data(i, data_id=data_id)
        else:
            data = self.get_data(i)

        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

        ax1 = plt.subplot(121)
        plt.axis('off')
        ax1.imshow(img)

        ax2 = plt.subplot(122)
        plt.axis('off')
        ax2.imshow(img)

        for cls, obj in objects:
            cad_index = obj['cad_index']
            cad = class_cads[cls][cad_index]

            vertices_3d = cad['vertices']
            faces = cad['faces']

            vertices_2d = utils.project_points_3d_to_2d(
                vertices_3d, **obj['viewpoint'])

            patches = []
            for face in faces:
                points = [vertices_2d[i_vertex - 1] for i_vertex in face]
                poly = Polygon(points, True)
                patches.append(poly)
            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
            ax2.add_collection(p)

        plt.tight_layout()
        plt.show()

    def create_patch_collection_from_vertices(self, vertices_2d, faces):
        """
        Create surface of CAD model in 2D from it's vertices and faces
        
            vertices_2d: Vertices of CAD model projected to 2D
            faces: Faces of the model 
        """
        patches = []
        for face in faces:
            points = [vertices_2d[i_vertex - 1] for i_vertex in face]
            poly = Polygon(points, True)
            patches.append(poly)

        return PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    def _get_pascal_data_ids(self):
        """
        Read data ids from pascal train, val and test text files
        """
        files = [PASCAL_IMAGESET_DIR + "train.txt", PASCAL_IMAGESET_DIR + "test.txt", PASCAL_IMAGESET_DIR + "val.txt"]
        return list(map(self._read_ids_from_file, files))

    def _get_imagenet_ids(self):
        """
        Read data ids from imagenet train and val text files
        """
        training_files = glob.glob(IMAGENET_IMAGESET_DIR + "*_imagenet_train.txt")
        validation_files = glob.glob(IMAGENET_IMAGESET_DIR + "*_imagenet_val.txt")

        training_ids = list(map(self._read_ids_from_file, training_files))
        all_training_ids = [data_id for id_list in training_ids for data_id in id_list]  # Flatten

        validation_ids = list(map(self._read_ids_from_file, validation_files))
        all_validation_ids = [data_id for id_list in validation_ids for data_id in id_list]

        return (all_training_ids, all_validation_ids)

    def _read_ids_from_file(self, filepath):
        """
        Reads ids from txt file where each line in file is a separate id
        """

        with open(filepath, 'r') as file:
            lines = file.readlines()

        return [line.rstrip() for line in lines]

    def create_tfrecords(self, tfrecord_directory, debug=False):
        """
        Create TF.Records file for traning or dataset
        """

        # Get all text files containing ids
        pascal_train_ids, pascal_test_ids, pascal_val_ids = self._get_pascal_data_ids()
        imagenet_train_ids, imagenet_val_ids = self._get_imagenet_ids()

        record_map = {
            # "pascal_train": pascal_train_ids,
            #             "pascal_test": pascal_test_ids,
            "pascal_val": pascal_val_ids,
            # "imagenet_train": imagenet_train_ids,
            # "imagenet_val": imagenet_val_ids
        }

        for name, id_list in record_map.items():
            tfrecords_filename = '{}.tfrecords'.format(name)
            print("Starting: {}".format(tfrecords_filename))
            self._create_tfrecords_from_data_ids(tfrecords_filename, id_list, tfrecord_directory, debug)
            print("Finished: {}".format(tfrecords_filename))

    def create_tfrecords_synth_domain(self, path_to_save_records, local):
        pascal_train_ids, pascal_test_ids, pascal_val_ids = self._get_pascal_data_ids()
        imagenet_train_ids, imagenet_val_ids = self._get_imagenet_ids()

        record_map = {
            "pascal_train": pascal_train_ids,
            # "pascal_val": pascal_val_ids,
            # "imagenet_train": imagenet_train_ids,
            # "imagenet_val": imagenet_val_ids
        }

        for name, id_list in record_map.items():
            tfrecords_filename = '{}.tfrecords'.format(name)
            print("Starting: {}".format(tfrecords_filename))
            descriptor_dict = self.get_rgb_descriptors(MODEL_DIR, tfrecords_filename)
            self._create_synth_tfrecords_from_data_ids(tfrecords_filename, descriptor_dict, id_list,
                                                       path_to_save_records, local)
            print("Finished: {}".format(tfrecords_filename))

    def get_rgb_descriptors(self, model_dir, tfrecords_filename):
        real_domain_cnn = tf.estimator.Estimator(
            model_fn=self.real_domain_cnn_model_fn_predict,
            model_dir=model_dir
        )
        tfrecords_dir = "/home/omarreid/selerio/datasets/real_domain_tfrecords/"
        tfrecord_path = tfrecords_dir + tfrecords_filename

        all_model_predictions = real_domain_cnn.predict(input_fn=lambda: self.predict_input_fn(tfrecord_path),
                                                        yield_single_examples=False)
        all_model_predictions = self.get_single_examples_from_batch(all_model_predictions)

        descriptor_dict = {}
        for counter, model_prediction in enumerate(all_model_predictions):
            image_descriptor = model_prediction["image_descriptor"]
            data_id = model_prediction["data_id"].decode('utf-8')
            object_idx = model_prediction["object_index"]

            descriptor_dict[(data_id, object_idx)] = image_descriptor

        return descriptor_dict

    def dataset_base(self, dataset, shuffle=True):
        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.map(map_func=self.tfrecord_parser,
                              num_parallel_calls=NUM_CPU_CORES)  # Parallelize data transformation
        dataset.apply(tf.contrib.data.ignore_errors())
        dataset = dataset.batch(batch_size=BATCH_SIZE)
        return dataset.prefetch(buffer_size=PRE_FETCH_BUFFER_SIZE)

    def predict_input_fn(self, tfrecords_file):
        dataset = tf.data.TFRecordDataset(tfrecords_file)
        dataset = self.dataset_base(dataset, shuffle=False)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def tfrecord_parser(self, serialized_example):
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
                'object_index': tf.FixedLenFeature([], tf.int64),
            }
        )

        # Convert Scalar String to uint8
        input_image = tf.decode_raw(features['object_image'], tf.uint8)
        input_image = tf.to_float(input_image)

        data_id = features['data_id']

        # Image is not in correct shape so
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
            "ground_truth_output": output_vector
        }

        return model_input, output_vector

    def get_single_examples_from_batch(self, all_model_predictions):
        single_examples = []
        for output_batch in all_model_predictions:
            data_ids = output_batch['data_id']
            image_descriptors = output_batch['image_descriptor']
            object_indices = output_batch['object_index']
            predictions_2d = output_batch['2d_prediction']
            number_in_batch = min(len(data_ids), len(object_indices), len(predictions_2d))

            if type(predictions_2d[0]) is not np.ndarray:
                single_examples.append({
                    "data_id": data_ids[0],
                    "object_index": object_indices[0],
                    "2d_prediction": predictions_2d,
                    "image_descriptor": image_descriptors
                })
            else:
                for index in range(number_in_batch):
                    single_examples.append({
                        "data_id": data_ids[index],
                        "object_index": object_indices[index],
                        "2d_prediction": predictions_2d[index],
                        "image_descriptor": image_descriptors[index]
                    })

        return single_examples

    def real_domain_cnn_model_fn_predict(self, features, labels, mode):
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
        logits = tf.layers.dense(image_descriptors, 19,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
        logits = tf.squeeze(logits, name='2d_predictions')

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(MODEL_DIR):
            tf.logging.info(
                'Ignoring RESNET50 CKPT because a checkpoint already exists in %s'
                % MODEL_DIR)

        checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)
        variables_to_restore = slim.get_variables_to_restore()
        tf.train.init_from_checkpoint(checkpoint_path, {v.name.split(':')[0]: v for v in variables_to_restore})

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "2d_prediction": logits,
            "image_descriptor": image_descriptors,
            "data_id": features['data_id'],
            "object_index": features['object_index']
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    def _create_synth_tfrecords_from_data_ids(self, record_name, descriptor_dict, ids, path_to_save_records, local):
        writer = tf.python_io.TFRecordWriter(path_to_save_records + record_name)
        skipped = []

        for counter, data_id in enumerate(ids):

            data = self.get_data(0, data_id=data_id)
            if data['img'] is None:
                print("No Image")
                # Annotation Not Find for Data ID
                skipped.append(str(data_id) + "\n")
                continue

            img = data['img']

            if (len(img.shape)) < 3:
                print("Greyscale")
                # Image is greyscale
                skipped.append(str(data_id) + "\n")
                continue

            print("Data ID: ")
            print(data_id)

            for obj_idx, (cls, obj) in enumerate(data['objects']):
                if obj['skip'] and record_name == 'pascal_val.tfrecords':
                    # We only want to evaluate on non truncated/occluded objects
                    # Skip object if it is truncated
                    skipped.append("Object: " + str(obj_idx) + " In Image: " + str(data_id) + "\n")
                    continue

                cad_index = obj['cad_index']
                bbox = obj['bbox']
                obj_id = str(obj_idx)

                cropped_img, square_bbox = self._crop_object_from_img(img, bbox)
                resized_img = scipy.misc.imresize(cropped_img, (224, 224))
                _, bbox_3d_dims = self._get_real_domain_output_vector(cls, data['class_cads'], obj)

                R, R_rot = utils.get_transformation_matrix(
                    obj['viewpoint']['azimuth'],
                    obj['viewpoint']['elevation'],
                    obj['viewpoint']['distance'],
                )

                rotation_tuple = self.mat2euler(R_rot)[::-1]
                cad_index = self.get_cad_number(cad_index)

                positive_depth_map_image_path, negative_depth_paths = self.render_for_dataset(cls, cad_index,
                                                                                             rotation_tuple,
                                                                                             bbox_3d_dims, data_id,
                                                                                             obj_id, OBJ_DIR,
                                                                                             local=local)

                print("PD Image Path: " + positive_depth_map_image_path)
                positive_depth_image = scipy.misc.imread(positive_depth_map_image_path)

                negative_depth_images = []
                for negative_depth_path in negative_depth_paths:
                    negative_depth_image = scipy.misc.imread(negative_depth_path, mode='RGB')
                    print(f"Original Neg Depth Image Dims: {negative_depth_image[0].shape}")
                    negative_depth_image = scipy.misc.imresize(negative_depth_image, (224, 224, 3))
                    print(f"New Neg Depth Image Dims: {negative_depth_image[0].shape}")
                    negative_depth_images.append(negative_depth_image)

                print(f"Neg Depth Image Dims: {len(negative_depth_images)}")
                rgb_descriptor = descriptor_dict[(data_id, obj_idx)]

                self._write_synth_record(writer, resized_img, rgb_descriptor, positive_depth_image, cad_index, cls,
                                         data_id, obj_id, negative_depth_images)
        writer.close()

    def _write_synth_record(self, record_writer, image, rgb_descriptor, positive_depth_map_image, cad_index,
                            object_class, data_id, object_index, negative_depth_images):

        num_neg_depth_imgs = len(negative_depth_images)
        negative_depth_imgs_raw = list(map(lambda x: x.tostring(), negative_depth_images))
        rgb_descriptor = rgb_descriptor.squeeze()
        img_raw = image.tostring()
        depth_img_raw = positive_depth_map_image.tostring()

        feature = {
            'object_image': self._bytes_feature(img_raw),
            'positive_depth_image': self._bytes_feature(depth_img_raw),
            'negative_depth_images': self._bytes_list_feature(negative_depth_imgs_raw),
            'num_negative_depth_images': self._int64_feature(num_neg_depth_imgs),
            'rgb_descriptor': self._floats_feature(rgb_descriptor),
            'object_class': self._bytes_feature(object_class.encode('utf-8')),
            'object_index': self._bytes_feature(object_index.encode('utf-8')),
            'data_id': self._bytes_feature(data_id.encode('utf-8')),
            'cad_index': self._bytes_feature(cad_index.encode('utf-8'))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())

    @staticmethod
    def get_cad_number(cad_index):
        index = int(cad_index) + 1
        if cad_index < 9:
            return '0' + str(index)
        else:
            return str(index)

    def render_for_dataset(self, image_class, cad_index, rotation_xyz, bbox_dims, record_id, obj_id, path_to_objs,
                           local=False):
        x_rotation, y_rotation, z_rotation = rotation_xyz
        x_dim, y_dim, z_dim = bbox_dims
        negative_depth_paths = []

        for object_path in glob.glob(path_to_objs + image_class + "/*.obj"):

            curr_obj_cad_index = object_path.split("/")[-1].split(".")[0]
            depth_path = "/home/omarreid/selerio/datasets/synth_renderings/" + str(
                record_id) + "/" + obj_id + "_" + str(curr_obj_cad_index) + "_0001.png"

            if os.path.isfile(depth_path):
                if cad_index != curr_obj_cad_index:
                    negative_depth_paths.append(depth_path)
                continue

            command = "nvidia-docker run -v /home/omarreid/selerio/:/workdir peterlauri/blender-python:latest blender " \
                      "-noaudio --background --python /workdir/pix3d/blender_render.py --  --specific_viewpoint True " \
                      "--cad_index " + curr_obj_cad_index + " --obj_id=" + obj_id + " --viewpoint=" + str(
                x_rotation) + "," + str(y_rotation) + "," + str(
                z_rotation) + " --output_folder /workdir/pix3d/synth_renderings/" + str(record_id) + " "

            if local:
                command = "blender -noaudio --background --python ./blender_render.py -- --specific_viewpoint=True " \
                          "--cad_index=" + curr_obj_cad_index + " --obj_id=" + obj_id + " --radians=True " \
                                                                                        "--viewpoint=" + str(
                    x_rotation) + "," + str(
                    y_rotation) + "," + str(
                    z_rotation) + " --bbox=" + str(
                    x_dim) + "," + str(
                    y_dim) + "," + str(
                    z_dim) + " --output_folder /home/omarreid/selerio/datasets/synth_renderings/" + str(record_id) + " "

            if not local:
                object_path = "/workdir/" + "/".join(object_path.split("/")[4:])

            full_command = command + object_path

            try:
                subprocess.run(full_command.split(), check=True)
            except subprocess.CalledProcessError as e:
                print(e)
                raise e

            print("Command: " + full_command)

        return "/home/omarreid/selerio/datasets/synth_renderings/" + str(record_id) + "/" + obj_id + "_" + str(
            cad_index) + "_0001.png", negative_depth_paths

    def mat2euler(self, M, cy_thresh=None):
        ''' Discover Euler angle vector from 3x3 matrix '''

        M = np.asarray(M)
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4
            except ValueError:
                cy_thresh = 1e-6
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
        return z, y, x

    def _create_tfrecords_from_data_ids(self, record_name, ids, tfrecord_directory, debug):
        """
            Creates TFRecords for a set of ids
            
                record_name: The name of the TFRecords file
                ids: list of data ids        
        """

        writer = tf.python_io.TFRecordWriter(record_name)
        skipped = []
        for data_id in tqdm.tqdm(ids):

            data = self.get_data(0, data_id=data_id)

            if data['img'] is None:
                # Annotation Not Find for Data ID
                skipped.append(str(data_id) + "\n")
                continue

            img = data['img']

            if (len(img.shape)) < 3:
                # Image is greyscale
                skipped.append(str(data_id) + "\n")
                continue

            # apply_blur, apply_random_crops = self._set_image_operations

            original_img = img
            objects = data['objects']
            class_cads = data['class_cads']

            if debug:
                fig = plt.figure()
                ax2 = plt.subplot(1, 1, 1)
                ax2.imshow(img)

            # Create a TF Record for each object in record
            for counter, (cls, obj) in enumerate(objects):
                if obj['skip'] and record_name == 'pascal_val.tfrecords':
                    # We only want to evaluate on non truncated/occluded objects
                    # Skip object if it is truncated
                    skipped.append("Object: " + str(counter) + " In Image: " + str(data_id) + "\n")
                    continue

                virtual_control_points_2d, bbox_3d_dims = self._get_real_domain_output_vector(
                    cls, class_cads, obj)

                bbox = obj['bbox']

                # Create a Rectangle patch
                if debug:
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                             edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    ax2.add_patch(rect)

                cropped_img, square_bbox = self._crop_object_from_img(img, bbox)

                if debug:
                    rect2 = patches.Rectangle((square_bbox[0], square_bbox[1]), square_bbox[2] - square_bbox[0],
                                              square_bbox[3] - square_bbox[1], linewidth=1, edgecolor='b',
                                              facecolor='none')
                    ax2.add_patch(rect2)

                normalized_virtual_control_points = self._normalize_2d_control_points(virtual_control_points_2d,
                                                                                      square_bbox)
                output_vector = np.append(normalized_virtual_control_points, bbox_3d_dims).astype(np.float)

                resized_img = scipy.misc.imresize(cropped_img, (224, 224))

                # One TF Record with normal image
                self._write_record(writer, resized_img, output_vector, cls, data_id, counter)

        writer.close()

        if debug:
            plt.show()

        if len(skipped) > 0:
            print("Skipped {} Objects".format(len(skipped)))
            print("*********************")
            print(skipped)

            with open(record_name + '_skipped.txt', 'w') as file:
                file.writelines(skipped)

    def _write_record(self, record_writer, image, output_vector, object_cls, data_id,
                      object_index):
        img_raw = image.tostring()

        feature = {
            'object_image': self._bytes_feature(img_raw),
            'output_vector': self._floats_feature(output_vector),
            'object_class': self._bytes_feature(object_cls.encode('utf-8')),
            'data_id': self._bytes_feature(data_id.encode('utf-8')),
            'object_index': self._int64_feature(object_index)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())

    def _create_random_crops(self, img, bbox, virtual_control_points_2d, bbox_3d_dims, number_of_crops=4):
        x_offset = y_offset = 16
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        jittered_bbox = (bbox_x1 - x_offset, bbox_y1 - y_offset, bbox_x2 + x_offset, bbox_y2 + y_offset)
        jittered_img, jittered_square_bbox = self._crop_object_from_img(img, jittered_bbox)
        j_h, j_w, _ = jittered_img.shape
        crop_dims = 5 * (224 // 8)

        jittered_img = scipy.misc.imresize(jittered_img, (224, 224))
        j_nvcp = self._normalize_2d_control_points(virtual_control_points_2d, jittered_square_bbox)
        jittered_ov = np.append(j_nvcp, bbox_3d_dims).astype(np.float)

        r_crops = []

        for i in range(number_of_crops):
            r_crop, _ = self._random_crop(jittered_img, crop_size=(crop_dims, crop_dims))
            r_crops.append(r_crop)

        return r_crops, jittered_ov

    def _random_crop(self, image, crop_size=(140, 140)):
        h, w, _ = image.shape
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        new_image = np.zeros((224, 224, 3), dtype=np.uint8)
        new_image.fill(255)
        image_section = image[top:bottom, left:right, :]

        new_image[top:bottom, left:right, :] = image_section
        return new_image, image_section

    def _crop_object_from_img(self, img, bbox):
        """
        Square crops image to the largest dimension of the bbox
        Args:
            img: Image containing the object
            bbox: bounding box for the object
        
        Returns:
            Cropped and padded image, and square bounding box
        """
        bbox = np.rint(bbox)
        center = self._get_bbox_center(bbox)
        max_dim = self._get_bbox_max_dim(bbox)

        square_bbox = self._get_square_bounding_box_dimensions(max_dim, center)

        return self._get_square_crop(square_bbox, img)

    def _get_bbox_center(self, bbox):
        """
        Calculates the center of a bounding box
        
        Args:
            bbox: Bounding box for the object
        Returns:
            center coordinates (x,y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _get_bbox_max_dim(self, bbox):
        """
        Calculates the maximum dimension of a bounding box
        
        Args:
            bbox: Bounding box for the object
        Returns:
            maximum dimension
        """
        x1, y1, x2, y2 = bbox
        return max(x2 - x1, y2 - y1)

    def _get_square_bounding_box_dimensions(self, max_dim, center):
        """
        Create new square bounding box dimensions to square crop object.
        Args:
            max_dim: The largest dimension of a bounding box
            center: Center of the bounding box
            img: image that contains the object
        Return: 
            New points of a square bounding box
        """
        center_x, center_y = center

        y_max = int(max_dim / 2 + center_y)
        y_min = int(center_y - max_dim / 2)

        x_max = int(max_dim / 2 + center_x)
        x_min = int(center_x - max_dim / 2)

        return x_min, y_min, x_max, y_max

    def _get_square_crop(self, square_bbox, img):
        """
        Create new square cropped and padded image
        Args:
            max_dim: The largest dimension of a bounding box
            center: Center of the bounding box
            img: image that contains the object
        Return: 
            Padded and square cropped image containing the object, and updated and bounding box
        """
        img_height, img_width, _ = img.shape

        # Square Bounding Box
        x_min, y_min, x_max, y_max = square_bbox

        padding_x = 0
        padding_y = 0

        if y_max > img_height:
            padding_y += (y_max - img_height)

        if y_min < 0:
            padding_y += (-y_min)
            y_min = 0

        if x_max > img_width:
            padding_x += (x_max - img_width)

        if x_min < 0:
            padding_x += (-x_min)
            x_min = 0

        pad_width = ((0, padding_y), (0, padding_x), (0, 0))
        padded_img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=255)

        return padded_img[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

    def _get_real_domain_output_vector(self, cls, class_cads, obj):
        """
        Gets output vector we need to train the real domain CNN 
        Args:
            cls: Class of object
            class_cads: Cad models for current class
            obj: data for the annotated object
        Returns:
            Array Containing:
            2D image locations of the projections of the eight 3D bounding box corners (16 values)
            3D bounding box dimensions (3 values)
        """
        cad_index = obj['cad_index']
        cad = class_cads[cls][cad_index]
        cad_vertices_3d = cad['vertices']
        bbox = obj['bbox']

        dx, dy, dz = self._compute_scalars(cad_vertices_3d)
        virtual_control_points_2d = self._get_bounding_box_corners_in_2d(cad_vertices_3d, obj['viewpoint'])

        return virtual_control_points_2d, np.array([dx, dy, dz])

    def _normalize_2d_control_points(self, control_points, bbox):
        """
        Normalize 2D Virtual Control Points 
        Args:
            control_points: list of non normalized 2D control points
            bbox: Bounding box of the object in the image. 
        Returns:
            Normalized control points (16 values)            
        """
        xmin, ymin, xmax, ymax = bbox

        new_control_points = []
        for control_point in control_points:
            cx, cy = control_point
            new_cx = (cx - xmin) / (xmax - xmin)
            new_cy = (cy - ymin) / (ymax - ymin)
            new_control_points.append((new_cx, new_cy))

        return np.array(new_control_points).flatten()

    def _compute_scalars(self, cad_model_vertices):
        """
        Compute How much we should scale each axis of the unit cube
            cad_model_vertices: The vertices of the cad model 
            
        Returns:
            scalar_x: magnitude that we must scale the unit cube in the x direction
            scalar_y: magnitude that we must scale the unit cube in the y direction
            scalar_z: magnitude that we must scale the unit cube in the z direction
        """
        x, y, z = zip(*cad_model_vertices)
        x_range = max(x) - min(x)
        y_range = max(y) - min(y)
        z_range = max(z) - min(z)

        cube_length = abs(UNIT_CUBE[7, 0] - UNIT_CUBE[0, 0])
        scalar_x = x_range / cube_length
        scalar_y = y_range / cube_length
        scalar_z = z_range / cube_length

        return scalar_x, scalar_y, scalar_z

    def _scale_unit_cube(self, cad_vertices_3d):
        """
        Scale unit cube to fit the cad model
        
        Args:
           cad_vertices_3d: The vertices of the cad model
        
        Returns:
            A unit cube scaled to fit the cad model
        """
        scalar_x, scalar_y, scalar_z = self._compute_scalars(cad_vertices_3d)
        scale_matrix = np.array([[scalar_x, 0, 0, 0], [0, scalar_y, 0, 0], [0, 0, scalar_z, 0], [0, 0, 0, 1]])
        return self.apply_transformation(UNIT_CUBE, scale_matrix)

    def _get_bounding_box_corners_in_2d(self, cad_vertices_3d, obj_viewpoint):
        """
            Scales a unit cube to fit the cad model then projects it to 2D 
        """
        scaled_cube_3d = self._scale_unit_cube(cad_vertices_3d)
        # The projection below takes care of the transformation of the unit cube into the correct angle
        return utils.project_points_3d_to_2d(scaled_cube_3d, **obj_viewpoint)

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _floats_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _augment_image(self, image, original_output_vector, blur, mirror):
        """
        Applies specified augmentations to the preprocessed image, and makes the corresponding changes
        to the output_vector
        """
        # Output Vector is Normalized!
        augmented_output_vector = original_output_vector
        augmented_image = image

        if mirror:
            augmented_image, augmented_output_vector = self._mirror_image(augmented_image, augmented_output_vector)

        if blur:
            std_x, std_y = np.random.randint(0, 20, size=2)
            augmented_image = cv2.GaussianBlur(augmented_image, (3, 3), sigmaX=std_x, sigmaY=std_y)

        return augmented_image, augmented_output_vector.astype(np.float)

    def _mirror_image(self, image, output_vector):
        """
        Flips image so that it becomes the mirror of the original image
        Also flips the virtual control points
        """
        vc_points = np.array(output_vector[:16]).reshape(8, 2)
        flipped_image = np.fliplr(image)
        flipped_vc_points = self._mirror_vc_points_2d(vc_points)
        output_vector[:16] = flipped_vc_points.flatten()
        return flipped_image, output_vector

    def _mirror_vc_points_2d(self, vc_points, image_width=1):
        """
        Flips 2d virtual control points so that they are consistent with the mirror image
        """
        vc_points[:, 0] = image_width - vc_points[:, 0]

        return vc_points

    def show_virtual_control_points(self, i, data_id):
        """
        Transform a 3D unit cube so that it becomes a 3D bounding box for CAD model
        Projects 3D bounding box to 2D to get virtual control points for objects in 2D image 
        """
        if data_id:
            data = self.get_data(i, data_id=data_id)
        else:
            data = self.get_data(i)

        img = data['img']
        print("Image")
        print(img)
        objects = data['objects']
        class_cads = data['class_cads']

        fig = plt.figure()
        ax2 = plt.subplot(1, 1, 1)
        ax2.imshow(img)

        line_colors = ['.r-', '.g-', '.b-', '.y-']

        for index, (cls, obj) in enumerate(objects):
            current_color = line_colors[index % 4]
            cad_index = obj['cad_index']
            cad = class_cads[cls][cad_index]

            # Overlays CAD Model on Image
            vertices_3d = cad['vertices']
            faces = cad['faces']
            vertices_2d = utils.project_points_3d_to_2d(vertices_3d, **obj['viewpoint'])
            p = self.create_patch_collection_from_vertices(vertices_2d, faces)
            ax2.add_collection(p)

            scaled_cube = self._scale_unit_cube(vertices_3d)

            verts = [[scaled_cube[0], scaled_cube[1], scaled_cube[3], scaled_cube[2]],
                     [scaled_cube[0], scaled_cube[1], scaled_cube[5], scaled_cube[4]],
                     [scaled_cube[4], scaled_cube[5], scaled_cube[7], scaled_cube[6]],
                     [scaled_cube[6], scaled_cube[7], scaled_cube[3], scaled_cube[2]],
                     [scaled_cube[0], scaled_cube[2], scaled_cube[6], scaled_cube[4]],
                     [scaled_cube[1], scaled_cube[3], scaled_cube[7], scaled_cube[5]]
                     ]

            collection = Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
            face_color = [0.5, 0.5, 1]
            collection.set_facecolor(face_color)

            # Projects 3D points of scaled cube to 2D points on Image
            cube_2d_projection = utils.project_points_3d_to_2d(scaled_cube, **obj['viewpoint'])

            # Connect Points to Visualize Cube
            # ACROSS
            ax2.plot(cube_2d_projection[:2, 0], cube_2d_projection[:2, 1], current_color)
            ax2.plot(cube_2d_projection[2:4, 0], cube_2d_projection[2:4, 1], current_color)
            ax2.plot(cube_2d_projection[4:6, 0], cube_2d_projection[4:6, 1], current_color)
            ax2.plot(cube_2d_projection[6:8, 0], cube_2d_projection[6:8, 1], current_color)
            # ALONG
            ax2.plot([cube_2d_projection[1, 0], cube_2d_projection[5, 0]],
                     [cube_2d_projection[1, 1], cube_2d_projection[5, 1]], current_color)
            ax2.plot([cube_2d_projection[2, 0], cube_2d_projection[6, 0]],
                     [cube_2d_projection[2, 1], cube_2d_projection[6, 1]], current_color)
            ax2.plot([cube_2d_projection[3, 0], cube_2d_projection[7, 0]],
                     [cube_2d_projection[3, 1], cube_2d_projection[7, 1]], current_color)
            ax2.plot([cube_2d_projection[0, 0], cube_2d_projection[4, 0]],
                     [cube_2d_projection[0, 1], cube_2d_projection[4, 1]], current_color)

            # VERTICAL
            ax2.plot([cube_2d_projection[2, 0], cube_2d_projection[0, 0]],
                     [cube_2d_projection[2, 1], cube_2d_projection[0, 1]], current_color)
            ax2.plot([cube_2d_projection[3, 0], cube_2d_projection[1, 0]],
                     [cube_2d_projection[3, 1], cube_2d_projection[1, 1]], current_color)
            ax2.plot([cube_2d_projection[5, 0], cube_2d_projection[7, 0]],
                     [cube_2d_projection[5, 1], cube_2d_projection[7, 1]], current_color)
            ax2.plot([cube_2d_projection[4, 0], cube_2d_projection[6, 0]],
                     [cube_2d_projection[4, 1], cube_2d_projection[6, 1]], current_color)

        plt.show()

    def apply_transformation(self, cube, transformation_matrix):
        transformed_cube = []
        for vector in cube:
            new_vector = np.dot(transformation_matrix, np.append(vector, [1]))
            transformed_cube.append(new_vector[:3])

        return np.array(transformed_cube)

    def compute_angle_between_vertices(self, cube):
        # Sanity Check Line Angles
        m1 = (cube[1, 1] - cube[0, 1]) / (cube[1, 0] - cube[0, 0])
        m2 = (cube[4, 1] - cube[0, 1]) / (cube[4, 0] - cube[0, 0])
        return np.arctan((m1 - m2) / (1 + m1 * m2))

    def show_pcd_overlay(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']

        ax1 = plt.subplot(121)
        plt.axis('off')
        ax1.imshow(img)

        ax2 = plt.subplot(122)
        plt.axis('off')

        n_classes = len(self.class_names)
        colormap = plt.cm.Spectral(
            np.linspace(0, 1, n_classes - 1))[:, :3]  # w/o background color
        colormap = np.vstack(([0, 0, 0], colormap))  # w/ background color
        for cls, obj in objects:
            cls_id = self.class_names.index(cls)
            pcd_file = obj['cad_basename'] + '.pcd'
            points_3d = utils.load_pcd(pcd_file)
            points_2d = utils.project_points_3d_to_2d(
                points_3d, **obj['viewpoint'])
            img = img.astype(np.float64)
            height, width = img.shape[:2]
            for x, y in points_2d:
                if x > width or x < 0 or y > height or y < 0:
                    continue
                img[y, x] = colormap[cls_id] * 255
            img = img.astype(np.uint8)

        ax2.imshow(img)
        plt.tight_layout()
        plt.show()

    def show_depth_by_pcd(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']

        ax1 = plt.subplot(131)
        plt.axis('off')
        plt.title('original image')
        ax1.imshow(img)

        height, width = img.shape[:2]
        depth = np.zeros((height, width), dtype=np.float64)
        depth[...] = np.nan
        max_depth = depth.copy()
        for cls, obj in objects:
            pcd_file = obj['cad_basename'] + '.pcd'
            points_3d = utils.load_pcd(pcd_file)
            points_3d_camframe = utils.transform_to_camera_frame(
                points_3d,
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            points_2d = utils.project_points_3d_to_2d(
                points_3d, **obj['viewpoint'])
            for (x, y), (_, _, z) in zip(points_2d, points_3d_camframe):
                x, y = int(x), int(y)
                if x >= width or x < 0 or y >= height or y < 0:
                    continue
                if np.isnan(depth[y, x]):
                    assert np.isnan(max_depth[y, x])
                    depth[y, x] = max_depth[y, x] = abs(z)
                else:
                    depth[y, x] = min(depth[y, x], abs(z))
                    max_depth[y, x] = max(max_depth[y, x], abs(z))

        obj_depth = max_depth - depth

        ax2 = plt.subplot(132)
        plt.axis('off')
        plt.title('depth')
        ax2.imshow(depth)

        ax2 = plt.subplot(133)
        plt.axis('off')
        plt.title('object depth')
        ax2.imshow(obj_depth)

        plt.tight_layout()
        plt.show()

    def convert_mesh_to_pcd(self, dry_run=False, replace=False):
        warnings.warn(
            'Note that this method needs pcl_mesh2pcd compiled with PCL1.8 '
            'to avoid being hanged by GUI.')
        # scrape off files
        off_files = []
        for cls in self.class_names[1:]:
            cad_dir = osp.join(self.dataset_dir, 'CAD', cls)
            for off_file in os.listdir(cad_dir):
                off_file = osp.join(cad_dir, off_file)
                if osp.splitext(off_file)[-1] == '.off':
                    off_files.append(off_file)
        # using pcl_mesh2pcd
        for off_file in off_files:
            cad_dir = osp.dirname(off_file)
            cad_id = osp.splitext(osp.basename(off_file))[0]
            obj_file = osp.join(cad_dir, cad_id + '.obj')
            pcd_file = osp.join(cad_dir, cad_id + '.pcd')
            if replace and osp.exists(pcd_file):
                os.remove(pcd_file)
            if osp.exists(pcd_file):
                if not dry_run:
                    print('PCD file exists, so skipping: {}'
                          .format(pcd_file))
                continue
            # off file -> obj file
            cmd = 'meshlabserver -i {} -o {}'.format(off_file, obj_file)
            if dry_run:
                print(cmd)
            else:
                subprocess.call(shlex.split(cmd))
            # obj file -> pcd file
            cmd = 'pcl_mesh2pcd {} {} -no_vis_result -leaf_size 0.0001' \
                .format(obj_file, pcd_file)
            if dry_run:
                print(cmd)
            else:
                subprocess.call(shlex.split(cmd))
        # using pcl_mesh_sampling
        # FIXME: sometimes pcl_mesh2pcd segfaults
        for off_file in off_files:
            cad_dir = osp.dirname(off_file)
            cad_id = osp.splitext(osp.basename(off_file))[0]
            obj_file = osp.join(cad_dir, cad_id + '.obj')
            pcd_file = osp.join(cad_dir, cad_id + '.pcd')
            if osp.exists(pcd_file):
                if not dry_run:
                    print('PCD file exists, so skipping: {}'
                          .format(pcd_file))
                continue
            # ply file -> pcd file
            cmd = 'pcl_mesh_sampling {} {} -no_vis_result -leaf_size 0.0001' \
                .format(obj_file, pcd_file)
            if dry_run:
                print(cmd)
            else:
                subprocess.call(shlex.split(cmd))

    def get_depth(self, i):
        data = self.get_data(i)

        img = data['img']
        height, width = img.shape[:2]
        objects = data['objects']
        class_cads = data['class_cads']

        depth = np.zeros((height, width), dtype=np.float64)
        depth[...] = np.inf
        max_depth = np.zeros((height, width), dtype=np.float64)
        max_depth[...] = np.inf

        for cls, obj in objects:
            cad = class_cads[cls][obj['cad_index']]
            vertices = cad['vertices']
            vertices_camframe = utils.transform_to_camera_frame(
                vertices,
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            vertices_2d = utils.project_points_3d_to_2d(
                vertices, **obj['viewpoint'])
            faces = cad['faces'] - 1

            polygons_z = np.abs(vertices_camframe[faces][:, :, 2])
            indices = np.argsort(polygons_z.max(axis=-1))

            depth_obj = np.zeros((height, width), dtype=np.float64)
            depth_obj.fill(np.nan)
            mask_obj = np.zeros((height, width), dtype=bool)
            for face in tqdm.tqdm(faces[indices]):
                xy = vertices_2d[face].ravel().tolist()
                mask_pil = PIL.Image.new('L', (width, height), 0)
                PIL.ImageDraw.Draw(mask_pil).polygon(xy=xy, outline=1, fill=1)
                mask_poly = np.array(mask_pil).astype(bool)
                mask = np.bitwise_and(~mask_obj, mask_poly)
                mask_obj[mask] = True
                #
                if mask.sum() == 0:
                    continue
                #
                ray1_xy = np.array(zip(*np.where(mask)))[:, ::-1]
                n_rays = len(ray1_xy)
                ray1_z = np.zeros((n_rays, 1), dtype=np.float64)
                ray1_xyz = np.hstack((ray1_xy, ray1_z))
                #
                ray0_z = np.ones((n_rays, 1), dtype=np.float64)
                ray0_xyz = np.hstack((ray1_xy, ray0_z))
                #
                tri0_xy = vertices_2d[face[0]]
                tri1_xy = vertices_2d[face[1]]
                tri2_xy = vertices_2d[face[2]]
                tri0_z = vertices_camframe[face[0]][2]
                tri1_z = vertices_camframe[face[1]][2]
                tri2_z = vertices_camframe[face[2]][2]
                tri0_xyz = np.hstack((tri0_xy, tri0_z))
                tri1_xyz = np.hstack((tri1_xy, tri1_z))
                tri2_xyz = np.hstack((tri2_xy, tri2_z))
                tri0_xyz = tri0_xyz.reshape(1, -1).repeat(n_rays, axis=0)
                tri1_xyz = tri1_xyz.reshape(1, -1).repeat(n_rays, axis=0)
                tri2_xyz = tri2_xyz.reshape(1, -1).repeat(n_rays, axis=0)
                #
                flags, intersection = utils.intersect3d_ray_triangle(
                    ray0_xyz, ray1_xyz, tri0_xyz, tri1_xyz, tri2_xyz)
                for x, y, z in intersection[flags == 1]:
                    depth_obj[int(y), int(x)] = -z

            max_depth_obj = np.zeros((height, width), dtype=np.float64)
            max_depth_obj.fill(np.nan)
            mask_obj = np.zeros((height, width), dtype=bool)
            for face in tqdm.tqdm(faces[indices[::-1]]):
                xy = vertices_2d[face].ravel().tolist()
                mask_pil = PIL.Image.new('L', (width, height), 0)
                PIL.ImageDraw.Draw(mask_pil).polygon(xy=xy, outline=1, fill=1)
                mask_poly = np.array(mask_pil).astype(bool)
                mask = np.bitwise_and(~mask_obj, mask_poly)
                mask_obj[mask_poly] = True
                #
                if mask.sum() == 0:
                    continue
                #
                ray1_xy = np.array(zip(*np.where(mask)))[:, ::-1]
                n_rays = len(ray1_xy)
                ray1_z = np.zeros((n_rays, 1), dtype=np.float64)
                ray1_xyz = np.hstack((ray1_xy, ray1_z))
                #
                ray0_z = np.ones((n_rays, 1), dtype=np.float64)
                ray0_xyz = np.hstack((ray1_xy, ray0_z))
                #
                tri0_xy = vertices_2d[face[0]]
                tri1_xy = vertices_2d[face[1]]
                tri2_xy = vertices_2d[face[2]]
                tri0_z = vertices_camframe[face[0]][2]
                tri1_z = vertices_camframe[face[1]][2]
                tri2_z = vertices_camframe[face[2]][2]
                tri0_xyz = np.hstack((tri0_xy, tri0_z))
                tri1_xyz = np.hstack((tri1_xy, tri1_z))
                tri2_xyz = np.hstack((tri2_xy, tri2_z))
                tri0_xyz = tri0_xyz.reshape(1, -1).repeat(n_rays, axis=0)
                tri1_xyz = tri1_xyz.reshape(1, -1).repeat(n_rays, axis=0)
                tri2_xyz = tri2_xyz.reshape(1, -1).repeat(n_rays, axis=0)
                #
                flags, intersection = utils.intersect3d_ray_triangle(
                    ray0_xyz, ray1_xyz, tri0_xyz, tri1_xyz, tri2_xyz)
                for x, y, z in intersection[flags == 1]:
                    max_depth_obj[int(y), int(x)] = -z

            depth[mask_obj] = np.minimum(
                depth[mask_obj], depth_obj[mask_obj])
            max_depth[mask_obj] = np.minimum(
                max_depth[mask_obj], max_depth_obj[mask_obj])

        depth[np.isinf(depth)] = np.nan
        max_depth[np.isinf(max_depth)] = np.nan

        return depth, max_depth
