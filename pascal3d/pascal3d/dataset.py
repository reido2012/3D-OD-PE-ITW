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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from itertools import product, combinations
from tqdm import tqdm
import numpy as np
import PIL.Image
import PIL.ImageDraw
import scipy.io
import scipy.misc
import skimage.color
import sklearn.model_selection
import tqdm

from pascal3d import utils


UNIT_CUBE = np.array(list(product([-1, 1], [-1, 1], [-1, 1])))
DATASET_DIR = osp.expanduser('/notebooks/selerio/datasets/pascal3d/PASCAL3D+_release1.1')
IMAGENET_IMAGESET_DIR = DATASET_DIR + "/Image_sets/"
PASCAL_IMAGESET_DIR = DATASET_DIR + "/PASCAL/VOCdevkit/VOC2012/ImageSets/Main/"

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

            self.objects.append({
                'cad_index': cad_index,
                'bbox': bbox,
                'anchors': anchors,
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

    def __init__(self, data_type, generate=True):
        assert data_type in ('train', 'val', 'all')
        self.dataset_dir = osp.expanduser(
            '/notebooks/selerio/datasets/pascal3d/PASCAL3D+_release1.1')
        # get all data ids
        if generate:
            print('Generating index for annotations...')
            data_ids = []
            for counter, cls in enumerate(self.class_names[1:]):
                pascal_cls_ann_dir = osp.join(self.dataset_dir, 'Annotations/{}_pascal'.format(cls))
                imagenet_class_ann_dir = osp.join(self.dataset_dir,'Annotations/{}_imagenet'.format(cls))
                all_annotation_dirs = glob.glob(pascal_cls_ann_dir) + glob.glob(imagenet_class_ann_dir)
                for annotation_directory in all_annotation_dirs:
                    for ann_file in glob.glob(annotation_directory + "/*.mat"):
                        ann = Pascal3DAnnotation(ann_file)
                        if not ann.segmented:
                            continue
                        #print("Ann File Path: {}".format(ann_file))
                        data_id = ann_file.split("/")[-1][:-4]
                        #print("Data Id: {}".format(data_id))
                        data_ids.append(data_id)
                    break
                break
                    

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
            ann_file = osp.join(self.dataset_dir,'Annotations/{}_pascal/{}.mat'.format(class_name, data_id))
            if not osp.exists(ann_file):
                ann_file = osp.join(self.dataset_dir,'Annotations/{}_imagenet/{}.mat'.format(class_name, data_id))
                if not osp.exists(ann_file):
                    continue
            
            #print("Ann File: {}".format(ann_file))
            ann = Pascal3DAnnotation(ann_file)
            #When creating the TF.Records we don't care about label cls
            if data['label_cls'] is None and ann.database != "ImageNet" and data_id is None:
                label_cls_file = osp.join(
                    self.dataset_dir,
                    'PASCAL/VOCdevkit/VOC2012/SegmentationClass/{}.png'.format(data_id))
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
                    needle  = 'Images/{}_imagenet'

                img_file = osp.join(
                    self.dataset_dir,
                    needle.format(class_name),
                    ann.img_filename)
                
                #print("Path To Image: {}".format(img_file))
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

    def show_cad_overlay(self, i):
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
                points = [vertices_2d[i_vertex-1] for i_vertex in face]
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
            points = [vertices_2d[i_vertex-1] for i_vertex in face]
            poly = Polygon(points, True)
            patches.append(poly)
    
        return PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    
    def _get_pascal_data_ids(self):
        """
        Read data ids from pascal train, val and test text files
        """
        files = [PASCAL_IMAGESET_DIR + "train.txt", PASCAL_IMAGESET_DIR + "test.txt",  PASCAL_IMAGESET_DIR + "val.txt"]        
        return list(map(self._read_ids_from_file, files))
    
    def _get_imagenet_ids(self):
        """
        Read data ids from imagenet train and val text files
        """
        training_files = glob.glob(IMAGENET_IMAGESET_DIR + "*_imagenet_train.txt")
        validation_files = glob.glob(IMAGENET_IMAGESET_DIR + "*_imagenet_val.txt")
        
        training_ids = list(map(self._read_ids_from_file, training_files))
        all_training_ids = [data_id for id_list in training_ids for data_id in id_list] #Flatten
        
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
    
        
    def create_tfrecords(self, dataset_type):
        """
        Create TF.Records file for traning or dataset
            dataset_type: traning or validation TF.Records
        """
        
        #Get all text files containing ids
        pascal_train_ids, pascal_test_ids, pascal_val_ids = self._get_pascal_data_ids()        
        imagenet_train_ids, imagenet_val_ids = self._get_imagenet_ids()
        
        record_map ={
            #"pascal_train": pascal_train_ids,
            "pascal_test": pascal_test_ids,
            "pascal_val": pascal_val_ids,
            #"imagenet_train": imagenet_train_ids,
            #"imagenet_val": imagenet_val_ids
        }
        
        for name, id_list in record_map.items():
            tfrecords_filename = '{}.tfrecords'.format(name)
            print("Starting: {}".format(tfrecords_filename))
            self._create_tfrecords_from_data_ids(tfrecords_filename, id_list)
            print("Finished: {}".format(tfrecords_filename))
            
            
    def _create_tfrecords_from_data_ids(self, record_name, ids):
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
                #print("Annotation Not Found -  Data ID: {}".format(data_id))
                skipped.append(str(data_id) + "\n")
            
            img = data['img']

            objects = data['objects']
            class_cads = data['class_cads']

            # Create a TF Record for each object in record
            for cls, obj in objects:
                output_vector = self._get_real_domain_output_vector(cls, class_cads, obj).astype(np.float)
                bbox = obj['bbox']
                cropped_img = self._crop_object_from_img_and_resize(img, bbox)        
                img_raw = cropped_img.tostring()

                feature = {
                    'object_image':  self._bytes_feature(img_raw),
                    'output_vector': self._floats_feature(output_vector)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                    
        writer.close()
        
        print("Skipped {} images".format(len(skipped)))
        print("*********************")
        print(skipped)
        
        with open(record_name + '_skipped.txt', 'w') as file:
            file.writelines(skipped)
        
    def _crop_object_from_img_and_resize(self, img, bbox):
        """
        Square crops image to the largest dimension of the bbox, then resizes to 224x224
        Args:
            img: Image containing the object
            bbox: bounding box for the object
        
        Returns:
            Cropped and resized image
        """
        x1, y1, x2, y2 = np.rint(bbox)
        center_x, center_y = ((x1+x2)//2 , (y1+y2)//2)
        width = x2 - x1
        height = y2 - y1
        max_dim = int(max(width, height))
        cropped_img = self._crop_to_bounding_box(max_dim, (center_x, center_y), img)
        return scipy.misc.imresize(cropped_img, (224,224))
    
    def _crop_to_bounding_box(self, max_dim, center, img):
        """
        Calculate the coordinates of the new square bounding box and use it to crop image
        Args:
            max_dim: The largest dimension of a bounding box
            center: Center of the bounding box
            img: image that contains the object
            
        Returns:
            Image of an object contained in square bounding box
        """
        new_y_min, new_y_max, new_x_min, new_x_max, max_padding = self._get_square_bounding_box_dimensions(max_dim, (center[0],center[1]), img)
        #print("Image Shape: {}".format(img.shape))       
        if max_padding > 0:
            if len(img.shape) == 3:
                pad_width = ((max_padding, max_padding), (max_padding, max_padding), (0,0))
            else:
                #Greyscale
                pad_width = ((max_padding, max_padding), (max_padding, max_padding))

            padded_img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=255)
            return padded_img[new_y_min:new_y_max , new_x_min:new_x_max]
        else:
            return img[new_y_min:new_y_max , new_x_min:new_x_max]
      
    def _get_square_bounding_box_dimensions(self, max_dim, center, img):
        """
        Create new square bounding box dimensions to square crop object.
        Args:
            max_dim: The largest dimension of a bounding box
            center: Center of the bounding box
            img: image that contains the object
        Return: 
            New points of a square bounding box
        """
        img_height = img.shape[0]
        img_width = img.shape[1]
        center_x, center_y = center
        
        new_y_max = int(max_dim//2 + center_y)
        new_y_min = int(center_y - max_dim//2)
        new_x_max = int(max_dim//2 + center_x)
        new_x_min = int(center_x - max_dim//2) 
        
        #Pad the image if the bounding box is out of bounds
        padding_x = 0
        padding_y = 0
        
        if new_y_max > img_height:
            padding_y = new_y_max - img_height
        if new_y_min < 0:
            padding_y = -new_y_min
        if new_x_max > img_width:
            padding_x = new_x_max - img_height
        if new_x_min < 0:
            padding_x = -new_x_min
      
        max_padding = max(padding_x, padding_y)
        
        new_x_min += max_padding
        new_x_max += max_padding
        new_y_min += max_padding
        new_y_max += max_padding
        
        return new_y_min, new_y_max, new_x_min, new_x_max, max_padding
    
    def _get_real_domain_output_vector(self, cls, class_cads, obj):
        """
        Gets output vector we need to train the real domain CNN 
        
        Returns:
            Array Containing:
            2D image locations of the projections of the eight 3D bounding box corners (16 values)
            3D bounding box dimensions (3 values)
        """
        cad_index = obj['cad_index']
        cad = class_cads[cls][cad_index]
        cad_vertices_3d = cad['vertices']
        
        dx, dy, dz = self._compute_scalars(cad_vertices_3d)
        virtual_control_points_2d = self._get_bounding_box_corners_in_2d(cad_vertices_3d, obj['viewpoint']) 
              
        return  np.append(virtual_control_points_2d.flatten(), np.array([dx, dy, dz]))
        
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
 
        cube_length = UNIT_CUBE[7,0] - UNIT_CUBE[0,0]
        scalar_x = x_range/cube_length
        scalar_y = y_range/cube_length
        scalar_z = z_range/cube_length
            
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
        scale_matrix = np.array([[scalar_x, 0, 0, 0], [0, scalar_y, 0 ,0 ], [0, 0, scalar_z, 0 ], [0, 0, 0, 1]])
        return self.apply_transformation(UNIT_CUBE, scale_matrix)
        
    def _get_bounding_box_corners_in_2d(self, cad_vertices_3d, obj_viewpoint):
        """
            Scales a unit cube to fit the cad model then projects it to 2D 
        """
        scaled_cube_3d = self._scale_unit_cube(cad_vertices_3d)
        #The projection below takes care of the transformation of the unit cube into the correct angle
        return utils.project_points_3d_to_2d(scaled_cube_3d, **obj_viewpoint)
        
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _floats_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    
    def show_virtual_control_points(self, i):
        """
        Transform a 3D unit cube so that it becomes a 3D bounding box for CAD model
        Projects 3D bounding box to 2D to get virtual control points for objects in 2D image 
        """
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

        fig = plt.figure()
        ax2 = plt.subplot(1,1,1)
        ax2.imshow(img)

        line_colors = ['.r-', '.g-', '.b-', '.y-']
        
        for index, (cls, obj) in enumerate(objects):
            current_color = line_colors[index%4]
            cad_index = obj['cad_index']
            cad = class_cads[cls][cad_index]
            
            # Overlays CAD Model on Image
            vertices_3d = cad['vertices']
            faces = cad['faces']
            vertices_2d = utils.project_points_3d_to_2d(vertices_3d, **obj['viewpoint'])
            p = self.create_patch_collection_from_vertices(vertices_2d, faces)
            ax2.add_collection(p)
        
            scaled_cube = self._scale_unit_cube(vertices_3d)

            verts = [[scaled_cube[0], scaled_cube[1],scaled_cube[3],scaled_cube[2]], 
                     [scaled_cube[0], scaled_cube[1], scaled_cube[5], scaled_cube[4]],
                     [scaled_cube[4], scaled_cube[5], scaled_cube[7], scaled_cube[6]],
                     [scaled_cube[6], scaled_cube[7], scaled_cube[3], scaled_cube[2]],
                     [scaled_cube[0], scaled_cube[2], scaled_cube[6], scaled_cube[4]],
                     [scaled_cube[1], scaled_cube[3], scaled_cube[7], scaled_cube[5]]
                    ]    
            
            collection = Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
            face_color = [0.5, 0.5, 1]
            collection.set_facecolor(face_color)
            
            #Projects 3D points of scaled cube to 2D points on Image
            cube_2d_projection = utils.project_points_3d_to_2d(scaled_cube, ** obj['viewpoint'])
            
            # Connect Points to Visualize Cube
            # ACROSS
            ax2.plot(cube_2d_projection[:2,0], cube_2d_projection[:2,1],current_color)
            ax2.plot(cube_2d_projection[2:4,0], cube_2d_projection[2:4,1],current_color)
            ax2.plot(cube_2d_projection[4:6,0], cube_2d_projection[4:6,1],current_color)
            ax2.plot(cube_2d_projection[6:8,0], cube_2d_projection[6:8,1],current_color)
            #ALONG
            ax2.plot([cube_2d_projection[1,0],cube_2d_projection[5,0]],
                     [cube_2d_projection[1,1],cube_2d_projection[5,1]], current_color)
            ax2.plot([cube_2d_projection[2,0],cube_2d_projection[6,0]],
                     [cube_2d_projection[2,1],cube_2d_projection[6,1]], current_color)
            ax2.plot([cube_2d_projection[3,0],cube_2d_projection[7,0]],
                     [cube_2d_projection[3,1],cube_2d_projection[7,1]], current_color)
            ax2.plot([cube_2d_projection[0,0],cube_2d_projection[4,0]],
                     [cube_2d_projection[0,1],cube_2d_projection[4,1]],current_color)
            
            #VERTICAL
            ax2.plot([cube_2d_projection[2,0],cube_2d_projection[0,0]],
                     [cube_2d_projection[2,1],cube_2d_projection[0,1]],current_color)
            ax2.plot([cube_2d_projection[3,0],cube_2d_projection[1,0]],
                     [cube_2d_projection[3,1],cube_2d_projection[1,1]],current_color)
            ax2.plot([cube_2d_projection[5,0],cube_2d_projection[7,0]],
                     [cube_2d_projection[5,1],cube_2d_projection[7,1]],current_color)
            ax2.plot([cube_2d_projection[4,0],cube_2d_projection[6,0]],
                     [cube_2d_projection[4,1],cube_2d_projection[6,1]], current_color)
            
        plt.show()

        
    def apply_transformation(self, cube, transformation_matrix):
        transformed_cube = []
        for vector in cube:
            new_vector = np.dot(transformation_matrix, np.append(vector,[1]))
            transformed_cube.append(new_vector[:3])

        return np.array(transformed_cube)
 
    
    def compute_angle_between_vertices(cube):
        #Sanity Check Line Angles
        m1 = (cube[1,1] - cube[0,1])/(cube[1,0] - cube[0, 0])
        m2 =(cube[4,1] - cube[0,1])/(cube[4,0] - cube[0, 0])
        return np.arctan((m1 - m2)/(1 + m1*m2))   
   
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
            np.linspace(0, 1, n_classes-1))[:, :3]   # w/o background color
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
            cmd = 'pcl_mesh2pcd {} {} -no_vis_result -leaf_size 0.0001'\
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
            cmd = 'pcl_mesh_sampling {} {} -no_vis_result -leaf_size 0.0001'\
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
