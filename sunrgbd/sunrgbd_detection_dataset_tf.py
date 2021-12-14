# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
from tensorflow import keras
import tensorflow as tf
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import json
environ_file = os.path.join(ROOT_DIR,'configs','environ.json')
environ = json.load(open(environ_file))['environ']

#DATA_DIR = os.path.dirname(ROOT_DIR)
if environ == 'server':    
    DATA_DIR = '/home/aiot/data'
elif environ == 'jetson':
    DATA_DIR='/media'
elif environ == 'server2':    
    DATA_DIR = '/data'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import sunrgbd_utils
from model_util_sunrgbd import SunrgbdDatasetConfig

MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

class SunrgbdDetectionVotesDataset(keras.utils.Sequence):
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, use_v1=False,
        augment=False, scan_idx_list=None, DC=None):

        assert(num_points<=50000)
        self.use_v1 = use_v1 
        if use_v1:
            #self.data_path = os.path.join(ROOT_DIR,
                #'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_%s'%(split_set))
            self.data_path = os.path.join(DATA_DIR,
                'sunrgbd_pc_bbox_votes_50k_v1_%s'%(split_set))
        else:
            #self.data_path = os.path.join(ROOT_DIR,
                #'sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_%s'%(split_set))
            self.data_path = os.path.join(DATA_DIR,
                'sunrgbd_pc_bbox_votes_50k_v2_%s'%(split_set))

        self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.DC = DC
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy') # K,8
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes'] # Nx10

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:,0] = -1 * bboxes[:,0]
                bboxes[:,6] = np.pi - bboxes[:,6]
                point_votes[:,[1,4,7]] = -1 * point_votes[:,[1,4,7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:,3:6] + MEAN_COLOR_RGB
                rgb_color *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                rgb_color += (0.1*np.random.random(3)-0.05) # color shift for each channel
                rgb_color += np.expand_dims((0.05*np.random.random(point_cloud.shape[0])-0.025), -1) # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0])>0.3,-1)
                point_cloud[:,3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random()*0.3+0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0)
            point_cloud[:,0:3] *= scale_ratio
            bboxes[:,0:3] *= scale_ratio
            bboxes[:,3:6] *= scale_ratio
            point_votes[:,1:4] *= scale_ratio
            point_votes[:,4:7] *= scale_ratio
            point_votes[:,7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio[0,0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0],:] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = self.DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6]*2
            size_class, size_residual = self.DC.size2class(box3d_size, self.DC.class2type[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size

        target_bboxes_mask = label_mask 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]
        
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        import tensorflow as tf
        ret_dict = {}
        ret_dict['point_clouds'] = tf.convert_to_tensor(point_cloud, dtype=tf.float32)
        ret_dict['center_label'] = tf.convert_to_tensor(target_bboxes[:,0:3], dtype=tf.float32)
        ret_dict['heading_class_label'] = tf.convert_to_tensor(angle_classes, dtype=tf.int32)
        ret_dict['heading_residual_label'] = tf.convert_to_tensor(angle_residuals, dtype=tf.float32)
        ret_dict['size_class_label'] = tf.convert_to_tensor(size_classes, dtype=tf.int32)
        ret_dict['size_residual_label'] = tf.convert_to_tensor(size_residuals, dtype=tf.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = tf.convert_to_tensor(target_bboxes_semcls, dtype=tf.int32)
        ret_dict['box_label_mask'] = tf.convert_to_tensor(target_bboxes_mask, dtype=tf.float32)
        ret_dict['vote_label'] = tf.convert_to_tensor(point_votes, dtype=tf.float32)
        ret_dict['vote_label_mask'] = tf.convert_to_tensor(point_votes_mask, dtype=tf.int32)
        ret_dict['scan_idx'] = tf.convert_to_tensor(np.array(idx), dtype=tf.int32)
        ret_dict['max_gt_bboxes'] = tf.convert_to_tensor(max_bboxes, dtype=tf.float32)
        return ret_dict



N_POINT = 50000
N_BOX = MAX_NUM_OBJ = 64 # same as MAX_NUM_OBJ in sunrgbd_data.py

class SunrgbdDetectionVotesDataset_tfrecord():
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, 
        augment=False, batch_size=8, shuffle=True,
        use_painted=False, DC=None):

        assert(num_points<=50000)
        self.use_painted = use_painted
        self.dim_features = 6        
        self.DC = DC
        self.num_class = DC.num_class

        if self.use_painted:
            self.dim_features = 3 + (self.num_class + 1) + 1 # xyz + num_class + 1(background) + 1(isPainted)

        if DC.include_person:
            self.data_path = os.path.join(DATA_DIR,'sunrgbd_pc_%s_painted_tf_person'%(split_set))
        elif self.use_painted:
            self.data_path = os.path.join(DATA_DIR,'sunrgbd_pc_%s_painted_tf3'%(split_set))
        else:
            self.data_path = os.path.join(DATA_DIR,'sunrgbd_pc_%s_tf'%(split_set))

        print(self.data_path)

        self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.tfrecords_pattern_path = "sunrgbd_*-of-*.records"
        self.files = tf.io.matching_files(os.path.join(self.data_path, self.tfrecords_pattern_path))
        #self.files = tf.io.matching_files(os.path.join(self.data_path, 'sunrgbd_00000-of-00052.records'))
        if self.shuffle:
            self.files = tf.random.shuffle(self.files)     
        shards = tf.data.Dataset.from_tensor_slices(self.files)
        dataset = shards.interleave(tf.data.TFRecordDataset)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size) 
        self.dataset = dataset       
        
        
        self.type_mean_size_np = np.zeros((DC.num_class, 3))
        for i in range(self.num_class):
            self.type_mean_size_np[i,:] = DC.type_mean_size[DC.class2type[i]]
        
        
        
    def preprocess(self):
           
        # Reshape
        self.dataset = self.dataset.map(map_func=self.reshape_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
 
        if self.use_color:
            self.dataset = self.dataset.map(self.preprocess_color, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.use_height:
            self.dataset = self.dataset.map(self.preprocess_height, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
        if self.augment:
            self.dataset = self.dataset.map(self.augment_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        self.dataset = self.dataset.map(self.sample_points, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.map(self.tf_get_output) #tf.data.experimental.AUTOTUNE)
        
        return self.dataset

    def _parse_function(self, example_proto):
        feature_description = {    
            'point_cloud': tf.io.FixedLenFeature([N_POINT*self.dim_features], tf.float32),                
            'bboxes': tf.io.FixedLenFeature([N_BOX*8],tf.float32),
            'point_votes': tf.io.FixedLenFeature([N_POINT*10], tf.float32),    
            'n_valid_box': tf.io.FixedLenFeature([], tf.int64)
        }
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    def reshape_tensor(self, features):        
        point_cloud = tf.reshape(features['point_cloud'], [-1, N_POINT, self.dim_features])            
        bboxes = tf.reshape(features['bboxes'], [-1, N_BOX, 8])
        point_votes = tf.reshape(features['point_votes'], [-1, N_POINT, 10])
        n_valid_box = tf.reshape(features['n_valid_box'], [-1])
        
        return point_cloud, bboxes, point_votes, n_valid_box

    def preprocess_color(self, point_cloud, bboxes, point_votes, n_valid_box):    
        # Not used
        MEAN_COLOR_RGB=tf.constant([0.5,0.5,0.5])
        point_cloud = point_cloud[:,:,0:6]
        pc_coord = point_cloud[:,:,:3]
        pc_RGB = point_cloud[:,:,3:]-MEAN_COLOR_RGB
        point_cloud = tf.concat((pc_coord, pc_RGB), axis=-1)
        return point_cloud, bboxes, point_votes, n_valid_box

    def preprocess_height(self, point_cloud, bboxes, votes, n_valid_box):
        y_coords = point_cloud[:, :, 2]
        y_coords = tf.sort(y_coords, direction='DESCENDING', axis=-1)
        floor_height = y_coords[:, int(0.99*N_POINT), None]         
        height = point_cloud[:,:,2] - tf.tile(floor_height, [1,N_POINT])
        if self.use_painted or self.use_color:
            point_cloud = tf.concat([point_cloud, tf.expand_dims(height, axis=-1)], axis=-1) # (N, C+1)
        #elif not self.augment:
        else:
            point_cloud = tf.concat([point_cloud[:,:,:3], tf.expand_dims(height, axis=-1)], axis=-1) # (N,4)
        return point_cloud, bboxes, votes, n_valid_box
    
    def augment_tensor(self, point_cloud, bboxes, point_votes, n_valid_box):
        pi = 3.141592653589793
        # Flipping along the YZ plane
        if tf.random.uniform(shape=[]) > 0.5:        
        #if True:
            pc_flip = -1 * point_cloud[:,:,0,None]
            point_cloud = tf.concat((pc_flip, point_cloud[:,:,1:]), axis=-1)
            bboxes_flip = -1 * bboxes[:,:,0,None]
            bboxes_angle = pi -1 * bboxes[:,:,6,None]
            bboxes = tf.concat((bboxes_flip, bboxes[:,:,1:6], bboxes_angle, bboxes[:,:,7:]), axis=-1)  
            votes1_flip = -1 * point_votes[:,:,1,None]
            votes4_flip = -1 * point_votes[:,:,4,None]
            votes7_flip = -1 * point_votes[:,:,7,None]
            point_votes = tf.concat((point_votes[:,:,0,None], votes1_flip, point_votes[:,:,2:4], votes4_flip, point_votes[:,:,5:7], votes7_flip, point_votes[:,:,8:]), axis=-1)
            
        pc_coords = point_cloud[:,:,0:3]    
        pc_height = point_cloud[:,:,-1,None]
        
        # Keep painting information separately
        if self.use_painted:            
            pc_painting = point_cloud[:,:,3:-1]
        
        # Rotation along up-axis/Z-axis
        rot_angle = (tf.random.uniform(shape=[])*pi/3) - pi/6    
        #rot_angle = (0.1*pi/3) - pi/6    
        c = tf.math.cos(rot_angle)
        s = tf.math.sin(rot_angle)
        rot_mat = tf.transpose(tf.convert_to_tensor([[c, -s, 0], [s, c, 0], [0,0,1]]), [1,0])
            
        point_votes_end1 = tf.matmul(pc_coords + point_votes[:,:,1:4], rot_mat)
        point_votes_end2 = tf.matmul(pc_coords + point_votes[:,:,4:7], rot_mat)
        point_votes_end3 = tf.matmul(pc_coords + point_votes[:,:,7:10], rot_mat)
        
        pc_coords = tf.matmul(pc_coords, rot_mat)    
        
        bboxes_coords = tf.matmul(bboxes[:,:,0:3], rot_mat)
        bboxes_angle = bboxes[:,:,6,None] - rot_angle    
        
        votes0 = point_votes[:,:,0,None]
        votes1 = point_votes_end1 - pc_coords
        votes2 = point_votes_end2 - pc_coords
        votes3 = point_votes_end3 - pc_coords

        # Augment point cloud scale: 0.85x-1.15x
        scale_ratio = tf.random.uniform(shape=[])*0.3+0.85
        #scale_ratio = 0.1*0.3+0.85    
        pc_coords = pc_coords * scale_ratio
        bboxes_coords = bboxes_coords * scale_ratio
        bboxes_edges = bboxes[:,:,3:6]
        bboxes_edges = bboxes_edges * scale_ratio
        votes1 = votes1 * scale_ratio
        votes2 = votes2 * scale_ratio
        votes3 = votes3 * scale_ratio    
        
        if self.use_height:
            pc_height = pc_height * scale_ratio
        
        # Augment RGB color
        if self.use_color:
            MEAN_COLOR_RGB=tf.constant([0.5,0.5,0.5])
            rgb_color = point_cloud[:,:,3:6] + MEAN_COLOR_RGB
            rgb_color = rgb_color * (1+0.4*tf.random.uniform([3])-0.2) # brightness change for each channel
            rgb_color = rgb_color + (0.1*tf.random.uniform([3])-0.05) # color shift for each channel
            rgb_color = rgb_color + tf.expand_dims((0.05*tf.random.uniform([B, N_POINT])-0.025), -1) # jittering on each pixel        
            rgb_color = tf.clip_by_value(rgb_color, 0, 1)
            # randomly drop out 30% of the points' colors
            rgb_color = rgb_color * tf.expand_dims(tf.where(tf.random.uniform([B, N_POINT])>0.3, 1.0, 0.0),-1)
            rgb_color = rgb_color - MEAN_COLOR_RGB
            point_cloud = tf.concat((pc_coords, rgb_color, pc_height), axis=-1)
        else:
            if self.use_painted:
                point_cloud = tf.concat((pc_coords, pc_painting, pc_height), axis=-1)
            else:
                point_cloud = tf.concat((pc_coords, pc_height), axis=-1)
        
        bboxes = tf.concat((bboxes_coords, bboxes_edges, bboxes_angle, bboxes[:,:,7:]), axis=-1)
        point_votes = tf.concat((votes0, votes1, votes2, votes3, point_votes[:,:,10:]), axis=-1)           
        
        return point_cloud, bboxes, point_votes, n_valid_box

    def sample_points(self, point_cloud, bboxes, point_votes, n_valid_box):        
        n_pc = point_cloud.shape[1]        

        choice_indices = tf.random.uniform([self.num_points], minval=0, maxval=n_pc, dtype=tf.int64)
        choice_indices = tf.tile(tf.expand_dims(choice_indices,0), [tf.shape(point_cloud)[0],1])
        
        point_cloud = tf.gather(point_cloud, choice_indices, axis=1, batch_dims=1)
        point_votes = tf.gather(point_votes, choice_indices, axis=1, batch_dims=1)        
    
        return point_cloud, bboxes, point_votes, n_valid_box

    def _get_output(self, point_cloud, bboxes, point_votes, n_valid_box):
        #print(point_cloud.shape)
        B = point_cloud.shape[0]
        
        box3d_centers = np.zeros((B, MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((B, MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((B, MAX_NUM_OBJ,))
        angle_residuals = np.zeros((B, MAX_NUM_OBJ,))
        size_classes = np.zeros((B, MAX_NUM_OBJ,))
        size_residuals = np.zeros((B, MAX_NUM_OBJ, 3))
        label_mask = np.zeros((B, MAX_NUM_OBJ))
        
        max_bboxes = np.zeros((B, MAX_NUM_OBJ, 8))
        target_bboxes = np.zeros((B, MAX_NUM_OBJ, 6)) 
        target_bboxes_semcls = np.zeros((B, MAX_NUM_OBJ))

        #point_cloud_sampled = np.zeros((B, self.num_points, point_cloud.shape[2]))
        #point_votes_sampled = np.zeros((B, self.num_points, point_votes.shape[2]-1))
        #point_votes_mask = np.zeros((B, self.num_points))

        for b in range(B):
            n_box = n_valid_box[b]
            
            label_mask[b,:n_box] = 1
            max_bboxes[b,:n_box,:] = bboxes[b,:n_box,:]

            for i in range(n_box):
                bbox = bboxes[b,i]
                semantic_class = bbox[7]
                box3d_center = bbox[0:3]
                angle_class, angle_residual = self.DC.angle2class(bbox[6])
                # NOTE: The mean size stored in size2class is of full length of box edges,
                # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
                #box3d_size = bbox[3:6]*2
                box3d_size = bbox[3:6]
                #box3d_size *= 2
                size_class = semantic_class         
                #mean_size =  self.type_mean_size_np[size_class.numpy().astype(np.int32)]         
                #size_residual = 2 * box3d_size - mean_size
                size_residual = 2 * box3d_size - self.type_mean_size_np[size_class.numpy().astype(np.int32)]
                #size_class, size_residual = size2class(box3d_size, class2type[semantic_class])
                box3d_centers[b,i,:] = box3d_center
                angle_classes[b,i] = angle_class
                angle_residuals[b,i] = angle_residual
                size_classes[b,i] = size_class
                size_residuals[b,i] = size_residual
                box3d_sizes[b,i,:] = 2 * box3d_size
            
            target_bboxes_mask = label_mask 
                   
            
            for i in range(n_box):
                bbox = bboxes[b,i]
                corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
                # compute axis aligned box
                xmin = np.min(corners_3d[:,0])
                ymin = np.min(corners_3d[:,1])
                zmin = np.min(corners_3d[:,2])
                xmax = np.max(corners_3d[:,0])
                ymax = np.max(corners_3d[:,1])
                zmax = np.max(corners_3d[:,2])
                target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
                target_bboxes[b,i,:] = target_bbox

            target_bboxes_semcls[b,:n_box] = bboxes[b,:n_box,-1] # from 0 to 9
        
        point_cloud = tf.constant(point_cloud, dtype=tf.float32)        
        center_label = tf.constant(target_bboxes[:,:,0:3], dtype=tf.float32)
        heading_class_label = tf.constant(angle_classes, dtype=tf.int64)
        heading_residual_label = tf.constant(angle_residuals, dtype=tf.float32)
        size_class_label = tf.constant(size_classes, dtype=tf.int64)
        
        size_residual_label = tf.constant(size_residuals, dtype=tf.float32)
        sem_cls_label = tf.constant(target_bboxes_semcls, dtype=tf.int64)
        box_label_mask = tf.constant(target_bboxes_mask, dtype=tf.float32)
        vote_label = tf.constant(point_votes[:,:,1:], dtype=tf.float32)
        vote_label_mask = tf.constant(tf.cast(point_votes[:,:,0],dtype=tf.int64), dtype=tf.int64)
        
        max_gt_bboxes = tf.constant(max_bboxes, dtype=tf.float32) 
        
        output = point_cloud, center_label, heading_class_label, heading_residual_label, size_class_label, \
            size_residual_label, sem_cls_label, box_label_mask, vote_label, vote_label_mask, max_gt_bboxes
        return output

    def tf_get_output(self, point_cloud, bboxes, point_votes, n_valid_box):

        [point_cloud, center_label, heading_class_label, heading_residual_label, size_class_label, \
            size_residual_label, sem_cls_label, box_label_mask, vote_label, vote_label_mask, max_gt_bboxes] \
                = tf.py_function(func=self._get_output, inp=[point_cloud, bboxes, point_votes, n_valid_box],
                                 Tout= [tf.float32, tf.float32, tf.int64, tf.float32, tf.int64,  
                                        tf.float32, tf.int64, tf.float32, tf.float32, tf.int64,
                                        tf.float32])
        #return tuple(output)
        return point_cloud, center_label, heading_class_label, heading_residual_label, size_class_label, \
            size_residual_label, sem_cls_label, box_label_mask, vote_label, vote_label_mask, max_gt_bboxes
    


def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i%10==0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)

if __name__=='__main__':
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sample = d[200]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])



