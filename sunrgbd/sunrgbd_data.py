# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: December, 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed depdency on mayavi).
Load depth with scipy.io
'''

import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../deeplab'))
import pc_util
import sunrgbd_utils
from tqdm import tqdm
import tensorflow as tf
import time
#from deeplab import run_semantic_segmentation_graph
import deeplab
import pickle

import json
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = 'data_dir'

DEFAULT_TYPE_WHITELIST = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training', use_v1=False):
        self.root_dir = root_dir
        self.split = split
        assert(self.split=='training') 
        self.split_dir = os.path.join(root_dir)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        if use_v1:
            self.label_dir = os.path.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return sunrgbd_utils.load_image(img_filename)

    def get_image2(self, idx):
        "This returns PIL.Image result"
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return Image.open(img_filename)

    def get_depth(self, idx): 
        depth_filename = os.path.join(self.depth_dir, '%06d.mat'%(idx))
        return sunrgbd_utils.load_depth_points_mat(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return sunrgbd_utils.SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return sunrgbd_utils.read_sunrgbd_label(label_filename)

def data_viz(data_dir, dump_dir=os.path.join(BASE_DIR, 'data_viz_dump')):  
    ''' Examine and visualize SUN RGB-D data. '''
    sunrgbd = sunrgbd_object(data_dir)
    idxs = np.array(range(1,len(sunrgbd)+1))
    np.random.seed(0)
    np.random.shuffle(idxs)
    for idx in range(len(sunrgbd)):
        data_idx = 4449#data_idx = idxs[idx]
        print('-'*10, 'data index: ', data_idx)
        pc = sunrgbd.get_depth(data_idx)
        print('Point cloud shape:', pc.shape)
        
        # Project points to image
        calib = sunrgbd.get_calibration(data_idx)
        uv,d = calib.project_upright_depth_to_image(pc[:,0:3])
        print('Point UV:', uv)
        print('Point depth:', d)
        
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
        
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i in range(uv.shape[0]):
            depth = d[i]
            color = cmap[int(120.0/depth),:]
            cv2.circle(img, (int(np.round(uv[i,0])), int(np.round(uv[i,1]))), 2,
                color=tuple(color), thickness=-1)
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        Image.fromarray(img).save(os.path.join(dump_dir,'img_depth.jpg'))
        
        # Load box labels
        objects = sunrgbd.get_label_objects(data_idx)
        print('Objects:', objects)
        
        # Draw 2D boxes on image
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i,obj in enumerate(objects):
            cv2.rectangle(img, (int(obj.xmin),int(obj.ymin)),
                (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
            cv2.putText(img, '%d %s'%(i,obj.classname), (max(int(obj.xmin),15),
                max(int(obj.ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,0,0), 2)
        Image.fromarray(img).save(os.path.join(dump_dir, 'img_box2d.jpg'))
       
        # Dump OBJ files for the colored point cloud 
        for num_point in [10000,20000,40000,80000]:
            sampled_pcrgb = pc_util.random_sampling(pc, num_point)
            pc_util.write_ply_rgb(sampled_pcrgb[:,0:3],
                (sampled_pcrgb[:,3:]*256).astype(np.int8),
                os.path.join(dump_dir, 'pcrgb_%dk.obj'%(num_point//1000)))
        # Dump OBJ files for 3D bounding boxes
        # l,w,h correspond to dx,dy,dz
        # heading angle is from +X rotating towards -Y
        # (+X is degree, -Y is 90 degrees)
        oriented_boxes = []
        for obj in objects:
            obb = np.zeros((7))
            obb[0:3] = obj.centroid
            # Some conversion to map with default setting of w,l,h
            # and angle in box dumping
            obb[3:6] = np.array([obj.l,obj.w,obj.h])*2
            obb[6] = -1 * obj.heading_angle
            print('Object cls, heading, l, w, h:',\
                 obj.classname, obj.heading_angle, obj.l, obj.w, obj.h)
            oriented_boxes.append(obb)
        if len(oriented_boxes)>0:
            oriented_boxes = np.vstack(tuple(oriented_boxes))
            pc_util.write_oriented_bbox(oriented_boxes,
                os.path.join(dump_dir, 'obbs.ply'))
        else:
            print('-'*30)
            continue

        # Draw 3D boxes on depth points
        box3d = []
        ori3d = []
        for obj in objects:
            corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d(obj, calib)
            ori_3d_image, ori_3d = sunrgbd_utils.compute_orientation_3d(obj, calib)
            print('Corners 3D: ', corners_3d)
            box3d.append(corners_3d)
            ori3d.append(ori_3d)
        pc_box3d = np.concatenate(box3d, 0)
        pc_ori3d = np.concatenate(ori3d, 0)
        print(pc_box3d.shape)
        print(pc_ori3d.shape)
        pc_util.write_ply(pc_box3d, os.path.join(dump_dir, 'box3d_corners.ply'))
        pc_util.write_ply(pc_ori3d, os.path.join(dump_dir, 'box3d_ori.ply'))
        print('-'*30)
        print('Point clouds and bounding boxes saved to PLY files under %s'%(dump_dir))
        print('Type anything to continue to the next sample...')
        input()

def extract_sunrgbd_data(idx_filename, split, output_folder, num_point=20000,
    type_whitelist=DEFAULT_TYPE_WHITELIST,
    save_votes=False, use_v1=False, skip_empty_scene=True,
    save_tfrecord=False):
    """ Extract scene point clouds and 
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        idx_filename: a TXT file where each line is an int number (index)
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.
        use_v1: use the SUN RGB-D V1 data
        skip_empty_scene: if True, skip scenes that contain no object (no objet in whitelist)

    Dumps:
        <id>_pc.npz of (N,6) where N is for number of subsampled points and 6 is
            for XYZ and RGB (in 0~1) in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 8 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """
    dataset = sunrgbd_object('./sunrgbd_trainval', split, use_v1=use_v1)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)

        # Skip scenes with 0 object
        if skip_empty_scene and (len(objects)==0 or \
            len([obj for obj in objects if obj.classname in type_whitelist])==0):
                continue

        object_list = []
        for obj in objects:
            if obj.classname not in type_whitelist: continue
            obb = np.zeros((8))
            obb[0:3] = obj.centroid
            # Note that compared with that in data_viz, we do not time 2 to l,w.h
            # neither do we flip the heading angle
            obb[3:6] = np.array([obj.l,obj.w,obj.h])
            obb[6] = obj.heading_angle
            obb[7] = sunrgbd_utils.type2class[obj.classname]
            object_list.append(obb)
        if len(object_list)==0:
            obbs = np.zeros((0,8))
        else:
            obbs = np.vstack(object_list) # (K,8)

        pc_upright_depth = dataset.get_depth(data_idx)
        pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)

        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(data_idx)),
            pc=pc_upright_depth_subsampled)
        np.save(os.path.join(output_folder, '%06d_bbox.npy'%(data_idx)), obbs)
       
        if save_votes:
            N = pc_upright_depth_subsampled.shape[0]
            point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
            point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
            indices = np.arange(N)
            for obj in objects:
                if obj.classname not in type_whitelist: continue
                try:
                    # Find all points in this object's OBB
                    box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(obj.centroid,
                        np.array([obj.l,obj.w,obj.h]), obj.heading_angle)
                    pc_in_box3d,inds = sunrgbd_utils.extract_pc_in_box3d(\
                        pc_upright_depth_subsampled, box3d_pts_3d)
                    # Assign first dimension to indicate it is in an object box
                    point_votes[inds,0] = 1
                    # Add the votes (all 0 if the point is not in any object's OBB)
                    votes = np.expand_dims(obj.centroid,0) - pc_in_box3d[:,0:3]
                    sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                    for i in range(len(sparse_inds)):
                        j = sparse_inds[i]
                        point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                        # Populate votes with the fisrt vote
                        if point_vote_idx[j] == 0:
                            point_votes[j,4:7] = votes[i,:]
                            point_votes[j,7:10] = votes[i,:]
                    point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
                except:
                    print('ERROR ----',  data_idx, obj.classname)
            np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz'%(data_idx)),
                point_votes = point_votes)

def extract_sunrgbd_data_tfrecord(idx_filename, split, output_folder, num_point=20000,
    type_whitelist=DEFAULT_TYPE_WHITELIST, use_v1=False, skip_empty_scene=True, pointpainting=False, use_gt=False,
    include_person=False, include_small=False):
    """ Same as extract_sunrgbd_data EXCEPT

    Args:
        save_votes is removed and assumed to be always True

    Dumps:
        TFRecords containing point_cloud, bboxes, point_votes

        point cloud: (N,6) where N is for number of subsampled points and 6 is
            for XYZ and RGB (in 0~1) in upright depth coord
        bboxes: (K,8) where K is the number of objects, 8 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
        point_votes: (N,2) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """
    def _bytes_feature(input_list):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=input_list))

    def _float_feature(input_list):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=input_list))

    def _int64_feature(input_list):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=input_list))
    
    def create_example(point_cloud, bboxes, point_labels, n_valid_box):    
        feature = {        
            'point_cloud': _float_feature(list(point_cloud.reshape((-1)))),
            'bboxes': _float_feature(list(bboxes.reshape((-1)))),
            'point_labels':_float_feature(list(point_labels.reshape((-1)))),
            'n_valid_box':_int64_feature([n_valid_box])
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    dataset = sunrgbd_object(os.path.join(DATA_DIR,'sunrgbd_trainval'), split, use_v1=use_v1)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    #output_folder = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), output_folder)
    #print(output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    n_pc_shard = 100
    n_shards = int(len(data_idx_list) / n_pc_shard) + (1 if len(data_idx_list) % n_pc_shard != 0 else 0)

    MAX_NUM_OBJ = 64

    if include_person:
        type_whitelist += ['person']

    if include_small:
        type_whitelist += ['garbage_bin','laptop','cup','back_pack','bottle']
    num_sunrgbd_class = len(type_whitelist)
    print(type_whitelist)

    class_map = {'bed':'bed', 'table':'table', 
                    'sofa':'sofa','sofa_chair':'sofa',
                    'chair':'chair','toilet':'toilet','desk':'desk','dresser':'dresser',
                    'night_stand':'night_stand', 'nightstand':'night_stand',
                    'bookshelf':'bookshelf','bathtub':'bathtub',
                    'person':'person',
                    'garbage_bin':'garbage_bin', 'garbagebin:':'garbage_bin', 'recycle_bin':'garbage_bin',
                    'laptop':'laptop',
                    'cup':'cup', 'cups':'cup', 'coffee_cup':'cup', 'paper_cup':'cup','plasticcup':'cup', 'glass':'cup',
                    'bottle':'bottle', 'bottles':'bottle', 'water_bottle':'bottle', 'wine_bottle':'bottle', 'plastic_bottle':'bottle',
                    'bottled_water':'bottle', 'mineral_bottle':'bottle', 'shampoo_bottle':'bottle', 'spray_bottle':'bottle',
                    'back_pack':'back_pack'}

    class_dict = {}
    for k, v in class_map.items():
        if v in type_whitelist:
            class_dict[k] = v


    # Load semantic segmentation model(Written and trained in TF1.15)
    if pointpainting:
        if use_gt: 
            from scipy import io
            sunrgbd_mat = io.loadmat(os.path.join(DATA_DIR,'OFFICIAL_SUNRGBD','SUNRGBDMeta3DBB_v2.mat')) 
            metadata = sunrgbd_mat['SUNRGBDMeta'][0] 
        else:
            INPUT_SIZE = (513, 513)
            with tf.compat.v1.gfile.GFile('../deeplab/saved_model/sunrgbd_COCO_15.pb', "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            myGraph = tf.compat.v1.Graph()
            with myGraph.as_default():
                tf.compat.v1.import_graph_def(graph_def, name='')

            sess = tf.compat.v1.Session(graph=myGraph)

    f = open('sunrgbd_semented_pts_stats.txt', 'a+')
    
    for shard in tqdm(range(n_shards)):
        tfrecords_shard_path = os.path.join(output_folder, "{}_{}.records".format("sunrgbd", '%.5d-of-%.5d' % (shard, n_shards - 1)))
        start_idx = shard * n_pc_shard
        end_idx = min((shard+1) * n_pc_shard, len(data_idx_list))
        data_idx_shard_list = data_idx_list[start_idx:end_idx]        

        with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
            for data_idx in data_idx_shard_list:                 
                if data_idx == 2983: continue   #Errorneous data
                #if data_idx < 20000: continue
                #print('------------- ', data_idx)
                objects = dataset.get_label_objects(data_idx)

                # Skip scenes with 0 object
                if skip_empty_scene and (len(objects)==0 or \
                    len([obj for obj in objects if obj.classname in class_dict])==0):
                        continue

                assert len(objects) <= MAX_NUM_OBJ

                obbs = np.zeros((MAX_NUM_OBJ,8))
                cnt = 0
                for i, obj in enumerate(objects):
                    if obj.classname not in class_dict: continue
                    obbs[cnt, 0:3] = obj.centroid
                    # Note that compared with that in data_viz, we do not time 2 to l,w.h
                    # neither do we flip the heading angle
                    obbs[cnt, 3:6] = np.array([obj.l,obj.w,obj.h])
                    obbs[cnt, 6] = obj.heading_angle
                    obbs[cnt, 7] = sunrgbd_utils.type2class[class_dict[obj.classname]]
                    cnt+=1

                n_valid_box = cnt
                    
                pc_upright_depth = dataset.get_depth(data_idx)
                pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)          

                if pointpainting:
                    ########## Add 2D segmentation result to point cloud(Point Painting) ##########
                    # Project points to image
                    calib = dataset.get_calibration(data_idx)
                    uv,d,_ = calib.project_upright_depth_to_image(pc_upright_depth_subsampled[:,0:3]) #uv: (N, 2)

                    # Run image segmentation result and get result
                    img = dataset.get_image2(data_idx)            
                    
                    # Round to the nearest integer; since uv starts from 1. subtract 1.
                    uv[:,0] = np.rint(uv[:,0] - 1)
                    uv[:,1] = np.rint(uv[:,1] - 1)

                    w, h = img.size

                    uv[:,0] = np.minimum(uv[:,0], w-1)
                    uv[:,1] = np.minimum(uv[:,1], h-1)

                    if use_gt: 
                        img_path = metadata[data_idx-1][4][0][12:] # indexing starts from 1
                        img_path = os.path.join('/home','aiot',img_path)
                        seg_path = os.path.join('/'.join(img_path.split('/')[:-2]),'seg.mat')

                        mat = io.loadmat(seg_path)
                        mask, names= mat['seglabel'], mat['names']
                        pred_class = np.zeros_like(mask)

                        if len(names) !=1:
                            names=names.swapaxes(0,1)[0]
                        else:
                            names=names[0]

                        #label = {'bed':1, 'table':2, 'sofa':3, 'chair':4, 'toilet':5, 'desk':6, 'dresser':7, 'night_stand':8, 'bookshelf':9, 'bathtub':10, 'person':11}
                        label = sunrgbd_utils.type2class
                        for idx, name in enumerate(names):            
                            name = name[0]            
                            if name in label:
                                label_idx = label[name]                
                                cur_seg = np.where(mask == idx+1, (label_idx), 0).astype(np.uint8)
                                pred_class += cur_seg
                        projected_class = pred_class[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]
                        pred_prob = np.eye(num_sunrgbd_class+1)[projected_class]                        
                        pred_prob = pred_prob[:,:(num_sunrgbd_class+1)]

                    else:
                        pred_prob, pred_class = deeplab.run_semantic_segmentation_graph(img, sess, INPUT_SIZE) # (w, h, num_class)     
                        pred_prob = pred_prob[:,:,:(num_sunrgbd_class+1)] # 0 is background class              
                        projected_class = pred_class[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]
                        pred_prob = pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]

                    
                    isPainted = np.where((projected_class > 0) & (projected_class < num_sunrgbd_class+1), 1, 0) # Point belongs to foreground?                    
                    isPainted = np.expand_dims(isPainted, axis=-1)

                    # Append segmentation score to each point
                    painted_pc = np.concatenate([pc_upright_depth_subsampled[:,:3],\
                                            isPainted,\
                                            pred_prob
                                            ], axis=-1)
                                                               
                    ######################################################################################################
            
                N = pc_upright_depth_subsampled.shape[0]
                point_votes = np.zeros((N,13)) # 3 votes and 1 vote mask 
                point_votes[:, 10:13] = -1
                point_vote_idx = np.zeros((N)).astype(np.int32)                
                indices = np.arange(N)
                i_obj = 0
                for obj in objects:
                    if obj.classname not in type_whitelist: continue
                    try:
                        # Find all points in this object's OBB
                        box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(obj.centroid,
                            np.array([obj.l,obj.w,obj.h]), obj.heading_angle)
                        pc_in_box3d,inds = sunrgbd_utils.extract_pc_in_box3d(\
                            pc_upright_depth_subsampled, box3d_pts_3d)
                        # Assign first dimension to indicate it is in an object box
                        point_votes[inds,0] = 1
                        # Add the votes (all 0 if the point is not in any object's OBB)
                        votes = np.expand_dims(obj.centroid,0) - pc_in_box3d[:,0:3]
                        sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                        for i in range(len(sparse_inds)):
                            j = sparse_inds[i]
                            point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                            point_votes[j, point_vote_idx[j] + 10] = i_obj

                            # Populate votes with the fisrt vote
                            if point_vote_idx[j] == 0:
                                point_votes[j,4:7] = votes[i,:]
                                point_votes[j,7:10] = votes[i,:]
                                point_votes[j, 10] = i_obj
                                point_votes[j, 11] = i_obj
                                point_votes[j, 12] = i_obj
                        point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
                        i_obj += 1
                    except:
                        print('ERROR ----',  data_idx, obj.classname)         

                # choose the nearest as the first gt for each point
                for ip in range(N):
                    is_pos = (point_votes[ip, 0] > 0)
                    if is_pos:
                        vote_delta1 = point_votes[ip, 1:4].copy()
                        vote_delta2 = point_votes[ip, 4:7].copy()
                        vote_delta3 = point_votes[ip, 7:10].copy()
                        dist1 = np.sum(vote_delta1 ** 2)
                        dist2 = np.sum(vote_delta2 ** 2)
                        dist3 = np.sum(vote_delta3 ** 2)

                        gt_ind1 = int(point_votes[ip, 10].copy())
                        # gt_ind2 = int(point_votes[ip, 11].copy())
                        # gt_ind3 = int(point_votes[ip, 12].copy())
                        # gt1 = obbs[gt_ind1]
                        # gt2 = obbs[gt_ind2]
                        # gt3 = obbs[gt_ind3]
                        # size_norm_vote_delta1 = vote_delta1 / gt1[3:6]
                        # size_norm_vote_delta2 = vote_delta2 / gt2[3:6]
                        # size_norm_vote_delta3 = vote_delta3 / gt3[3:6]
                        # size_norm_dist1 = np.sum(size_norm_vote_delta1 ** 2)
                        # size_norm_dist2 = np.sum(size_norm_vote_delta2 ** 2)
                        # size_norm_dist3 = np.sum(size_norm_vote_delta3 ** 2)

                        near_ind = np.argmin([dist1, dist2, dist3])
                        # near_ind = np.argmin([size_norm_dist1, size_norm_dist2, size_norm_dist3])

                        point_votes[ip, 10] = point_votes[ip, 10 + near_ind].copy()
                        point_votes[ip, 10 + near_ind] = gt_ind1
                        point_votes[ip, 1:4] = point_votes[ip, int(near_ind * 3 + 1):int((near_ind + 1) * 3 + 1)].copy()
                        point_votes[ip, int(near_ind * 3 + 1):int((near_ind + 1) * 3 + 1)] = vote_delta1
                    else:
                        assert point_votes[ip, 10] == -1, "error"
                        assert point_votes[ip, 11] == -1, "error"
                        assert point_votes[ip, 12] == -1, "error" 

                point_labels = point_votes[:,[0,10]]
                

                #f.write(str(data_idx) + '\t' + str(np.sum(isPainted)) + '\t' 
                #        + str(np.sum(point_votes[:,0])) + '\t' + str(np.sum(point_votes[:,0] * isPainted)) +'\n')                                
                if pointpainting:
                    tf_example = create_example(painted_pc, obbs, point_labels, n_valid_box) 
                else:
                    tf_example = create_example(pc_upright_depth_subsampled, obbs, point_labels, n_valid_box)
                                
                writer.write(tf_example.SerializeToString())                
                
    f.close()



def gf_extract_sunrgbd_data_tfrecord(idx_filename, split, output_folder, num_point=20000,
                         type_whitelist=DEFAULT_TYPE_WHITELIST,
                         save_votes=False, use_v1=False, skip_empty_scene=True, pointpainting=False):

    """ Extract scene point clouds and 
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        idx_filename: a TXT file where each line is an int number (index)
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.
        use_v1: use the SUN RGB-D V1 data
        skip_empty_scene: if True, skip scenes that contain no object (no objet in whitelist)

    Dumps:
        <id>_pc.npz of (N,6) where N is for number of subsampled points and 6 is
            for XYZ and RGB (in 0~1) in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 8 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """

    def _bytes_feature(input_list):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=input_list))

    def _float_feature(input_list):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=input_list))

    def _int64_feature(input_list):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=input_list))
    
    def create_example(point_cloud, bboxes, point_labels, n_valid_box):    
        feature = {        
            'point_cloud': _float_feature(list(point_cloud.reshape((-1)))),
            'bboxes': _float_feature(list(bboxes.reshape((-1)))),
            'point_labels':_float_feature(list(point_labels.reshape((-1)))),
            'n_valid_box':_int64_feature([n_valid_box])
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    dataset = sunrgbd_object(os.path.join(DATA_DIR,'sunrgbd_trainval'), split, use_v1=use_v1)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    n_pc_shard = 100
    n_shards = int(len(data_idx_list) / n_pc_shard) + (1 if len(data_idx_list) % n_pc_shard != 0 else 0)

    MAX_NUM_OBJ = 64
    num_sunrgbd_class = len(type_whitelist)
    
    all_obbs = []
    all_pc_upright_depth_subsampled = []
    all_point_votes = []

    # Load semantic segmentation model(Written and trained in TF1.15)
    if pointpainting:
        INPUT_SIZE = (513, 513)
        with tf.compat.v1.gfile.GFile('../deeplab/saved_model/sunrgbd_COCO_15.pb', "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        myGraph = tf.compat.v1.Graph()
        with myGraph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name='')

        sess = tf.compat.v1.Session(graph=myGraph)

    for shard in tqdm(range(n_shards)):
        
        tfrecords_shard_path = os.path.join(output_folder, "{}_{}.records".format("sunrgbd", '%.5d-of-%.5d' % (shard, n_shards - 1)))
        start_idx = shard * n_pc_shard
        end_idx = min((shard+1) * n_pc_shard, len(data_idx_list))
        data_idx_shard_list = data_idx_list[start_idx:end_idx]     
        with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
            for data_idx in data_idx_shard_list:
                if data_idx == 2983: continue   #Errorneous data
                print('------------- ', data_idx)
                objects = dataset.get_label_objects(data_idx)

                # Skip scenes with 0 object
                if skip_empty_scene and (len(objects) == 0 or
                                        len([obj for obj in objects if obj.classname in type_whitelist]) == 0):
                    continue
                
                assert len(objects) <= MAX_NUM_OBJ

                
                cnt = 0

                object_list = []
                for obj in objects:
                    if obj.classname not in type_whitelist: continue
                    obb = np.zeros((8))
                    obb[0:3] = obj.centroid
                    # Note that compared with that in data_viz, we do not time 2 to l,w.h
                    # neither do we flip the heading angle
                    obb[3:6] = np.array([obj.l, obj.w, obj.h])
                    obb[6] = obj.heading_angle
                    obb[7] = sunrgbd_utils.type2class[obj.classname]
                    object_list.append(obb)
                    cnt+=1
                
                n_valid_box = cnt
                obbs = np.zeros((MAX_NUM_OBJ,8))
                if len(object_list) > 0:                    
                    obbs[:n_valid_box] = np.vstack(object_list)  # (K,8)
                
                # print(f"{data_idx} has {obbs.shape[0]} gt bboxes")

                pc_upright_depth = dataset.get_depth(data_idx)
                pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)

                np.savez_compressed(os.path.join(output_folder, '%06d_pc.npz' % (data_idx)),
                                    pc=pc_upright_depth_subsampled)
                np.save(os.path.join(output_folder, '%06d_bbox.npy' % (data_idx)), obbs)
                # pickle save
                with open(os.path.join(output_folder, '%06d_pc.pkl' % (data_idx)), 'wb') as f:
                    pickle.dump(pc_upright_depth_subsampled, f)
                    print(f"{os.path.join(output_folder, '%06d_pc.pkl' % (data_idx))} saved successfully !!")
                with open(os.path.join(output_folder, '%06d_bbox.pkl' % (data_idx)), 'wb') as f:
                    pickle.dump(obbs, f)
                    print(f"{os.path.join(output_folder, '%06d_bbox.pkl' % (data_idx))} saved successfully !!")
                # add to collection
                all_pc_upright_depth_subsampled.append(pc_upright_depth_subsampled)
                all_obbs.append(obbs)

                if pointpainting:
                    ########## Add 2D segmentation result to point cloud(Point Painting) ##########
                    # Project points to image
                    calib = dataset.get_calibration(data_idx)
                    uv,d,_ = calib.project_upright_depth_to_image(pc_upright_depth_subsampled[:,0:3]) #uv: (N, 2)

                    # Run image segmentation result and get result
                    img = dataset.get_image2(data_idx)            
                    
                    # Round to the nearest integer; since uv starts from 1. subtract 1.
                    uv[:,0] = np.rint(uv[:,0] - 1)
                    uv[:,1] = np.rint(uv[:,1] - 1)

                    w, h = img.size

                    uv[:,0] = np.minimum(uv[:,0], w-1)
                    uv[:,1] = np.minimum(uv[:,1], h-1)

                    pred_prob, pred_class = deeplab.run_semantic_segmentation_graph(img, sess, INPUT_SIZE) # (w, h, num_class)     
                    pred_prob = pred_prob[:,:,:(num_sunrgbd_class+1)] # 0 is background class              
                    projected_class = pred_class[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]
                    pred_prob = pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]
                    
                    isPainted = np.where((projected_class > 0) & (projected_class < num_sunrgbd_class+1), 1, 0) # Point belongs to foreground?                    
                    isPainted = np.expand_dims(isPainted, axis=-1)

                    # Append segmentation score to each point
                    painted_pc = np.concatenate([pc_upright_depth_subsampled[:,:3],\
                                            isPainted,\
                                            pred_prob
                                            ], axis=-1)
                                                               
                    ######################################################################################################

                N = pc_upright_depth_subsampled.shape[0]
                point_votes = np.zeros((N, 13))  # 1 vote mask + 3 votes and + 3 votes gt ind
                point_votes[:, 10:13] = -1
                point_vote_idx = np.zeros((N)).astype(np.int32)  # in the range of [0,2]
                indices = np.arange(N)
                i_obj = 0
                for obj in objects:
                    if obj.classname not in type_whitelist: continue
                    try:
                        # Find all points in this object's OBB
                        box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(obj.centroid,
                                                                    np.array([obj.l, obj.w, obj.h]), obj.heading_angle)
                        pc_in_box3d, inds = sunrgbd_utils.extract_pc_in_box3d( \
                            pc_upright_depth_subsampled, box3d_pts_3d)
                        # Assign first dimension to indicate it is in an object box
                        point_votes[inds, 0] = 1
                        # Add the votes (all 0 if the point is not in any object's OBB)
                        votes = np.expand_dims(obj.centroid, 0) - pc_in_box3d[:, 0:3]
                        sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
                        for i in range(len(sparse_inds)):
                            j = sparse_inds[i]
                            point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i,:]
                            point_votes[j, point_vote_idx[j] + 10] = i_obj
                            # Populate votes with the fisrt vote
                            if point_vote_idx[j] == 0:
                                point_votes[j, 4:7] = votes[i, :]
                                point_votes[j, 7:10] = votes[i, :]
                                point_votes[j, 10] = i_obj
                                point_votes[j, 11] = i_obj
                                point_votes[j, 12] = i_obj
                        point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
                        i_obj += 1
                    except:
                        print('ERROR ----', data_idx, obj.classname)

                # choose the nearest as the first gt for each point
                for ip in range(N):
                    is_pos = (point_votes[ip, 0] > 0)
                    if is_pos:
                        vote_delta1 = point_votes[ip, 1:4].copy()
                        vote_delta2 = point_votes[ip, 4:7].copy()
                        vote_delta3 = point_votes[ip, 7:10].copy()
                        dist1 = np.sum(vote_delta1 ** 2)
                        dist2 = np.sum(vote_delta2 ** 2)
                        dist3 = np.sum(vote_delta3 ** 2)

                        gt_ind1 = int(point_votes[ip, 10].copy())

                        near_ind = np.argmin([dist1, dist2, dist3])

                        point_votes[ip, 10] = point_votes[ip, 10 + near_ind].copy()
                        point_votes[ip, 10 + near_ind] = gt_ind1
                        point_votes[ip, 1:4] = point_votes[ip, int(near_ind * 3 + 1):int((near_ind + 1) * 3 + 1)].copy()
                        point_votes[ip, int(near_ind * 3 + 1):int((near_ind + 1) * 3 + 1)] = vote_delta1
                    else:
                        assert point_votes[ip, 10] == -1, "error"
                        assert point_votes[ip, 11] == -1, "error"
                        assert point_votes[ip, 12] == -1, "error"

                print(f"{data_idx}_votes.npz has {i_obj} gt bboxes")
                np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz' % (data_idx)),
                                    point_votes=point_votes)
                with open(os.path.join(output_folder, '%06d_votes.pkl' % (data_idx)), 'wb') as f:
                    pickle.dump(point_votes, f)
                    print(f"{os.path.join(output_folder, '%06d_votes.pkl' % (data_idx))} saved successfully !!")
                all_point_votes.append(point_votes)

                point_labels = point_votes[:,[0,10]]
                
                if pointpainting:
                    tf_example = create_example(painted_pc, obbs, point_labels, n_valid_box) 
                else:
                    tf_example = create_example(pc_upright_depth_subsampled, obbs, point_labels, n_valid_box)
                                
                writer.write(tf_example.SerializeToString())   

    pickle_filename = os.path.join(output_folder, 'all_obbs_modified_nearest_has_empty.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_obbs, f)
        print(f"{pickle_filename} saved successfully !!")

    pickle_filename = os.path.join(output_folder, 'all_pc_modified_nearest_has_empty.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_pc_upright_depth_subsampled, f)
        print(f"{pickle_filename} saved successfully !!")

    pickle_filename = os.path.join(output_folder, 'all_point_votes_nearest_has_empty.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_point_votes, f)
        print(f"{pickle_filename} saved successfully !!")

    all_point_labels = []
    for point_votes in all_point_votes:
        point_labels = point_votes[:, [0, 10]]
        all_point_labels.append(point_labels)
    pickle_filename = os.path.join(output_folder, 'all_point_labels_nearest_has_empty.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_point_labels, f)
        print(f"{pickle_filename} saved successfully !!")



def get_box3d_dim_statistics(idx_filename,
    type_whitelist=['garbage_bin','laptop','cup','cups','coffee_cup','paper_cup','bottle','bottles','water_bottle','wine_bottle','back_pack'],
    save_path=None):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    
    dataset = sunrgbd_object(os.path.join(DATA_DIR,'sunrgbd_trainval'), use_v1=True)
    dimension_list = []
    type_list = []
    ry_list = []
    pts_cnt_in_box = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    # Number of objects per class
    from collections import defaultdict
    obj_stats = defaultdict(int)
    
    # This is to map different class names to one class.
    class_type_dict = {'garbage_bin':'garbage_bin', 'garbagebin:':'garbage_bin', 'recycle_bin':'bin',
                        'laptop':'laptop',
                        'cup':'cup', 'cups':'cup', 'coffee_cup':'cup', 'paper_cup':'cup','plasticcup':'cup', 'glass':'cup',
                        'bottle':'bottle', 'bottles':'bottle', 'water_bottle':'bottle', 'wine_bottle':'bottle', 'plastic_bottle':'bottle',
                        'bottled_water':'bottle', 'mineral_bottle':'bottle', 'shampoo_bottle':'bottle', 'spray_bottle':'bottle',
                        'back_pack':'back_pack'}
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        if data_idx == 2983: continue   #Errorneous data
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_upright_depth = dataset.get_depth(data_idx)
        pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, 50000)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            obj_stats[obj.classname] += 1
            if obj.classname not in type_whitelist: continue
            heading_angle = -1 * np.arctan2(obj.orientation[1], obj.orientation[0])
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(class_type_dict[obj.classname]) 
            ry_list.append(heading_angle)

            try:
                # Find all points in this object's OBB
                box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(obj.centroid,
                    np.array([obj.l,obj.w,obj.h]), obj.heading_angle)
                pc_in_box3d,inds = sunrgbd_utils.extract_pc_in_box3d(\
                    pc_upright_depth_subsampled, box3d_pts_3d)
                pts_cnt_in_box.append(np.sum(inds))
            except:
                print("-- ERROR IN POINT COUNT, replacing it to 0 -- ")
                print(obj.classname)
                pts_cnt_in_box.append(0)

    '''
    import cPickle as pickle
    if save_path is not None:
        with open(save_path,'wb') as fp:
            pickle.dump(type_list, fp)
            pickle.dump(dimension_list, fp)
            pickle.dump(ry_list, fp)
    '''

    #Save object counts per class
    save_path = os.path.join('sunrgbd_3d_obj_stats_val.txt')
    f = open(save_path, 'w+')
    for k, v in sorted(obj_stats.items(), key=lambda x:-x[1]) :
        f.write("%s %d\n"%(k, v))
    f.close()

    # Get average box size for different catgories
    box3d_pts = np.vstack(dimension_list)
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        pts_cnt_class = []
        for i in range(len(dimension_list)):
            if type_list[i]==class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])
                pts_cnt_class.append(pts_cnt_in_box[i])
        median_box3d = np.median(box3d_list,0)
        mean_box3d = np.mean(box3d_list,0)
        pts_cnt_class = np.array(pts_cnt_class)
        mean_pts = np.mean(pts_cnt_class,0)
        min_pts = np.min(pts_cnt_class[pts_cnt_class>0],0)
        percent10 = np.percentile(pts_cnt_class, 10)

        print("Median size: \'%s\': np.array([%f,%f,%f])," % \
            (class_type, median_box3d[0]*2, median_box3d[1]*2, median_box3d[2]*2))

        print("Mean size: \'%s\': np.array([%f,%f,%f])," % \
            (class_type, mean_box3d[0], mean_box3d[1], mean_box3d[2]))
        print("Mean points count: %f" % (mean_pts))
        print("Minimum points count: %f" % (min_pts))
        print("10 percentile points count: %f" % (percent10))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true', help='Run data visualization.')
    parser.add_argument('--compute_median_size', action='store_true', help='Compute median 3D bounding box sizes for each class.')
    parser.add_argument('--gen_v1_data', action='store_true', help='Generate V1 dataset.')
    parser.add_argument('--gen_v2_data', action='store_true', help='Generate V2 dataset.')
    parser.add_argument('--tfrecord', action='store_true', help='Generate TFRecord dataset.')
    parser.add_argument('--painted', action='store_true', help='Generate point painted TFRecord dataset.')
    parser.add_argument('--use_gt', action='store_true', help='When pointpainting, use ground truth segmentation.')    
    args = parser.parse_args()

    if args.viz:
        data_viz(os.path.join(DATA_DIR, 'sunrgbd_trainval'))
        exit()

    if args.compute_median_size:
        get_box3d_dim_statistics(os.path.join(DATA_DIR, 'sunrgbd_trainval/val_data_idx.txt'))
        exit()

    if args.gen_v1_data:
        extract_sunrgbd_data(os.path.join(DATA_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, 'sunrgbd_pc_bbox_votes_50k_v1_train'),
            save_votes=True, num_point=50000, use_v1=True, skip_empty_scene=False)
        extract_sunrgbd_data(os.path.join(DATA_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, 'sunrgbd_pc_bbox_votes_50k_v1_val'),
            save_votes=True, num_point=50000, use_v1=True, skip_empty_scene=False)
    
    if args.gen_v2_data:
        extract_sunrgbd_data(os.path.join(DATA_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, 'sunrgbd_pc_bbox_votes_50k_v2_train'),
            save_votes=True, num_point=50000, use_v1=False, skip_empty_scene=False)
        extract_sunrgbd_data(os.path.join(DATA_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, 'sunrgbd_pc_bbox_votes_50k_v2_val'),
            save_votes=True, num_point=50000, use_v1=False, skip_empty_scene=False)

    if args.tfrecord and not args.painted:
        gf_extract_sunrgbd_data_tfrecord(os.path.join(DATA_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, 'gf_sunrgbd_pc_train_tf2'),
            num_point=50000, use_v1=True, skip_empty_scene=False)
        gf_extract_sunrgbd_data_tfrecord(os.path.join(DATA_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, 'gf_sunrgbd_pc_val_tf2'),
            num_point=50000, use_v1=True, skip_empty_scene=False)

    if args.painted:
        assert args.tfrecord, "Need to set tfrecord flag as True"
        train_data_idx_file = 'train_data_idx.txt'
        val_data_idx_file = 'val_data_idx.txt'
        output_folder_train = 'gf_sunrgbd_pc_train_painted_tf'
        output_folder_val = 'gf_sunrgbd_pc_val_painted_tf'


        gf_extract_sunrgbd_data_tfrecord(os.path.join(DATA_DIR, 'sunrgbd_trainval', train_data_idx_file),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, output_folder_train),
            num_point=50000, use_v1=True, skip_empty_scene=False, pointpainting=True)
        gf_extract_sunrgbd_data_tfrecord(os.path.join(DATA_DIR, 'sunrgbd_trainval', val_data_idx_file),
            split = 'training',
            output_folder = os.path.join(DATA_DIR, output_folder_val),
            num_point=50000, use_v1=True, skip_empty_scene=False, pointpainting=True)