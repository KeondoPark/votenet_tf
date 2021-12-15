# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import importlib
import time

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

import json
environ_file = os.path.join(ROOT_DIR,'configs','environ.json')
environ = json.load(open(environ_file))['environ']

if environ == 'server':    
    DATA_DIR = '/home/aiot/data'
elif environ == 'jetson':
    DATA_DIR= 'sunrgbd'
elif environ == 'server2':    
    DATA_DIR = '/data'

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper_tf import parse_predictions

import votenet_tf
from votenet_tf import dump_results
from PIL import Image
from deeplab.deeplab import run_semantic_seg, run_semantic_seg_tflite
import json
from multiprocessing import Queue


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    if not model_config['use_painted']:
        point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    if point_cloud.shape[0] > 20000:
        point_cloud = random_sampling(point_cloud, 20000)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,20000,4)
    return pc

# Assume Edgetpu is available!
def votenet_inference(queue):

    model_config = json.load(open(os.path.join(ROOT_DIR,'configs','inf_211213_sep.json')))
    DEFAULT_CHECKPOINT_PATH = os.path.join('tf_ckpt', model_config['model_id'])

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')     
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from model_util_sunrgbd import SunrgbdDatasetConfig
    if 'include_person' in model_config and model_config['include_person']:
        DATASET_CONFIG = SunrgbdDatasetConfig(include_person=True)
    else:
        DATASET_CONFIG = SunrgbdDatasetConfig()
    from sunrgbd_data import sunrgbd_object
    
    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DATASET_CONFIG}

    # Init the model and optimzier    
    net = votenet_tf.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        #sampling='seed_fps', num_class=DC.num_class,
        sampling='vote_fps', num_class=DATASET_CONFIG.num_class,
        num_heading_bin=DATASET_CONFIG.num_heading_bin,
        num_size_cluster=DATASET_CONFIG.num_size_cluster,
        mean_size_arr=DATASET_CONFIG.mean_size_arr,
        model_config=model_config)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    
   
    point_cloud = queue.get()        
    
    time_record = [('Start: ', time.time())]
    point_cloud_sampled = random_sampling(point_cloud[:,0:3], 20000)
    pc = preprocess_point_cloud(point_cloud_sampled)    
        
    time_record.append(('Votenet data preprocessing time:', time.time()))

    inputs = {'point_clouds': tf.convert_to_tensor(pc)}
   
    # Model inference    
    if model_config['use_painted']:        
        img = queue.get() 

        import sunrgbd_utils  
        Rtilt, K = queue.get()        
        calib = sunrgbd_utils.SUNRGBD_Calib_FromArr(Rtilt, K)        
                      
        if model_config['use_multiThr']:
            end_points = net(inputs['point_clouds'], training=False, img=img, calib=calib)        
        else:
            xyz = pc[0,:,:3]
            if model_config['use_edgetpu']:
                pred_prob = run_semantic_seg_tflite(img, save_result=False)                
            else:                
                pred_prob, pred_class = run_semantic_seg(img, save_result=False)  
            
            time_record.append(('Deeplab inference time:', time.time()))

            uv,d = calib.project_upright_depth_to_image(xyz) #uv: (N, 2)
            uv = np.rint(uv - 1)
            
            pred_prob = pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)] # (npoint, num_class + 1 + 1 )
            projected_class = np.argmax(pred_prob, axis=-1) # (npoint, 1) 
            isPainted = np.where((projected_class > 0) & (projected_class < DATASET_CONFIG.num_class+1), 1, 0) # Point belongs to background?                    
            isPainted = np.expand_dims(isPainted, axis=-1)

            # 0 is background class, deeplab is trained with "person" included, (height, width, num_class)
            pred_prob = pred_prob[:,:(DATASET_CONFIG.num_class+1)] #(npoint, num_class)
            pointcloud = np.concatenate([xyz, isPainted, pred_prob, pc[0,:,3:]], axis=-1)
            time_record.append(('Pointpainting time:', time.time()))

            inputs['point_clouds'] = tf.convert_to_tensor(np.expand_dims(pointcloud, axis=0))

            print(inputs['point_clouds'].shape)
            end_points = net(inputs['point_clouds'])        
        
    else:        
        end_points = net(inputs['point_clouds'], training=False)                
    

    time_record += end_points['time_record']    
    time_record = time_record + [('Voting and Proposal time:', time.time())]

    for idx, (desc, t) in enumerate(time_record):
        if idx == 0:                                 
            print(desc, t)
            prev_time = t
            continue
        print(desc, t - prev_time)            
        prev_time = t
    print('Total inference time: %f \n'%(time_record[-1][1] - time_record[0][1]))
    

    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    
    type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9, 'person':10}
    class2type = {type2class[t]:t for t in type2class}

    #print(pred_map_cls[0])
    #for pred in pred_map_cls[0]:
    #    print('-'*20)        
    #    print('class:', class2type[pred[0].numpy()])
    #    print('conf:', pred[2])
        #print('coords', pred[1])

    print('Finished detection. %d object detected.'%(len(pred_map_cls[0][0])))
    """
    dump_dir = os.path.join(demo_dir, 'sunrgbd_results')
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    dump_results(end_points, dump_dir, DATASET_CONFIG, True)
    print('Dumped detection results to folder %s'%(dump_dir))
    """
