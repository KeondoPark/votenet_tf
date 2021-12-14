# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')
parser.add_argument('--inf_time_file', default=None, help='Record inference time')
parser.add_argument('--config_path', default=None, required=True, help='Model configuration path')
FLAGS = parser.parse_args()

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

model_config = json.load(open(FLAGS.config_path))
DEFAULT_CHECKPOINT_PATH = os.path.join('tf_ckpt', model_config['model_id'])

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    if not model_config['use_painted']:
        point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    if point_cloud.shape[0] > FLAGS.num_point:
        point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,20000,4)
    return pc

if __name__=='__main__':
    
    # Limit GPU Memory usage, 256MB suffices in jetson nano
    if FLAGS.gpu_mem_limit:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.gpu_mem_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from model_util_sunrgbd import SunrgbdDatasetConfig
        if 'include_person' in model_config and model_config['include_person']:
            DATASET_CONFIG = SunrgbdDatasetConfig(include_person=True)
        else:
            DATASET_CONFIG = SunrgbdDatasetConfig()
        from sunrgbd_data import sunrgbd_object
        checkpoint_path = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH          
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
        #pc_path = os.path.join(demo_dir, 'pc_person2.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    else:
        print('Unkown dataset. Exiting.')
        exit(-1)

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

    if model_config['use_tflite']:          
        restore_list = []  
        #restore_list.append(tf.train.Checkpoint(pnet=net.pnet))
        #restore_list.append(tf.train.Checkpoint(vgen=net.vgen))
        
        for layer in restore_list:
            new_root = tf.train.Checkpoint(net=layer)
            new_root.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    else:    
        ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        epoch = ckpt.epoch.numpy()

        print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))  
   
    # Load and preprocess input point cloud     
    #point_cloud = read_ply(pc_path)
    #pc = preprocess_point_cloud(point_cloud)
    #print('Loaded point cloud data: %s'%(pc_path))
    ## TODO: NEED TO BE REPLACED
    data_idx = 5051
    dataset = sunrgbd_object(os.path.join(DATA_DIR,'sunrgbd_trainval'), 'training', use_v1=True)
    point_cloud = dataset.get_depth(data_idx)        
    
    
    time_record = [('Start: ', time.time())]
    point_cloud_sampled = random_sampling(point_cloud[:,0:3], FLAGS.num_point)
    pc = preprocess_point_cloud(point_cloud_sampled)    
        
    time_record.append(('Votenet data preprocessing time:', time.time()))

    inputs = {'point_clouds': tf.convert_to_tensor(pc)}
   
    # Model inference    
    if model_config['use_painted']:
        ## TODO: NEED TO BE REPLACED
        img = dataset.get_image2(data_idx)
        calib = dataset.get_calibration(data_idx)                
        if model_config['use_multiThr']:
            end_points = net(inputs['point_clouds'], training=False, img=img, calib=calib)        
        else:
            xyz = pc[0,:,:3]
            if model_config['use_tflite']:
                pred_prob = run_semantic_seg_tflite(img, save_result=False)                
            else:                
                pred_prob, pred_class = run_semantic_seg(img, save_result=False)  
            time_record.append(('Deeplab inference time:', time.time()))

            uv,d = calib.project_upright_depth_to_image(xyz) #uv: (N, 2)
            uv = np.rint(uv - 1)
            
            pred_prob = pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)] # (npoint, num_class + 1 + 1 )
            projected_class = np.argmax(pred_prob, axis=-1) # (npoint, 1) 
            isPainted = np.where((projected_class > 0) & (projected_class < 11), 1, 0) # Point belongs to background?                    
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

    if FLAGS.inf_time_file:
        inf_time_log = open(FLAGS.inf_time_file, 'a+')

        for idx, (desc, t) in enumerate(time_record):
            if idx == 0:                 
                inf_time_log.write(desc + str(t) + '\n')
                prev_time = t
                continue
            inf_time_log.write(desc + str(t - prev_time) + '\n')
            prev_time = t
        
        inf_time_log.write('Total inference time: %f \n'%(time_record[-1][1] - time_record[0][1]))
        inf_time_log.close()
    else:
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
  
    dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    #dump_results(end_points, dump_dir, DC, True)
    #print('Dumped detection results to folder %s'%(dump_dir))

