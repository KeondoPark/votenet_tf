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
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))

from pc_util import random_sampling, read_ply
from ap_helper_tf import parse_predictions

import votenet_tf
from votenet_tf import dump_results
from PIL import Image
from deeplab.deeplab import run_semantic_seg, run_semantic_seg_tflite
import json
from multiprocessing import Queue

class VoteNetInf:
    def __init__(self):        
        s = time.time()
        self.model_config = json.load(open(os.path.join(ROOT_DIR,'configs','inf_211213_sep.json')))    

        # Set file paths and dataset config                
        from model_util_sunrgbd import SunrgbdDatasetConfig
        if 'include_person' in self.model_config and self.model_config['include_person']:
            self.DATASET_CONFIG = SunrgbdDatasetConfig(include_person=True)
        else:
            self.DATASET_CONFIG = SunrgbdDatasetConfig()
        from sunrgbd_data import sunrgbd_object
        
        self.eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
            'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
            'conf_thresh': 0.5, 'dataset_config': self.DATASET_CONFIG}

        # Init the model and optimzier    
        self.net = votenet_tf.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
            #sampling='seed_fps', num_class=DC.num_class,
            sampling='vote_fps', num_class=self.DATASET_CONFIG.num_class,
            num_heading_bin=self.DATASET_CONFIG.num_heading_bin,
            num_size_cluster=self.DATASET_CONFIG.num_size_cluster,
            mean_size_arr=self.DATASET_CONFIG.mean_size_arr,
            model_config=self.model_config)
        print('Constructed model.', time.time() - s)

    # Assume Edgetpu is available!
    def inference(self, conn):
        
        
        pc_path = conn.recv()
        time_record = [('Start: ', time.time())]    
        point_cloud = np.load(pc_path)                    
        pc = preprocess_point_cloud(point_cloud)                
        time_record.append(('Votenet data preprocessing time:', time.time()))
        inputs = {'point_clouds': tf.convert_to_tensor(pc)}
    
        # Model inference    
        if self.model_config['use_painted']:        
            
            img_path = conn.recv()
            img = np.load(img_path)            
            img = Image.fromarray(img, 'RGB')        

            import sunrgbd_utils       
            Rtilt, K = conn.recv()                                             
            calib = sunrgbd_utils.SUNRGBD_Calib_FromArr(Rtilt, K)        
                        
            if self.model_config['use_multiThr']:
                end_points = self.net(inputs['point_clouds'], training=False, img=img, calib=calib)        
            else:
                xyz = pc[0,:,:3]
                if self.model_config['use_edgetpu']:
                    pred_prob = run_semantic_seg_tflite(img, save_result=False)                
                else:                
                    pred_prob, pred_class = run_semantic_seg(img, save_result=False)  
                
                time_record.append(('Deeplab inference time:', time.time()))

                uv,d = calib.project_upright_depth_to_image(xyz) #uv: (N, 2)
                uv = np.rint(uv - 1)            

                pred_prob = pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)] # (npoint, num_class + 1 + 1 )
                projected_class = np.argmax(pred_prob, axis=-1) # (npoint, 1) 
                isPainted = np.where((projected_class > 0) & (projected_class < self.DATASET_CONFIG.num_class+1), 1, 0) # Point belongs to background?                    
                isPainted = np.expand_dims(isPainted, axis=-1)

                # 0 is background class, deeplab is trained with "person" included, (height, width, num_class)
                pred_prob = pred_prob[:,:(self.DATASET_CONFIG.num_class+1)] #(npoint, num_class)
                pointcloud = np.concatenate([xyz, isPainted, pred_prob, pc[0,:,3:]], axis=-1)
                time_record.append(('Pointpainting time:', time.time()))

                inputs['point_clouds'] = tf.convert_to_tensor(np.expand_dims(pointcloud, axis=0))
                
                end_points = self.net(inputs['point_clouds'])        
            
        else:        
            end_points = self.net(inputs['point_clouds'], training=False)                
        

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
        pred_map_cls, pred_mask = parse_predictions(end_points, self.eval_config_dict)
        
        
        type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9, 'person':10}
        class2type = {type2class[t]:t for t in type2class}
        
        print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))

        print("Inference time in the class", time.time() - time_record[0][1])  
        
        #demo_dir = os.path.join(BASE_DIR, 'demo_files')
        #dump_dir = os.path.join(demo_dir, 'sunrgbd_results')
        #if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
        #dump_results(end_points, dump_dir, self.DATASET_CONFIG, True)
        #print('Dumped detection results to folder %s'%(dump_dir))    
        conn.send(pred_map_cls)

        


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''    
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)    
    if point_cloud.shape[0] > 20000:        
        point_cloud = random_sampling(point_cloud, 20000)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,20000,4)
    return pc

def run_inference(conn):  
    votenet = VoteNetInf()
    while True:
        votenet.inference(conn)

from multiprocessing import Process, Queue

if __name__ == '__main__':

    while True:
        start = time.time()
        q0 = Queue()                             
        p0 = Process(target=votenet_inference, args=(q0,))
        p0.start()

        q0.put('/home/jetson/Documents/luxmea/point_cloud/pc.npy')
        q0.put('/home/jetson/Documents/luxmea/rgb_image/rgb.npy')
        Rtilt = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        K = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        q0.put((Rtilt, K))

        p0.join()
        predicted_class = q0.get()    
        end = time.time()
        print("Processing time:", end - start)
        print(predicted_class)


