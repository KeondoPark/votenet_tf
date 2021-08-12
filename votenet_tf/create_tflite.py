import os
import sys
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply

import votenet_tf
from pointnet2 import tf_utils

import tensorflow as tf
from tensorflow.keras import layers

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, 20000)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc



if __name__=='__main__':
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import DC # dataset config
    checkpoint_path = os.path.join(demo_dir, 'tv_ckpt_210810')        
    pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')    
    

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier    
    net = votenet_tf.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        #sampling='seed_fps', num_class=DC.num_class,
        sampling='vote_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    epoch = ckpt.epoch.numpy()

    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud     
    point_cloud = read_ply(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': tf.convert_to_tensor(pc)}

    tic = time.time()
    end_points = net(inputs['point_clouds'], training=False)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    
    class SharedMLPModel(tf.keras.Model):
        def __init__(self, mlp_spec, nsample):
            super().__init__()
            self.sharedMLP = tf_utils.SharedMLP(mlp_spec, bn=True)
            self.max_pool = layers.MaxPooling2D(pool_size=(1, nsample), strides=1, data_format="channels_last")

        def call(self, grouped_features):
            new_features = self.max_pool(self.sharedMLP(grouped_features))

            return grouped_features

    sa1_model = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64)
    dummy_in = tf.convert_to_tensor(np.random.random([1,2048,64,4])) # (B, npoint, nsample, C+3)
    dummy_out = sa1_model(dummy_in)

    
    layer = sa1_model.sharedMLP
    to_wght = layer.get_weights()
    for w in to_wght:
        print(w.shape)
    print("=" * 20)
    from_wght = net.backbone_net.sa1.mlp_module.get_weights()
    for w in from_wght:
        print(w.shape)
    print("=" * 20)
    layer.set_weights(net.backbone_net.sa1.mlp_module.get_weights())


    converter = tf.lite.TFLiteConverter.from_keras_model(sa1_model)
    tflite_model = converter.convert()

    # Save the model.
    with open('sa1_model.tflite', 'wb') as f:
        f.write(tflite_model)


