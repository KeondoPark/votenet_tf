import os
import sys
import numpy as np
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from pc_util import random_sampling, read_ply

import votenet_tf
from pointnet2 import tf_utils

import tensorflow as tf
from tensorflow.keras import layers

from sunrgbd_detection_dataset_tf import SunrgbdDetectionVotesDataset_tfrecord, MAX_NUM_OBJ
from model_util_sunrgbd import SunrgbdDatasetConfig
from sunrgbd_detection_dataset_tf import DC # dataset config
import voting_module_tf

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--out_dir', default="tflite_rep_data", help='Folder name where output tflite files are saved')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')
FLAGS = parser.parse_args()

# Limit GPU Memory usage, 256MB suffices
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

BATCH_SIZE = 1
TEST_DATASET = SunrgbdDetectionVotesDataset_tfrecord('val', num_points=20000,
    augment=False,  shuffle=True, batch_size=BATCH_SIZE,
    use_color=False, use_height=True)

test_ds = TEST_DATASET.preprocess()
test_ds = test_ds.prefetch(BATCH_SIZE)


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
    #checkpoint_path = os.path.join(demo_dir, 'tv_ckpt_210810')        
    checkpoint_path = FLAGS.checkpoint_path    

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier    
    net = votenet_tf.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        #sampling='seed_fps', num_class=DC.num_class,
        sampling='vote_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        use_tflite=True)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    epoch = ckpt.epoch.numpy()
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
    
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)

    for batch_idx, batch_data in enumerate(test_ds):
        if batch_idx >= 800 / BATCH_SIZE: break
        inputs = batch_data[0]
        end_points = net(inputs, training=False)
        print(batch_idx, "-th batch...")
        if (batch_idx * BATCH_SIZE) % 200 == 0:
            #sa1_feats = end_points['sa1_grouped_features']
            #sa2_feats = end_points['sa2_grouped_features']
            #sa3_feats = end_points['sa3_grouped_features']
            #sa4_feats = end_points['sa4_grouped_features']
            #fp1_feats = end_points['fp1_grouped_features']            
            fp2_feats = end_points['fp1_grouped_features']            

        else:
            #sa1_feats = tf.concat([sa1_feats, end_points['sa1_grouped_features']], axis=0)
            #sa2_feats = tf.concat([sa2_feats, end_points['sa2_grouped_features']], axis=0)
            #sa3_feats = tf.concat([sa3_feats, end_points['sa3_grouped_features']], axis=0)
            #sa4_feats = tf.concat([sa4_feats, end_points['sa4_grouped_features']], axis=0)
            #fp1_feats = tf.concat([fp1_feats, end_points['fp1_grouped_features']], axis=0)
            fp2_feats = tf.concat([fp2_feats, end_points['fp1_grouped_features']], axis=0)
        
        if ((batch_idx+1) * BATCH_SIZE) % 200 == 0:
            end = (batch_idx+1) * BATCH_SIZE
            start = end - 200
            #np.save(os.path.join(FLAGS.out_dir, 'sa1_rep_' + str(start) + '_to_' + str(end) + '.npy'), sa1_feats.numpy())
            #np.save(os.path.join(FLAGS.out_dir, 'sa2_rep_' + str(start) + '_to_' + str(end) + '.npy'), sa2_feats.numpy())
            #np.save(os.path.join(FLAGS.out_dir, 'sa3_rep_' + str(start) + '_to_' + str(end) + '.npy'), sa3_feats.numpy())
            #np.save(os.path.join(FLAGS.out_dir, 'sa4_rep_' + str(start) + '_to_' + str(end) + '.npy'), sa4_feats.numpy())
            #np.save(os.path.join(FLAGS.out_dir, 'fp1_rep_' + str(start) + '_to_' + str(end) + '.npy'), fp1_feats.numpy())
            np.save(os.path.join(FLAGS.out_dir, 'fp2_rep_' + str(start) + '_to_' + str(end) + '.npy'), fp2_feats.numpy())
    
    
    
    
    
    
