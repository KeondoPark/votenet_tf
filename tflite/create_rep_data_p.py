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
TRAIN_DATASET = SunrgbdDetectionVotesDataset_tfrecord('train', num_points=20000,
        augment=False, shuffle=True, batch_size=BATCH_SIZE,
        use_color=False, use_height=True,
        use_painted=True)

ds = TRAIN_DATASET.preprocess()
ds = ds.prefetch(BATCH_SIZE)

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

    for batch_idx, batch_data in enumerate(ds):
        if batch_idx >= 800 / BATCH_SIZE: break
        inputs = batch_data[0]
        end_points = net(inputs, training=False)
        print(batch_idx, "-th batch...")
        rnd = np.random.rand(1)
        sa1_grouped_features = end_points['sa1_grouped_features1'] if rnd > 0.5  else end_points['sa1_grouped_features2']
        sa2_grouped_features = end_points['sa2_grouped_features1'] if rnd > 0.5  else end_points['sa2_grouped_features2']
        sa3_grouped_features = end_points['sa3_grouped_features1'] if rnd > 0.5  else end_points['sa3_grouped_features2']
        sa4_grouped_features = end_points['sa4_grouped_features1'] if rnd > 0.5  else end_points['sa4_grouped_features2']
        seed_features = end_points['seed_features']
        va_grouped_features = end_points['va_grouped_features']

        if (batch_idx * BATCH_SIZE) % 200 == 0:
            rnd = np.random.rand(1)
            #sa1_feats = sa1_grouped_features
            #sa2_feats = sa2_grouped_features
            #sa3_feats = sa3_grouped_features
            #sa4_feats = sa4_grouped_features        
            #voting_feats = seed_features 
            va_feats = va_grouped_features

        else:
            #sa1_feats = tf.concat([sa1_feats, sa1_grouped_features], axis=0)
            #sa2_feats = tf.concat([sa2_feats, sa2_grouped_features], axis=0)
            #sa3_feats = tf.concat([sa3_feats, sa3_grouped_features], axis=0)
            #sa4_feats = tf.concat([sa4_feats, sa4_grouped_features], axis=0)
            #voting_feats = tf.concat([voting_feats, seed_features], axis=0)
            va_feats = tf.concat([va_feats, va_grouped_features], axis=0)
        
        if ((batch_idx+1) * BATCH_SIZE) % 200 == 0:
            end = (batch_idx+1) * BATCH_SIZE
            start = end - 200
            """
            npy_filename = 'sa1_rep_' + str(start) + '_to_' + str(end) + '.npy'
            np.save(os.path.join(FLAGS.out_dir, npy_filename), sa1_feats.numpy())
            print(npy_filename, 'saved.')
            npy_filename = 'sa2_rep_' + str(start) + '_to_' + str(end) + '.npy'
            np.save(os.path.join(FLAGS.out_dir, npy_filename), sa2_feats.numpy())
            print(npy_filename, 'saved.')
            npy_filename = 'sa3_rep_' + str(start) + '_to_' + str(end) + '.npy'
            np.save(os.path.join(FLAGS.out_dir, npy_filename), sa3_feats.numpy())
            print(npy_filename, 'saved.')
            npy_filename = 'sa4_rep_' + str(start) + '_to_' + str(end) + '.npy'
            np.save(os.path.join(FLAGS.out_dir, npy_filename), sa4_feats.numpy())            
            print(npy_filename, 'saved.')
            
            npy_filename = 'voting_rep_' + str(start) + '_to_' + str(end) + '.npy'
            np.save(os.path.join(FLAGS.out_dir, npy_filename), voting_feats.numpy())
            print(npy_filename, 'saved.')
            """
            npy_filename = 'va_rep_' + str(start) + '_to_' + str(end) + '.npy'
            np.save(os.path.join(FLAGS.out_dir, npy_filename), va_feats.numpy())
            print(npy_filename, 'saved.')
    
    
    
    
    
    
