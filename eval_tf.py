# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.optim import lr_scheduler
#from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper_tf import APCalculator, parse_predictions, parse_groundtruths

import votenet_tf
from votenet_tf import dump_results
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
parser.add_argument('--config_path', default=None, required=True, help='Model configuration path')

FLAGS = parser.parse_args()

import json
model_config = json.load(open(FLAGS.config_path))

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
DEFAULT_CHECKPOINT_PATH = os.path.join('tf_ckpt', model_config['model_id'])
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
assert(CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]


# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)


# Create Dataset and Dataloader
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import SunrgbdDetectionVotesDataset_tfrecord
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TEST_DATASET = SunrgbdDetectionVotesDataset_tfrecord('val', num_points=NUM_POINT,
        augment=False,  shuffle=FLAGS.shuffle_dataset, batch_size=BATCH_SIZE,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=model_config['use_painted'])
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)

# Init the model and optimzier
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

### Point Paiting : Sementation score is appended at the end of point cloud
if model_config['use_painted']:
    num_input_channel += DATASET_CONFIG.num_class


net = votenet_tf.VoteNet(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling,
               model_config=model_config)

import loss_helper_tf
criterion = loss_helper_tf.get_loss

# Load the Adam optimizer # No weight decay in tf basic adam optimizer... so ignore
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)

if not model_config['use_tflite']:
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = ckpt.epoch.numpy()
    else:
        print("Failed to restore.")
        exit(-1)

test_ds = TEST_DATASET.preprocess()
test_ds = test_ds.prefetch(BATCH_SIZE)

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms':FLAGS.use_3d_nms,
    'nms_iou':FLAGS.nms_iou, 'use_old_type_nms':FLAGS.use_old_type_nms, 'cls_nms':FLAGS.use_cls_nms,
    'per_class_proposal': FLAGS.per_class_proposal, 'conf_thresh':FLAGS.conf_thresh,
    'dataset_config':DATASET_CONFIG}

label_dict = {0:'point_cloud', 1:'center_label', 2:'heading_class_label', 3:'heading_residual_label', 4:'size_class_label',\
    5:'size_residual_label', 6:'sem_cls_label', 7:'box_label_mask', 8:'vote_label', 9:'vote_label_mask', 10: 'max_gt_bboxes'}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def evaluate_one_epoch():
    stat_dict = defaultdict(int) # collect statistics            
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    start = time.time()
    start2 = time.time()
    for batch_idx, batch_data in enumerate(test_ds):        
        #if batch_idx*BATCH_SIZE >= 100: break
        if batch_idx % 10 == 0:
            end = time.time()
            log_string('---------- Eval batch: %d ----------'%(batch_idx) + str(end - start))
            start = time.time()
        start2 = time.time()
        # Forward pass
        
        point_cloud = batch_data[0]
        end_points = net(point_cloud, training=False)
        for i, label in label_dict.items():
            if label_dict[i] not in end_points:
                end_points[label_dict[i]] = batch_data[i] 

        end_points['point_clouds'] = point_cloud

        config = tf.constant(DATASET_CONFIG.num_heading_bin, dtype=tf.int32), \
            tf.constant(DATASET_CONFIG.num_size_cluster, dtype=tf.int32), \
            tf.constant(DATASET_CONFIG.num_class, dtype=tf.int32), \
            tf.constant(DATASET_CONFIG.mean_size_arr, dtype=tf.float32)
        loss, end_points = criterion(end_points, config)        

        stat_dict['box_loss'] += end_points['box_loss']
        stat_dict['vote_loss'] += end_points['vote_loss']
        stat_dict['objectness_loss'] += end_points['objectness_loss']
        stat_dict['sem_cls_loss'] += end_points['sem_cls_loss']
        stat_dict['loss'] += end_points['loss']
        stat_dict['obj_acc'] += end_points['obj_acc']
        stat_dict['pos_ratio'] += end_points['pos_ratio']
        stat_dict['neg_ratio'] += end_points['neg_ratio']
        end2 = time.time()
        log_string('inference time: ' +  str(end2 - start2))        

        batch_pred_map_cls, pred_mask = parse_predictions(end_points, CONFIG_DICT)        
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)            
        
        # Dump evaluation results for visualization
        if batch_idx == 0:
            dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    # Log statistics
    #TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
    #    EPOCH_CNT*len(TRAIN_DATASET)*BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss
def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    eval()
