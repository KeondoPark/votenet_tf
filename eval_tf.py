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

import groupfree_tf
from groupfree_tf import dump_results
from collections import defaultdict
from torch.utils.data import DataLoader
import tensorflow_addons as tfa


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
parser.add_argument('--conf_thresh', type=float, default=0.0, help='Filter out predictions with obj prob less than it. [default: 0.0]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
parser.add_argument('--config_path', default=None, required=True, help='Model configuration path')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')

parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')
parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
parser.add_argument('--clip_norm', default=0.1, type=float, help='gradient clipping max norm')
parser.add_argument('--decoder_normalization', default='layer', help='Which normalization method to use in decoder [layer or batch]')
parser.add_argument('--light', action='store_true', help='Use light version of detector')

FLAGS = parser.parse_args()

import json
model_config = json.load(open(FLAGS.config_path))

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

DEFAULT_CHECKPOINT_PATH = os.path.join('tf_ckpt', model_config['model_id'])
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
assert(CHECKPOINT_PATH is not None)

if FLAGS.dump_dir is None:
    DUMP_DIR = os.path.join('logs', model_config['model_id'])
else:
    DUMP_DIR = FLAGS.dump_dir

AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]


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

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

use_painted = model_config['use_painted']

if 'dataset' in model_config:
    DATASET = model_config['dataset']
else:
    DATASET =  FLAGS.dataset

# Create Dataset and Dataloader
if DATASET == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import SunrgbdDetectionDataset_tfrecord
    from model_util_sunrgbd import SunrgbdDatasetConfig
    include_person = 'include_person' in model_config and model_config['include_person']
    DATASET_CONFIG = SunrgbdDatasetConfig(include_person=include_person)
    
    TEST_DATASET = SunrgbdDetectionDataset_tfrecord('val', num_points=NUM_POINT,
        augment=False,  shuffle=FLAGS.shuffle_dataset, batch_size=BATCH_SIZE,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=use_painted, DC=DATASET_CONFIG)

elif DATASET == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()    

    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=use_painted)

    # Init datasets and dataloaders 
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    test_ds =DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)

else:
    print('Unknown dataset %s. Exiting...'%(DATASET))
    exit(-1)

# Init the model and optimzier
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

### Point Paiting : Sementation score is appended at the end of point cloud
if model_config['use_painted']:
    num_input_channel += DATASET_CONFIG.num_class + 1 + 1


net = groupfree_tf.GroupFreeNet(num_class=DATASET_CONFIG.num_class,
            num_heading_bin=DATASET_CONFIG.num_heading_bin,
            num_size_cluster=DATASET_CONFIG.num_size_cluster,
            mean_size_arr=DATASET_CONFIG.mean_size_arr,
            num_proposal=FLAGS.num_target,
            input_feature_dim=num_input_channel,                                
            model_config=model_config,
            size_cls_agnostic=FLAGS.size_cls_agnostic,
            decoder_normalization=FLAGS.decoder_normalization,
            light_detector=FLAGS.light)

import loss_helper_tf
criterion = loss_helper_tf.get_loss

# Load the optimizer
optimizer1 = tfa.optimizers.AdamW(learning_rate=0.006, weight_decay=0.0005, epsilon=1e-08)
optimizer2 = tfa.optimizers.AdamW(learning_rate=0.0006, weight_decay=0.0005, epsilon=1e-08)    

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer1=optimizer1, optimizer2=optimizer2, net=net)

if not model_config['use_tflite']:
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = ckpt.epoch.numpy()
    else:
        print("Failed to restore.")
        exit(-1)

if DATASET == 'sunrgbd':
    test_ds = TEST_DATASET.preprocess()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms':FLAGS.use_3d_nms,
    'nms_iou':FLAGS.nms_iou, 'use_old_type_nms':FLAGS.use_old_type_nms, 'cls_nms':FLAGS.use_cls_nms,
    'per_class_proposal': FLAGS.per_class_proposal, 'conf_thresh':FLAGS.conf_thresh,
    'dataset_config':DATASET_CONFIG}

label_dict = {0:'point_cloud', 1:'center_label', 2:'heading_class_label', 3:'heading_residual_label', 4:'size_class_label',\
    5:'size_residual_label', 6:'sem_cls_label', 7:'box_label_mask', 8:'point_obj_mask', 9:'point_instance_label', 10: 'max_gt_bboxes', 11: 'size_gts'}


def torch_to_tf_data(batch_data):
    point_clouds = tf.convert_to_tensor(batch_data['point_clouds'], dtype=tf.float32)
    center_label = tf.convert_to_tensor(batch_data['center_label'], dtype=tf.float32)
    heading_class_label = tf.convert_to_tensor(batch_data['heading_class_label'], dtype=tf.int64)
    heading_residual_label = tf.convert_to_tensor(batch_data['heading_residual_label'],dtype=tf.float32)
    size_class_label = tf.convert_to_tensor(batch_data['size_class_label'], dtype=tf.int64)
    size_residual_label = tf.convert_to_tensor(batch_data['size_residual_label'], dtype=tf.float32)                
    sem_cls_label = tf.convert_to_tensor(batch_data['sem_cls_label'], dtype=tf.int64)
    box_label_mask = tf.convert_to_tensor(batch_data['box_label_mask'], dtype=tf.float32)
    point_obj_mask = tf.convert_to_tensor(batch_data['point_obj_mask'], tf.int64)
    point_instance_label = tf.convert_to_tensor(batch_data['point_instance_label'], tf.int64)
    max_gt_bboxes = tf.convert_to_tensor(np.zeros((BATCH_SIZE, 64, 8)), dtype=tf.float32) 
    size_gts = tf.convert_to_tensor(batch_data['size_gts'], dtype=tf.float32)
    batch_data = point_clouds, center_label, heading_class_label, heading_residual_label, size_class_label, \
        size_residual_label, sem_cls_label, box_label_mask, point_obj_mask, point_instance_label, max_gt_bboxes, size_gts                   

    return batch_data

# ------------------------------------------------------------------------- GLOBAL CONFIG END

@tf.function
def evaluate_one_epoch(batch_data):     
    
    # For type match
    config = tf.constant(DATASET_CONFIG.num_heading_bin, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.num_size_cluster, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.num_class, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.mean_size_arr, dtype=tf.float32)
    
    # Forward pass
    point_cloud = batch_data[0]    
    
    end_points = net(point_cloud, training=False)

    for i, label in label_dict.items():
        if label_dict[i] not in end_points:
            end_points[label_dict[i]] = batch_data[i] 

    loss, end_points = criterion(end_points, config, num_decoder_layers=FLAGS.num_decoder_layers,
                                    query_points_generator_loss_coef=FLAGS.query_points_generator_loss_coef,
                                    obj_loss_coef=FLAGS.obj_loss_coef,
                                    box_loss_coef=FLAGS.box_loss_coef,
                                    sem_cls_loss_coef=FLAGS.sem_cls_loss_coef,
                                    query_points_obj_topk=FLAGS.query_points_obj_topk,
                                    center_delta=FLAGS.center_delta,                                     
                                    size_delta=FLAGS.size_delta,                                     
                                    heading_delta=FLAGS.heading_delta,
                                    size_cls_agnostic=FLAGS.size_cls_agnostic)     

    end_points['point_clouds'] = point_cloud
    return loss, end_points


def run_eval():
    stat_dict = defaultdict(int) # collect statistics            

    all_prefix = 'all_layers_'
    _prefixes = ['last_', 'proposal_']
    _prefixes += [f'{i}head_' for i in range(FLAGS.num_decoder_layers - 1)]

    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]

    all_ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    
    start = time.time()    
    total_start = start
    for batch_idx, batch_data in enumerate(test_ds):        
        # if batch_idx*BATCH_SIZE >= 400: break
        if DATASET == 'scannet':                
            batch_data = torch_to_tf_data(batch_data)

        if batch_idx % 10 == 0:
            end = time.time()
            print('---------- Eval batch: %d ----------'%(batch_idx) + str(end - start))
            start = time.time() 

        curr_loss, end_points = evaluate_one_epoch(batch_data)       
          

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float) or isinstance(end_points[key], int):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].numpy()

        batch_pred_map_cls, pred_mask = parse_predictions(end_points, 
                                                                CONFIG_DICT, 
                                                                prefix='last_', 
                                                                size_cls_agnostic=FLAGS.size_cls_agnostic) 

        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 



        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)                 
     
        end_points[f'{all_prefix}center'] = tf.concat([end_points[f'{ppx}center']
                                                    for ppx in _prefixes], 1)
        end_points[f'{all_prefix}heading_scores'] = tf.concat([end_points[f'{ppx}heading_scores']
                                                            for ppx in _prefixes], 1)
        end_points[f'{all_prefix}heading_residuals'] = tf.concat([end_points[f'{ppx}heading_residuals']
                                                                for ppx in _prefixes], 1)
        if FLAGS.size_cls_agnostic:
            end_points[f'{all_prefix}pred_size'] = tf.concat([end_points[f'{ppx}pred_size']
                                                            for ppx in _prefixes], 1)
        else:
            end_points[f'{all_prefix}size_scores'] = tf.concat([end_points[f'{ppx}size_scores']
                                                            for ppx in _prefixes], 1)
            end_points[f'{all_prefix}size_residuals'] = tf.concat([end_points[f'{ppx}size_residuals']
                                                                for ppx in _prefixes], 1)
        end_points[f'{all_prefix}sem_cls_scores'] = tf.concat([end_points[f'{ppx}sem_cls_scores']
                                                            for ppx in _prefixes], 1)
        end_points[f'{all_prefix}objectness_scores'] = tf.concat([end_points[f'{ppx}objectness_scores']
                                                                for ppx in _prefixes], 1)

        all_batch_pred_map_cls, all_pred_mask = parse_predictions(end_points, 
                                                                CONFIG_DICT, 
                                                                prefix=all_prefix, 
                                                                size_cls_agnostic=FLAGS.size_cls_agnostic) 
        
        for ap_calculator in all_ap_calculator_list:
            ap_calculator.step(all_batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        # if batch_idx == 0:
        #     dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

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

    log_string('---------- Eval All Layers ----------')

    # Evaluate average precision: All layers
    for i, ap_calculator in enumerate(all_ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    log_string('total eval time: '+ str(time.time() - total_start))
    return mean_loss

def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = run_eval()

if __name__=='__main__':
    eval()
