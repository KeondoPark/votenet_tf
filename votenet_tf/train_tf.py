# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

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
from tf_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper_tf import APCalculator, parse_predictions, parse_groundtruths

import votenet_tf


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--use_painted', action='store_true', help='Use Point painting')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import SunrgbdDetectionVotesDataset_tfrecord, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset_tfrecord('train', num_points=NUM_POINT,
        augment=True, shuffle=True, batch_size=BATCH_SIZE,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
        #use_painted=FLAGS.use_painted)
    TEST_DATASET = SunrgbdDetectionVotesDataset_tfrecord('val', num_points=NUM_POINT,
        augment=False,  shuffle=False, batch_size=BATCH_SIZE,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
        #use_painted=FLAGS.use_painted)
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)

# Init the model and optimzier
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

### Point Paiting : Sementation score is appended at the end of point cloud
if FLAGS.use_painted:
    num_input_channel += DATASET_CONFIG.num_class


net = votenet_tf.VoteNet(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

#if torch.cuda.device_count() > 1:
#  log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#  net = nn.DataParallel(net)
#net.to(device)


#criterion = votenet_tf.get_loss
import loss_helper_tf
criterion = loss_helper_tf.get_loss

# Load the Adam optimizer # No weight decay in tf basic adam optimizer... so ignore
optimizer = tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)

if CHECKPOINT_PATH is None or not os.path.isdir(CHECKPOINT_PATH):
    print("Use defualt checkpoint path")
    CHECKPOINT_PATH = './log/tf_ckpt'

manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
print("Start epoch:", ckpt.epoch)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    start_epoch = ckpt.epoch.numpy()
else:
    print("Initializing from scratch.")

    #net.load_weights(CHECKPOINT_PATH)    
    #log_string("-> loaded checkpoint %s"%(CHECKPOINT_PATH))


# Decay Batchnorm momentum from 0.5 to 0.001
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: 1 - max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    optimizer.learning_rate = lr
"""
def lr_schedule(epoch, lr):
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr
"""

train_ds = TRAIN_DATASET.preprocess()
train_ds = train_ds.prefetch(BATCH_SIZE)

test_ds = TEST_DATASET.preprocess()
test_ds = test_ds.prefetch(BATCH_SIZE)

"""
train_ds = tf.data.Dataset.from_generator(lambda: (ret for ret in TRAIN_DATASET),
    output_types=({
                    'point_clouds': tf.float32,    
                    'center_label': tf.float32,
                    'heading_class_label': tf.int32,
                    'heading_residual_label': tf.float32,
                    'size_class_label': tf.int32,
                    'size_residual_label': tf.float32,
                    'sem_cls_label': tf.int32,
                    'box_label_mask': tf.float32,
                    'vote_label': tf.float32,
                    'vote_label_mask': tf.int32,
                    'scan_idx': tf.int32,
                    'max_gt_bboxes': tf.float32
                }),
    output_shapes = ({
        'point_clouds':tf.TensorShape((NUM_POINT,num_input_channel+3)),    
        'center_label':tf.TensorShape((64,3)),
        'heading_class_label':tf.TensorShape((64,)),
        'heading_residual_label': tf.TensorShape((64,)),
        'size_class_label': tf.TensorShape((64,)),
        'size_residual_label': tf.TensorShape((64,3)),
        'sem_cls_label': tf.TensorShape((64,)),
        'box_label_mask': tf.TensorShape((64,)),
        'vote_label': tf.TensorShape((NUM_POINT,9)),
        'vote_label_mask': tf.TensorShape((NUM_POINT,)),
        'scan_idx': tf.TensorShape(()),
        'max_gt_bboxes': tf.TensorShape((64,8))
    })
    )
                
train_ds = train_ds.batch(BATCH_SIZE).prefetch(2).shuffle(len(TRAIN_DATASET))

test_ds = tf.data.Dataset.from_generator(lambda: (ret for ret in TEST_DATASET),
    output_types=({
                    'point_clouds': tf.float32,    
                    'center_label': tf.float32,
                    'heading_class_label': tf.int32,
                    'heading_residual_label': tf.float32,
                    'size_class_label': tf.int32,
                    'size_residual_label': tf.float32,
                    'sem_cls_label': tf.int32,
                    'box_label_mask': tf.float32,
                    'vote_label': tf.float32,
                    'vote_label_mask': tf.int32,
                    'scan_idx': tf.int32,
                    'max_gt_bboxes': tf.float32
                }),
    output_shapes = ({
        'point_clouds':tf.TensorShape((NUM_POINT,num_input_channel+3)),    
        'center_label':tf.TensorShape((64,3)),
        'heading_class_label':tf.TensorShape((64,)),
        'heading_residual_label': tf.TensorShape((64,)),
        'size_class_label': tf.TensorShape((64,)),
        'size_residual_label': tf.TensorShape((64,3)),
        'sem_cls_label': tf.TensorShape((64,)),
        'box_label_mask': tf.TensorShape((64,)),
        'vote_label': tf.TensorShape((NUM_POINT,9)),
        'vote_label_mask': tf.TensorShape((NUM_POINT,)),
        'scan_idx': tf.TensorShape(()),
        'max_gt_bboxes': tf.TensorShape((64,8))
    })
    )
                
test_ds = test_ds.batch(BATCH_SIZE).prefetch(2)
"""
# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

label_dict = {0:'point_cloud', 1:'center_label', 2:'heading_class_label', 3:'heading_residual_label', 4:'size_class_label',\
    5:'size_residual_label', 6:'sem_cls_label', 7:'box_label_mask', 8:'vote_label', 9:'vote_label_mask', 10: 'max_gt_bboxes'}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum    

    #inputs = tf.constant([[1.0,2.0,3.0,1.0],[3.0,2.0,1.0,3.0]])
    #inputs = tf.constant([[1.0,2.0,3.0,1.0]])
    #inputs = {'point_clouds':tf.expand_dims(inputs, axis=0)}
    #output = net(inputs, training=True)
    #loss, end_points = criterion(output, DATASET_CONFIG)        \
    
    t_load = 0
    t_fwd = 0
    t_bwd = 0

    start = time.time()

    for batch_idx, batch_data in enumerate(train_ds):                 
        
        # Forward pass
        with tf.GradientTape() as tape:
            inputs = batch_data[0]            
            t_load += time.time() - start
            start = time.time()
            #for i, data in enumerate(batch_data):
            #    print("==============================",label_dict[i])
            #    if i == 8:
            #        np.savetxt("tf_votes.csv", data[0].numpy(), delimiter=",")
                            
                #else:
                #    print(data[0])

            #exit(0)


            end_points = net(inputs, training=True)
            t_fwd += time.time() - start
            start = time.time()
            # Compute loss and gradients, update parameters.
            for i, data in enumerate(batch_data):
                if i == 0: continue #pass point cloud
                end_points[label_dict[i]] = data
            loss, end_points = criterion(end_points, DATASET_CONFIG)        

        grads = tape.gradient(loss, net.trainable_weights)

        optimizer.apply_gradients(zip(grads, net.trainable_weights))
        
        t_bwd += time.time() - start        

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key]

        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:            
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            #TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
            #    (EPOCH_CNT*len(TRAIN_DATASET)+(batch_idx))*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
        #if batch_idx == 9:
        #    break

        start = time.time()

    log_string("Loading time:" + str(t_load))
    log_string("Forward time:" + str(t_fwd))
    log_string("Backward time:" +  str(t_bwd))


def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    
    for batch_idx, batch_data in enumerate(test_ds):        
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        
        # Forward pass
        inputs = batch_data[0] 
        end_points = net(inputs, training=False)

        # Compute loss
        for i, data in enumerate(batch_data):
            if i == 0: continue #pass point cloud
            end_points[label_dict[i]] = data
        loss, end_points = criterion(end_points, DATASET_CONFIG)        

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key]

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)        
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)    
         

    # Log statistics
    #TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
    #    EPOCH_CNT*len(TRAIN_DATASET)*BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 

    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))  

        train_one_epoch()
        ckpt.epoch.assign_add(1)
        
        save_path = manager.save()
        log_string("Saved checkpoint for step {}: {}".format(int(ckpt.epoch), save_path))

        #if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
        if EPOCH_CNT % 20 == 19: # Eval every 20 epochs
            loss = evaluate_one_epoch()            
            print("loss {:1.2f}".format(loss.numpy()))
        

if __name__=='__main__':
    #train(start_epoch)
    #print(len(TRAIN_DATASET), len(TEST_DATASET))
    #ret_dict = next(iter(TRAIN_DATASET))
    #print(ret_dict)
    train(start_epoch)
