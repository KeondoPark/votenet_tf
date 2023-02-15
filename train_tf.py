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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

import tensorflow_addons as tfa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from tf_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper_tf import APCalculator, parse_predictions, parse_groundtruths
from collections import defaultdict
from torch.utils.data import DataLoader

import groupfree_tf
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--load_from', default=None, help='Loading checkpoint path [default: None]')
parser.add_argument('--log_dir', default=None, help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 400]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')

parser.add_argument('--bn_decay_step', type=int, default=40, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')

parser.add_argument('--lr-scheduler', type=str, default='step', choices=["step", "cosine"], help="learning rate scheduler")
parser.add_argument('--cosine_alpha', type=float, default=0.01, help='Final lr to the beginning lr when cosine learning schedule is used.')
parser.add_argument('--lr_decay_steps', default='280,340', help='When to decay the learning rate (in epochs) [default: 280,340]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1]')
parser.add_argument('--decoder_lr_decay_steps', default='280,340', help='When to decay the learning rate (in epochs) [default: 280,340]')
parser.add_argument('--decoder_lr_decay_rates', default='0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1]')

parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--config_path', default=None, required=True, help='Model configuration path')

parser.add_argument('--optimizer', default='adam', help='Which optimizer to use [adam or adamw]')
parser.add_argument('--learning_rate', type=float, default=0.004, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--decoder_learning_rate', type=float, default=0.0004, help='Initial learning rate for decoder [default: 0.0004]')

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
parser.add_argument('--clip_norm', default=0.0, type=float, help='gradient clipping max norm')
parser.add_argument('--decoder_normalization', default='layer', help='Which normalization method to use in decoder [layer or batch]')
parser.add_argument('--light', action='store_true', help='Use light version of detector')

FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
model_config = json.load(open(FLAGS.config_path))

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate

LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
DECODER_LR_DECAY_STEPS = [int(x) for x in FLAGS.decoder_lr_decay_steps.split(',')]
DECODER_LR_DECAY_RATES = [float(x) for x in FLAGS.decoder_lr_decay_rates.split(',')]

assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
assert(len(DECODER_LR_DECAY_STEPS)==len(DECODER_LR_DECAY_RATES))

if FLAGS.log_dir is None:
    LOG_DIR = os.path.join('logs', model_config['model_id'])
else:
    LOG_DIR = FLAGS.log_dir
#DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DEFAULT_DUMP_DIR = LOG_DIR
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join('tf_ckpt', model_config['model_id'])
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR
use_painted = model_config['use_painted']
if 'dataset' in model_config:
    DATASET = model_config['dataset']
else:
    DATASET =  FLAGS.dataset

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
LOG_FOUT.write(str(model_config)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if DATASET == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import SunrgbdDetectionDataset_tfrecord, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    include_person = 'include_person' in model_config and model_config['include_person']
    include_small = 'include_small' in model_config and model_config['include_small']
    DATASET_CONFIG = SunrgbdDatasetConfig(include_person=include_person, include_small=include_small)
    TRAIN_DATASET = SunrgbdDetectionDataset_tfrecord('train', num_points=NUM_POINT,
        augment=True, shuffle=True, batch_size=BATCH_SIZE,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=use_painted, use_repsurf=True, DC=DATASET_CONFIG)
    TEST_DATASET = SunrgbdDetectionDataset_tfrecord('val', num_points=NUM_POINT,
        augment=False,  shuffle=False, batch_size=BATCH_SIZE,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=use_painted, use_repsurf=True, DC=DATASET_CONFIG)
elif DATASET == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    # NUM_POINT = 40000
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=use_painted)
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_painted=use_painted)

    # Init datasets and dataloaders 
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def val_worker_init_fn(worker_id):
        np.random.seed(2481757)

    train_ds = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn, drop_last=True)
    test_ds =DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, worker_init_fn=val_worker_init_fn)
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)

# Init the model and optimzier
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

### Point Paiting : Sementation score is appended at the end of point cloud
if use_painted:
    # Probabilties that each point belongs to each class + is the point belong to background(Boolean)
    num_input_channel += DATASET_CONFIG.num_class + 1 + 1

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:    
        print(e)

import loss_helper_tf

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
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

criterion = loss_helper_tf.get_loss

# Load the optimizer
if FLAGS.optimizer == 'adamw':
    optimizer1 = tfa.optimizers.AdamW(learning_rate=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, epsilon=1e-08)
    optimizer2 = tfa.optimizers.AdamW(learning_rate=FLAGS.decoder_learning_rate, weight_decay=FLAGS.weight_decay, epsilon=1e-08)
    if FLAGS.clip_norm > 0:
        optimizer1.global_clipnorm = FLAGS.clip_norm
        optimizer2.global_clipnorm = FLAGS.clip_norm  
    
else:
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=FLAGS.decoder_learning_rate)

if DATASET == 'sunrgbd':
    train_ds = TRAIN_DATASET.preprocess()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = TEST_DATASET.preprocess()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
    # test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)


# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer1=optimizer1, optimizer2=optimizer2, net=net)

if CHECKPOINT_PATH is None:
    print("Use defualt checkpoint path")
    CHECKPOINT_PATH = './tf_ckpt'

if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)


# with mirrored_strategy.scope():
if FLAGS.load_from is not None:
    manager = tf.train.CheckpointManager(ckpt, FLAGS.load_from, max_to_keep=3)
    print("Restored from {}".format(manager.latest_checkpoint))
    ckpt.restore(manager.latest_checkpoint)
    start_epoch = ckpt.epoch.numpy()
else:
    print("Initializing from scratch.")

manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=3)
print("Start epoch:", ckpt.epoch)

    


        
# Decay Batchnorm momentum from 0.5 to 0.001
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lmbd = lambda it: 1 - max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
# bn_lmbd = lambda it: 0.9
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lmbd, last_epoch=start_epoch-1)

def get_current_lr(epoch, base_lr, lr_decay_rates, lr_decay_steps):
    lr = base_lr
    for i,lr_decay_epoch in enumerate(lr_decay_steps):
        if epoch >= lr_decay_epoch:
            lr *= lr_decay_rates[i]
    return lr

def adjust_cosine_lr(optimizer, epoch, base_lr, base_wd, decay_steps, alpha=0.01):
    step = min(epoch, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    # cosine_decay = 1 + np.cos(np.pi/2 * (1 + step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    curr_lr = base_lr * decayed      
    optimizer.learning_rate = float(curr_lr)
    if FLAGS.optimizer == 'adamw':
        curr_wd = base_wd * decayed      
        optimizer.weight_decay = float(curr_wd)
        return curr_lr, curr_wd
    else:
        return curr_lr, None    

def adjust_learning_rate(optimizer, epoch, base_lr, base_wd, lr_decay_rates, lr_decay_steps):
    lr = get_current_lr(epoch, base_lr, lr_decay_rates, lr_decay_steps)
    optimizer.learning_rate = lr
    if FLAGS.optimizer == 'adamw':
        wd = get_current_lr(epoch, base_wd, lr_decay_rates, lr_decay_steps)
        optimizer.weight_decay = float(wd)
    return lr



# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(LOG_DIR, 'train')
TEST_VISUALIZER = TfVisualizer(LOG_DIR, 'test')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.0,
    'dataset_config':DATASET_CONFIG}

label_dict = {0:'point_cloud', 1:'center_label', 2:'heading_class_label', 3:'heading_residual_label', \
              4:'size_class_label', 5:'size_residual_label', 6:'sem_cls_label', 7:'box_label_mask', \
              8:'point_obj_mask', 9:'point_instance_label', 10: 'max_gt_bboxes', 11: 'size_gts', \
              12:'repsurf_feature'}


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
    repsurf_feature = tf.convert_to_tensor(batch_data['repsurf_feature'], dtype=tf.float32)
    batch_data = point_clouds, center_label, heading_class_label, heading_residual_label, size_class_label, \
        size_residual_label, sem_cls_label, box_label_mask, point_obj_mask, point_instance_label, max_gt_bboxes, size_gts, repsurf_feature    

    return batch_data

# ------------------------------------------------------------------------- GLOBAL CONFIG END

@tf.function
def train_one_epoch(batch_data):     

    # For type match 
    config = tf.constant(DATASET_CONFIG.num_heading_bin, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.num_size_cluster, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.num_class, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.mean_size_arr, dtype=tf.float32)

    # Forward pass
    with tf.GradientTape() as tape:

        point_cloud = batch_data[0]
        repsurf_feature = batch_data[-1]
        end_points = net(point_cloud, repsurf_feature, training=True)    
                    
        # Compute loss and gradients, update parameters.
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

    grads = tape.gradient(loss, net.trainable_weights)

    # DIfferent learning rate by layers
    varlist1, varlist2 = [], []
    gradlist1, gradlist2 = [], []
    for v, g in zip(net.trainable_weights, grads):  
        if 'decoder' in v.name:
            varlist2.append(v)
            gradlist2.append(g)
        else:
            varlist1.append(v)
            gradlist1.append(g)
    # optimizer1.apply_gradients(zip(grads, net.trainable_weights))      
    optimizer1.apply_gradients(zip(gradlist1, varlist1))    
    optimizer2.apply_gradients(zip(gradlist2, varlist2))

    return loss

@tf.function
def evaluate_one_epoch(batch_data):     
    
    # For type match
    config = tf.constant(DATASET_CONFIG.num_heading_bin, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.num_size_cluster, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.num_class, dtype=tf.int32), \
        tf.constant(DATASET_CONFIG.mean_size_arr, dtype=tf.float32)
    
    # Forward pass
    point_cloud = batch_data[0]
    repsurf_feature = batch_data[-1]
    end_points = net(point_cloud, repsurf_feature, training=False)

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

    return loss, end_points
         
def train(start_epoch):
    global EPOCH_CNT   

    # input_signature=[[     
    #     tf.TensorSpec(shape=(None, NUM_POINT, 3+num_input_channel), dtype=tf.float32), #point cloud        
    #     tf.TensorSpec(shape=(None, 64, 3), dtype=tf.float32), #center label
    #     tf.TensorSpec(shape=(None, 64), dtype=tf.int64), #heading class label
    #     tf.TensorSpec(shape=(None, 64), dtype=tf.float32), #heading residual label
    #     tf.TensorSpec(shape=(None, 64), dtype=tf.int64), #size_class_label
    #     tf.TensorSpec(shape=(None, 64, 3), dtype=tf.float32), #size_residual_label
    #     tf.TensorSpec(shape=(None, 64), dtype=tf.int64), #sem_cls_label
    #     tf.TensorSpec(shape=(None, 64), dtype=tf.float32), #box_label_mask
    #     tf.TensorSpec(shape=(None, NUM_POINT), dtype=tf.int64), #point_obj_mask
    #     tf.TensorSpec(shape=(None, NUM_POINT), dtype=tf.int64), #point_instance_label
    #     tf.TensorSpec(shape=(None, 64,8), dtype=tf.float32), #max_gt_bboxes                           
    #     tf.TensorSpec(shape=(None, 64,3), dtype=tf.float32), #size_gts
    # ]]

    # `run` replicates the provided computation and runs it
    # with the distributed input.
    # @tf.function #(experimental_relax_shapes=True, input_signature=input_signature)
    # def distributed_train_step(batch_data):
    #     per_replica_losses = mirrored_strategy.run(train_one_epoch, args=(batch_data, ))
    #     return mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        

    # @tf.function #(experimental_relax_shapes=True, input_signature=input_signature)
    # def distributed_eval_step(batch_data):
    #     per_replica_losses, end_points = mirrored_strategy.run(evaluate_one_epoch, args=(batch_data,))
    #     return mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, end_points, axis=None)
    
    best_eval_mAP = 0.0
    
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))

        if FLAGS.lr_scheduler == 'cosine':                
            curr_lr, curr_wd = adjust_cosine_lr(optimizer1, EPOCH_CNT, FLAGS.learning_rate, FLAGS.weight_decay, MAX_EPOCH, alpha=FLAGS.cosine_alpha) # int(MAX_EPOCH * 0.75))                        
            curr_decoder_lr, curr_decoder_wd = adjust_cosine_lr(optimizer2, EPOCH_CNT, FLAGS.decoder_learning_rate, FLAGS.weight_decay, MAX_EPOCH, alpha=FLAGS.cosine_alpha) #int(MAX_EPOCH * 0.75))            
            
        else:
            curr_lr = adjust_learning_rate(optimizer1, EPOCH_CNT, FLAGS.learning_rate, FLAGS.weight_decay, LR_DECAY_RATES, LR_DECAY_STEPS)
            curr_decoder_lr = adjust_learning_rate(optimizer2, EPOCH_CNT, FLAGS.decoder_learning_rate, FLAGS.weight_decay, DECODER_LR_DECAY_RATES, DECODER_LR_DECAY_STEPS)
        
        log_string('Current learning rate:%f'%(curr_lr))
        log_string('Current decoder learning rate:%f'%(curr_decoder_lr))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))  

        bnm_scheduler.step() # decay BN momentum  
        train_loss = tf.constant(0.0, tf.float32)
        eval_loss = tf.constant(0.0, tf.float32)

        t_epoch = 0
        start = time.time()
        
        for batch_idx, batch_data in enumerate(train_ds):                   
               
            if DATASET == 'scannet':                
                batch_data = torch_to_tf_data(batch_data)

            # train_loss += distributed_train_step(batch_data)                      
            train_loss += train_one_epoch(batch_data)  
            
            
            # Accumulate statistics and print out
            #for key in end_points:
            #    if 'loss' in key or 'acc' in key or 'ratio' in key:
            #        if key not in stat_dict: stat_dict[key] = 0
            #        stat_dict[key] += end_points[key]            
            batch_interval = 50
            if (batch_idx+1) % batch_interval == 0:            
                log_string(' ---- batch: %03d ----' % (batch_idx+1))
                log_string('mean loss: %f'%(train_loss/batch_interval))
                train_loss = tf.constant(0.0, tf.float32)
                #TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
                #    (EPOCH_CNT*len(TRAIN_DATASET)+(batch_idx))*BATCH_SIZE)
                #for key in sorted(stat_dict.keys()):
                #    log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                #    stat_dict[key] = 0
        
        t_epoch = time.time() - start
        log_string("1 Epoch training time:" + str(t_epoch))        
        
        
        # if EPOCH_CNT+1 >= 100 and (EPOCH_CNT % 10 == 9 or EPOCH_CNT == 0): # Eval every 10 epochs                                
        if (EPOCH_CNT % 10 == 9 or EPOCH_CNT == 0): # Eval every 10 epochs                                
            stat_dict = defaultdict(float) # collect statistics            
            ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
                class2type_map=DATASET_CONFIG.class2type)     

            if 1+EPOCH_CNT == 150:
                # Reset best mAP at 100 epochs
                best_eval_mAP = 0.0 

            stop_eval_early = False 
            if 1+EPOCH_CNT < 150:
                # at early stage of training, does not do the evaluation fully.
                stop_eval_early = True

            for batch_idx, batch_data in enumerate(test_ds):                               
                if DATASET == 'scannet':      
                    batch_data = torch_to_tf_data(batch_data)                          
                
                if batch_idx % 10 == 0:
                    print('Eval batch: %d'%(batch_idx))

                if stop_eval_early and batch_idx * BATCH_SIZE >= 160:
                    break
                
                # curr_loss, end_points = distributed_eval_step(batch_data)
                curr_loss, end_points = evaluate_one_epoch(batch_data)
                
                eval_loss += curr_loss

                # Accumulate statistics and print out
                for key in end_points:
                    if 'loss' in key or 'acc' in key or 'ratio' in key or 'sa1_obj_sum' in key:
                        if key not in stat_dict: stat_dict[key] = 0
                        if isinstance(end_points[key], float) or isinstance(end_points[key], int):
                            stat_dict[key] += end_points[key]
                        else:
                            stat_dict[key] += end_points[key].numpy()

                batch_pred_map_cls, pred_mask = parse_predictions(end_points, 
                                                                CONFIG_DICT, 
                                                                prefix='last_', 
                                                                size_cls_agnostic=FLAGS.size_cls_agnostic)        
                batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT, size_cls_agnostic=FLAGS.size_cls_agnostic)
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

            
            eval_mAP = metrics_dict['mAP']
            if eval_mAP > best_eval_mAP:                
                save_path = manager.save(checkpoint_number=ckpt.epoch)
                log_string("Saved checkpoint for step {}: {}".format(int(ckpt.epoch), save_path))
                best_eval_mAP = eval_mAP
                

        ckpt.epoch.assign_add(1)

if __name__=='__main__':   

    train(start_epoch)
