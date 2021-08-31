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
parser.add_argument('--out_dir', default="tflite_models", help='Folder name where output tflite files are saved')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')
parser.add_argument('--use_rep_data', action='store_true', help='When iterating representative dataset, use saved data')
parser.add_argument('--rep_data_dir', default='tflite_rep_data', help='Saved representative data directory')
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

if not FLAGS.use_rep_data:
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

def wrapper_representative_data_gen_mlp(keyword, base_model):
    def representative_data_gen_mlp():
        for i in range(int(800 / BATCH_SIZE)):
            if not FLAGS.use_rep_data:            
                batch_data = next(iter(test_ds))
                inputs = batch_data[0]
                end_points = base_model(inputs, training=False)
                print(i, "-th batch...")
                yield [end_points[keyword + '_grouped_features']]
            else:
                if (i * BATCH_SIZE) % 200 == 0:
                    start = i * BATCH_SIZE
                    end = start + 200                            
                    np_feats = np.load(os.path.join(FLAGS.rep_data_dir, keyword + '_rep_' + str(start) + '_to_' + str(end) + '.npy'))
                    feats = tf.convert_to_tensor(np_feats)
                print(i, "-th batch...")
                idx = (i * BATCH_SIZE) % 200
                yield [feats[idx: idx+BATCH_SIZE,:,:,:]]
    return representative_data_gen_mlp

def wrapper_representative_data_gen_voting(base_model):
    def representative_data_gen_voting():
        for i in range(int(800 / BATCH_SIZE)):
            batch_data = next(iter(test_ds))
            inputs = batch_data[0]
            end_points = base_model(inputs, training=False)
            print(i, "-th batch...")
            yield [tf.expand_dims(end_points['seed_features'], axis=-2)]
    return representative_data_gen_voting

# TFlite conversion
def tflite_convert(keyword, model, base_model, out_dir, mlp=True):
    # A generator that provides a representative dataset

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    if mlp:
        converter.representative_dataset = wrapper_representative_data_gen_mlp(keyword, base_model)
    else:        
        converter.representative_dataset = wrapper_representative_data_gen_voting(base_model)
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open(os.path.join(out_dir, keyword + '_quant.tflite'), 'wb') as f:
        f.write(tflite_model)

if __name__=='__main__':
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')         
    #checkpoint_path = os.path.join(demo_dir, 'tv_ckpt_210810')        
    checkpoint_path = FLAGS.checkpoint_path
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
        mean_size_arr=DC.mean_size_arr,
        use_tflite=False)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    epoch = ckpt.epoch.numpy()

    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud     
    pc = tf.convert_to_tensor(np.random.random([BATCH_SIZE,20000,4]))
    #pc = preprocess_point_cloud(point_cloud)
    #print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': tf.convert_to_tensor(pc)}

    tic = time.time()
    end_points = net(inputs['point_clouds'], training=False)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))

    ############################## Above is base model inference ######################################
    ############################## Below is tflite conversion #########################################

    # Build Shard MLP parts of the pointnet backbone as a model
    class SharedMLPModel(tf.keras.Model):
        def __init__(self, mlp_spec, input_shape, nsample=0):
            super().__init__()
            self.sharedMLP = tf_utils.SharedMLP(mlp_spec, bn=True, input_shape=input_shape)
            self.nsample = nsample
            if nsample:
                self.max_pool = layers.MaxPooling2D(pool_size=(1, 16), strides=(1,16), data_format="channels_last")
                self.max_pool2 = layers.MaxPooling2D(pool_size=(1, int(self.nsample/16)), strides=(1,int(self.nsample/16)), data_format="channels_last")

        def call(self, grouped_features):
            if self.nsample:
                if self.nsample == 16:
                    new_features = self.max_pool(self.sharedMLP(grouped_features))
                elif self.nsample > 16:
                    new_features = self.max_pool2(self.max_pool(self.sharedMLP(grouped_features)))
            else:
                new_features = self.sharedMLP(grouped_features)

            return new_features

    class nnInVotingModule(tf.keras.Model):
        def __init__(self, vote_factor, seed_feature_dim):            
            super().__init__()
            
            self.vote_factor = vote_factor
            self.in_dim = seed_feature_dim
            self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
            self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1)        
            self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv3 = layers.Conv2D(filters=(3+self.out_dim) * self.vote_factor, kernel_size=1) 
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)
            self.relu1 = layers.Activation('relu')
            self.relu2 = layers.Activation('relu')
        
        def call(self, seed_features):
            net = self.relu1(self.bn1(self.conv1(seed_features))) 
            net = self.relu2(self.bn2(self.conv2(net))) 
            net = self.conv3(net) # (batch_size, num_seed, (3+out_dim)*vote_factor)   
            return net             

    converting_layers = ['fp1']

    if 'sa1' in converting_layers:    
        sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=[2048,64,1+3])
        dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,2048,64,4])) # (B, npoint, nsample, C+3)
        dummy_out = sa1_mlp(dummy_in_sa1)
        # Copy weights from the base model    
        layer = sa1_mlp.sharedMLP
        layer.set_weights(net.backbone_net.sa1.mlp_module.get_weights()) 
        tflite_convert('sa1', sa1_mlp, net, FLAGS.out_dir)

    if 'sa2' in converting_layers:
        sa2_mlp = SharedMLPModel(mlp_spec=[128, 128, 128, 256], nsample=32, input_shape=[1024,32,128+3])
        dummy_in_sa2 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,32,128+3])) # (B, npoint, nsample, C+3)
        dummy_out = sa2_mlp(dummy_in_sa2)
        layer = sa2_mlp.sharedMLP
        layer.set_weights(net.backbone_net.sa2.mlp_module.get_weights())
        tflite_convert('sa2', sa2_mlp, net, FLAGS.out_dir)

    if 'sa3' in converting_layers:
        sa3_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16, input_shape=[512,16,256+3])
        dummy_in_sa3 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,512,16,256+3])) # (B, npoint, nsample, C+3)
        dummy_out = sa3_mlp(dummy_in_sa3)
        layer = sa3_mlp.sharedMLP
        layer.set_weights(net.backbone_net.sa3.mlp_module.get_weights())
        tflite_convert('sa3', sa3_mlp, net, FLAGS.out_dir)

    if 'sa4' in converting_layers:
        sa4_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16, input_shape=[256,16,256+3])
        dummy_in_sa4 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,16,256+3])) # (B, npoint, nsample, C+3)
        dummy_out = sa4_mlp(dummy_in_sa4)
        layer = sa4_mlp.sharedMLP
        layer.set_weights(net.backbone_net.sa4.mlp_module.get_weights())
        tflite_convert('sa4', sa4_mlp, net, FLAGS.out_dir)


    if 'fp1' in converting_layers:
        fp1_mlp = SharedMLPModel(mlp_spec=[256+256,256,256], input_shape=[512,1,512])
        dummy_in_fp1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,512,1,512])) # (B, npoint, 1, C)
        dummy_out = fp1_mlp(dummy_in_fp1)
        layer = fp1_mlp.sharedMLP
        layer.set_weights(net.backbone_net.fp1.mlp.get_weights()) 
        tflite_convert('fp1', fp1_mlp, net, FLAGS.out_dir) 


    if 'fp2' in converting_layers:
        fp2_mlp = SharedMLPModel(mlp_spec=[256+256,256,256], input_shape=[1024,1,512])
        dummy_in_fp2 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,512])) # (B, npoint, 1, C)
        dummy_out = fp2_mlp(dummy_in_fp2)
        layer = fp2_mlp.sharedMLP
        layer.set_weights(net.backbone_net.fp2.mlp.get_weights())
        tflite_convert('fp2', fp2_mlp, net, FLAGS.out_dir)


    if 'voting' in converting_layers:
        voting = nnInVotingModule(vote_factor=1, seed_feature_dim=256)
        dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,256])) # (B, num_seed, 1, 3)
        dummy_out = voting(dummy_in_voting_features)
        layer = voting
        layer.conv1.set_weights(net.vgen.conv1.get_weights())
        layer.conv2.set_weights(net.vgen.conv2.get_weights())
        layer.conv3.set_weights(net.vgen.conv3.get_weights())
        layer.bn1.set_weights(net.vgen.bn1.get_weights())
        layer.bn2.set_weights(net.vgen.bn2.get_weights())
        tflite_convert('voting', voting, net, FLAGS.out_dir, mlp=False)


    if 'va' in converting_layers:
        va_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 128], nsample=16, input_shape=[256,16,256+3])
        dummy_in_va = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,16,256+3])) # (B, npoint, nsample, C+3)
        dummy_out = va_mlp(dummy_in_va)
        layer = va_mlp.sharedMLP
        layer.set_weights(net.pnet.vote_aggregation.mlp_module.get_weights())
        tflite_convert('va', va_mlp, net, FLAGS.out_dir)

    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)

    
    
    
    
    
    
    
    
