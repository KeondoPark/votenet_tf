import os
import sys
import numpy as np
import time
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
#from sunrgbd_detection_dataset_tf import DC # dataset config
import voting_module_tf
import json

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--out_dir', default=None, help='Folder name where output tflite files are saved')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')
parser.add_argument('--use_rep_data', action='store_true', help='When iterating representative dataset, use saved data')
parser.add_argument('--rep_data_dir', default='tflite/tflite_rep_data', help='Saved representative data directory')
parser.add_argument('--config_path', default=None, required=True, help='Model configuration path')
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

model_config = json.load(open(FLAGS.config_path))

if not FLAGS.use_rep_data:
    if 'include_person' in model_config and model_config['include_person']:
        DATASET_CONFIG = SunrgbdDatasetConfig(include_person=True)
    else:
        DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset_tfrecord('train', num_points=20000,
        augment=False, shuffle=True, batch_size=BATCH_SIZE,
        use_color=False, use_height=True,
        use_painted=model_config['use_painted'], DC=DATASET_CONFIG)

    ds = TRAIN_DATASET.preprocess()
    ds = ds.prefetch(BATCH_SIZE)


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
                batch_data = next(iter(ds))
                inputs = batch_data[0]
                end_points = base_model(inputs, training=False)
                print("Using inference results", i, "-th batch...")

                rnd = np.random.rand(1)
                if 'sa1_grouped_features1' in end_points:
                    sa1_grouped_features = end_points['sa1_grouped_features1'] if rnd > 0.5  else end_points['sa1_grouped_features2']
                    sa2_grouped_features = end_points['sa2_grouped_features1'] if rnd > 0.5  else end_points['sa2_grouped_features2']
                    sa3_grouped_features = end_points['sa3_grouped_features1'] if rnd > 0.5  else end_points['sa3_grouped_features2']
                    sa4_grouped_features = end_points['sa4_grouped_features1'] if rnd > 0.5  else end_points['sa4_grouped_features2']
                else:
                    sa1_grouped_features = end_points['sa1_grouped_features']
                    sa2_grouped_features = end_points['sa2_grouped_features']
                    sa3_grouped_features = end_points['sa3_grouped_features']
                    sa4_grouped_features = end_points['sa4_grouped_features']

                fp1_grouped_features = end_points['fp1_grouped_features']
                fp2_grouped_features = end_points['fp2_grouped_features']
                va_grouped_features = end_points['va_grouped_features']

                feature_dict = {'sa1':sa1_grouped_features,
                        'sa2':sa2_grouped_features,
                        'sa3':sa3_grouped_features,
                        'sa4':sa4_grouped_features,
                        'fp1':fp1_grouped_features,
                        'fp2':fp2_grouped_features
                        }
                if keyword == 'va':                    
                    va_xyz = end_points['aggregated_vote_xyz']
                    #va_features = layers.Reshape((256,-1))(va_grouped_features)
                    #va_input = layers.Concatenate(axis=-1)([va_xyz, va_features])
                    yield [va_grouped_features, va_xyz]
                    
                    #yield [va_grouped_features]
                else:
                    yield [feature_dict[keyword]]
            else:
                if (i * BATCH_SIZE) % 200 == 0:
                    start = i * BATCH_SIZE
                    end = start + 200                            
                    np_feats = np.load(os.path.join(FLAGS.rep_data_dir, keyword + '_rep_' + str(start) + '_to_' + str(end) + '.npy'))
                    feats = tf.convert_to_tensor(np_feats)
                print("Using saved rep data", i, "-th batch...")
                idx = (i * BATCH_SIZE) % 200
                yield [feats[idx: idx+BATCH_SIZE,:,:,:]]
    return representative_data_gen_mlp

def wrapper_representative_data_gen_voting(base_model):
    def representative_data_gen_voting():
        for i in range(int(800 / BATCH_SIZE)):
            if not FLAGS.use_rep_data:
                batch_data = next(iter(ds))
                inputs = batch_data[0]
                end_points = base_model(inputs, training=False)
                print(i, "-th batch...")
                voting_input = [tf.expand_dims(end_points['seed_features'], axis=-2),
                                tf.expand_dims(end_points['seed_xyz'], axis=-2)]

                yield voting_input
                #yield [tf.expand_dims(seed_features, axis=-2)]
            else:
                if (i * BATCH_SIZE) % 200 == 0:
                    start = i * BATCH_SIZE
                    end = start + 200                            
                    np_feats = np.load(os.path.join(FLAGS.rep_data_dir, 'voting_rep_' + str(start) + '_to_' + str(end) + '.npy'))
                    feats = tf.convert_to_tensor(np_feats)
                    feats = tf.reshape(feats, (feats.shape[0], feats.shape[1], 1, feats.shape[2]))
                print("Using saved rep data", i, "-th batch...")
                idx = (i * BATCH_SIZE) % 200
                
                yield [feats[idx: idx+BATCH_SIZE,:,:,:]]
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
    if FLAGS.checkpoint_path is None:
        checkpoint_path = os.path.join(BASE_DIR, 'tf_ckpt', model_config['model_id'])
    else:
        checkpoint_path = FLAGS.checkpoint_path

    if FLAGS.out_dir is None:
        OUT_DIR = os.path.join(BASE_DIR, model_config['tflite_folder'])

        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)    
    else:
        OUT_DIR = FLAGS.out_dir

    #Use separate layer for coordinates in voting and va layer
    sep_coords = model_config['sep_coords']

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DATASET_CONFIG}

    # Init the model and optimzier    
    net = votenet_tf.VoteNet(num_proposal=256, 
        input_feature_dim=DATASET_CONFIG.num_class+1 if model_config['use_painted'] else 1, 
        vote_factor=1,
        #sampling='seed_fps', num_class=DATASET_CONFIG.num_class,
        sampling='vote_fps', num_class=DATASET_CONFIG.num_class,
        num_heading_bin=DATASET_CONFIG.num_heading_bin,
        num_size_cluster=DATASET_CONFIG.num_size_cluster,
        mean_size_arr=DATASET_CONFIG.mean_size_arr,
        model_config=model_config)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    epoch = ckpt.epoch.numpy()

    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud
    if not model_config['use_painted']:
        pc = tf.convert_to_tensor(np.random.random([BATCH_SIZE,20000,3+1]))
    elif model_config['two_way']:
        pc = tf.convert_to_tensor(np.random.random([BATCH_SIZE,20000,3+1+DATASET_CONFIG.num_class+1+1]))
    else:
        pc = tf.convert_to_tensor(np.random.random([BATCH_SIZE,20000,3+1+DATASET_CONFIG.num_class+1]))
   
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
            self.npoint = input_shape[0]
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

            new_features = layers.Reshape((self.npoint, new_features.shape[-1]))(new_features)

            return new_features

    class nnInVotingModule(tf.keras.Model):
        def __init__(self, vote_factor, seed_feature_dim, sep_coords=sep_coords):            
            super().__init__()
            
            self.vote_factor = vote_factor
            self.in_dim = seed_feature_dim            
            self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
            self.sep_coords = sep_coords
            self.conv0 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1)        
            self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            if self.sep_coords:
                self.conv3_1 = layers.Conv2D(filters=(3) * self.vote_factor, kernel_size=1) 
                self.conv3_2 = layers.Conv2D(filters=(self.out_dim) * self.vote_factor, kernel_size=1) 
            else:
                self.conv3 = layers.Conv2D(filters=(self.out_dim+3) * self.vote_factor, kernel_size=1)
            self.bn0 = layers.BatchNormalization(axis=-1)
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)
            self.relu0 = layers.ReLU(6)
            self.relu1 = layers.ReLU(6)
            self.relu2 = layers.ReLU(6)
        
        def call(self, voting_input):

            num_seed = 1024
            seed_features = voting_input[0]
            xyz = voting_input[1]           

            net0 = self.relu0(self.bn0(self.conv0(seed_features)))
            net = self.relu1(self.bn1(self.conv1(net0))) 
            net = self.relu2(self.bn2(self.conv2(net))) 
            net0 = layers.Reshape((num_seed, self.vote_factor, net0.shape[-1]))(net0)
            if self.sep_coords:
                offset = self.conv3_1(net)
                net = self.conv3_2(net) # (batch_size, num_seed, (3+out_dim)*vote_factor)
                residual_features = layers.Reshape((num_seed, self.vote_factor, self.out_dim))(net)                
                vote_features = net0 + residual_features 
                #vote_xyz = xyz + offset 
                #return [vote_xyz, vote_features]
                return [offset, vote_features]
            else:
                net = self.conv3(net)
                offset = net[:,:,:,0:3]
                net = net[:,:,:,3:]
                residual_features = layers.Reshape((num_seed, self.vote_factor, self.out_dim))(net)                
                vote_features = net0 + residual_features 
                return [offset, vote_features]

    class vaModule(tf.keras.Model):
        def __init__(self, mlp_spec, input_shape, nsample=0, sep_coords=sep_coords):
            super().__init__()
            self.sharedMLP = tf_utils.SharedMLP(mlp_spec, bn=True, input_shape=input_shape)
            self.npoint = 256
            self.nsample = nsample            
            self.sep_coords = sep_coords
            self.max_pool = layers.MaxPooling2D(pool_size=(1, 16), strides=(1,16), data_format="channels_last")            

            self.conv1 = layers.Conv2D(filters=128, kernel_size=1)        
            self.conv2 = layers.Conv2D(filters=128, kernel_size=1)
            if self.sep_coords:
                self.conv3_1 = layers.Conv2D(filters=3, kernel_size=1) 
                self.conv3_2 = layers.Conv2D(filters=2+DATASET_CONFIG.num_heading_bin*2+DATASET_CONFIG.num_size_cluster*4+DATASET_CONFIG.num_class, kernel_size=1) 
            else:
                self.conv3 = layers.Conv2D(filters=2+3+DATASET_CONFIG.num_heading_bin*2+DATASET_CONFIG.num_size_cluster*4+DATASET_CONFIG.num_class, kernel_size=1)
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)
            self.relu1 = layers.ReLU(6)
            self.relu2 = layers.ReLU(6)
            
        def call(self, va_input):            
            
            grouped_features = va_input[0]
            xyz = va_input[1]

            new_features = self.max_pool(self.sharedMLP(grouped_features))     
            # --------- PROPOSAL GENERATION ---------
            net = self.relu1(self.bn1(self.conv1(new_features)))
            net = self.relu2(self.bn2(self.conv2(net))) 
            if self.sep_coords:
                offset = self.conv3_1(net)                                
                net = self.conv3_2(net) # (batch_size, num_proposal, 2+3+num_heading_bin*2+num_size_cluster*4)
                offset = layers.Reshape((self.npoint,3))(offset)
                #center = xyz + offset
                net = layers.Reshape((self.npoint, net.shape[-1]))(net)   

                return [offset, net]            
            else:
                net = self.conv3(net)
                return net

    converting_layers = ['sa1','sa2','sa3','sa4','fp1','fp2','voting','va']
    #converting_layers = ['voting','va']
    if 'sa1' in converting_layers:    
        if not model_config['use_painted']:
            sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=[2048,64,1+3])
            dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,2048,64,1+3])) # (B, npoint, nsample, C+3)
        elif model_config['two_way']:
            sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=[1024,64,3+1+DATASET_CONFIG.num_class+1])
            dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,64,3+1+DATASET_CONFIG.num_class+1])) # (B, npoint, nsample, C+3)
        else:
            sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=[2048,64,3+1+DATASET_CONFIG.num_class+1+1])
            dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,2048,64,1+10+3+1])) # (B, npoint, nsample, C+3)
        dummy_out = sa1_mlp(dummy_in_sa1)
        # Copy weights from the base model    
        layer = sa1_mlp.sharedMLP
        if model_config['two_way']:
            layer.set_weights(net.backbone_net.sa1_mlp.mlp_module.get_weights()) 
        else:
            layer.set_weights(net.backbone_net.sa1.mlp_module.get_weights()) 
        print("=" * 30, "Converting SA1 layer", "=" * 30)
        tflite_convert('sa1', sa1_mlp, net, OUT_DIR)

    if 'sa2' in converting_layers:
        if model_config['two_way']:
            sa2_mlp = SharedMLPModel(mlp_spec=[128, 128, 128, 256], nsample=32, input_shape=[512,32,128+3])
            dummy_in_sa2 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,512,32,128+3])) # (B, npoint, nsample, C+3)
        else:
            sa2_mlp = SharedMLPModel(mlp_spec=[128, 128, 128, 256], nsample=32, input_shape=[1024,32,128+3])
            dummy_in_sa2 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,32,128+3])) # (B, npoint, nsample, C+3)
        dummy_out = sa2_mlp(dummy_in_sa2)
        layer = sa2_mlp.sharedMLP
        if model_config['two_way']:
            layer.set_weights(net.backbone_net.sa2_mlp.mlp_module.get_weights()) 
        else:
            layer.set_weights(net.backbone_net.sa2.mlp_module.get_weights()) 
        print("=" * 30, "Converting SA2 layer", "=" * 30)
        tflite_convert('sa2', sa2_mlp, net, OUT_DIR)

    if 'sa3' in converting_layers:
        if model_config['two_way']:
            sa3_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16, input_shape=[256,16,256+3])
            dummy_in_sa3 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,16,256+3])) # (B, npoint, nsample, C+3)
        else:
            sa3_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16, input_shape=[512,16,256+3])
            dummy_in_sa3 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,512,16,256+3])) # (B, npoint, nsample, C+3)
        dummy_out = sa3_mlp(dummy_in_sa3)
        layer = sa3_mlp.sharedMLP
        if model_config['two_way']:
            layer.set_weights(net.backbone_net.sa3_mlp.mlp_module.get_weights()) 
        else:
            layer.set_weights(net.backbone_net.sa3.mlp_module.get_weights()) 
        print("=" * 30, "Converting SA3 layer", "=" * 30)
        tflite_convert('sa3', sa3_mlp, net, OUT_DIR)

    if 'sa4' in converting_layers:
        if model_config['two_way']:
            sa4_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16, input_shape=[128,16,256+3])
            dummy_in_sa4 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,128,16,256+3])) # (B, npoint, nsample, C+3)
        else:
            sa4_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16, input_shape=[256,16,256+3])
            dummy_in_sa4 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,16,256+3])) # (B, npoint, nsample, C+3)

        dummy_out = sa4_mlp(dummy_in_sa4)
        layer = sa4_mlp.sharedMLP
        if model_config['two_way']:
            layer.set_weights(net.backbone_net.sa4_mlp.mlp_module.get_weights()) 
        else:
            layer.set_weights(net.backbone_net.sa4.mlp_module.get_weights()) 
        print("=" * 30, "Converting SA4 layer", "=" * 30)
        tflite_convert('sa4', sa4_mlp, net, OUT_DIR)

    if 'fp1' in converting_layers and model_config['use_fp_mlp']:
        fp1_mlp = SharedMLPModel(mlp_spec=[256+256,256,256], input_shape=[512,1,512])
        dummy_in_fp1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,512,1,512])) # (B, npoint, nsample, C+3)
        dummy_out = fp1_mlp(dummy_in_fp1)
        layer = fp1_mlp.sharedMLP
        layer.set_weights(net.backbone_net.fp1.mlp.get_weights())
        print("=" * 30, "Converting FP1 layer", "=" * 30)
        tflite_convert('fp1', fp1_mlp, net, OUT_DIR)
    
    if 'fp2' in converting_layers and model_config['use_fp_mlp']:
        fp2_mlp = SharedMLPModel(mlp_spec=[256+256,256,256], input_shape=[1024,1,512])
        dummy_in_fp2 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,512])) # (B, npoint, nsample, C+3)
        dummy_out = fp2_mlp(dummy_in_fp2)
        layer = fp2_mlp.sharedMLP
        layer.set_weights(net.backbone_net.fp2.mlp.get_weights())
        print("=" * 30, "Converting FP2 layer", "=" * 30)
        tflite_convert('fp2', fp2_mlp, net, OUT_DIR)

    if 'voting' in converting_layers:
        voting = nnInVotingModule(vote_factor=1, seed_feature_dim=256, sep_coords=sep_coords)        
        if model_config['use_fp_mlp']:
            dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,256])) # (B, num_seed, 1, 256*3)
        else: 
            dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,256*3])) # (B, num_seed, 1, 256*3)
        dummy_in_voting_xyz =  tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,3])) # (B, num_seed, 1, 3)        
        dummy_in_voting = [dummy_in_voting_features, dummy_in_voting_xyz]        
        dummy_out = voting(dummy_in_voting)
        layer = voting
        layer.conv0.set_weights(net.vgen.conv0.get_weights())
        layer.conv1.set_weights(net.vgen.conv1.get_weights())
        layer.conv2.set_weights(net.vgen.conv2.get_weights())
        if sep_coords:
            w, b = net.vgen.conv3.get_weights()
            layer.conv3_1.set_weights([w[:,:,:,:3], b[:3]])
            layer.conv3_2.set_weights([w[:,:,:,3:], b[3:]])                       
        else:
            layer.conv3.set_weights(net.vgen.conv3.get_weights())

        layer.bn0.set_weights(net.vgen.bn0.get_weights())
        layer.bn1.set_weights(net.vgen.bn1.get_weights())
        layer.bn2.set_weights(net.vgen.bn2.get_weights())
        print("=" * 30, "Converting Voting layer", "=" * 30)
        tflite_convert('voting', voting, net, OUT_DIR, mlp=False)


    if 'va' in converting_layers:
        va_mlp = vaModule(mlp_spec=[128, 128, 128, 128], nsample=16, input_shape=[256,16,256+3], sep_coords=sep_coords)
        #dummy_in_va = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,3 + (16*(128+3))]), dtype=tf.float32) # (B, npoint, nsample, C+3)                
        dummy_va_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,16,(256+3)]), dtype=tf.float32) # (B, npoint, 3 + nsample*(C+3)) 
        dummy_va_xyz = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,3]), dtype=tf.float32) # (B, npoint, 3 + nsample*(C+3)) 
        dummy_in_va = [dummy_va_features, dummy_va_xyz]
        dummy_out = va_mlp(dummy_in_va)
        layer = va_mlp.sharedMLP
        layer.set_weights(net.pnet.mlp_module.get_weights())

        layer = va_mlp
        layer.conv1.set_weights(net.pnet.conv1.get_weights())
        layer.conv2.set_weights(net.pnet.conv2.get_weights())
        if sep_coords:
            w, b = net.pnet.conv3.get_weights()
            layer.conv3_1.set_weights([w[:,:,:,:3], b[:3]])
            layer.conv3_2.set_weights([w[:,:,:,3:], b[3:]])
        
        else:
            layer.conv3.set_weights(net.pnet.conv3.get_weights())
        layer.bn1.set_weights(net.pnet.bn1.get_weights())
        layer.bn2.set_weights(net.pnet.bn2.get_weights())
        
        print("=" * 30, "Converting VA layer", "=" * 30)
        tflite_convert('va', va_mlp, net, OUT_DIR)
        