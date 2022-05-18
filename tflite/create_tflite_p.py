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
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
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
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--out_dir', default=None, help='Folder name where output tflite files are saved')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')
parser.add_argument('--use_rep_data', action='store_true', help='When iterating representative dataset, use saved data')
parser.add_argument('--rep_data_dir', default='tflite/tflite_rep_data', help='Saved representative data directory')
parser.add_argument('--config_path', default=None, required=True, help='Model configuration path')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--q_gran', default='semantic', help='Quantization granularity(Channelwise, Groupwise, Semanticwise). [default: semantic]')
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

use_painted = model_config['use_painted']

if 'dataset' in model_config:
    DATASET = model_config['dataset']
else:
    DATASET =  FLAGS.dataset

#Use separate layer for coordinates in voting and va layer
q_gran = 'semantic' if 'q_gran' not in model_config else model_config['q_gran']

if not FLAGS.use_rep_data:
    if DATASET == 'sunrgbd':
        if 'include_person' in model_config and model_config['include_person']:
            DATASET_CONFIG = SunrgbdDatasetConfig(include_person=True)
        else:
            DATASET_CONFIG = SunrgbdDatasetConfig()
        NUM_POINT = 20000
        TRAIN_DATASET = SunrgbdDetectionVotesDataset_tfrecord('train', num_points=NUM_POINT,
            augment=False, shuffle=True, batch_size=BATCH_SIZE,
            use_color=False, use_height=True,
            use_painted=use_painted, DC=DATASET_CONFIG)

        ds = TRAIN_DATASET.preprocess()
        ds = ds.prefetch(BATCH_SIZE)
    elif DATASET == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
        from model_util_scannet import ScannetDatasetConfig
        DATASET_CONFIG = ScannetDatasetConfig()
        NUM_POINT = 40000

        TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
                augment=True,
                use_color=False, use_height=True,
                use_painted=use_painted)

        # Init datasets and dataloaders 
        def my_worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        
        ds = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn, drop_last=True)

if not use_painted:
    num_input_channel = 1
#elif model_config['two_way']:
#    num_input_channel = 1 + DATASET_CONFIG.num_class + 1
else:
    num_input_channel = 1 + 1 + DATASET_CONFIG.num_class + 1

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, NUM_POINT)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

'''
def wrapper_representative_data_gen_mlp(keyword, base_model):
    def representative_data_gen_mlp():        
        for i in range(int(10 / BATCH_SIZE)):            
            if not FLAGS.use_rep_data:            
                batch_data = next(iter(ds))
                
                if DATASET == 'sunrgbd':
                    inputs = batch_data[0]
                else:
                    inputs = batch_data['point_clouds']
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
                    #yield [va_grouped_features, va_xyz]
                    yield [va_grouped_features]
                    
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
                
                if DATASET == 'sunrgbd':
                    inputs = batch_data[0]
                else:
                    inputs = batch_data['point_clouds']

                end_points = base_model(inputs, training=False)
                print(i, "-th batch...")
                #voting_input = [tf.expand_dims(end_points['seed_features'], axis=-2),
                #                tf.expand_dims(end_points['seed_xyz'], axis=-2)]
                #voting_input = [end_points['seed_features'], end_points['seed_xyz']]
                voting_input = [tf.expand_dims(end_points['seed_features'],axis=2)]

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
'''


def simulate_run(base_model, keyword_list):
    output_dict = {}

    for i in range(int(100 / BATCH_SIZE)):                               
        batch_data = next(iter(ds))
        
        if DATASET == 'sunrgbd':
            inputs = batch_data[0]
        else:
            inputs = batch_data['point_clouds']
        end_points = base_model(inputs, training=False)
        print("Saving inference results", i, "-th batch...")

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

        seed_features = end_points['seed_features']        

        feature_dict = {'sa1':sa1_grouped_features,
                'sa2':sa2_grouped_features,
                'sa3':sa3_grouped_features,
                'sa4':sa4_grouped_features,                
                'fp1':fp1_grouped_features,
                'fp2':fp2_grouped_features,
                'voting':tf.expand_dims(seed_features, axis=2),
                'va': va_grouped_features
                }
        for k in keyword_list:
            if i == 0:                            
                output_dict[k] = [feature_dict[k]]
            else:                
                output_dict[k].append(feature_dict[k])                
    return output_dict                             

def data_gen_wrapper(keyword, features_dict):
    def data_gen():
        for feat in features_dict[keyword]:
            yield [feat]

    return data_gen


# TFlite conversion for multi layers(models)
def tflite_convert_multi(keyword_list, model_list, base_model, out_dir, mlp=True):

    features_dict = simulate_run(base_model, keyword_list)

    for k, model in zip(keyword_list, model_list):
        print("=" * 30, f"Converting {k} layer", "=" * 30)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # This enables quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # This sets the representative dataset for quantization        
        converter.representative_dataset = data_gen_wrapper(k, features_dict)        
        # This ensures that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
        converter.target_spec.supported_types = [tf.int8]
        # These set the input and output tensors to uint8 (added in r2.3)
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()

        if q_gran != 'semantic':
            tflite_name = k + '_quant_%s.tflite'%(q_gran)
        else:
            tflite_name = k + '_quant.tflite'
        with open(os.path.join(out_dir, tflite_name), 'wb') as f:
            f.write(tflite_model)
    del features_dict

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

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DATASET_CONFIG}

    # Init the model and optimzier    
    net = votenet_tf.VoteNet(num_proposal=256, 
        input_feature_dim=DATASET_CONFIG.num_class+1 if use_painted else 1, 
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
   
    
    pc = tf.convert_to_tensor(np.random.random([BATCH_SIZE,NUM_POINT,3+num_input_channel]))    
   
    # Model inference
    inputs = {'point_clouds': pc}

    tic = time.time()
    end_points = net(inputs['point_clouds'], training=False)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))    

    ############################## Above is base model inference ######################################
    ############################## Below is tflite conversion #########################################

    act = model_config['activation'] if 'activation' in model_config['activation'] else 'relu6'
    if act == 'relu6':
        maxval = 6
    else:
        maxval = None

    # Build Shard MLP parts of the pointnet backbone as a model
    class SharedMLPModel(tf.keras.Model):
        def __init__(self, mlp_spec, input_shape, nsample=0):
            super().__init__()
            self.sharedMLP = tf_utils.SharedMLP(mlp_spec, bn=True, activation=act, input_shape=input_shape)
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
        def __init__(self, vote_factor, seed_feature_dim, q_gran='semantic'):            
            super().__init__()
            
            self.vote_factor = vote_factor
            self.in_dim = seed_feature_dim            
            self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim            
            self.use_fp_mlp = model_config['use_fp_mlp']
            self.conv0 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            #self.conv0 = layers.Dense(self.in_dim)
            #self.conv1 = layers.Dense(self.in_dim)
            #self.conv2 = layers.Dense(self.in_dim)
            self.q_gran = q_gran
            
            if self.q_gran=='channel':
                self.conv3_chn_list = []
                for i in range((3 + self.out_dim) * self.vote_factor):
                    self.conv3_chn_list.append(layers.Conv2D(filters=1, kernel_size=1))

            elif self.q_gran=='semantic':
                self.conv3_1 = layers.Conv2D(filters=(3) * self.vote_factor, kernel_size=1) 
                self.conv3_2 = layers.Conv2D(filters=(self.out_dim) * self.vote_factor, kernel_size=1) 
                #self.conv3_1 = layers.Dense(3 * self.vote_factor)
                #self.conv3_2 = layers.Dense(self.out_dim * self.vote_factor)

            elif self.q_gran=='group':
                self.conv3_chn_list = []
                grp1_size = ((3+self.out_dim) * self.vote_factor) // 2
                grp2_size = (3+self.out_dim) * self.vote_factor  - grp1_size
                self.conv3_chn_list.append(layers.Conv2D(filters=grp1_size, kernel_size=1))
                self.conv3_chn_list.append(layers.Conv2D(filters=grp2_size, kernel_size=1))

            else:
                self.conv3 = layers.Conv2D(filters=(self.out_dim+3) * self.vote_factor, kernel_size=1)
            
                #self.conv3 = layers.Dense((self.out_dim+3) * self.vote_factor)
            self.bn0 = layers.BatchNormalization(axis=-1)
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)

            self.relu0 = layers.ReLU(maxval)
            self.relu1 = layers.ReLU(maxval)
            self.relu2 = layers.ReLU(maxval)
        
        def call(self, voting_input):

            num_seed = 1024
            seed_features = voting_input            

            if not self.use_fp_mlp:
                net0 = self.relu0(self.bn0(self.conv0(seed_features)))
            else:
                net0 = seed_features
            net = self.relu1(self.bn1(self.conv1(net0))) 
            net = self.relu2(self.bn2(self.conv2(net))) 
            net0 = layers.Reshape((num_seed, self.vote_factor, net0.shape[-1]))(net0)
            
            if self.q_gran=='channel':
                out = []
                for i in range((3 + self.out_dim) * self.vote_factor):
                    out.append(self.conv3_chn_list[i](net))
                # offset = layers.Concatenate(axis=-1)(out[:3])
                # residual_features = layers.Concatenate(axis=-1)(out[3:])
                # vote_features = net0 + residual_features 
                # return [offset, vote_features]
                out.append(net0)
                return out

            
            elif self.q_gran=='semantic':
                offset = self.conv3_1(net)
                net = self.conv3_2(net) # (batch_size, num_seed, (3+out_dim)*vote_factor)
                residual_features = layers.Reshape((num_seed, self.vote_factor, self.out_dim))(net)                
                vote_features = net0 + residual_features 
                #vote_xyz = xyz + offset 
                #return [vote_xyz, vote_features]
                return [offset, vote_features]

            elif self.q_gran=='group':
                out = []
                for i in range(2):
                    out.append(self.conv3_chn_list[i](net))
                out.append(net0)
                return out
            
            else:
                net = self.conv3(net)
                # offset = net[:,:,:,0:3]
                # net = net[:,:,:,3:]                
                
                # residual_features = layers.Reshape((num_seed, self.vote_factor, self.out_dim))(net)                
                # vote_features = net0 + residual_features 
                # return [offset, vote_features]
                return [net, net0]

    class vaModule(tf.keras.Model):
        def __init__(self, mlp_spec, input_shape, nsample=0, q_gran='semantic'):
            super().__init__()
            self.sharedMLP = tf_utils.SharedMLP(mlp_spec, bn=True, input_shape=input_shape)
            self.npoint = 256
            self.nsample = nsample            
            self.q_gran = q_gran
            self.max_pool = layers.MaxPooling2D(pool_size=(1, 16), strides=(1,16), data_format="channels_last")            

            self.conv1 = layers.Conv2D(filters=128, kernel_size=1)        
            self.conv2 = layers.Conv2D(filters=128, kernel_size=1)
            #self.conv1 = layers.Dense(128)
            #self.conv2 = layers.Dense(128)
            self.NH = DATASET_CONFIG.num_heading_bin
            self.NC = DATASET_CONFIG.num_size_cluster
            self.num_class = DATASET_CONFIG.num_class
                  
            if self.q_gran=='channel':
                self.conv3_chn_list = []
                for i in range(2+3+self.NH*2+self.NC*4+self.num_class):
                    self.conv3_chn_list.append(layers.Conv2D(filters=1, kernel_size=1))

            elif self.q_gran=='semantic':
                self.conv3_1 = layers.Conv2D(filters=3, kernel_size=1) 
                self.conv3_2 = layers.Conv2D(filters=2 + self.NH + self.NC  + self.num_class, kernel_size=1) 
                self.conv3_3 = layers.Conv2D(filters=self.NH + self.NC * 3, kernel_size=1)                
                #self.conv3_2 = layers.Conv2D(filters=2+DATASET_CONFIG.num_heading_bin*2+DATASET_CONFIG.num_size_cluster*4+DATASET_CONFIG.num_class, kernel_size=1) 
                #self.conv3_1 = layers.Dense(3)
                #self.conv3_2 = layers.Dense(2 + DATASET_CONFIG.num_heading_bin*2 + DATASET_CONFIG.num_size_cluster*4 + DATASET_CONFIG.num_class)

            elif self.q_gran=='group':
                self.conv3_chn_list = []
                grp1_size = (2+3+self.NH*2+self.NC*4+self.num_class) // 3
                grp2_size = grp1_size
                grp3_size = 2+3+self.NH*2+self.NC*4+self.num_class - grp1_size - grp2_size                
                self.conv3_chn_list.append(layers.Conv2D(filters=grp1_size, kernel_size=1))
                self.conv3_chn_list.append(layers.Conv2D(filters=grp2_size, kernel_size=1))
                self.conv3_chn_list.append(layers.Conv2D(filters=grp3_size, kernel_size=1))

            else:
                self.conv3 = layers.Conv2D(filters=2+3+self.NH*2+self.NC*4+self.num_class, kernel_size=1)
                #self.conv3 = layers.Dense(3 + 2 + DATASET_CONFIG.num_heading_bin*2 + DATASET_CONFIG.num_size_cluster*4 + DATASET_CONFIG.num_class)
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)
            self.relu1 = layers.ReLU(maxval)
            self.relu2 = layers.ReLU(maxval)
            
        def call(self, va_input):            
            
            grouped_features = va_input            

            new_features = self.max_pool(self.sharedMLP(grouped_features))     
            
            #For Dense Layer
            #new_features = layers.Reshape((self.npoint, new_features.shape[-1]))(new_features)

            # --------- PROPOSAL GENERATION ---------
            net = self.relu1(self.bn1(self.conv1(new_features)))
            net = self.relu2(self.bn2(self.conv2(net))) 

            if self.q_gran=='channel':
                out = []
                for i in range(2+3+self.NH*2+self.NC*4+self.num_class):
                    out.append(self.conv3_chn_list[i](net))
                #offset = layers.Concatenate(axis=-1)(out[:3])

                #net2 = layers.Concatenate(axis=-1)(out[3:3+2+self.NH+self.NC] + out[-self.num_class:])
                #net3 = layers.Concatenate(axis=-1)(out[3+2+self.NH+self.NC:3+2+self.NH*2+self.NC*4])

                #return [offset, net2, net3]            
                return out

            elif self.q_gran=='semantic':
                offset = self.conv3_1(net)                                
                offset = layers.Reshape((self.npoint,3))(offset)
               
                net2 = self.conv3_2(net)
                net3 = self.conv3_3(net)

                net2 = layers.Reshape((self.npoint, net2.shape[-1]))(net2)
                net3 = layers.Reshape((self.npoint, net3.shape[-1]))(net3)                

                return [offset, net2, net3] 

            elif self.q_gran=='group':                
                out = []
                for i in range(3):
                    out.append(self.conv3_chn_list[i](net))
                return out

            else:
                net = self.conv3(net)
                net = layers.Reshape((self.npoint, net.shape[-1]))(net)
                return net

    converting_layers = ['sa1','sa2','sa3','sa4','fp1','fp2','voting','va']
    #converting_layers = ['voting','va']    
    model_list = []

    if 'sa1' in converting_layers:    
        if not use_painted:
            sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=[2048,64,1+3])
            dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,2048,64,1+3])) # (B, npoint, nsample, C+3)
        elif model_config['two_way']:
            input_shape=[1024,64,3+num_input_channel-1] # xyz + height + (num_class + background) (isPainted removed)
            sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=input_shape)
            dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE] + input_shape)) # (B, npoint, nsample, C+3)
        else:
            input_shape=[2048,64,3+num_input_channel] # xyz + height + (num_class + background) + isPainted
            sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64, input_shape=input_shape)
            dummy_in_sa1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE] + input_shape)) # (B, npoint, nsample, C+3)
        dummy_out = sa1_mlp(dummy_in_sa1)
        # Copy weights from the base model    
        layer = sa1_mlp.sharedMLP
        if model_config['two_way']:
            layer.set_weights(net.backbone_net.sa1_mlp.mlp_module.get_weights()) 
        else:
            layer.set_weights(net.backbone_net.sa1.mlp_module.get_weights()) 
        print("=" * 30, "Converting SA1 layer", "=" * 30)
        #tflite_convert('sa1', sa1_mlp, net, OUT_DIR)
        model_list.append(sa1_mlp)

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
        #tflite_convert('sa2', sa2_mlp, net, OUT_DIR)
        model_list.append(sa2_mlp)

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
        #tflite_convert('sa3', sa3_mlp, net, OUT_DIR)
        model_list.append(sa3_mlp)

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
        #tflite_convert('sa4', sa4_mlp, net, OUT_DIR)
        model_list.append(sa4_mlp)

    if 'fp1' in converting_layers and model_config['use_fp_mlp']:
        fp1_mlp = SharedMLPModel(mlp_spec=[256+256,256,256], input_shape=[512,1,512])
        dummy_in_fp1 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,512,1,512])) # (B, npoint, nsample, C+3)
        dummy_out = fp1_mlp(dummy_in_fp1)
        layer = fp1_mlp.sharedMLP
        layer.set_weights(net.backbone_net.fp1.mlp.get_weights())
        print("=" * 30, "Converting FP1 layer", "=" * 30)
        #tflite_convert('fp1', fp1_mlp, net, OUT_DIR)
        model_list.append(fp1_mlp)
    
    if 'fp2' in converting_layers and model_config['use_fp_mlp']:
        fp2_mlp = SharedMLPModel(mlp_spec=[256+256,256,256], input_shape=[1024,1,512])
        dummy_in_fp2 = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,512])) # (B, npoint, nsample, C+3)
        dummy_out = fp2_mlp(dummy_in_fp2)
        layer = fp2_mlp.sharedMLP
        layer.set_weights(net.backbone_net.fp2.mlp.get_weights())
        print("=" * 30, "Converting FP2 layer", "=" * 30)
        #tflite_convert('fp2', fp2_mlp, net, OUT_DIR)
        model_list.append(fp2_mlp)

    if 'voting' in converting_layers:
        voting = nnInVotingModule(vote_factor=1, seed_feature_dim=256, q_gran=q_gran)        
        if model_config['use_fp_mlp']:
            dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,256])) # (B, num_seed, 1, 256*3)
            #dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,256]))
        else: 
            dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,256*3])) # (B, num_seed, 1, 256*3)
            #dummy_in_voting_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,256*3]))
        dummy_in_voting_xyz =  tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,1,3])) # (B, num_seed, 1, 3)        
        #dummy_in_voting_xyz =  tf.convert_to_tensor(np.random.random([BATCH_SIZE,1024,3])) # (B, num_seed, 1, 3)        
        #dummy_in_voting = [dummy_in_voting_features, dummy_in_voting_xyz]        
        dummy_out = voting(dummy_in_voting_features)
        layer = voting
        layer.conv0.set_weights(net.vgen.conv0.get_weights())
        layer.conv1.set_weights(net.vgen.conv1.get_weights())
        layer.conv2.set_weights(net.vgen.conv2.get_weights())

        if q_gran=='channel':
            w, b = net.vgen.conv3.get_weights()
            for i in range(w.shape[-1]):                    
                layer.conv3_chn_list[i].set_weights([w[:,:,:,i:i+1], b[i:i+1]])
                             

        elif q_gran=='semantic':
            w, b = net.vgen.conv3.get_weights()
            layer.conv3_1.set_weights([w[:,:,:,:3], b[:3]])
            layer.conv3_2.set_weights([w[:,:,:,3:], b[3:]]) 

        elif q_gran=='group':
            w, b = net.vgen.conv3.get_weights()                        

            grp1_size = ((3+layer.out_dim) * layer.vote_factor) // 2
            grp2_size = (3+layer.out_dim) * layer.vote_factor  - grp1_size
            
            layer.conv3_chn_list[0].set_weights([w[:,:,:,:grp1_size], b[:grp1_size]])                   
            layer.conv3_chn_list[1].set_weights([w[:,:,:,grp1_size:], b[grp1_size:]])                               

        else:
            layer.conv3.set_weights(net.vgen.conv3.get_weights())

        layer.bn0.set_weights(net.vgen.bn0.get_weights())
        layer.bn1.set_weights(net.vgen.bn1.get_weights())
        layer.bn2.set_weights(net.vgen.bn2.get_weights())
        print("=" * 30, "Converting Voting layer", "=" * 30)
        #tflite_convert('voting', voting, net, OUT_DIR, mlp=False)
        model_list.append(voting)


    if 'va' in converting_layers:
        va_mlp = vaModule(mlp_spec=[128, 128, 128, 128], nsample=16, input_shape=[256,16,256+3], q_gran=q_gran)
        #dummy_in_va = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,3 + (16*(128+3))]), dtype=tf.float32) # (B, npoint, nsample, C+3)
        dummy_va_features = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,16,(256+3)]), dtype=tf.float32) # (B, npoint, 3 + nsample*(C+3)) 
        #dummy_va_xyz = tf.convert_to_tensor(np.random.random([BATCH_SIZE,256,3]), dtype=tf.float32) # (B, npoint, 3 + nsample*(C+3)) 
        #dummy_in_va = [dummy_va_features, dummy_va_xyz]
        dummy_out = va_mlp(dummy_va_features)
        layer = va_mlp.sharedMLP
        layer.set_weights(net.pnet.mlp_module.get_weights())

        layer = va_mlp
        layer.conv1.set_weights(net.pnet.conv1.get_weights())
        layer.conv2.set_weights(net.pnet.conv2.get_weights())
        
        NH = DATASET_CONFIG.num_heading_bin
        NC = DATASET_CONFIG.num_size_cluster
        num_class = DATASET_CONFIG.num_class

        if q_gran=='channel':
            w, b = net.pnet.conv3.get_weights()
            for i in range(w.shape[-1]):                    
                layer.conv3_chn_list[i].set_weights([w[:,:,:,i:i+1], b[i:i+1]])

        elif q_gran=='semantic':
            w, b = net.pnet.conv3.get_weights()
            layer.conv3_1.set_weights([w[:,:,:,:3], b[:3]])
            #layer.conv3_2.set_weights([w[:,:,:,3:], b[3:]])
            w_2 = np.concatenate([w[:,:,:,3:3+2+NH+NC], w[:,:,:,3+2+NH*2+NC*4:]], axis=-1)
            b_2 = np.concatenate([b[3:3+2+NH+NC], b[3+2+NH*2+NC*4:]], axis=-1)
            layer.conv3_2.set_weights([w_2, b_2])
            w_3 = w[:,:,:,3+2+NH+NC:3+2+NH*2+NC*4]
            b_3 = b[3+2+NH+NC:3+2+NH*2+NC*4]
            layer.conv3_3.set_weights([w_3, b_3])
            
            #layer.conv3_1.set_weights([w[:,:3], b[:3]])
            #layer.conv3_2.set_weights([w[:,3:], b[3:]])     
        
        elif q_gran=='group':
            w, b = net.pnet.conv3.get_weights()

            grp1_size = (2+3+layer.NH*2+layer.NC*4+layer.num_class) // 3
            grp2_size = grp1_size
            grp3_size = 2+3+layer.NH*2+layer.NC*4+layer.num_class - grp1_size - grp2_size                
            
            layer.conv3_chn_list[0].set_weights([w[:,:,:,:grp1_size], b[:grp1_size]])                   
            layer.conv3_chn_list[1].set_weights([w[:,:,:,grp1_size:grp1_size+grp2_size], b[grp1_size:grp1_size+grp2_size]])                   
            layer.conv3_chn_list[2].set_weights([w[:,:,:,-grp3_size:], b[-grp3_size:]])     

        else:
            layer.conv3.set_weights(net.pnet.conv3.get_weights())
        layer.bn1.set_weights(net.pnet.bn1.get_weights())
        layer.bn2.set_weights(net.pnet.bn2.get_weights())
        
        print("=" * 30, "Converting VA layer", "=" * 30)
        #tflite_convert('va', va_mlp, net, OUT_DIR)
        model_list.append(va_mlp)
        
    
    if len(converting_layers) > 4:
        tflite_convert_multi(converting_layers[:4], model_list[:4], net, OUT_DIR)
        tflite_convert_multi(converting_layers[4:], model_list[4:], net, OUT_DIR)
    else:
        tflite_convert_multi(converting_layers, model_list, net, OUT_DIR)