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

BATCH_SIZE = 1
TEST_DATASET = SunrgbdDetectionVotesDataset_tfrecord('val', num_points=20000,
    augment=False,  shuffle=False, batch_size=BATCH_SIZE,
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

# TFlite conversion
def tflite_convert(keyword, model, base_model, out_dir):
    # A generator that provides a representative dataset
    def representative_data_gen():
        for i in range(100):
            batch_data = next(iter(test_ds))
            inputs = batch_data[0]
            end_points = base_model(inputs, training=False)
            yield [end_points[keyword + '_grouped_features']]

        """
        dataset_list = tf.data.Dataset.list_files(flowers_dir + '/*/*')
        for i in range(100):
            image = next(iter(dataset_list))
            image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.cast(image / 255., tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]
        """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with open(os.path.join(out_dir, keyword + '_quant.tflite'), 'wb') as f:
        f.write(tflite_model)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--out_dir', default=None, help='Folder name where output tflite files are saved')
FLAGS = parser.parse_args()



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

    ############################## Above is base model inference ######################################
    ############################## Below is tflite conversion #########################################

    # Build Shard MLP parts of the pointnet backbone as a model
    class SharedMLPModel(tf.keras.Model):
        def __init__(self, mlp_spec, nsample):
            super().__init__()
            self.sharedMLP = tf_utils.SharedMLP(mlp_spec, bn=True)
            self.max_pool = layers.MaxPooling2D(pool_size=(1, nsample), strides=1, data_format="channels_last")

        def call(self, grouped_features):
            new_features = self.max_pool(self.sharedMLP(grouped_features))

            return grouped_features

    sa1_mlp = SharedMLPModel(mlp_spec=[1, 64, 64, 128], nsample=64)
    sa2_mlp = SharedMLPModel(mlp_spec=[128, 128, 128, 256], nsample=32)
    sa3_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16)
    sa4_mlp = SharedMLPModel(mlp_spec=[256, 128, 128, 256], nsample=16)
    dummy_in_sa1 = tf.convert_to_tensor(np.random.random([1,2048,64,4])) # (B, npoint, nsample, C+3)
    dummy_in_sa2 = tf.convert_to_tensor(np.random.random([1,1024,32,128+3])) # (B, npoint, nsample, C+3)
    dummy_in_sa3 = tf.convert_to_tensor(np.random.random([1,512,16,256+3])) # (B, npoint, nsample, C+3)
    dummy_in_sa4 = tf.convert_to_tensor(np.random.random([1,256,16,256+3])) # (B, npoint, nsample, C+3)
    dummy_out = sa1_mlp(dummy_in_sa1)
    dummy_out = sa2_mlp(dummy_in_sa2)
    dummy_out = sa3_mlp(dummy_in_sa3)
    dummy_out = sa4_mlp(dummy_in_sa4)
    
    
    # Copy weights from the base model
    layer = sa1_mlp.sharedMLP
    to_wght = layer.get_weights()
    for w in to_wght:
        print(w.shape)
    print("=" * 20)
    from_wght = net.backbone_net.sa1.mlp_module.get_weights()
    for w in from_wght:
        print(w.shape)
    print("=" * 20)
    layer.set_weights(net.backbone_net.sa1.mlp_module.get_weights())

    layer = sa2_mlp.sharedMLP
    layer.set_weights(net.backbone_net.sa2.mlp_module.get_weights())
    layer = sa3_mlp.sharedMLP
    layer.set_weights(net.backbone_net.sa3.mlp_module.get_weights())
    layer = sa4_mlp.sharedMLP
    layer.set_weights(net.backbone_net.sa4.mlp_module.get_weights())


    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)

    tflite_convert('sa1', sa1_mlp, net, FLAGS.out_dir)
    tflite_convert('sa2', sa2_mlp, net, FLAGS.out_dir)
    tflite_convert('sa3', sa3_mlp, net, FLAGS.out_dir)
    tflite_convert('sa4', sa4_mlp, net, FLAGS.out_dir)
