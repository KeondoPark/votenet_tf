# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
import time

import tensorflow as tf
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules_tf import PointnetSAModuleVotes, PointnetFPModule, SamplingAndGrouping, PointnetMLP

class Pointnet2Backbone(layers.Layer):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, use_tflite=False):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa1_quant.tflite'
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa2_quant.tflite'
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa3_quant.tflite'
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa4_quant.tflite'
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256], m=512)
                #use_tflite=use_tflite, tflite_name='fp1_quant.tflite')
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256], m=1024)
                #use_tflite=use_tflite, tflite_name='fp2_quant_b8.tflite')

    def _break_up_pc(self, pc):
        #xyz = pc[..., 0:3]
        #features = pc[..., 3:] if pc.shape[-1] > 3 else None        

        xyz = pc[:,:,0:3]
        features =  pc[:,:, 3:]

        return xyz, features

    def call(self, pointcloud, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """        
        if not end_points: end_points = {}
        xyz, features = self._break_up_pc(pointcloud)
        print(features.shape)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        #xyz, features, fps_inds = self.sa1(xyz, features)
        #print("========================== SA1 ===============================")
        sa1_xyz, sa1_features, sa1_inds, sa1_ball_query_idx, sa1_grouped_features = self.sa1(xyz, features, sample_type='fps')
        end_points['sa1_xyz'] = sa1_xyz
        end_points['sa1_features'] = sa1_features
        end_points['sa1_inds'] = sa1_inds        
        #end_points['sa1_ball_query_idx'] = ball_query_idx
        end_points['sa1_grouped_features'] = sa1_grouped_features

        #print("========================== SA2 ===============================")
        #xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        sa2_xyz, sa2_features, sa2_inds, sa2_ball_query_idx, sa2_grouped_features = self.sa2(sa1_xyz, sa1_features, sample_type='fps') # this fps_inds is just 0,1,...,1023
        end_points['sa2_xyz'] = sa2_xyz
        end_points['sa2_features'] = sa2_features
        end_points['sa2_inds'] = sa2_inds        
        #end_points['sa2_ball_query_idx'] = ball_query_idx
        end_points['sa2_grouped_features'] = sa2_grouped_features

        #print("========================== SA3 ===============================")
        #xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        sa3_xyz, sa3_features, sa3_inds, sa3_ball_query_idx, sa3_grouped_features = self.sa3(sa2_xyz, sa2_features, sample_type='fps') # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = sa3_xyz
        end_points['sa3_features'] = sa3_features
        end_points['sa3_inds'] = sa3_inds        
        #end_points['sa3_ball_query_idx'] = ball_query_idx
        end_points['sa3_grouped_features'] = sa3_grouped_features


        #print("========================== SA4 ===============================")
        #xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        sa4_xyz, sa4_features, sa4_inds, sa4_ball_query_idx, sa4_grouped_features = self.sa4(sa3_xyz, sa3_features, sample_type='fps') # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = sa4_xyz
        end_points['sa4_features'] = sa4_features
        end_points['sa4_inds'] = sa4_inds        
        #end_points['sa4_ball_query_idx'] = ball_query_idx
        end_points['sa4_grouped_features'] = sa4_grouped_features


        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        #features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        #print("========================== FP1 ===============================")
        features, prop_features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        #fp1_features, fp1_grouped_features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features, sa4_ball_query_idx, sa4_inds)
        end_points['fp1_grouped_features'] = prop_features
        #end_points.append(prop_features) #20
        
        #print("========================== FP2 ===============================")
        fp2_features, fp2_grouped_features = self.fp1(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)        
        end_points['fp2_features'] = fp2_features
        end_points['fp2_grouped_features'] = fp2_grouped_features
        end_points['fp2_xyz'] = end_points['sa2_xyz']        
        num_seed = sa2_inds.shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds        
        
        return end_points


from tf_ops.sampling import tf_sampling
import pointnet2_utils_tf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

class Pointnet2Backbone_p(layers.Layer):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, use_tflite=False):
        super().__init__()

        self.sa1 = SamplingAndGrouping(
                npoint=1024,
                radius=0.2,
                nsample=64,                
                use_xyz=True,
                normalize_xyz=True                
            )
        self.sa1_mlp = PointnetMLP(mlp=[input_feature_dim, 64, 64, 128], nsample=64)
        
        self.sa2 = SamplingAndGrouping(
                npoint=512,
                radius=0.4,
                nsample=32,                
                use_xyz=True,
                normalize_xyz=True                
            )
        self.sa2_mlp = PointnetMLP(mlp=[128, 128, 128, 128], nsample=32)

        self.sa3 = SamplingAndGrouping(
                npoint=256,
                radius=0.8,
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True                
            )
        self.sa3_mlp = PointnetMLP(mlp=[128, 128, 128, 128], nsample=16)

        self.sa4 = SamplingAndGrouping(
                npoint=128,
                radius=1.2,
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True                
            )
        self.sa4_mlp = PointnetMLP(mlp=[128, 128, 128, 128], nsample=16)

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256], m=512)
                #use_tflite=use_tflite, tflite_name='fp1_quant.tflite')
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256], m=1024)
                #use_tflite=use_tflite, tflite_name='fp2_quant_b8.tflite')

        """
        self.sa2_interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR,os.path.join("tflite_models",'sa3_quant.tflite')))
        self.sa2_interpreter.allocate_tensors()
        self.input_details = self.sa2_interpreter.get_input_details()
        self.output_details = self.sa2_interpreter.get_output_details()


        self.sa2_1_started = asyncio.Event()
        self.sa2_2_started = asyncio.Event()
        self._executor = ThreadPoolExecutor(2)
        self.loop = asyncio.get_event_loop()
        """
    def _break_up_pc(self, pc):
        #xyz = pc[..., 0:3]
        #features = pc[..., 3:] if pc.shape[-1] > 3 else None        

        xyz1 = pc[:, ::2,0:3]
        features1 =  pc[:, ::2, 3:]
        xyz2 = pc[:, 1::2, 0:3]
        features2 =  pc[:, 1::2, 3:]

        return xyz1, features1, xyz2, features2

    def call_tflite(self, grouped_features):
        self.sa2_interpreter.set_tensor(self.input_details[0]['index'], grouped_features)
        self.sa2_interpreter.invoke()
        return self.sa2_interpreter.get_tensor(self.output_details[0]['index'])

    async def wrapper_tflite(self, grouped_features):
        features = await self.loop.run_in_executor(self._executor, \
                    functools.partial(self.call_tflite, grouped_features))
        return features

    def call(self, pointcloud, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """        
        if not end_points: end_points = {}
        xyz1, features1, xyz2, features2 = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        #xyz, features, fps_inds = self.sa1(xyz, features)
        #print("========================== SA1 ===============================")
        sa1_xyz1, sa1_inds1, sa1_ball_query_idx1, sa1_grouped_features1 = self.sa1(xyz1, features1, sample_type='fps')
        sa1_features1 = self.sa1_mlp(sa1_grouped_features1)
        sa1_xyz2, sa1_inds2, sa1_ball_query_idx2, sa1_grouped_features2 = self.sa1(xyz2, features2, sample_type='fps')
        sa1_features2 = self.sa1_mlp(sa1_grouped_features2)
        end_points['sa1_xyz1'] = sa1_xyz1
        end_points['sa1_features1'] = sa1_features1
        end_points['sa1_inds1'] = sa1_inds1               
        end_points['sa1_grouped_features1'] = sa1_grouped_features1
        end_points['sa1_xyz2'] = sa1_xyz2
        end_points['sa1_features2'] = sa1_features2
        end_points['sa1_inds2'] = sa1_inds2               
        end_points['sa1_grouped_features1'] = sa1_grouped_features2

        sa2_xyz1, sa2_inds1, sa2_ball_query_idx1, sa2_grouped_features1 = self.sa2(sa1_xyz1, sa1_features1, sample_type='fps')
        sa2_features1 = self.sa2_mlp(sa2_grouped_features1)
        sa2_xyz2, sa2_inds2, sa2_ball_query_idx2, sa2_grouped_features2 = self.sa2(sa1_xyz2, sa1_features2, sample_type='fps')
        sa2_features2 = self.sa2_mlp(sa2_grouped_features2)
        end_points['sa2_xyz1'] = sa2_xyz1
        end_points['sa2_features1'] = sa2_features1
        end_points['sa2_inds1'] = sa2_inds1               
        end_points['sa2_grouped_features1'] = sa2_grouped_features1
        end_points['sa2_xyz2'] = sa2_xyz2
        end_points['sa2_features2'] = sa2_features2
        end_points['sa2_inds2'] = sa2_inds2               
        end_points['sa2_grouped_features1'] = sa2_grouped_features2

        sa3_xyz1, sa3_inds1, sa3_ball_query_idx1, sa3_grouped_features1 = self.sa3(sa2_xyz1, sa2_features1, sample_type='fps')
        sa3_features1 = self.sa3_mlp(sa3_grouped_features1)
        sa3_xyz2, sa3_inds2, sa3_ball_query_idx2, sa3_grouped_features2 = self.sa3(sa2_xyz2, sa2_features2, sample_type='fps')
        sa3_features2 = self.sa3_mlp(sa3_grouped_features2)
        end_points['sa3_xyz1'] = sa3_xyz1
        end_points['sa3_features1'] = sa3_features1
        end_points['sa3_inds1'] = sa3_inds1               
        end_points['sa3_grouped_features1'] = sa3_grouped_features1
        end_points['sa3_xyz2'] = sa3_xyz2
        end_points['sa3_features2'] = sa3_features2
        end_points['sa3_inds2'] = sa3_inds2               
        end_points['sa3_grouped_features1'] = sa3_grouped_features2

        sa4_xyz1, sa4_inds1, sa4_ball_query_idx1, sa4_grouped_features1 = self.sa4(sa3_xyz1, sa3_features1, sample_type='fps')
        sa4_features1 = self.sa4_mlp(sa4_grouped_features1)
        sa4_xyz2, sa4_inds2, sa4_ball_query_idx2, sa4_grouped_features2 = self.sa4(sa3_xyz2, sa3_features2, sample_type='fps')
        sa4_features2 = self.sa4_mlp(sa4_grouped_features2)
        end_points['sa4_xyz1'] = sa4_xyz1
        end_points['sa4_features1'] = sa4_features1
        end_points['sa4_inds1'] = sa4_inds1               
        end_points['sa4_grouped_features1'] = sa4_grouped_features1
        end_points['sa4_xyz2'] = sa4_xyz2
        end_points['sa4_features2'] = sa4_features2
        end_points['sa4_inds2'] = sa4_inds2               
        end_points['sa4_grouped_features1'] = sa4_grouped_features2

        sa4_features = layers.Concatenate(axis=1)([sa4_features1, sa4_features2])
        sa3_features = layers.Concatenate(axis=1)([sa3_features1, sa3_features2])
        sa2_features = layers.Concatenate(axis=1)([sa2_features1, sa2_features2])

        sa4_xyz = layers.Concatenate(axis=1)([sa4_xyz1, sa4_xyz2])
        sa3_xyz = layers.Concatenate(axis=1)([sa3_xyz1, sa3_xyz2])
        sa2_xyz = layers.Concatenate(axis=1)([sa2_xyz1, sa2_xyz2])

        sa2_inds = layers.Concatenate(axis=1)([sa2_inds1, sa2_inds2])
        sa1_inds = layers.Concatenate(axis=1)([sa1_inds1, sa1_inds2])

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #print("========================== FP1 ===============================")
        fp1_features, fp1_grouped_features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        #fp1_features1, fp1_grouped_features1 = self.fp1(sa3_xyz1, sa4_xyz1, sa3_features1, sa4_features1)
        #fp1_features2, fp1_grouped_features2 = self.fp1(sa3_xyz2, sa4_xyz2, sa3_features2, sa4_features2)
        #fp1_features, fp1_grouped_features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features, sa4_ball_query_idx, sa4_inds)
        #end_points.append(prop_features) #20
        
        #print("========================== FP2 ===============================")
        fp2_features, fp2_grouped_features = self.fp2(sa2_xyz, sa3_xyz, sa2_features, fp1_features)        
        #fp2_features1, fp2_grouped_features1 = self.fp2(sa2_xyz1, sa3_xyz1, sa2_features1, fp1_features1)        
        #fp2_features2, fp2_grouped_features2 = self.fp2(sa2_xyz2, sa3_xyz2, sa2_features2, fp1_features2) 
        
        
        #fp2_features = layers.Concatenate(axis=1)([fp2_features1, fp2_features2])
        #fp2_grouped_features = layers.Concatenate(axis=1)([fp2_grouped_features1, fp2_grouped_features2])
        #fp1_grouped_features = layers.Concatenate(axis=1)([fp1_grouped_features1, fp1_grouped_features2])

        end_points['fp1_grouped_features'] = fp1_grouped_features
        end_points['fp2_features'] = fp2_features
        end_points['fp2_grouped_features'] = fp2_grouped_features
        end_points['fp2_xyz'] = sa2_xyz        
        num_seed = sa2_inds.shape[1]
        end_points['fp2_inds'] = sa1_inds[:,0:num_seed] # indices among the entire input point clouds        

        return end_points
        """        
        sa2_xyz_1, sa2_features_1, sa2_inds_1, sa2_ball_query_idx_1, sa2_grouped_features_1 = self.sa2_half(sa1_xyz_1, sa1_features_1, sample_type='fps') # this fps_inds is just 0,1,...,1023
        sa2_xyz_2, sa2_features_2, sa2_inds_2, sa2_ball_query_idx_2, sa2_grouped_features_2 = self.sa2_half(sa1_xyz_2, sa1_features_2, sample_type='fps') # this fps_inds is just 0,1,...,1023

        sa2_xyz = layers.Concatenate(axis=1)([sa2_xyz_1, sa2_xyz_2], )
        sa2_features = layers.Concatenate(axis=1)([sa2_features_1, sa2_features_2])
        sa2_inds = layers.Concatenate(axis=1)([sa2_inds_1, sa2_inds_2])
        sa2_ball_query_idx = layers.Concatenate(axis=1)([sa2_ball_query_idx_1, sa2_ball_query_idx_2])
        sa2_grouped_features = layers.Concatenate(axis=1)([sa2_grouped_features_1, sa2_grouped_features_2])
        print(sa2_features.shape)
        
        sa1_xyz_1 = sa1_xyz[:,:1024,:]
        sa1_features_1 = sa1_features[:,:1024,:]
        sa1_xyz_2 = sa1_xyz[:,1024:,:]
        sa1_features_2 = sa1_features[:,1024:,:]

        #print("========================== SA2 ===============================")        
        
        
        sa2_inds_1 = tf_sampling.farthest_point_sample(512, sa1_xyz_1)
        sa2_xyz_1 = tf_sampling.gather_point(sa1_xyz_1, sa2_inds_1)
        sa2_grouper = pointnet2_utils_tf.QueryAndGroup(radius=0.4, nsample=16, use_xyz=True, ret_grouped_xyz=True, normalize_xyz=True,
                sample_uniformly=False, ret_unique_cnt=False)
        sa2_grouped_features_1, sa2_ball_query_idx_1, sa2_grouped_xyz_1 = sa2_grouper(sa1_xyz_1, sa2_xyz_1, sa1_features_1)

        start = time.time()
        self.sa2_interpreter.set_tensor(self.input_details[0]['index'], sa2_grouped_features_1)
        self.sa2_interpreter.invoke()
        sa2_features_1 = self.sa2_interpreter.get_tensor(self.output_details[0]['index'])
        end = time.time()
        print("Synchronous inference", end - start)

        sa2_inds_2 = tf_sampling.farthest_point_sample(512, sa1_xyz_2)
        sa2_xyz_2 = tf_sampling.gather_point(sa1_xyz_2, sa2_inds_2)        
        sa2_grouped_features_2, sa2_ball_query_idx_2, sa2_grouped_xyz_2 = sa2_grouper(sa1_xyz_2, sa2_xyz_2, sa1_features_2)

        start = time.time()
        #task1 = asyncio.create_task(call_tflite(sa2_grouped_features_2))
        #sa2_features_2 = await task1
        #sa2_features_2 = self.loop.run_until_complete(self.wrapper_tflite(sa2_grouped_features_2))
        sa2_features_2 = self.wrapper_tflite(sa2_grouped_features_2)
        end = time.time()
        print("Asynchronous inference", end - start)
        print(sa2_features_2)

        sa2_features_2 = self.loop.run_until_complete(sa2_features_2)
        
        print(sa2_features_2.shape)
        end = time.time()
        print("Asynchronous inference 2", end - start)
        self.loop.close()
        """
        
        


if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3)
    #print(backbone_net)
    #backbone_net.eval()

    xyz = np.random.random((16,20000,6)).astype('float32')
    out = backbone_net(tf.constant(xyz))
    print("========================out=========================")
    print(out)
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
