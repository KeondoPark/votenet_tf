# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os

import tensorflow as tf
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules_tf import PointnetSAModuleVotes, PointnetFPModule

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
                tflite_name='sa1_quant_b8.tflite'
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa2_quant_b8.tflite'
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa3_quant_b8.tflite'
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                use_tflite=use_tflite,
                tflite_name='sa4_quant_b8.tflite'
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256], m=512,
                use_tflite=use_tflite, tflite_name='fp1_quant_b8.tflite')
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256], m=1024,
                use_tflite=use_tflite, tflite_name='fp2_quant_b8.tflite')

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3]
        features = pc[..., 3:] if pc.shape[-1] > 3 else None        

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
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        #xyz, features, fps_inds = self.sa1(xyz, features)
        #print("========================== SA1 ===============================")
        xyz, features, fps_inds, ball_query_idx, grouped_features = self.sa1(xyz, features, sample_type='fps')
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features
        end_points['sa1_ball_query_idx'] = ball_query_idx
        end_points['sa1_grouped_features'] = grouped_features

        #print("========================== SA2 ===============================")
        #xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        xyz, features, fps_inds, ball_query_idx, grouped_features = self.sa2(xyz, features, sample_type='fps') # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features
        end_points['sa2_ball_query_idx'] = ball_query_idx
        end_points['sa2_grouped_features'] = grouped_features

        #print("========================== SA3 ===============================")
        #xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        xyz, features, fps_inds, ball_query_idx, grouped_features = self.sa3(xyz, features, sample_type='fps') # this fps_inds is just 0,1,...,511
        end_points['sa3_inds'] = fps_inds
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features
        end_points['sa3_ball_query_idx'] = ball_query_idx
        end_points['sa3_grouped_features'] = grouped_features

        #print("========================== SA4 ===============================")
        #xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        xyz, features, fps_inds, ball_query_idx, grouped_features = self.sa4(xyz, features, sample_type='fps') # this fps_inds is just 0,1,...,255
        end_points['sa4_inds'] = fps_inds
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features
        end_points['sa4_ball_query_idx'] = ball_query_idx
        end_points['sa4_grouped_features'] = grouped_features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        #features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        #print("========================== FP1 ===============================")
        features, prop_features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'], end_points['sa4_ball_query_idx'], end_points['sa4_inds'])
        end_points['fp1_grouped_features'] = prop_features
        #print("========================== FP2 ===============================")
        features, prop_features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features, end_points['sa3_ball_query_idx'], end_points['sa3_inds'])
        end_points['fp2_features'] = features
        end_points['fp2_grouped_features'] = prop_features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        return end_points


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
