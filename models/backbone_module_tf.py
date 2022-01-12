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

from pointnet2_modules_tf import PointnetSAModuleVotes, PointnetFPModule
from deeplab.deeplab import run_semantic_seg, run_semantic_seg_tflite
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping

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
    def __init__(self, input_feature_dim=0, model_config=None):
        super().__init__()

        use_tflite = model_config['use_tflite']
        use_fp_mlp = model_config['use_fp_mlp']
        self.use_painted = model_config['use_painted']

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True,
                model_config=model_config,
                layer_name='sa1'
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                model_config=model_config,
                layer_name='sa2'
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                model_config=model_config,
                layer_name='sa3'
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                model_config=model_config,
                layer_name='sa4'
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256], m=512, model_config=model_config, layer_name='fp1')
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256], m=1024, model_config=model_config, layer_name='fp2')

    def _break_up_pc(self, pc):
        #xyz = pc[..., 0:3]
        #features = pc[..., 3:] if pc.shape[-1] > 3 else None        

        xyz = pc[:,:,0:3]
        features =  pc[:,:,3:]
        
        return xyz, features

    def call(self, pointcloud, end_points=None, img=None, calib=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run  predicts on
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
        time_record = []
        time_record.append(("Start:", time.time()))
        sa1_xyz, sa1_features, sa1_inds, sa1_grouped_features = self.sa1(xyz, features, time_record)        
        end_points['sa1_xyz'] = sa1_xyz
        end_points['sa1_features'] = sa1_features
        end_points['sa1_inds'] = sa1_inds        
        end_points['sa1_grouped_features'] = sa1_grouped_features

        #print("========================== SA2 ===============================")
        #xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        sa2_xyz, sa2_features, sa2_inds, sa2_grouped_features = self.sa2(sa1_xyz, sa1_features, time_record) # this fps_inds is just 0,1,...,1023        
        end_points['sa2_xyz'] = sa2_xyz
        end_points['sa2_features'] = sa2_features
        end_points['sa2_inds'] = sa2_inds        
        end_points['sa2_grouped_features'] = sa2_grouped_features

        #print("========================== SA3 ===============================")
        #xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        sa3_xyz, sa3_features, sa3_inds, sa3_grouped_features = self.sa3(sa2_xyz, sa2_features, time_record) # this fps_inds is just 0,1,...,511        
        end_points['sa3_xyz'] = sa3_xyz
        end_points['sa3_features'] = sa3_features
        end_points['sa3_inds'] = sa3_inds        
        end_points['sa3_grouped_features'] = sa3_grouped_features


        #print("========================== SA4 ===============================")
        #xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        sa4_xyz, sa4_features, sa4_inds, sa4_grouped_features = self.sa4(sa3_xyz, sa3_features, time_record) # this fps_inds is just 0,1,...,255        
        end_points['sa4_xyz'] = sa4_xyz
        end_points['sa4_features'] = sa4_features
        end_points['sa4_inds'] = sa4_inds        
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
        fp2_features, fp2_grouped_features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)        
        end_points['fp2_features'] = fp2_features
        end_points['fp2_grouped_features'] = fp2_grouped_features
        end_points['fp2_xyz'] = end_points['sa2_xyz']        
        num_seed = sa2_inds.shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds  

        time_record.append(("SA End:", time.time()))
        end_points['time_record'] = time_record 
        
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
