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

from pointnet2_modules_tf import PointnetSAModuleVotes, PointnetFPModule, SamplingAndGrouping, PointnetMLP, SurfPointnetMLP
from deeplab.deeplab import run_semantic_seg, run_semantic_seg_tflite
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
import repsurf_utils_tf

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
        self.umb_learner = repsurf_utils_tf.UmbrellaSurface_Learner(act=model_config['activation'])
        # self.umb_constructor = repsurf_utils_tf.UmbrellaSurfaceConstructor(k=9, out_channel=10, act=model_config['activation'])

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True,
                model_config=model_config,
                use_repsurf=True,
                layer_name='sa1',
                repsurf_channel=10
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                model_config=model_config,
                use_repsurf=True,
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
                use_repsurf=True,
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
                use_repsurf=True,
                layer_name='sa4'
            )

        if use_fp_mlp:
            self.fp1 = PointnetFPModule(mlp=[256+256,256,256], m=512, model_config=model_config, layer_name='fp1')
            self.fp2 = PointnetFPModule(mlp=[256+256,256,288], m=1024, model_config=model_config, layer_name='fp2')
        else:
            self.fp1 = PointnetFPModule(mlp=None, m=512, model_config=model_config)
            self.fp2 = PointnetFPModule(mlp=None, m=1024, model_config=model_config)

    def _break_up_pc(self, pc):
        xyz = pc[:,:,0:3]
        features =  pc[:,:,3:]
        
        return xyz, features

    def call(self, pointcloud, repsurf_feature, end_points=None, img=None, calib=None):
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
        
        repsurf_feature = self.umb_learner(repsurf_feature)        
        # B = tf.shape(xyz)[0]
        # N = tf.shape(xyz)[1]
        
        # offset = tf.cast(tf.range(N // 5000)*5000 + 5000, dtype=tf.int32)
        # offset = tf.tile(tf.expand_dims(offset,0), [B,1])
        # repsurf_feature = self.umb_constructor(xyz, offset)

        features = layers.Concatenate(axis=-1)([features, repsurf_feature])

        # --------- 4 SET ABSTRACTION LAYERS ---------
        #print("========================== SA1 ===============================")
        time_record = []
        time_record.append(("Start:", time.time()))
        sa1_xyz, sa1_features, sa1_inds, sa1_grp_feats = self.sa1(xyz, features, time_record)        
        end_points['sa1_xyz'] = sa1_xyz
        end_points['sa1_features'] = sa1_features
        end_points['sa1_inds'] = sa1_inds        
        # end_points['sa1_grouped_features'] = sa1_grouped_features

        #print("========================== SA2 ===============================")
        #xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        sa2_xyz, sa2_features, sa2_inds, sa2_grp_feats = self.sa2(sa1_xyz, sa1_features, time_record) # this fps_inds is just 0,1,...,1023        
        end_points['sa2_xyz'] = sa2_xyz
        end_points['sa2_features'] = sa2_features
        end_points['sa2_inds'] = sa2_inds        
        # end_points['sa2_grouped_features'] = sa2_grouped_features

        #print("========================== SA3 ===============================")
        #xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        sa3_xyz, sa3_features, sa3_inds, sa3_grp_feats = self.sa3(sa2_xyz, sa2_features, time_record) # this fps_inds is just 0,1,...,511        
        end_points['sa3_xyz'] = sa3_xyz
        end_points['sa3_features'] = sa3_features
        end_points['sa3_inds'] = sa3_inds        
        # end_points['sa3_grouped_features'] = sa3_grouped_features


        #print("========================== SA4 ===============================")        
        sa4_xyz, sa4_features, sa4_inds, sa4_grp_feats = self.sa4(sa3_xyz, sa3_features, time_record) # this fps_inds is just 0,1,...,255        
        end_points['sa4_xyz'] = sa4_xyz
        end_points['sa4_features'] = sa4_features
        end_points['sa4_inds'] = sa4_inds        
        # end_points['sa4_grouped_features'] = sa4_grouped_features
              

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #print("========================== FP1 ===============================")
        features, prop_features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        #fp1_features, fp1_grouped_features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features, sa4_ball_query_idx, sa4_inds)
        # end_points['fp1_grouped_features'] = prop_features
        
        #print("========================== FP2 ===============================")
        fp2_features, fp2_grp_feats = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)        
        end_points['fp2_features'] = fp2_features
        # end_points['fp2_grouped_features'] = fp2_grouped_features
        end_points['fp2_xyz'] = end_points['sa2_xyz']        
        num_seed = sa2_inds.shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds  

        time_record.append(("SA End:", time.time()))
        end_points['time_record'] = time_record 
        
        return end_points


from tf_ops.sampling import tf_sampling
import pointnet2_utils_tf
import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import functools

class Pointnet2Backbone_p(layers.Layer):
    r"""
       Backbone network using 2way SA layers
    """
    def __init__(self, input_feature_dim=0, model_config=None):
        super().__init__()

        use_fp_mlp = model_config['use_fp_mlp']
        self.bfps_wght = model_config["bfps_wght"]
        self.umb_learner = repsurf_utils_tf.UmbrellaSurface_Learner(act=model_config['activation'])
        radius = model_config["radius"] if "radius" in model_config else [0.2,0.4,0.8,1.2]

        self.sa1 = SamplingAndGrouping(
                npoint=1024,
                radius=radius[0],
                nsample=64,                
                use_xyz=True,
                normalize_xyz=True,
                return_polar=True
            )
        self.sa1_mlp = SurfPointnetMLP(mlp=[input_feature_dim, 64, 64, 128], nsample=64, model_config=model_config, repsurf_channel=10)        
        
        self.sa2 = SamplingAndGrouping(
                npoint=512,
                radius=radius[1],
                nsample=32,                
                use_xyz=True,
                normalize_xyz=True,
                return_polar=True
            )
        self.sa2_mlp = SurfPointnetMLP(mlp=[128, 128, 128, 256], nsample=32, model_config=model_config)

        self.sa3 = SamplingAndGrouping(
                npoint=256,
                radius=radius[2],
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True,
                return_polar=True
            )
        self.sa3_mlp = SurfPointnetMLP(mlp=[256, 128, 128, 256], nsample=16, model_config=model_config)
        
        self.sa4 = SamplingAndGrouping(
                npoint=256,
                radius=radius[3],
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True,
                return_polar=True
            )
        self.sa4_mlp = SurfPointnetMLP(mlp=[256, 128, 128, 256], nsample=16, model_config=model_config)

        
        self.fp1 = PointnetFPModule(mlp=[256+256,256,256], m=512, model_config=model_config)
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256], m=1024, model_config=model_config)


    def _break_up_pc(self, pc):
        xyz = pc[:,:,0:3]        
        isPainted = tf.cast(pc[:,:,3], dtype=tf.int32)        
        features =  pc[:,:, 4:]

        return xyz, isPainted, features

    def _remove_sampled(self, xyz, inds, isPainted, features):
        '''
        Removes sampled points from the input point cloud
        '''
        # Batch index. i in (i,j) index type
        B = tf.shape(xyz)[0]
        N = tf.shape(xyz)[1]        
        
        npoint = tf.shape(inds)[1]
        batch_inds = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(B),-1), [1,npoint]), -1)
        new_inds = tf.concat([batch_inds, tf.expand_dims(inds, -1)], -1)        
        updates = tf.cast(tf.ones([B,npoint]), tf.bool)
        mask = tf.scatter_nd(new_inds, updates, [B,N]) # mask where sa1_inds = True

        new_xyz, new_isPainted, new_features = [], [], []

        new_xyz = tf.boolean_mask(xyz, tf.logical_not(mask))
        new_xyz = tf.reshape(new_xyz, [B, N - npoint, -1])

        new_isPainted = tf.boolean_mask(isPainted, tf.logical_not(mask))
        new_isPainted = tf.reshape(new_isPainted, [B, N - npoint])

        new_features = tf.boolean_mask(features, tf.logical_not(mask))
        new_features = tf.reshape(new_features, [B, N - npoint, -1])
        
        return new_xyz, new_isPainted, new_features, mask


    def call(self, pointcloud, repsurf_feature, end_points=None):
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
        xyz, isPainted, features = self._break_up_pc(pointcloud)

        # --------- RepSurf preprocessing ---------
        
        B = tf.shape(xyz)[0]
        # offset = tf.convert_to_tensor(np.arange(num_point // 5000)*5000 + 5000, dtype=tf.int32)
        # offset = tf.tile(tf.expand_dims(offset,0), [B,1])
        # repsurf_feature = self.umb_constructor(xyz, offset)

        repsurf_feature = self.umb_learner(repsurf_feature)  
        features = layers.Concatenate(axis=-1)([features, repsurf_feature])
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        # ------------------------------- SA1-------------------------------                
        
        time_record = []
        time_record.append(("SA Start:", time.time()))
        sa1_xyz1, sa1_inds1, sa1_grp_feats1, sa1_painted1 = self.sa1(xyz, isPainted, features, bg1=True, wght1=1)
        time_record.append(("SA1 sampling and grouping 1:", time.time()))        

        # Remove sampled points from xyz
        new_xyz, new_isPainted, new_features, mask = self._remove_sampled(xyz, sa1_inds1, isPainted, features)             
        sa1_features1 = self.sa1_mlp(sa1_grp_feats1)        

        time_record.append(("SA1 MLP 1:", time.time()))
        
        sa1_xyz2, sa1_inds2, sa1_grp_feats2, sa1_painted2 \
            = self.sa1(new_xyz, new_isPainted, new_features, bg1=True, wght1=self.bfps_wght[0], 
                xyz_ball=xyz, features_ball=features)        
        time_record.append(("SA1 sampling and grouping 2:", time.time()))        
        
        sa1_features2 = self.sa1_mlp(sa1_grp_feats2)          
        time_record.append(("SA1 MLP 2:", time.time()))     

        
        # end_points['sa1_painted1'] = sa1_painted1
        # end_points['sa1_painted2'] = sa1_painted2

        sa1_xyz = layers.Concatenate(axis=1)([sa1_xyz1, sa1_xyz2])
        sa1_features = layers.Concatenate(axis=1)([sa1_features1, sa1_features2])        
                
        
        # ------------------------------- SA2-------------------------------        
        sa2_xyz1, sa2_inds1, sa2_grp_feats1, sa2_painted1 \
            = self.sa2(sa1_xyz1, sa1_painted1, sa1_features1, bg1=True, wght1=1)
        time_record.append(("SA2 sampling and grouping 1:", time.time()))        
        
        sa2_features1 = self.sa2_mlp(sa2_grp_feats1)
        time_record.append(("SA2 MLP 1:", time.time()))
        
        sa1_xyz = layers.Concatenate(axis=1)([sa1_xyz1, sa1_xyz2])
        sa1_features = layers.Concatenate(axis=1)([sa1_features1, sa1_features2])        

        sa2_xyz2, sa2_inds2, sa2_grp_feats2, sa2_painted2 \
            = self.sa2(sa1_xyz2, sa1_painted2, sa1_features2, bg1=True, wght1=self.bfps_wght[1], 
                    xyz_ball=sa1_xyz, features_ball=sa1_features)
        time_record.append(("SA2 sampling and grouping 2:", time.time()))        

        sa2_features2 = self.sa2_mlp(sa2_grp_feats2)
        time_record.append(("SA2 MLP 2:", time.time()))

        # ------------------------------- SA3-------------------------------        
        sa3_xyz1, sa3_inds1, sa3_grp_feats1, sa3_painted1 \
            = self.sa3(sa2_xyz1, sa2_painted1, sa2_features1, bg1=True, wght1=1)
        time_record.append(("SA3 sampling and grouping 1:", time.time()))

        sa3_features1 = self.sa3_mlp(sa3_grp_feats1)
        time_record.append(("SA3 MLP 1:", time.time()))

        sa2_xyz = layers.Concatenate(axis=1)([sa2_xyz1, sa2_xyz2])
        sa2_features = layers.Concatenate(axis=1)([sa2_features1, sa2_features2])        

        sa3_xyz2, sa3_inds2, sa3_grp_feats2, sa3_painted2 \
            = self.sa3(sa2_xyz2, sa2_painted2, sa2_features2, bg1=True, wght1=self.bfps_wght[2], 
                    xyz_ball=sa2_xyz, features_ball=sa2_features)        
        time_record.append(("SA3 sampling and grouping 2:", time.time()))

        sa3_features2 = self.sa3_mlp(sa3_grp_feats2)
        time_record.append(("SA3 MLP 2:", time.time()))

        # ------------------------------- SA4-------------------------------       

        sa3_xyz = layers.Concatenate(axis=1)([sa3_xyz1, sa3_xyz2])
        sa3_features = layers.Concatenate(axis=1)([sa3_features1, sa3_features2])        
        sa3_painted = layers.Concatenate(axis=1)([sa3_painted1, sa3_painted2])

        sa4_xyz, sa4_inds, sa4_grp_feats, sa4_painted \
            = self.sa4(sa3_xyz, sa3_painted, sa3_features, bg1=True, wght1=self.bfps_wght[3])
        sa4_features = self.sa4_mlp(sa4_grp_feats)

        # end_points['sa1_grouped_features1'] = sa1_grouped_features1
        # end_points['sa1_grouped_features2'] = sa1_grouped_features2

        # end_points['sa2_grouped_features1'] = sa2_grouped_features1
        # end_points['sa2_grouped_features2'] = sa2_grouped_features2
       
        # end_points['sa3_grouped_features1'] = sa3_grouped_features1
        # end_points['sa3_grouped_features2'] = sa3_grouped_features2

        # end_points['sa4_grouped_features'] = sa4_grouped_features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #print("========================== FP1 ===============================")
        fp1_features, fp1_grp_feats = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        
        #print("========================== FP2 ===============================")
        fp2_features, fp2_grp_feats = self.fp2(sa2_xyz, sa3_xyz, sa2_features, fp1_features)        

        # end_points['fp1_grouped_features'] = fp1_grouped_features
        end_points['fp2_features'] = fp2_features
        # end_points['fp2_grouped_features'] = fp2_grouped_features
        end_points['fp2_xyz'] = sa2_xyz
        seed_inds1 = tf.gather(sa1_inds1, axis=1, indices=sa2_inds1, batch_dims=1)
        
        # Necessary if excluding first sampling points
        B = tf.shape(xyz)[0]
        N = tf.shape(xyz)[1]
        
        all_inds = tf.tile(tf.expand_dims(tf.range(N), 0), [B,1])
        rem_inds = tf.boolean_mask(all_inds, tf.logical_not(mask))
        rem_inds = tf.reshape(rem_inds, [B,-1])
        sa1_2_inds2 = tf.gather(sa1_inds2, axis=1, indices=sa2_inds2, batch_dims=1)
        seed_inds2 = tf.gather(rem_inds, indices=sa1_2_inds2, batch_dims=1)   

        sa1_inds2_from_orig = tf.gather(rem_inds, indices=sa1_inds2, batch_dims=1)         
        sa1_inds = layers.Concatenate(axis=1)([sa1_inds1, sa1_inds2_from_orig])

        
        end_points['sa1_xyz'] = sa1_xyz        
        end_points['sa1_inds'] = sa1_inds        
        
        
        end_points['fp2_inds'] = layers.Concatenate(axis=1)([seed_inds1, seed_inds2])      
        
        time_record.append(("SA End:", time.time()))
        end_points['time_record'] = time_record   

        return end_points


class Pointnet2Backbone_tflite(layers.Layer):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, model_config=None, num_class=10):
        super().__init__()

        self.sa1 = SamplingAndGrouping(
                npoint=1024,
                radius=0.2,
                nsample=64,                
                use_xyz=True,
                normalize_xyz=True                
            )
        
        self.sa2 = SamplingAndGrouping(
                npoint=512,
                radius=0.4,
                nsample=32,                
                use_xyz=True,
                normalize_xyz=True                
            )        

        self.sa3 = SamplingAndGrouping(
                npoint=256,
                radius=0.8,
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True                
            )        
        
        self.sa4 = SamplingAndGrouping(
                npoint=256,
                radius=1.2,
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True                
            )        

        
        self.use_multiThr = model_config['use_multiThr']
        self.use_edgetpu = model_config['use_edgetpu']
        
        self.fp1 = PointnetFPModule(mlp=None, m=512, model_config=model_config, layer_name='fp1')
        self.fp2 = PointnetFPModule(mlp=None, m=1024, model_config=model_config, layer_name='fp2')                        
        
        tflite_folder = model_config['tflite_folder']
        
        #Preparation to use tflite
        if self.use_edgetpu:
            from pycoral.utils.edgetpu import make_interpreter                        
            self.sa1_interpreter = make_interpreter(os.path.join(ROOT_DIR, tflite_folder, 'sa1_quant_edgetpu.tflite'))
            self.sa2_interpreter = make_interpreter(os.path.join(ROOT_DIR, tflite_folder, 'sa2_quant_edgetpu.tflite'))
            self.sa3_interpreter = make_interpreter(os.path.join(ROOT_DIR, tflite_folder, 'sa3_quant_edgetpu.tflite'))
            self.sa4_interpreter = make_interpreter(os.path.join(ROOT_DIR, tflite_folder, 'sa4_quant_edgetpu.tflite'))
        else:
            self.sa1_interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR, tflite_folder, 'sa1_quant.tflite'))
            self.sa2_interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR, tflite_folder, 'sa2_quant.tflite'))
            self.sa3_interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR, tflite_folder, 'sa3_quant.tflite'))
            self.sa4_interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR, tflite_folder, 'sa4_quant.tflite'))
        
        self.sa1_interpreter.allocate_tensors()
        self.sa2_interpreter.allocate_tensors()
        self.sa3_interpreter.allocate_tensors()
        self.sa4_interpreter.allocate_tensors()

        self.sa1_input_details = self.sa1_interpreter.get_input_details()
        self.sa2_input_details = self.sa2_interpreter.get_input_details()
        self.sa3_input_details = self.sa3_interpreter.get_input_details()
        self.sa4_input_details = self.sa4_interpreter.get_input_details()

        self.sa1_output_details = self.sa1_interpreter.get_output_details()
        self.sa2_output_details = self.sa2_interpreter.get_output_details()
        self.sa3_output_details = self.sa3_interpreter.get_output_details()
        self.sa4_output_details = self.sa4_interpreter.get_output_details()

        # For multithreading
        self._executor = ThreadPoolExecutor(2)
        self.num_class = num_class

    def _break_up_pc(self, pc):
        xyz = pc[:,:,0:3]        
        isPainted = tf.cast(pc[:,:,3], dtype=tf.int32)        
        features =  pc[:,:, 4:]

        return xyz, isPainted, features

    def call_tflite(self, interpreter, grouped_features, input_details, output_details):        
        interpreter.set_tensor(input_details[0]['index'], grouped_features)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    async def wrapper_tflite(self, interpreter, grouped_features):
        features = await self.loop.run_in_executor(self._executor, \
                    functools.partial(self.call_tflite, interpreter, grouped_features))
        return features

    def _remove_sampled(self, xyz, inds, isPainted, features):
        '''
        Removes sampled points from the input point cloud
        '''

        # Batch index. i in (i,j) index type
        B = tf.shape(xyz)[0]
        N = tf.shape(xyz)[1]
        npoint = tf.shape(inds)[1]
        batch_inds = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(B),-1), [1,npoint]), -1) # (B, npoint,1)
        new_inds = tf.concat([batch_inds, tf.expand_dims(inds, -1)], -1) # (B, npoint, 2)
        updates = tf.cast(tf.ones([B,npoint]), tf.bool)
        mask = tf.scatter_nd(new_inds, updates, [B,N]) # mask where sa1_inds = True

        new_xyz, new_isPainted, new_features = [], [], []

        new_xyz = tf.boolean_mask(xyz, tf.logical_not(mask))
        new_xyz = tf.reshape(new_xyz, [B, N - npoint, -1])

        new_isPainted = tf.boolean_mask(isPainted, tf.logical_not(mask))
        new_isPainted = tf.reshape(new_isPainted, [B, N - npoint])

        new_features = tf.boolean_mask(features, tf.logical_not(mask))
        new_features = tf.reshape(new_features, [B, N - npoint, -1])

        return new_xyz, new_isPainted, new_features, mask

    def call(self, pointcloud, end_points=None, imgs=None, calibs=None, deeplab_tflite_file=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            imgs: RGB images for semantic segmentation
                Used only pipelining is used

            calibs: 2d-3d projection
                Used only pipelining is used

            deeplab_tflite_file: Used only pipelining is used

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """        
        if not end_points: end_points = {}
        time_record = []
        time_record.append(("Pointpainted Votenet start:", time.time()))

        sa1_inds1 = None
        sa1_new_xyz1 = None
        sa1_ball_inds1 = None

        # Run image segmentation result and get result
        if imgs is not None and self.use_multiThr:
            xyz = pointcloud[:,:,:3]      

            # Run deeplab with different thread
            future0 = self._executor.submit(run_semantic_seg_tflite, imgs, False, deeplab_tflite_file)  

            # While EdgeTPU is working on 2D semantic segmentation, do point sampling, ball query
            sa1_inds1 = tf_sampling.farthest_point_sample(1024, xyz) # First sampling is not biased FPS, i.e. weight = 1
            sa1_new_xyz1 = tf_sampling.gather_point(xyz, sa1_inds1)
            sa1_ball_inds1, _ = tf_grouping.query_ball_point(0.2, 64, xyz, sa1_new_xyz1)
            
            # Prepare projection betwen 2d-3d
            uv_list, filter_idx_list = [], []
            for calib in calibs:
                uv, d, filter_idx = calib.project_upright_depth_to_image(xyz[0]) #uv: (N, 2)
                uv = np.rint(uv - 1)
                uv_list.append(uv)
                filter_idx_list.append(filter_idx)
            
            # Wait for pointpainting results
            pred_prob_list = future0.result()            
            time_record.append(('Deeplab inference time:', time.time()))      

            features = np.zeros((1, xyz.shape[1], self.num_class+1+1))
            isPainted = np.zeros((1, xyz.shape[1]), dtype=np.int32)
            features[:,:,-1] = pointcloud[:,:,-1]

            for pred_prob, uv, filter_idx in zip(pred_prob_list, uv_list, filter_idx_list): 
                pred_prob = pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)] # (npoint, num_class + 1 + 1 )
                projected_class = np.argmax(pred_prob, axis=-1) # (npoint, 1) 
                _isPainted = np.where(projected_class > 0, 1, 0) # Point belongs to background?                                    

                # 0 is background class, (height, width, num_class)
                pred_prob = pred_prob[:,:(self.num_class+1)] # (npoint, num_class+1)
                isPainted[0, filter_idx] = _isPainted
                features[0, filter_idx, :-1] = pred_prob/255
            
            time_record.append(('Pointpainting time:', time.time()))
        else:
            xyz, isPainted, features = self._break_up_pc(pointcloud)
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        # ------------------------------- SA1-------------------------------        

        # Initial FPS, in multithreading case, this simply concatenates xyz-coordinates and features of sampled points
        sa1_xyz1, sa1_inds1, sa1_grp_feats1, sa1_painted1 = self.sa1(xyz, isPainted, features, sa1_inds1, new_xyz=sa1_new_xyz1, ball_inds=sa1_ball_inds1, bg1=True, wght1=1)        
        with tf.device('cpu'):
            sa1_xyz1, sa1_inds1, sa1_grp_feats1, sa1_painted1 = tf.identity(sa1_xyz1), tf.identity(sa1_inds1), tf.identity(sa1_grp_feats1), tf.identity(sa1_painted1)
        time_record.append(("SA1 sampling and grouping 1:", time.time()))   
        
        # Pointnet
        if self.use_multiThr:
            future1 = self._executor.submit(self.call_tflite, self.sa1_interpreter, sa1_grp_feats1, self.sa1_input_details, self.sa1_output_details)            
        else:
            sa1_features1 = self.call_tflite(self.sa1_interpreter, sa1_grp_feats1, self.sa1_input_details, self.sa1_output_details)                        
                
        time_record.append(("SA1 MLP 1:", time.time()))

        # Exclude the first sampled points from input, for the second sampling
        new_xyz, new_isPainted, new_features, mask = self._remove_sampled(xyz, sa1_inds1, isPainted, features)             
        
        # Biased FPS
        sa1_xyz2, sa1_inds2, sa1_grp_feats2, sa1_painted2 = self.sa1(new_xyz, new_isPainted, new_features, bg1=True, wght1=4, xyz_ball=xyz, features_ball=features)        
        with tf.device('cpu'):
            sa1_xyz2, sa1_inds2, sa1_grp_feats2, sa1_painted2 = tf.identity(sa1_xyz2), tf.identity(sa1_inds2), tf.identity(sa1_grp_feats2), tf.identity(sa1_painted2)
        time_record.append(("SA1 sampling and grouping 2:", time.time()))
        
        # Pointnet
        if self.use_multiThr:
            sa1_features1 = future1.result()        
            future2 = self._executor.submit(self.call_tflite, self.sa1_interpreter, sa1_grp_feats2, self.sa1_input_details, self.sa1_output_details)
        else:
            sa1_features2 = self.call_tflite(self.sa1_interpreter, sa1_grp_feats2, self.sa1_input_details, self.sa1_output_details)                        
                
        time_record.append(("SA1 MLP 2:", time.time()))               

        # ------------------------------- SA2-------------------------------        
        # Normal FPS
        sa2_xyz1, sa2_inds1, sa2_grp_feats1, sa2_painted1 = self.sa2(sa1_xyz1, sa1_painted1, sa1_features1, bg1=True, wght1=1)        
        with tf.device('cpu'):
            sa2_xyz1, sa2_inds1, sa2_grp_feats1, sa2_painted1 = tf.identity(sa2_xyz1), tf.identity(sa2_inds1), tf.identity(sa2_grp_feats1), tf.identity(sa2_painted1)
        time_record.append(("SA2 sampling and grouping 1:", time.time()))                 
        
        
        if self.use_multiThr:
            sa1_features2 = future2.result()
            future1 = self._executor.submit(self.call_tflite, self.sa2_interpreter, sa2_grp_feats1, self.sa2_input_details, self.sa2_output_details)                
        else:
            sa2_features1 = self.call_tflite(self.sa2_interpreter, sa2_grp_feats1, self.sa2_input_details, self.sa2_output_details)
                
        time_record.append(("SA2 MLP 1:", time.time()))
        
        sa1_xyz = layers.Concatenate(axis=1)([sa1_xyz1, sa1_xyz2])
        sa1_features = layers.Concatenate(axis=1)([sa1_features1, sa1_features2])        

        # Biased FPS
        sa2_xyz2, sa2_inds2, sa2_grp_feats2, sa2_painted2 = self.sa2(sa1_xyz2, sa1_painted2, sa1_features2, bg1=True, wght1=4, xyz_ball=sa1_xyz, features_ball=sa1_features)        
        with tf.device('cpu'):
            sa2_xyz2, sa2_inds2, sa2_grp_feats2, sa2_painted2 = tf.identity(sa2_xyz2), tf.identity(sa2_inds2), tf.identity(sa2_grp_feats2), tf.identity(sa2_painted2)
        time_record.append(("SA2 sampling and grouping 2:", time.time()))        

        
        if self.use_multiThr:
            sa2_features1 = future1.result()
            future2 = self._executor.submit(self.call_tflite, self.sa2_interpreter, sa2_grp_feats2, self.sa2_input_details, self.sa2_output_details)                
        else:
            sa2_features2 = self.call_tflite(self.sa2_interpreter, sa2_grp_feats2, self.sa2_input_details, self.sa2_output_details)        
        time_record.append(("SA2 MLP 2:", time.time()))

        # ------------------------------- SA3-------------------------------        
         # Normal FPS
        sa3_xyz1, sa3_inds1, sa3_grp_feats1, sa3_painted1 = self.sa3(sa2_xyz1, sa2_painted1, sa2_features1, bg1=True, wght1=1)
        with tf.device('cpu'):
            sa3_xyz1, sa3_inds1, sa3_grp_feats1, sa3_painted1 = tf.identity(sa3_xyz1), tf.identity(sa3_inds1), tf.identity(sa3_grp_feats1), tf.identity(sa3_painted1)
        time_record.append(("SA3 sampling and grouping 1:", time.time()))

        
        if self.use_multiThr:            
            sa2_features2 = future2.result()
            future1 = self._executor.submit(self.call_tflite, self.sa3_interpreter, sa3_grp_feats1, self.sa3_input_details, self.sa3_output_details)
        else:
            sa3_features1 = self.call_tflite(self.sa3_interpreter, sa3_grp_feats1, self.sa3_input_details, self.sa3_output_details)
        time_record.append(("SA3 MLP 1:", time.time()))

        sa2_xyz = layers.Concatenate(axis=1)([sa2_xyz1, sa2_xyz2])
        sa2_features = layers.Concatenate(axis=1)([sa2_features1, sa2_features2])        

        # Second sampling, but Normal FPS
        sa3_xyz2, sa3_inds2, sa3_grp_feats2, sa3_painted2 = self.sa3(sa2_xyz2, sa2_painted2, sa2_features2, bg1=True, wght1=1, xyz_ball=sa2_xyz, features_ball=sa2_features)        
        with tf.device('cpu'):
            sa3_xyz2, sa3_inds2, sa3_grp_feats2, sa3_painted2 = tf.identity(sa3_xyz2), tf.identity(sa3_inds2), tf.identity(sa3_grp_feats2), tf.identity(sa3_painted2)
        time_record.append(("SA3 sampling and grouping 2:", time.time()))

        
        if self.use_multiThr:
            sa3_features1 = future1.result()
            future2 = self._executor.submit(self.call_tflite, self.sa3_interpreter, sa3_grp_feats2, self.sa3_input_details, self.sa3_output_details)
        else:
            sa3_features2 = self.call_tflite(self.sa3_interpreter, sa3_grp_feats2, self.sa3_input_details, self.sa3_output_details)            
        
        time_record.append(("SA3 MLP 2:", time.time()))

                     
        if self.use_multiThr:
            sa3_features2 = future2.result()               

        # Fuse two pointsets
        sa3_xyz = layers.Concatenate(axis=1)([sa3_xyz1, sa3_xyz2])
        sa3_features = layers.Concatenate(axis=1)([sa3_features1, sa3_features2])        
        sa3_painted = layers.Concatenate(axis=1)([sa3_painted1, sa3_painted2])

         # ------------------------------- SA4-------------------------------  
        sa4_xyz, sa4_inds, sa4_grp_feats, sa4_painted = self.sa4(sa3_xyz, sa3_painted, sa3_features, bg1=True, wght1=1)
        time_record.append(("SA4 sampling and grouping:", time.time()))             

        sa4_features = self.call_tflite(self.sa4_interpreter, sa4_grp_feats, self.sa4_input_details, self.sa4_output_details)

        time_record.append(("SA4 MLP:", time.time()))             
        
        # end_points['sa1_grouped_features1'] = sa1_grouped_features1        
        # end_points['sa1_grouped_features2'] = sa1_grouped_features2
        # end_points['sa2_grouped_features1'] = sa2_grouped_features1        
        # end_points['sa2_grouped_features2'] = sa2_grouped_features2
        # end_points['sa3_grouped_features1'] = sa3_grouped_features1        
        # end_points['sa3_grouped_features2'] = sa3_grouped_features2

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #print("========================== FP1 ===============================")
        fp1_features, fp1_grp_feats = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        
        #print("========================== FP2 ===============================")
        fp2_features, fp2_grp_feats = self.fp2(sa2_xyz, sa3_xyz, sa2_features, fp1_features)        

        
        end_points['fp2_features'] = fp2_features        
        end_points['fp2_xyz'] = sa2_xyz
        #num_seed = sa2_inds.shape[1]
        #end_points['fp2_inds'] = sa1_inds[:,0:num_seed] # indices among the entire input point clouds
        seed_inds1 = tf.gather(sa1_inds1, axis=1, indices=sa2_inds1, batch_dims=1)
        
        # Necessary if excluding first sampling points
        B = tf.shape(xyz)[0]
        N = tf.shape(xyz)[1]
        
        all_inds = tf.tile(tf.expand_dims(tf.range(N), 0), [B,1])
        rem_inds = tf.boolean_mask(all_inds, tf.logical_not(mask))
        rem_inds = tf.reshape(rem_inds, [B,-1])
        sa1_2_inds2 = tf.gather(sa1_inds2, axis=1, indices=sa2_inds2, batch_dims=1)
        seed_inds2 = tf.gather(rem_inds, indices=sa1_2_inds2, batch_dims=1)        

        end_points['fp2_inds'] = layers.Concatenate(axis=1)([seed_inds1, seed_inds2])      

        
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

