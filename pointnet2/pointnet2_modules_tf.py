# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Pointnet2 layers.
Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
Extended with the following:
1. Uniform sampling in each local region (sample_uniformly)
2. Return sampled points indices to support votenet.
'''
import tensorflow as tf
from tensorflow.keras import layers

from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
from tf_ops.interpolation import tf_interpolate


import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

import pointnet2_utils_tf
import tf_utils
from typing import List

import time

class PointnetSAModuleVotes(layers.Layer):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            model_config = None,
            layer_name = 'sa1'
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.layer_name = layer_name

        if npoint is not None:
            self.grouper = pointnet2_utils_tf.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils_tf.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3  
        
        self.use_tflite = model_config['use_tflite']
        
        if self.use_tflite:
            self.use_edgetpu = model_config['use_edgetpu']
            tflite_folder = model_config['tflite_folder']
            
            if self.use_edgetpu:            
                tflite_file = layer_name + '_quant_edgetpu.tflite'
                from pycoral.utils.edgetpu import make_interpreter
                self.interpreter = make_interpreter(os.path.join(ROOT_DIR,os.path.join(tflite_folder, tflite_file)))
            else:
                tflite_file = layer_name + '_quant.tflite'
                self.interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR,os.path.join(tflite_folder, tflite_file)))                             
            
            self.interpreter.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            act = model_config['activation'] if 'activation' in model_config['activation'] else 'relu6'
            self.mlp_module = tf_utils.SharedMLP(mlp_spec, bn=bn, activation=act, input_shape=[npoint, nsample, mlp_spec[0]])        
            #self.max_pool = layers.MaxPooling2D(pool_size=(1, self.nsample), strides=1, data_format="channels_last")
            self.max_pool = layers.MaxPooling2D(pool_size=(1, 16), strides=(1,16), data_format="channels_last")
            self.max_pool2 = layers.MaxPooling2D(pool_size=(1, int(self.nsample/16)), strides=(1,int(self.nsample/16)), data_format="channels_last")
            self.avg_pool = layers.AveragePooling2D(pool_size=(1, self.nsample), strides=1, data_format="channels_last")        

    def call(self, xyz, features, time_record, inds=None, bg=False):
        r"""
        Parameters
        ----------
        xyz : (B, N, 3) tensor of the xyz coordinates of the features
        features : (B, N, C) tensor of the descriptors of the the features
        inds : (Optinal)(B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : (B, npoint, 3) tensor of the new features' xyz
        new_features : (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: (B, npoint) tensor of the inds
        ball_quer_idx: (B, npoint, nsample) Index of ball queried points
        """        
        if inds is None:           
            if bg:
                isPainted = tf.cast(features[:,:,0], dtype=tf.int32) 
                inds, isPainted = tf_sampling.farthest_point_sample_bg(self.npoint, xyz, isPainted, weight=1)
                xyz = xyz[:,:,:3]
            else:
                inds = tf_sampling.farthest_point_sample(self.npoint, xyz)                
            #inds, batch_distances = pointnet2_utils.fps_light(xyz, self.npoint)            
        else:
            assert(inds.shape[1] == self.npoint)   

        
        new_xyz = tf_sampling.gather_point(
            xyz, inds
        ) if self.npoint is not None else None
        
        

        if not self.ret_unique_cnt:
            #grouped_features, grouped_xyz = self.grouper(
            grouped_features= self.grouper(
                xyz, new_xyz, features
                #xyz, new_xyz, batch_distances, inds, features
            )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)
        else:
            grouped_features, unique_cnt = self.grouper(
            #grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)        
        time_record.append((f"{self.layer_name} Runtime for Sampling and Grouping:", time.time()))
         
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], grouped_features)
            self.interpreter.invoke()
            new_features = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:              
            new_features = self.mlp_module(
                    grouped_features
                )  # (B, npoint, nsample, mlp[-1])
            #end = time.time()
            #print("Runtime for shared MLP", end - start)

            if self.pooling == 'max':
                #new_features = layers.MaxPooling2D(pool_size=(1, tf.shape(new_features)[2]), strides=1, data_format="channels_last")(new_features)  # (B, npoint, 1, mlp[-1])
                #new_features = self.max_pool(new_features)  # (B, npoint, 1, mlp[-1])
                if self.nsample == 16:
                    new_features = self.max_pool(new_features)  # (B, npoint, 1, mlp[-1])
                elif self.nsample > 16:
                    new_features = self.max_pool2(self.max_pool(new_features))  # (B, npoint, 1, mlp[-1])
            elif self.pooling == 'avg':
                #new_features = layers.AvgPooling2D(pool_size=(1, tf.shape(new_features)[2]), strides=1, data_format="channels_last")(new_features) # (B, npoint, 1, mlp[-1])
                new_features = self.avg_pool(new_features)  # (B, npoint, 1, mlp[-1])
            '''
            elif self.pooling == 'rbf': 
                # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
                # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
                rbf = tf.math.exp(-1 * grouped_xyz.pow(2).sum(1,keepdim=False) / (self.sigma**2) / 2) # (B, npoint, nsample)
                new_features = tf.reduce_sum(new_features * rbf.unsqueeze(1), axis=-1, keepdims=True) / float(self.nsample) # (B, mlp[-1], npoint, 1)
            '''
        #new_features = tf.squeeze(new_features, axis=-2)  # (B, npoint, mlp[-1])
        new_features = layers.Reshape((self.npoint, new_features.shape[-1]))(new_features)        
        time_record.append((f"{self.layer_name} Runtime for shared MLP:", time.time()))

        if not self.ret_unique_cnt:
            #return new_xyz, new_features, inds
            return new_xyz, new_features, inds, grouped_features
        else:
            return new_xyz, new_features, inds, unique_cnt


class SamplingAndGrouping(layers.Layer):
    ''' Only sampling and grouping part in PointnetSAModuleVotes '''

    def __init__(
            self,
            *,            
            npoint: int = None,
            radius: float = None,
            nsample: int = None,            
            use_xyz: bool = True,                        
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample        
        self.mlp_module = None
        self.use_xyz = use_xyz                
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = pointnet2_utils_tf.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils_tf.GroupAll(use_xyz, ret_grouped_xyz=True)
        
    def call(self, xyz, isPainted, features, inds=None, new_xyz=None, ball_inds=None, bg1=False, bg2=False, wght1=1, wght2=1, xyz_ball=None, features_ball=None):
    #def call(self, xyz, features, inds=None, bg=False, wght=1, isFront=0, xyz_ball=None, features_ball=None):
        r"""
        Parameters
        ----------
        xyz : (B, N, 3) tensor of the xyz coordinates of the features
        features : (B, N, C) tensor of the descriptors of the the features
        inds : (Optinal)(B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : (B, npoint, 3) tensor of the new features' xyz        
        inds: (B, npoint) tensor of the inds
        ball_quer_idx: (B, npoint, nsample) Index of ball queried points
        grouped_features: (B, npoint, nsample, C+3) Required to create tflite
        """
        
        if inds is None:                        
            if bg2:                                    
                #inds = tf_sampling.farthest_point_sample_bg(self.npoint, xyz, wght, isFront)                                
                inds, isPainted = tf_sampling.farthest_point_sample_bg2(self.npoint, xyz, isPainted, wght1, wght2)                                                          
            elif bg1:
                inds, isPainted = tf_sampling.farthest_point_sample_bg(self.npoint, xyz, isPainted, wght1)
            else:
                inds = tf_sampling.farthest_point_sample(self.npoint, xyz)     
        else:
            assert(inds.shape[1] == self.npoint)   
        
        if new_xyz is None:
            new_xyz = tf_sampling.gather_point(
                xyz, inds
            ) if self.npoint is not None else None

        if not self.ret_unique_cnt:      
            if xyz_ball is None and features_ball is None:
                grouped_features = self.grouper(
                    xyz, new_xyz, features, ball_inds=ball_inds            
                )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)
            else:
                grouped_features = self.grouper(
                    xyz_ball, new_xyz, features_ball, ball_inds=ball_inds        
                )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)
        else:
            grouped_features, unique_cnt = self.grouper(            
                xyz, new_xyz, features
            )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)
        

        if not self.ret_unique_cnt:            
            return new_xyz, inds, grouped_features, isPainted
        else:
            return new_xyz, inds, unique_cnt, isPainted

class PointnetMLP(layers.Layer):
    ''' Only shareMLP and maxpooling in PointnetSAModuleVotes '''

    def __init__(
            self,
            *,            
            npoint: int = None,            
            nsample: int = None,            
            use_xyz: bool = True,                                    
            mlp: List[int], 
            bn: bool = True, 
            model_config = None
    ):
        super().__init__()

        self.nsample = nsample        
        self.npoint = npoint

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3  
        act = model_config['activation'] if 'activation' in model_config['activation'] else 'relu6'
        self.mlp_module = tf_utils.SharedMLP(mlp_spec, bn=bn, activation=act, input_shape=[npoint, nsample, mlp_spec[0]])        
        self.max_pool = layers.MaxPooling2D(pool_size=(1, 16), strides=(1,16), data_format="channels_last")
        self.max_pool2 = layers.MaxPooling2D(pool_size=(1, int(self.nsample/16)), strides=(1,int(self.nsample/16)), data_format="channels_last")        
        
    def call(self, features):
        new_features = self.mlp_module(features)

        if self.nsample == 16:
            new_features = self.max_pool(new_features)  # (B, npoint, 1, mlp[-1])
        elif self.nsample > 16:
            new_features = self.max_pool2(self.max_pool(new_features))  # (B, npoint, 1, mlp[-1])

        new_features = layers.Reshape((-1, new_features.shape[-1]))(new_features)

        return new_features

class PointnetFPModule(layers.Layer):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True, m: int, model_config = None, layer_name = 'fp1'):
        super().__init__()
        self.use_fp_mlp = model_config['use_fp_mlp']
        self.use_tflite = model_config['use_tflite']
        
        if self.use_fp_mlp:
            if self.use_tflite:
                self.use_edgetpu = model_config['use_edgetpu']
                tflite_folder = model_config['tflite_folder']                

                if self.use_edgetpu:
                    from pycoral.utils.edgetpu import make_interpreter            
                    tflite_file = layer_name  + '_quant_edgetpu.tflite'
                    self.interpreter = make_interpreter(os.path.join(ROOT_DIR, tflite_folder, tflite_file))
                else:
                    tflite_file = layer_name  + '_quant.tflite'
                    self.interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR, tflite_folder, tflite_file))
                
                self.interpreter.allocate_tensors()

                # Get input and output tensors.
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            else:
                act = model_config['activation'] if 'activation' in model_config['activation'] else 'relu6'
                self.mlp = tf_utils.SharedMLP(mlp, bn=bn, activation=act, input_shape=[m,1,mlp[0]])
        else: 
            self.mlp = None
        

    def call(
            self, unknown, known,
            unknow_feats, known_feats
    ):
        r"""
        Parameters
        ----------
        unknown : (B, n, 3) tensor of the xyz positions of the unknown features
        known : (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : (B, n, C1) tensor of the features to be propigated to
        known_feats : (B, m, C2) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, n, mlp[-1]) tensor of the features of the unknown features
        """

        if known is not None:            
            #start = time.time()
            dist, idx = tf_interpolate.three_nn(unknown, known)
            dist_recip = tf.divide(tf.constant(1.0, dtype=tf.float32), (dist + 1e-8))
            norm = tf.reduce_sum(dist_recip, axis=2, keepdims=True)
            weight = tf.divide(dist_recip, norm)  # (B, n, 3)
            #end = time.time()
            #print("Runtime for Threenn original", end - start)
            
            #start = time.time() 
            interpolated_feats = tf_interpolate.three_interpolate(
                known_feats, idx, weight
            )
            #end = time.time()
            #print("Runtime for Inverse three_interpolate original: ", end - start)

        else:
            interpolated_feats = tf.tile(known_feats, [1, tf.shape(unknow_feats)[1] / tf.shape(known_feats)[1], 1])

        if unknow_feats is not None:
            prop_features = layers.concatenate([interpolated_feats, unknow_feats], axis=2)  #(B, n, C2 + C1)
        else:
            prop_features = interpolated_feats
        
        
        #new_features = tf.expand_dims(new_features, axis=-2)
        if self.use_fp_mlp:
            print("FP MLP!")
            prop_features = layers.Reshape((prop_features.shape[1], 1, prop_features.shape[2]))(prop_features)
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], prop_features)
                self.interpreter.invoke()
                res_features = self.interpreter.get_tensor(self.output_details[0]['index'])
            else:
                res_features = self.mlp(prop_features)
            return layers.Reshape((res_features.shape[1], res_features.shape[-1]))(res_features), prop_features
        else:
            return prop_features, prop_features
        


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, npoint, n_heads, embed_dim):
        super().__init__()
        self.npoint = npoint
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.emb = layers.Dense(embed_dim * n_heads * 3, use_bias=False)         
        self.avg_pool = layers.AveragePooling2D(pool_size=(self.n_heads,1), strides=(self.n_heads,1))
        d = tf.cast(embed_dim, dtype=tf.float32)    
        self.scaling = 1/tf.math.sqrt(d)

    def call(self, inputs):    
        """
        inputs: (B, num_seed, features)
        """
        #num_seed = tf.shape(inputs)[1]
        embedding = self.emb(inputs) # (B, n, d * h * 3)            
        heads = layers.Reshape((self.npoint, self.embed_dim, self.n_heads, 3))(embedding) #(B, n, d, h, 3)
        

        heads = tf.transpose(heads, perm=[0,4,3,1,2]) # (B, 3, h, n, d)
        q = heads[:,0,:,:,:] #(B, h, n, d)
        k = heads[:,1,:,:,:] #(B, h, n, d)
        v = heads[:,2,:,:,:] #(B, h, n, d)
        
        qk = tf.matmul(q, k, transpose_b=True) # (B, h, n, n)    
        qk = tf.keras.backend.softmax(qk) * self.scaling # (B, h, n, n)            
        attn = self.avg_pool(qk) #(B, 1, n, n)
        attn = layers.Reshape((self.npoint, self.npoint))(attn)        
        
        output = tf.matmul(qk, v) # (B, h, n, d)
        output = tf.transpose(output, perm=[0,2,1,3]) #(B, n, h, d)
        output = layers.Reshape((self.npoint, self.embed_dim * self.n_heads))(output)
        return attn, output

class MultiheadAttention2(tf.keras.layers.Layer):
    def __init__(self, npoint, n_heads, embed_dim):
        super().__init__()
        self.npoint = npoint
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.emb_q = [layers.Dense(embed_dim, use_bias=False) for _ in range(n_heads)]
        self.emb_k = [layers.Dense(embed_dim, use_bias=False) for _ in range(n_heads)]
        self.emb_v = [layers.Dense(embed_dim, use_bias=False) for _ in range(n_heads)]
        self.avg_pool = layers.AveragePooling2D(pool_size=(self.n_heads,1), strides=(self.n_heads,1))
        d = tf.cast(embed_dim, dtype=tf.float32)    
        self.scaling = 1/tf.math.sqrt(d)

    def call(self, inputs):    
        """
        inputs: (B, num_seed, features)
        """
        #num_seed = tf.shape(inputs)[1]
        embedding_q = [emb(inputs) for emb in self.emb_q] # (B, n, d) * h
        embedding_k = [emb(inputs) for emb in self.emb_k] # (B, n, d) * h
        embedding_v = [emb(inputs) for emb in self.emb_v] # (B, n, d) * h
        attn = [tf.keras.backend.softmax(tf.matmul(q, k, transpose_b=True)) for q, k in zip(embedding_q, embedding_k)]

        qk_list = [a * self.scaling for a in attn] # (B, n, n) * h

        qkv_list = [tf.matmul(qk, v) for qk, v in zip(qk_list, embedding_v)] #(B, n, d) * h

        output = layers.Concatenate(axis=-1)(qkv_list)        #(B, n, d * h)
        return attn, output

class SamplingAndAttention(layers.Layer):
    ''' Only sampling and grouping part in PointnetSAModuleVotes '''

    def __init__(
            self,
            *,            
            npoint: int = None,  
            nsample: int = None, 
            radius: float = None, 
            use_xyz: bool = True,                        
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False            
    ):
        super().__init__()

        self.npoint = npoint  
        self.nsample = nsample 
        self.radius = radius     
        self.mlp_module = None
        self.use_xyz = use_xyz                
        self.normalize_xyz = normalize_xyz        

        if npoint is not None:
            self.grouper = pointnet2_utils_tf.QueryAndGroup(radius=self.radius, nsample=self.nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=False)
        else:
            self.grouper = pointnet2_utils_tf.GroupAll(use_xyz, ret_grouped_xyz=True)

        self.attention = MultiheadAttention2(self.npoint, n_heads=8, embed_dim=16)
        
        
    def call(self, xyz, features, inds=None, sample_type = 'fps_light', bg=False, wght=1):
        r"""
        Parameters
        ----------
        xyz : (B, N, 3) tensor of the xyz coordinates of the features
        features : (B, N, C) tensor of the descriptors of the the features
        inds : (Optinal)(B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : (B, npoint, 3) tensor of the new features' xyz        
        inds: (B, npoint) tensor of the inds
        ball_quer_idx: (B, npoint, nsample) Index of ball queried points
        grouped_features: (B, npoint, nsample, C+3) Required to create tflite
        """
        
        if inds is None:            
            if sample_type == 'fps':
                if bg:                    
                    inds = tf_sampling.farthest_point_sample_bg(self.npoint, xyz, wght)
                    xyz = xyz[:,:,:3]                    
                else:
                    inds = tf_sampling.farthest_point_sample(self.npoint, xyz)     
        else:
            assert(inds.shape[1] == self.npoint)   

        #start = time.time()     
        new_xyz = tf_sampling.gather_point(
            xyz, inds
        ) if self.npoint is not None else None
        #end = time.time()
        #print("Runtime for gather_op original", end - start)

        attn, features = self.attention(features)
        attn = tf.gather(attn, axis=1, indices=inds, batch_dims=1)
        print("Attention shape", attn.shape)

        grouped_features, ball_query_idx, grouped_xyz = self.grouper(
            xyz, new_xyz, features, knn=False
        )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)

        #features = grouped_features[:,:,:,3:]
        #attn, features = self.attention(features)
        #grouped_features = tf.concat([grouped_xyz, features], axis=-1)
               
        return new_xyz, inds, ball_query_idx, grouped_features
        