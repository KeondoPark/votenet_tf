# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Testing customized ops. '''

import tensorflow as tf
from tensorflow.keras import layers

from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
from tf_ops.interpolation import tf_interpolate
import pointnet2_utils_tf
import tf_utils
import numpy as np
from typing import List


import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#import pointnet2_utils_tf

'''
def test_interpolation_grad():
    batch_size = 1
    feat_dim = 2
    m = 4
    feats = torch.randn(batch_size, feat_dim, m, requires_grad=True).float().cuda()
    
    def interpolate_func(inputs):
        idx = torch.from_numpy(np.array([[[0,1,2],[1,2,3]]])).int().cuda()
        weight = torch.from_numpy(np.array([[[1,1,1],[2,2,2]]])).float().cuda()
        interpolated_feats = pointnet2_utils.three_interpolate(inputs, idx, weight)
        return interpolated_feats
    
    assert (gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1))
'''

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
            ret_unique_cnt: bool = False
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

        if npoint is not None:
            self.grouper = pointnet2_utils_tf.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils_tf.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = tf_utils.SharedMLP(mlp_spec, bn=bn)        
        self.max_pool = layers.MaxPooling2D(pool_size=(1, self.nsample), strides=1, data_format="channels_last")
        self.avg_pool = layers.AveragePooling2D(pool_size=(1, self.nsample), strides=1, data_format="channels_last")        

    def call(self, xyz, features, inds=None, sample_type = 'fps_light'):
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
            
            if sample_type == 'fps':
                #start = time.time()
                inds = tf_sampling.farthest_point_sample(self.npoint, xyz)                
                #inds, batch_distances = pointnet2_utils.fps_light(xyz, self.npoint)
                #end = time.time()
                #print("Runtime for FPS original", end - start)
        else:
            assert(inds.shape[1] == self.npoint)   
        
        print(inds)
        #start = time.time()     
        new_xyz = tf_sampling.gather_point(
            xyz, inds
        ) if self.npoint is not None else None
        #end = time.time()
        #print("Runtime for gather_op original", end - start)
        print(new_xyz)

        if not self.ret_unique_cnt:
        #if not self.ret_unique_cnt:
            #grouped_features, grouped_xyz = self.grouper(
            grouped_features, ball_query_idx, grouped_xyz = self.grouper(
                xyz, new_xyz, features
                #xyz, new_xyz, batch_distances, inds, features
            )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)
        else:
            grouped_features, ball_query_idx, grouped_xyz, unique_cnt = self.grouper(
            #grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, npoint, nsample, C+3), (B,npoint,nsample), (B,npoint,nsample,3)

        new_features = self.mlp_module(
                grouped_features
            )  # (B, npoint, nsample, mlp[-1])

        print(grouped_features)
        print(new_features)
        print(tf.shape(new_features))
        print(tf.shape(new_features)[2])

        if self.pooling == 'max':
            #new_features = layers.MaxPooling2D(pool_size=(1, tf.shape(new_features)[2]), strides=1, data_format="channels_last")(new_features)  # (B, npoint, 1, mlp[-1])
            new_features = self.max_pool(new_features)  # (B, npoint, 1, mlp[-1])
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
        new_features = tf.squeeze(new_features, axis=-2)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            #return new_xyz, new_features, inds
            return new_xyz, new_features, inds, ball_query_idx
        else:
            return new_xyz, new_features, inds, unique_cnt

class PointnetFPModule(layers.Layer):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = tf_utils.SharedMLP(mlp, bn=bn)

    def call(
            self, unknown, known,
            unknow_feats, known_feats,
            grouped_xyz, inds
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
            dist_recip = 1.0 / (dist + 1e-8)
            print("=====Distance and Index=====")
            print(dist)
            print(idx)
            norm = tf.reduce_sum(dist_recip, axis=2, keepdims=True)
            weight = dist_recip / norm  # (B, n, 3)
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
            new_features = tf.concat([interpolated_feats, unknow_feats],
                                   axis=2)  #(B, n, C2 + C1)
        else:
            new_features = interpolated_feats

        new_features = tf.expand_dims(new_features, axis=-2)
        new_features = self.mlp(new_features)

        return tf.squeeze(new_features, axis=-2)            


if __name__=='__main__':
    #test_interpolation_grad()
    #xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=False)
    xyz = tf.constant([[[ 0.0234, -0.9751,  1.4011],
         [-0.2499,  1.6626, -0.3811],
         [ 0.8417, -0.5145,  0.7864],
         [ 0.2058,  1.1784,  0.3849],
         [ 0.5544, -0.6031, -0.0555],
         [-1.7716,  0.5014, -0.6067],
         [ 2.5301,  0.5504, -0.4435],
         [-1.1078,  0.3823,  0.2507],
         [-0.7827,  0.9522,  0.4905]],

        [[ 0.7612, -1.1062, -1.7496],
         [ 1.1352,  0.5164,  0.7210],
         [ 0.8828,  1.0338, -0.7701],
         [-0.4920,  0.8126, -0.2284],
         [ 0.1904, -2.3758, -0.3887],
         [ 0.0711, -2.1078, -0.5276],
         [ 0.4946,  1.5537,  0.2516],
         [ 0.2828,  0.6082, -1.1267],
         [-1.0092, -2.4192, -0.2919]]])

    sa1 = PointnetSAModuleVotes(mlp=[4,4,4], npoint=3, radius=2, nsample=2)

    features = tf.constant([[[-0.6665,  0.0526],
         [ 0.0153,  1.0042],
         [-0.9345, -0.4614],
         [ 0.3896,  1.3611],
         [-0.6665,  0.0526],
         [ 0.0153,  1.0042],
         [-0.9345, -0.4614],
         [ 0.3896,  1.3611],
         [-0.6665,  0.0526]],

        [[ 1.6132, -0.4626],
         [ 0.2349, -1.2845],
         [ 0.8065, -0.4818],
         [-0.2079, -2.4726],
         [ 1.6132, -0.4626],
         [ 0.2349, -1.2845],
         [ 0.8065, -0.4818],
         [-0.2079, -2.4726],
         [ 1.6132, -0.4626]]])

    new_xyz, new_features, fps_inds, ball_query_idx = sa1(xyz, features, sample_type='fps')

    print("======Sampled points=====")
    print(xyz)
    print("======Sampled features=====")
    print(features)
    print("======Sampled point index=====")
    print(fps_inds)
    print("======Ball query index=====")
    print(ball_query_idx)

    fp1 = PointnetFPModule(mlp=[2,2,2])
    interp_features = fp1(xyz, new_xyz, features, new_features, ball_query_idx, fps_inds)    

    print("======Interpolated features=====")
    print(interp_features)




    '''
    inds = tf_sampling.farthest_point_sample(3, xyz)
    print("======Point cloud information=====")
    print(xyz)
    print("======Sampling center points distance from avg=====")
    print(inds)
    #print(dist)
        
    new_xyz = tf_sampling.gather_point(xyz, inds)
    print("======Sampling and gathering distance from avg=====")
    print(new_xyz)
    
    #Ball query
    inds3, pts_cnt = tf_grouping.query_ball_point(2, 2, xyz, new_xyz)
    print("======Ball query=====")
    print(inds3)

    #Ball query group
    xyz_grouped = tf_grouping.group_point(xyz, inds3)
    print(xyz_grouped)


    
    #Inverse ball query
    inds4 = pointnet2_utils.inv_ball_query_cpu(xyz, new_xyz, inds3, inds)
    print("======Inverse Ball query=====")
    print(inds4)

    #features = Variable(torch.randn(2, 4, 2).cuda(), requires_grad=False)
    features = torch.Tensor([[[-0.6665,  0.0526],
         [ 0.0153,  1.0042],
         [-0.9345, -0.4614],
         [ 0.3896,  1.3611]],

        [[ 1.6132, -0.4626],
         [ 0.2349, -1.2845],
         [ 0.8065, -0.4818],
         [-0.2079, -2.4726]]]).to('cuda:0')
    print("======Randomly generated features=====")
    print(features)
    
    #Interpolate based on inverse ball query result
    prop_features = pointnet2_utils.inv_interpolate_cpu(features, inds4)
    print("======Propagated features=====")
    print(prop_features)
    '''