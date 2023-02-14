# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import tf_utils
import sys

import tensorflow as tf
from tensorflow.keras import layers

from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
from tf_ops.interpolation import tf_interpolate


import time

try:
    import builtins
except:
    import __builtin__ as builtins

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *
'''
class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)
'''

class QueryAndGroup(layers.Layer):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    #def forward(self, xyz, new_xyz, batch_distances, inds, features=None):
    def call(self, xyz, new_xyz, features=None, ball_inds=None, knn=False, attn=None, run_cpu=False):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : xyz coordinates of the features (B, N, 3)
        new_xyz : centriods (B, npoint, 3)
        features : Descriptors of the features (B, N, C)

        Returns
        -------
        new_features : (B, 3 + C, npoint, nsample) tensor
        """

        
        
        if ball_inds is None:
            if not knn:
                if run_cpu:
                    idx, pts_cnt = tf_grouping.query_ball_point_cpu(self.radius, self.nsample, xyz, new_xyz)
                else:
                    idx, pts_cnt = tf_grouping.query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            else:
                idx, _ = tf_grouping.knn_with_attention(self.nsample, xyz, new_xyz, attn)
        else:
            idx = ball_inds
            
        if self.sample_uniformly:
            unique_cnt = tf.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = tf.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = tf.random.int(shape=[self.nsample - num_unique], minval=0, maxval=num_unique, dtype=tf.dtypes.int32)
                    all_ind = tf.concat([unique_ind, unique_ind[sample_ind]], axis=0)
                    idx[i_batch, i_region, :] = all_ind
        start = time.time()
        if run_cpu:
            grouped_xyz = tf_grouping.group_point_cpu(xyz, idx)  # (B, npoint, nsample, 3)        
        else:
            grouped_xyz = tf_grouping.group_point(xyz, idx)  # (B, npoint, nsample, 3)        
        grouped_xyz -= tf.expand_dims(new_xyz, axis=-2)
        
        if self.normalize_xyz and not knn:            
            grouped_xyz = tf.divide(grouped_xyz, self.radius)
        
        #dist = tf.math.sqrt(tf.reduce_sum(grouped_xyz * grouped_xyz, axis=-1, keepdims=True))        

        if features is not None:
            if run_cpu:
                grouped_features = tf_grouping.group_point_cpu(features, idx)
            else:
                grouped_features = tf_grouping.group_point(features, idx)
            #grouped_features = grouping_operation_nocuda(features, idx)
            if self.use_xyz:
                new_features = tf.concat(
                    [grouped_xyz, grouped_features], axis=-1
                )  # (B, npoint, nsample, C + 3)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        #print("Runtime for group_point: ", time.time() - start)        
        
        if self.ret_grouped_xyz:
            ret = new_features            
        
        if self.ret_unique_cnt:
            ret = new_features, unique_cnt            
        
        return ret


class GroupAll(layers.Layer):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = tf.concat(
                    [grouped_xyz, grouped_features], axis=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
