# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Voting module: generate votes from XYZ and features of seed points.

Date: July, 2019
Author: Charles R. Qi and Or Litany
'''

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class VotingModule(layers.Layer):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = layers.Conv1D(filters=self.in_dim, kernel_size=1)
        self.conv2 = layers.Conv1D(filters=self.in_dim, kernel_size=1)
        self.conv3 = layers.Conv1D(filters=(3+self.out_dim) * self.vote_factor, kernel_size=1) 
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu1 = layers.Activation('relu')
        self.relu2 = layers.Activation('relu')
        
    def call(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) tensor
            seed_features: (batch_size, num_seed, feature_dim) tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, num_seed*vote_factor, vote_feature_dim)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = self.relu1(self.bn1(self.conv1(seed_features))) 
        net = self.relu2(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, (3+out_dim)*vote_factor, num_seed)                
        
        print(net)

        net = tf.reshape(net, shape=tf.convert_to_tensor([num_seed, self.vote_factor, 3+self.out_dim]))
        #net = tf.expand_dims(net, axis=1)
        print(net)

        offset = net[:,:,:,0:3]
        vote_xyz = tf.expand_dims(seed_xyz, axis = 2) + offset
        vote_xyz = tf.reshape(vote_xyz, shape=[batch_size, num_vote, 3])
        
        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)        
        vote_features = tf.expand_dims(seed_features, axis=2) + residual_features
        vote_features = tf.reshape(vote_features, shape=[batch_size, num_vote, self.out_dim])

        return vote_xyz, vote_features
 
if __name__=='__main__':
    net = VotingModule(2, 256)
    xyz = np.random.random((8,1024,3)).astype('float32')
    features = np.random.random((8,1024,256)).astype('float32')
    xyz, features = net(tf.constant(xyz), tf.constant(features))
    print('xyz', xyz.shape)
    print('features', features.shape)
