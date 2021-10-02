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
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class VotingModule(layers.Layer):
    def __init__(self, vote_factor, seed_feature_dim, use_tflite=False, tflite_name=None):
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
        self.use_tflite = use_tflite

        if self.use_tflite:
            #from pycoral.utils.edgetpu import make_interpreter
            self.interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR,os.path.join("tflite_models", tflite_name)))                             
            #self.interpreter = make_interpreter(os.path.join(ROOT_DIR,os.path.join("tflite_models",tflite_name)))
            self.interpreter.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            #self.conv1 = layers.Conv1D(filters=self.in_dim, kernel_size=1)        
            #self.conv2 = layers.Conv1D(filters=self.in_dim, kernel_size=1)
            #self.conv3 = layers.Conv1D(filters=(3+self.out_dim) * self.vote_factor, kernel_size=1)
            self.conv0 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1)        
            self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv3 = layers.Conv2D(filters=(3+self.out_dim) * self.vote_factor, kernel_size=1) 
            self.bn0 = layers.BatchNormalization(axis=-1)
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)
            self.relu0 = layers.ReLU(6)
            self.relu1 = layers.ReLU(6)
            self.relu2 = layers.ReLU(6)
        
    def call(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) tensor
            seed_features: (batch_size, num_seed, feature_dim) tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, num_seed*vote_factor, vote_feature_dim)
        """        
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor

        seed_features = layers.Reshape((num_seed, 1, seed_features.shape[-1]))(seed_features) # Expand to use Conv2D 
              

        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], seed_features)
            self.interpreter.invoke()
            res = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            net0 = self.relu0(self.bn0(self.conv0(seed_features)))
            net = self.relu1(self.bn1(self.conv1(net0))) 
            net = self.relu2(self.bn2(self.conv2(net))) 
            net = self.conv3(net) # (batch_size, num_seed, (3+out_dim)*vote_factor)

            net = layers.Reshape((num_seed, self.vote_factor, 3+self.out_dim))(net)

            residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)        
            #vote_features = tf.expand_dims(seed_features, axis=2) + residual_features
            net0 = layers.Reshape((num_seed, self.vote_factor, net0.shape[-1]))(net0)
            vote_features = net0 + residual_features              

            offset = net[:,:,:,0:3]  
            res = layers.Concatenate(axis=-1)([offset, vote_features])

        # Return from expansion
        #net = layers.Reshape((num_seed, self.vote_factor * (3+self.out_dim)))(net)    
        
        # No need for transpose in Tensorflow, because it uses H,W,C indexing        
        
        offset = res[:,:,:,0:3]
        #vote_xyz = tf.expand_dims(seed_xyz, axis = 2) + offset
        seed_xyz = layers.Reshape((num_seed, 1, 3))(seed_xyz)
        vote_xyz = seed_xyz + offset
        vote_xyz = layers.Reshape((num_vote, 3))(vote_xyz)

        vote_features = res[:,:,:,3:]       
        vote_features = layers.Reshape((num_vote, self.out_dim))(vote_features)

        return vote_xyz, vote_features
 
if __name__=='__main__':
    net = VotingModule(2, 256)
    xyz = np.random.random((8,1024,3)).astype('float32')
    features = np.random.random((8,1024,256)).astype('float32')
    xyz, features = net(tf.constant(xyz), tf.constant(features))
    print('xyz', xyz.shape)
    print('features', features.shape)