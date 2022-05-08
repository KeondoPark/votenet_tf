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
    def __init__(self, vote_factor, seed_feature_dim, model_config):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
            sep_coords: boolean
                If True, use separate layer for coordinate 
            use_tflite: boolean
                If True, use tflite
            tflite_name: string
                The name of tflite file
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim

        self.use_tflite = model_config['use_tflite']
        #self.sep_coords = model_config['sep_coords']
        self.q_gran = model_config['q_gran']
        self.use_fp_mlp = model_config['use_fp_mlp']

        if self.use_tflite:
            self.use_edgetpu = model_config['use_edgetpu']
            tflite_folder = model_config['tflite_folder']            

            if self.use_edgetpu:            
                tflite_file = 'voting_quant_edgetpu.tflite'
                from pycoral.utils.edgetpu import make_interpreter            
                self.interpreter = make_interpreter(os.path.join(ROOT_DIR,os.path.join(tflite_folder,tflite_file)))
            else:
                tflite_file = 'voting_quant.tflite'
                self.interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR,os.path.join(tflite_folder, tflite_file)))                             
            
            self.interpreter.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.conv0 = layers.Conv2D(filters=self.in_dim, kernel_size=1)
            self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1) #, kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Zeros())
            self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1) #, kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Zeros())            
            self.conv3 = layers.Conv2D(filters=(self.out_dim+3) * self.vote_factor, kernel_size=1) #, kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Zeros())

            #self.conv0 = layers.Dense(self.in_dim)
            #self.conv1 = layers.Dense(self.in_dim)        
            #self.conv2 = layers.Dense(self.in_dim)            
            #self.conv3 = layers.Dense((self.out_dim+3) * self.vote_factor)
            
            self.bn0 = layers.BatchNormalization(axis=-1)
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)
            act = model_config['activation'] if 'activation' in model_config['activation'] else 'relu6'
            if act == 'relu6':
                maxval = 6
            else:
                maxval = None
            self.relu0 = layers.ReLU(maxval)
            self.relu1 = layers.ReLU(maxval)
            self.relu2 = layers.ReLU(maxval)
        
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

        seed_xyz = layers.Reshape((num_seed, 1, 3))(seed_xyz)
        seed_features = layers.Reshape((num_seed, 1, seed_features.shape[-1]))(seed_features) # Expand to use Conv2D               

        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], seed_features)
            if len(self.input_details) > 1:
                self.interpreter.set_tensor(self.input_details[1]['index'], seed_xyz)
            self.interpreter.invoke()
            if self.q_gran == 'semantic':
                offset = self.interpreter.get_tensor(self.output_details[0]['index'])
                vote_xyz = seed_xyz + offset
                vote_features = self.interpreter.get_tensor(self.output_details[1]['index'])                                

            elif self.q_gran == 'channel':
                out = []
                for i in range((self.out_dim+3) * self.vote_factor):
                    out.append(self.output_details[i]['index'])

                offset = layers.Concatenate(axis=-1)(out[:3])
                vote_xyz = seed_xyz + offset

                residual_features = layers.Concatenate(axis=-1)(out[3:-1])
                net0 = out[-1]
                net0 = layers.Reshape((num_seed, self.vote_factor, net0.shape[-1]))(net0)
                vote_features = net0 + residual_features 

            else:
                net = self.interpreter.get_tensor(self.output_details[0]['index'])
                net0 = self.interpreter.get_tensor(self.output_details[1]['index'])

                offset = net[:,:,:,0:3]            
                vote_xyz = seed_xyz + offset

                residual_features = layers.Reshape((num_seed, self.vote_factor, self.out_dim))(net[:,:,:,3:]) # (batch_size, num_seed, vote_factor, out_dim)
                vote_features = net0 + residual_features                

        else:
            if not self.use_fp_mlp:
                net0 = self.relu0(self.bn0(self.conv0(seed_features))) #(B, num_seed, 1, in_dim)
            else:
                net0 = seed_features
            net = self.relu1(self.bn1(self.conv1(net0))) 
            net = self.relu2(self.bn2(self.conv2(net))) 
            net = self.conv3(net)

            offset = net[:,:,:,0:3]            
            residual_features = layers.Reshape((num_seed, self.vote_factor, self.out_dim))(net[:,:,:,3:]) # (batch_size, num_seed, vote_factor, out_dim)        
            
            
            net0 = layers.Reshape((num_seed, self.vote_factor, net0.shape[-1]))(net0)
            vote_features = net0 + residual_features 
            vote_xyz = seed_xyz + offset 
        
        vote_xyz = layers.Reshape((num_vote, 3))(vote_xyz)
        vote_features = layers.Reshape((num_vote, self.out_dim))(vote_features)

        #np.savetxt(os.path.join(ROOT_DIR, '..', 'votenet_test','tf_voting_xyz.txt'), vote_xyz[0].numpy())
        #np.savetxt(os.path.join(ROOT_DIR, '..', 'votenet_test','tf_voting_features.txt'), vote_features[0].numpy())

        return vote_xyz, vote_features
 
if __name__=='__main__':
    net = VotingModule(2, 256)
    xyz = np.random.random((8,1024,3)).astype('float32')
    features = np.random.random((8,1024,256)).astype('float32')
    xyz, features = net(tf.constant(xyz), tf.constant(features))
    print('xyz', xyz.shape)
    print('features', features.shape)
