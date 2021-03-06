# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules_tf import PointnetSAModuleVotes, PointnetSAModuleVotes_NoMLP
import pointnet2_utils_tf
from tf_ops.sampling import tf_sampling
import tf_utils

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    """
    Args:
        net: (B, num_proposal, 2+3+num_heading_bin*2+num_size_cluster*4)
        end_points: dictionary of 'aggregated_vote_xyz', 'aggregated_vote_inds', 
                                  'fp2_xyz', 'fp2_features', 'seed_inds', 'seed_xyz', 'seed_features',
                                  'vote_xyz', 'vote_features'
    return:
        Add 'objectness_scores', 'center', 'heading_scores', 'heading_residuals_normalized', 'heading_residuals', 
            'size_scores', 'size_residuals_normalized', 'size_residuals', 'sem_cls_scores' to end_points
                
    """
    #net_transposed = tf.transpose(net, perm=[0,2,1]) # (batch_size, 1024, ..)    
    batch_size = net.shape[0]
    num_proposal = net.shape[1]

    objectness_scores = net[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # B x num_proposal x num_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # B x num_proposal x num_heading_bin

    size_scores = net[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals = net[:,:,5 + num_heading_bin*2 + num_size_cluster : 5 + num_heading_bin*2 + num_size_cluster*4]
    size_residuals_normalized = tf.reshape(size_residuals, [batch_size,num_proposal, num_size_cluster, 3]) # B x num_proposal x num_size_cluster x 3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(mean_size_arr, dtype=tf.float32), axis=0), axis=0)

    sem_cls_scores = net[:,:,5+num_heading_bin*2+num_size_cluster*4:] # B x num_proposal x 10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(layers.Layer):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, use_tflite=False):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        
        # Vote clustering        
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
                #use_tflite=use_tflite,
                #tflite_name='va_quant_b8.tflite'
            )
        """
        self.npoint = self.num_proposal
        self.nsample = 16
        self.vote_aggregation = PointnetSAModuleVotes_NoMLP( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,                
                use_xyz=True,
                normalize_xyz=True
            )
        self.use_tflite = use_tflite
        
        mlp_spec = [self.seed_feat_dim, 128, 128, 128]
        mlp_spec[0] += 3  

        if self.use_tflite:
            #from pycoral.utils.edgetpu import make_interpreter
            self.interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR,os.path.join("tflite_models",'proposal_quant.tflite')))                             
            #self.interpreter = make_interpreter(os.path.join(ROOT_DIR,os.path.join("tflite_models",tflite_name)))
            self.interpreter.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.mlp_module = tf_utils.SharedMLP(mlp_spec, bn=True, input_shape=[self.npoint, self.nsample, 3+mlp_spec[0]])        
            self.max_pool = layers.MaxPooling2D(pool_size=(1, self.nsample), strides=1, data_format="channels_last")
        """        
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = layers.Conv1D(filters=128, kernel_size=1)
        self.conv2 = layers.Conv1D(filters=128, kernel_size=1)
        self.conv3 = layers.Conv1D(filters=2+3+num_heading_bin*2+num_size_cluster*4+self.num_class, kernel_size=1)
        #### Changed to Conv2D to be compatible with EdgeTPU compiler
        #self.conv1 = layers.Conv2D(filters=128, kernel_size=1)
        #self.conv2 = layers.Conv2D(filters=128, kernel_size=1)
        #self.conv3 = layers.Conv2D(filters=2+3+num_heading_bin*2+num_size_cluster*4+self.num_class, kernel_size=1)
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu1 = layers.Activation('relu')
        self.relu2 = layers.Activation('relu')

    def call(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,K,C)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds, _, grouped_features = self.vote_aggregation(xyz, features, sample_type='fps')
            #xyz, fps_inds, _, grouped_features = self.vote_aggregation(xyz, features, sample_type='fps') #NoMLP version
            end_points['va_grouped_features'] = grouped_features
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = tf_sampling.farthest_point_sample(self.num_proposal, end_points['seed_xyz'])
            xyz, features, fps_inds, _, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            seed_pts = np.random.randint(low=0, high=num_seed, size=(batch_size, self.num_proposal)).astype('float32')
            sample_inds = tf.constant(seed_pts)
            xyz, features, fps_inds, _, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()        
        """
        new_features = self.mlp_module(grouped_features)
        features = self.max_pool(new_features)
        features = layers.Reshape((self.npoint, 1, features.shape[-1]))(features) # Expand to use Conv2D
        """
        # --------- PROPOSAL GENERATION ---------
        net = self.relu1(self.bn1(self.conv1(features)))
        net = self.relu2(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, num_proposal, 2+3+num_heading_bin*2+num_size_cluster*4)
        
        # Return from expanded shape
        #net = layers.Reshape((self.npoint, net.shape[-1]))(net)

        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal


        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)        
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps')
    end_points = {'seed_xyz': tf.constant(np.random.random((8,1024,3)).astype('float32'))}
    xyz = tf.constant(np.random.random((8,1024,3)).astype('float32'))
    features = tf.constant(np.random.random((8,256,1024)).astype('float32'))

    out = net(xyz, features, end_points)
    for key in out:
        print(key, out[key].shape)
'''
seed_xyz (8, 1024, 3)
aggregated_vote_xyz (8, 128, 3)
aggregated_vote_inds (8, 128)
objectness_scores (8, 128, 2)
center (8, 128, 3) #(B, num_proposal, 3)
heading_scores (8, 128, 12) #(B, num_proposal, num_heading_bin)
heading_residuals_normalized (8, 128, 12) #(B, num_proposal, num_heading_bin)
heading_residuals (8, 128, 12) #(B, num_proposal, num_heading_bin)
size_scores (8, 128, 10) # (B, num_proposal, num_size_cluster)
size_residuals_normalized (8, 128, 10, 3) # (B, num_proposal, num_size_cluster, 3)
size_residuals (8, 128, 10, 3)
sem_cls_scores (8, 128, 10)
'''