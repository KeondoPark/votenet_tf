# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module_tf import Pointnet2Backbone, Pointnet2Backbone_p
from voting_module_tf import VotingModule
from proposal_module_tf import ProposalModule
from dump_helper_tf import dump_results
from loss_helper_tf import get_loss

import time

class VoteNet(tf.keras.Model):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=256, vote_factor=1, sampling='vote_fps', use_tflite=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal =  num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone_p(input_feature_dim=self.input_feature_dim, use_tflite=use_tflite)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, seed_feature_dim=128, use_tflite=use_tflite, tflite_name='voting_quant_test_edgetpu.tflite')

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, seed_feat_dim=128, use_tflite=use_tflite, tflite_name='va_quant_test_edgetpu.tflite')

    def call(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: list
        """
        end_points = self.backbone_net(inputs)
        
        # --------- HOUGH VOTING ---------
        #xyz = end_points['fp2_xyz']
        #features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']
        
        #start = time.time() 
        xyz, features = self.vgen(end_points['seed_xyz'], end_points['seed_features'])
        #print("Runtime for Voting module:", time.time() - start)
        features_norm = tf.norm(features, ord=2, axis=1)
        features = tf.divide(features, tf.expand_dims(features_norm, axis=1))        
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features        
        
        #start = time.time() 
        end_points = self.pnet(xyz, features, end_points)
        #print("Runtime for Proposal module:", time.time() - start)

        #return end_points
        return end_points



if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3)))
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': tf.expand_dims(tf.convert_to_tensor(sample['point_clouds']), axis=0)}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': tf.expand_dims(tf.random.normal(shape=[20000,3]), axis=0)}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = tf.expand_dims(tf.convert_to_tensor(sample[key]), axis=0)
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')
