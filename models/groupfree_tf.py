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
from backbone_module_tf import Pointnet2Backbone, Pointnet2Backbone_p, Pointnet2Backbone_tflite
from groupfree_detector import GroupFreeDetector
from groupfree_detector_light import GroupFreeDetector_Light
from dump_helper_tf import dump_results
from loss_helper_tf import get_loss

import time

class GroupFreeNet(tf.keras.Model):
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
        model_config: Configuration information including "use_tflite", "use_multiThre", "two_way", "q_gran" etc.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        model_config,
        input_feature_dim=0, num_proposal=256, size_cls_agnostic=False,
        decoder_normalization='layer', light_detector=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal =  num_proposal       
        self.use_tflite = model_config['use_tflite']
        self.use_multiThr = model_config['use_multiThr']
        two_way = model_config['two_way']

        if two_way:
            if self.use_tflite:
                # inference only, pipelining is implemented here.
                self.backbone_net = Pointnet2Backbone_tflite(input_feature_dim=self.input_feature_dim, model_config=model_config, num_class=num_class)
            else:
                # 2-way set abstraction + no pointnnet in feature propagation
                self.backbone_net = Pointnet2Backbone_p(input_feature_dim=self.input_feature_dim, model_config=model_config)
        else:
            # Original votenet architecture
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, model_config=model_config)
        
        if light_detector:
            self.detector = GroupFreeDetector_Light(num_class, 
                                        num_heading_bin, 
                                        num_size_cluster, 
                                        mean_size_arr, 
                                        model_config, 
                                        num_proposal=num_proposal, 
                                        size_cls_agnostic=size_cls_agnostic, 
                                        decoder_normalization=decoder_normalization)            

        else:
            self.detector = GroupFreeDetector(num_class, 
                                            num_heading_bin, 
                                            num_size_cluster, 
                                            mean_size_arr, 
                                            model_config, 
                                            num_proposal=num_proposal, 
                                            size_cls_agnostic=size_cls_agnostic, 
                                            decoder_normalization=decoder_normalization)            

    def call(self, point_cloud, repsurf_feature, imgs=None, calibs=None, deeplab_tflite_file=None):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)

                imgs: 2D images for pointpainting
                    Only used when pipelining(multithreading) is used.
                    Otherwise, already processed before this module starts

                calibs: 2D - 3D projection
                    Only used when pipelining(multithreading) is used.
                    Otherwise, already processed before this module starts
                        
        Returns:
            end_points: list
        """
        if self.use_tflite and self.use_multiThr:
            end_points = self.backbone_net(point_cloud, repsurf_feature, imgs=imgs, calibs=calibs, deeplab_tflite_file=deeplab_tflite_file)
        else:
            end_points = self.backbone_net(point_cloud, repsurf_feature)
        # --------- HOUGH VOTING ---------
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']
        
        
        end_points = self.detector(end_points['seed_xyz'], end_points['seed_features'], end_points)
        
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
