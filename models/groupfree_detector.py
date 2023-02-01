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

from modules_tf import PointsObjClsModule, PredictHead, PositionEmbeddingLearned, ClsAgnosticPredictHead
from transformer_tf import TransformerDecoderLayer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class GroupFreeDetector(layers.Layer):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, model_config,
                 num_proposal=128, dropout=0.1,
                 nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='loc_learned', cross_position_embedding='xyz_learned',
                 size_cls_agnostic=False):
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
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)        
        self.num_proposal = num_proposal
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.size_cls_agnostic = size_cls_agnostic

        if model_config is not None and 'activation' in model_config:
            act = model_config['activation']
        else:
            act = 'relu'
        self.points_obj_cls = PointsObjClsModule(288, activation=act) # 3 Conv layers
        # self.gsample_module = GeneralSamplingModule() # Gather operation
        

        # Proposal
        if self.size_cls_agnostic:            
            self.proposal_head = ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, activation=act)
        else:
            self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                             mean_size_arr, num_proposal, 288, activation=act)

        # Transformer Decoder Projection
        self.decoder_key_proj = layers.Conv2D(filters=288, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())
        self.decoder_query_proj = layers.Conv2D(filters=288, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(self.num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':            
            self.decoder_self_posembeds = []
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(288, activation=act))
        elif self.self_position_embedding == 'loc_learned':    
            self.decoder_self_posembeds = []        
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(288, activation=act))

         # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = []
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(288, activation=act))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        
        self.decoder = []
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Prediction Head
        self.prediction_heads = []
        for i in range(self.num_decoder_layers):
            if self.size_cls_agnostic:                
                self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, activation=act))
            else:
                self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                         mean_size_arr, num_proposal, 288, activation=act))
        
    def call(self, seed_xyz, seed_features, end_points):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) tensor
            seed_features: (batch_size, num_seed, feature_dim) tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, num_seed*vote_factor, vote_feature_dim)
        """        
        
        N = seed_features.shape[1]
        C = seed_features.shape[2]
        seed_features = layers.Reshape((N, 1, C))(seed_features)


        # Query Points Generation; Object candidate sampling 
        points_obj_cls_logits = self.points_obj_cls(seed_features)  # (batch_size, num_seed, 1, 1)
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        points_obj_cls_scores = tf.math.sigmoid(points_obj_cls_logits)
        points_obj_cls_scores = layers.Reshape((N,))(points_obj_cls_logits)
        values, sample_inds = tf.math.top_k(points_obj_cls_scores, self.num_proposal)
        xyz = tf.gather(seed_xyz, axis=1, indices=sample_inds, batch_dims=1)
        features = tf.gather(seed_features, axis=1, indices=sample_inds, batch_dims=1) # (batch_size, num_proposal, 1, C)
        cluster_feature = features # (batch_size, num_proposal, 1, C)
        cluster_xyz = xyz
        end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['query_points_feature'] = features  # (batch_size, num_proposal, 1, C)
        end_points['query_points_sample_inds'] = sample_inds  # (batch_size, num_proposal) # should be 0,1,...,num_proposal

        # Proposal
        proposal_center, proposal_size, end_points = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center
        base_size = proposal_size

        # Transformer Decoder and Prediction
        query = self.decoder_query_proj(cluster_feature) # (batch_size, num_proposal, 1, C)
        key = self.decoder_key_proj(seed_features) if self.decoder_key_proj is not None else None # (batch_size, num_proposal, 1, C)

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = seed_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = layers.Concatenate(axis=-1)([base_xyz, base_size])
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size, end_points = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz
            base_size = base_size

        return end_points
 
if __name__=='__main__':

    mean_size_arr = np.array([np.array([0.765840,1.398258,0.472728]),
        np.array([2.114256,1.620300,0.927272]),
        np.array([0.404671,1.071108,1.688889]),
        np.array([0.591958,0.552978,0.827272]),
        np.array([0.695190,1.346299,0.736364]),
        np.array([0.528526,1.002642,1.172878]),
        np.array([0.500618,0.632163,0.683424]),
        np.array([0.923508,1.867419,0.845495]),
        np.array([0.791118,1.279516,0.718182]),
        np.array([0.699104,0.454178,0.756250])])

    net = GroupFreeDetector(num_class=10, num_heading_bin=12, num_size_cluster=10, mean_size_arr=mean_size_arr,
                            model_config=None, num_proposal=256)
    xyz = np.random.random((1,1024,3)).astype('float32')
    features = np.random.random((1,1024,256)).astype('float32')
    end_points = {}
    end_points = net(tf.constant(xyz), tf.constant(features), end_points)
    print(end_points)
