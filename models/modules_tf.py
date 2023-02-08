import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)



class PositionEmbeddingLearned(tf.keras.layers.Layer):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=288, activation='relu'):
        super().__init__()
        maxval = None if activation=='relu' else 6        
        self.conv1 = layers.Conv2D(filters=num_pos_feats, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.act1 = layers.ReLU(maxval)
        self.conv2 = layers.Conv2D(filters=num_pos_feats, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())

    def call(self, xyz):        
        position_embedding = self.conv2(self.act1(self.bn1(self.conv1(xyz))))
        
        return position_embedding


class PointsObjClsModule(tf.keras.layers.Layer):
    def __init__(self, seed_feature_dim, activation='relu'):
        """ object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.conv3 = layers.Conv2D(filters=1, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())

        maxval = None if activation=='relu' else 6
        self.relu1 = layers.ReLU(maxval)
        self.relu2 = layers.ReLU(maxval)

    def call(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, num_seed, 1, feature_dim)
        Returns:
            logits: (batch_size, num_seed, 1)
        """
        net = self.relu1(self.bn1(self.conv1(seed_features)))
        net = self.relu2(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, num_seed, 1, 1)

        return logits

class PointsObjClsModule2(tf.keras.layers.Layer):
    def __init__(self, seed_feature_dim, activation='relu'):
        """ object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = layers.Conv2D(filters=self.in_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=self.in_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.conv3 = layers.Conv2D(filters=1 + 288, kernel_size=1, kernel_initializer=tf.keras.initializers.he_normal())

        maxval = None if activation=='relu' else 6
        self.relu1 = layers.ReLU(maxval)
        self.relu2 = layers.ReLU(maxval)

    def call(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, num_seed, 1, feature_dim)
        Returns:
            logits: (batch_size, num_seed, 1)
        """
        net = self.relu1(self.bn1(self.conv1(seed_features)))
        net = self.relu2(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, num_seed, 1, 288 + 1)

        logits = net[:,:,:,:1] # (batch_size, num_seed, 1, 1)
        features = net[:,:,:,1:] # (batch_size, num_seed, 1, 288)

        return logits, features


class PredictHead(tf.keras.layers.Layer):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, seed_feat_dim=256, activation='relu'):
        super().__init__()

        self.nc = num_class
        self.NH = num_heading_bin
        self.NC = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = layers.Conv2D(filters=seed_feat_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_uniform())
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=seed_feat_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_uniform())
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.conv3 = layers.Conv2D(filters=1 + 3 + self.NH*2 + self.NC*4 + self.nc, kernel_size=1, kernel_initializer=tf.keras.initializers.he_uniform())

        maxval = None if activation=='relu' else 6
        self.relu1 = layers.ReLU(maxval)
        self.relu2 = layers.ReLU(maxval)

    def decode_scores(self, net, end_points, prefix=''):
        """
        Args:
            net: (B, num_proposal, 2+num_heading_bin*2+num_size_cluster*4)
            end_points: dictionary of 'aggregated_vote_xyz', 'aggregated_vote_inds', 
                                    'fp2_xyz', 'fp2_features', 'seed_inds', 'seed_xyz', 'seed_features',
                                    'vote_xyz', 'vote_features'
        return:
            Add 'objectness_scores', 'center', 'heading_scores', 'heading_residuals_normalized', 'heading_residuals', 
                'size_scores', 'size_residuals_normalized', 'size_residuals', 'sem_cls_scores' to end_points
                    
        """
        objectness_scores = net[:,:,0:1]        
        
        heading_scores = net[:,:, 1:1+self.NH]
        size_scores = net[:,:, 1 + self.NH:1 + self.NH + self.NC]

        heading_residuals_normalized = net[:,:,1 + self.NH + self.NC:1 + self.NH + self.NC + self.NH]
        pi = tf.constant(3.14159265359, dtype=tf.float32)
        
        heading_residuals = heading_residuals_normalized * (pi/tf.cast(self.NH, tf.float32)) # B x num_proposal x num_heading_bin

        size_residuals = net[:,:,1 + self.NH*2 + self.NC:1 + self.NH*2 + self.NC*4]
        size_residuals_normalized = layers.Reshape((self.num_proposal, self.NC, 3))(size_residuals) # B x num_proposal x num_size_cluster x 3    
        mean_size_arr = tf.expand_dims(tf.expand_dims(tf.cast(self.mean_size_arr,dtype=tf.float32), axis=0), axis=0) #(1, 1, NC, 3)
        size_residuals = size_residuals_normalized * mean_size_arr        

        sem_cls_scores = net[:,:,1+self.NH*2+self.NC*4:] # B x num_proposal x 10        

        size_recover = size_residuals + mean_size_arr  # (B, num_proposal, num_size_cluster, 3)
        pred_size_class = tf.math.argmax(size_scores, -1)  # batch_size, num_proposal
        pred_size_class = layers.Reshape((self.num_proposal,1))(pred_size_class)


        pred_size = tf.gather(size_recover, axis=2, indices=pred_size_class, batch_dims=2)  # batch_size, num_proposal, 1, 3        
        pred_size = layers.Reshape((self.num_proposal, 3))(pred_size)  # batch_size, num_proposal, 3
       
        end_points[f'{prefix}objectness_scores'] = objectness_scores        
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}size_scores'] = size_scores
        end_points[f'{prefix}size_residuals_normalized'] = size_residuals_normalized
        end_points[f'{prefix}size_residuals'] = size_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores
        
        return end_points

    def call(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,num_proposal,1,C)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NC*4+nc)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[1]
        num_features = features.shape[-1]
        features = layers.Reshape((num_proposal, 1, num_features))(features)

        net = self.relu1(self.bn1(self.conv1(features)))
        net = self.relu2(self.bn2(self.conv2(net)))
        net = self.conv3(net)

        offset = net[:,:,:,0:3]
        net = net[:,:,:,3:]

        offset = layers.Reshape((num_proposal, 3))(offset)                
        center = base_xyz + offset            
        net = layers.Reshape((num_proposal, net.shape[-1]))(net)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}center'] = center
        end_points = self.decode_scores(net, end_points, prefix)
        pred_size = end_points[f'{prefix}pred_size']

        return center, pred_size, end_points


class ClsAgnosticPredictHead(tf.keras.layers.Layer):

    def __init__(self, num_class, num_heading_bin, num_proposal, seed_feat_dim=256, activation='relu'):
        super().__init__()

        self.nc = num_class
        self.NH = num_heading_bin        
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = layers.Conv2D(filters=seed_feat_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_uniform())
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=seed_feat_dim, kernel_size=1, kernel_initializer=tf.keras.initializers.he_uniform())
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.conv3 = layers.Conv2D(filters=1 + 3 + self.NH*2 + 3 + self.nc, kernel_size=1, kernel_initializer=tf.keras.initializers.he_uniform())

        maxval = None if activation=='relu' else 6
        self.relu1 = layers.ReLU(maxval)
        self.relu2 = layers.ReLU(maxval)


    def decode_scores(self, net, end_points, prefix=''):

        objectness_scores = net[:,:,0:1]       
        heading_scores = net[:,:, 1:1+self.NH]

        pred_size = net[:,:, 1 + self.NH:1 + self.NH + 3]

        heading_residuals_normalized = net[:,:,1 + self.NH + 3:1 + self.NH + 3 + self.NH]
        pi = tf.constant(3.14159265359, dtype=tf.float32)        
        heading_residuals = heading_residuals_normalized * (pi/tf.cast(self.NH, tf.float32)) # B x num_proposal x num_heading_bin        

        sem_cls_scores = net[:,:,1+self.NH*2+3:] # B x num_proposal x 10        
       
        end_points[f'{prefix}objectness_scores'] = objectness_scores        
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals        
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        return end_points

    def call(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,num_proposal,1,C)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NC*4+nc)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[1]
        num_features = features.shape[-1]
        features = layers.Reshape((num_proposal, 1, num_features))(features)

        net = self.relu1(self.bn1(self.conv1(features)))
        net = self.relu2(self.bn2(self.conv2(net)))
        net = self.conv3(net)

        offset = net[:,:,:,0:3]
        net = net[:,:,:,3:]

        offset = layers.Reshape((num_proposal, 3))(offset)                
        center = base_xyz + offset            
        net = layers.Reshape((num_proposal, net.shape[-1]))(net)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}center'] = center
        end_points = self.decode_scores(net, end_points, prefix)
        pred_size = end_points[f'{prefix}pred_size']

        return center, pred_size, end_points