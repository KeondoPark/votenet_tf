# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance_tf import nn_distance, huber_loss
import tensorflow as tf
import torch
import torch.nn as nn

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = tf.cast(end_points['seed_inds'], dtype=tf.int32) # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = tf.gather(end_points['vote_label_mask'], axis=1, indices=seed_inds, batch_dims = 1) #(B, 20000) -> (B, num_seed)
    #seed_inds_expand = tf.tile(tf.reshape(seed_inds, shape=[batch_size, num_seed, 1]), multiples=[1,1,3*GT_VOTE_FACTOR])
    #print("seed_inds_expand shape: ", seed_inds_expand.shape)
    seed_gt_votes = tf.gather(end_points['vote_label'], axis=1, indices=seed_inds, batch_dims=1)
    seed_gt_votes += tf.tile(end_points['seed_xyz'], multiples=[1,1,3])

    # Compute the min of min of distance
    vote_xyz_reshape = tf.reshape(vote_xyz, shape=[batch_size*num_seed, -1, 3]) # from (B,num_seed*vote_factor,3) to (B*num_seed,vote_factor,3)
    seed_gt_votes_reshape = tf.reshape(seed_gt_votes, [batch_size*num_seed, GT_VOTE_FACTOR, 3]) # from (B,num_seed,3*GT_VOTE_FACTOR) to (B*num_seed,GT_VOTE_FACTOR,3)
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True) # dist1: (B*num_seed, GT_VOTE_FACTOR), dist2: (B*num_seed, vote_factor)
    votes_dist= tf.reduce_min(dist2, axis=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = tf.reshape(votes_dist, shape=[batch_size, num_seed])
    vote_loss = tf.reduce_sum(votes_dist*tf.cast(seed_gt_votes_mask, dtype=tf.float32))/(tf.reduce_sum(tf.cast(seed_gt_votes_mask, dtype=tf.float32))+1e-6)
    
    """
    For validation
    
    seed_inds_torch = torch.tensor(seed_inds.numpy()).long()
    seed_gt_votes_mask_torch = torch.gather(torch.tensor(end_points['vote_label_mask'].numpy()), 1, seed_inds_torch)
    print("(Tensorflow) seed_gt_votes_mask[0]", seed_gt_votes_mask[0])
    print("(Torch) seed_gt_votes_mask[0]", seed_gt_votes_mask_torch[0])
    seed_inds_expand = seed_inds_torch.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    vote_label_torch = torch.tensor(end_points['vote_label'].numpy())
    seed_gt_votes_torch = torch.gather(vote_label_torch, 1, seed_inds_expand)
    print("(Tensorflow) seed_gt_votes[0,0]", seed_gt_votes[0,0])
    print("(Torch) seed_gt_votes[0,0]", seed_gt_votes_torch[0,0])
    seed_xyz_torch = torch.tensor(end_points['seed_xyz'].numpy())
    seed_gt_votes_torch += seed_xyz_torch.repeat(1,1,3)
    print("(Tensorflow) seed_gt_votes_torch[0,0] recoverd", seed_gt_votes[0,0])
    print("(Torch) seed_gt_votes_torch[0,0] recovered", seed_gt_votes_torch[0,0])

    # Compute the min of min of distance
    vote_xyz_torch = torch.tensor(vote_xyz.numpy())
    vote_xyz_reshape_torch = vote_xyz_torch.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_torch = seed_gt_votes_torch.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    #dist1, _, dist2, _ = nn_distance(vote_xyz_reshape_torch, seed_gt_votes_reshape_torch, l1=True)
    dist2_torch = torch.tensor(dist2.numpy())
    votes_dist_torch, _ = torch.min(dist2_torch, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_torch = votes_dist_torch.view(batch_size, num_seed)
    vote_loss_torch = torch.sum(votes_dist_torch*seed_gt_votes_mask_torch.float())/(torch.sum(seed_gt_votes_mask_torch.float())+1e-6)

    print("(Tensorflow) vote loss", vote_loss)
    print("(Torch) vote loss", vote_loss_torch)
    """
    
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'] #(B, K, 3)
    gt_center = end_points['center_label'][:,:,0:3] # (batch_size, MAX_NUM_OBJ, 3)
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1] #num_proposal
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = tf.math.sqrt(dist1+1e-6)
    objectness_label = tf.zeros([B,K], dtype=tf.int32)
    objectness_mask = tf.zeros([B,K], dtype=tf.float32)

    print(euclidean_dist1)
    print(euclidean_dist1<NEAR_THRESHOLD)
    print(objectness_label.shape)
    objectness_label = tf.where(euclidean_dist1<NEAR_THRESHOLD, 1, 0)
    objectness_mask = tf.where(euclidean_dist1<NEAR_THRESHOLD, 1.0, 0.0)
    objectness_mask = tf.where(euclidean_dist1>FAR_THRESHOLD, 1.0, 0.0)
    #objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    #objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    #objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores'] #(B, num_proposal, 2)
    objectness_label_one_hot = tf.one_hot(objectness_label, depth=2, axis=-1)

    """
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) # softmax not applied
    objectness_loss = criterion(objectness_label, objectness_scores)

    print("objectness_loss shape:", objectness_loss.shape)
    print(objectness_loss)
    """
    def crossEntropyWithClassWeights(y_true, y_pred, weight, n_class=2):
        y_pred_softmax = tf.nn.softmax(y_pred)
        loss = - y_true * tf.math.log(y_pred_softmax+1e-10) * weight
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    objectness_loss = crossEntropyWithClassWeights(objectness_label_one_hot, objectness_scores, weight=tf.constant(OBJECTNESS_CLS_WEIGHTS), n_class=2)
    objectness_loss = tf.reduce_sum(objectness_loss * objectness_mask)/(tf.reduce_sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center'] # (batch_size, num_proposal, 3)
    gt_center = end_points['center_label'][:,:,0:3] # (batch_size, MAX_NUM_OBJ, 3)
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: (batch_size, MAX_NUM_OBJ), dist2: (batch_size, num_proposal)
    box_label_mask = end_points['box_label_mask']  #(batch_size, MAX_NUM_OBJ)
    objectness_label = tf.cast(end_points['objectness_label'], dtype=tf.float32) #(batch_size, MAX_NUM_OBJ)
    centroid_reg_loss1 = \
        tf.reduce_sum(dist1*objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        tf.reduce_sum(dist2*box_label_mask)/(tf.reduce_sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    # Change object_assignment to be compatible with tf.gather_nd
    K = object_assignment.shape[1]
    object_assignment_exp = tf.expand_dims(object_assignment, axis = -1) #(B,K,1)
    row_id = tf.expand_dims(tf.expand_dims(np.array(list(range(batch_size))), axis=-1), axis=-1) #(B,1,1)
    row_id_tile = tf.tile(row_id, multiples=(1,K,1)) #(B,K,1)
    ind_tf = tf.concat([row_id_tile, object_assignment_exp], axis=-1) #(B,K,2)
    heading_class_label = tf.gather_nd(end_points['heading_class_label'], ind_tf)
    #heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)

    criterion_heading_class = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) #SparseCategoricalCrossentropy is used because heading_class_label is NOT one-hot
    heading_class_loss = criterion_heading_class(heading_class_label, tf.transpose(end_points['heading_scores'], perm=[0,2,1])) # (B,K)    
    heading_class_loss = tf.reduce_sum(heading_class_loss * objectness_label) / (tf.reduce_sum(objectness_label)+1e-6)

    heading_residual_label = tf.gather_nd(end_points['heading_residual_label'], ind_tf) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    heading_label_one_hot = tf.one_hot(heading_class_label, depth=num_heading_bin)
    huber_loss = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)()
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(end_points['heading_residuals_normalized']*heading_label_one_hot - heading_residual_normalized_label, axis=-1))
    heading_residual_normalized_loss = tf.reduce_sum(heading_residual_normalized_loss*objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = tf.gather_nd(end_points['size_class_label'], ind_tf) # select (B,K) from (B,K2)
    criterion_size_class = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    size_class_loss = criterion_size_class(size_class_label, tf.transpose(end_points['size_scores'], perm=[0,2,1])) # (B,K)
    size_class_loss = tf.reduce_sum(size_class_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    #Create index used for tensorflow gather_nd... not very convenient compared to pytorch
    a = tf.tile(tf.expand_dims(ind_tf, axis=2), multiples=(1,1,3,1)) # (B,K,3,2)    
    b = tf.tile(tf.expand_dims(np.array(list(range(3))), axis=0), multiples=(K,1)) #(K,3)
    b = tf.expand_dims(tf.tile(tf.expand_dims(b, axis=0), multiples=(batch_size,1,1)), axis=-1) #(B,K,3,1)
    ind_size_residual = tf.concat([a,b], axis=-1) #(B, K, 3, 3) First 3 is the number of elements to gather / Second 3 is the number of dimension
    size_residual_label = tf.gather_nd(end_points['size_residual_label'], ind_size_residual) # select (B,K,3) from (B,K2,3)

    
    size_label_one_hot = tf.one_hot(heading_class_label, depth=num_size_cluster) #(B, K, num_size_cluster)    
    size_label_one_hot_tiled = tf.tile(tf.expand_dims(size_label_one_hot, axis=-1), multiples=(1,1,1,3)) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = tf.reduce_sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, axis=2) # (B,K,3)
    
    mean_size_arr_expanded = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(mean_size_arr.astype(np.float32)), axis=0), axis=0) # (1,1,num_size_cluster,3) 
    mean_size_label = tf.reduce_sum(size_label_one_hot_tiled * mean_size_arr_expanded, axis=2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = tf.math.reduce_mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), axis=-1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = tf.reduce_sum(size_residual_normalized_loss*objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    #sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    sem_cls_label = tf.gather_nd(end_points['sem_cls_label'], ind_tf) # select (B,K) from (B,K2)
    criterion_sem_cls = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    sem_cls_loss = criterion_sem_cls(sem_cls_label, tf.transpose(end_points['sem_cls_scores'], perm=[0,2,1])) # (B,K)
    sem_cls_loss = tf.reduce_sum(sem_cls_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """
    print("========Calc loss!!!=========")       

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
    end_points['pos_ratio'] = \
        tf.reduce_sum(tf.cast(objectness_label, dtype=tf.float32))/float(total_num_proposal)
    end_points['neg_ratio'] = \
        tf.reduce_sum(tf.cast(objectness_mask, dtype=tf.float32))/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = tf.math.argmax(end_points['objectness_scores'], axis=2) # B,K
    obj_acc = tf.reduce_sum(tf.cast(obj_pred_val==tf.cast(objectness_label, dtype=tf.int32), dtype=tf.float32)*objectness_mask)/(tf.reduce_sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    print("loss and end points", loss, end_points)
    return loss, end_points
