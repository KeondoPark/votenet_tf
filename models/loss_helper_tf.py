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
from nn_distance_tf import nn_distance, huber_loss, huber_loss_torch, SigmoidFocalClassificationLoss
import tensorflow as tf

def compute_points_obj_cls_loss_hard_topk(end_points, topk):

    box_label_mask = end_points['box_label_mask']
    box_label_mask = tf.cast(box_label_mask, tf.int32)
    seed_inds = end_points['seed_inds']  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
    B = tf.shape(gt_center)[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = tf.gather(point_instance_label, axis=1, indices=seed_inds, batch_dims=1)  # B, num_seed
    # object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment = tf.where(object_assignment < 0, tf.constant(K2-1, tf.int64), object_assignment)
    object_assignment_one_hot = tf.one_hot(object_assignment, depth=K2, axis=-1) #(B, K, K2)    

    delta_xyz = tf.expand_dims(seed_xyz,2) - tf.expand_dims(gt_center, 1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (tf.expand_dims(gt_size,1) + 1e-6)  # (B, K, K2, 3)
    new_dist = tf.reduce_sum(delta_xyz ** 2, axis=-1) # (B, K, K2)

    euclidean_dist1 = tf.math.sqrt(new_dist + 1e-6)  # (B, K, K2)
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # (B, K, K2)
    euclidean_dist1 = tf.transpose(euclidean_dist1, perm=[0,2,1])  # (B, K2, K)    
    topk_inds = tf.math.top_k(-euclidean_dist1, topk)[1] * box_label_mask[:, :, None] + (box_label_mask[:, :, None] - 1)  # (B, K2, topk)
    topk_inds = tf.reshape(topk_inds, [B,-1])  # B, K2xtopk

    topk_inds = tf.where(topk_inds < 0, tf.constant(K, tf.int32), topk_inds)

    batch_inds = tf.tile(tf.expand_dims(tf.range(B),-1), [1, K2 * topk]) 
    batch_topk_inds = tf.stack([batch_inds, topk_inds], -1) # (B, K2 * topk, 2)
    batch_topk_inds = tf.cast(batch_topk_inds, dtype=tf.int64)
    updates = tf.ones([B,K2 * topk], dtype=tf.int64)
    objectness_label = tf.scatter_nd(indices=batch_topk_inds, updates=updates, shape=[B,K+1])

    # WHen there are duplicates in topk, object labels are greater than 1... Need to make it 1 or 0
    objectness_label = tf.where(objectness_label > 0, tf.constant(1, tf.int64), tf.constant(0, tf.int64))
    objectness_label = objectness_label[:, :K]


    objectness_label_mask = tf.gather(point_instance_label, axis=1, indices=seed_inds, batch_dims=1)  # B, num_seed
    objectness_label = tf.where(objectness_label_mask < 0, tf.constant(0, dtype=tf.int64), objectness_label)


    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        tf.cast(tf.reduce_sum(objectness_label),tf.float32) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']


    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = tf.where(objectness_label >= 0, tf.constant(1.0, tf.float32), tf.constant(0.0, tf.float32))
    cls_normalizer = tf.reduce_sum(cls_weights, axis=1, keepdims=True)
    cls_weights /= tf.clip_by_value(cls_normalizer, clip_value_min=1.0, clip_value_max=tf.float32.max)
    objectness_label = tf.cast(objectness_label, tf.float32)    
    cls_loss_src = criterion.forward(tf.reshape(seeds_obj_cls_logits,[B, K, 1]), tf.expand_dims(objectness_label, -1), weights=cls_weights)
    objectness_loss = tf.reduce_sum(cls_loss_src) / float(B)
    
    
    # Below part causes an error in tf.function.. ignore.
    # # Compute recall upper bound
    # padding_array = tf.range(0, B) * 10000
    # padding_array = tf.expand_dims(padding_array, 1)  # B,1
    # point_instance_label = point_instance_label + tf.cast(padding_array, dtype=tf.int64)  # B,num_points
    # # point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    # point_instance_label = tf.where(point_instance_label < 0, tf.constant(-1, dtype=tf.int64), point_instance_label)
    

    # seed_instance_label = tf.gather(point_instance_label, axis=1, indices=seed_inds, batch_dims=1)  # B, num_seed
    # # point_instance_label = tf.cast(point_instance_label, tf.float32)
    # objectness_label = tf.cast(objectness_label, tf.int64)
    # pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    
    # point_instance_label = tf.reshape(point_instance_label, (-1,))
    # num_gt_bboxes = tf.unique_with_counts(point_instance_label)[2] - 1
    
    # pos_points_instance_label = tf.reshape(pos_points_instance_label, (-1,))     
    # num_query_bboxes = tf.unique_with_counts(pos_points_instance_label)[2] - 1
    # if num_gt_bboxes > 0:
    #     end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = tf.divide(tf.cast(num_query_bboxes, tf.float32), tf.cast(num_gt_bboxes, tf.float32))

    return objectness_loss


def compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers):
    """ 
    Compute objectness loss for the proposals.
    """
    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    objectness_loss_sum = 0.0

    for prefix in prefixes:
        # Associate proposal and GT objects
        seed_inds = end_points['seed_inds']  # B,num_seed in [0,num_points-1]
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        query_points_sample_inds = end_points['query_points_sample_inds']
        
        B = tf.shape(seed_inds)[0]
        K = query_points_sample_inds.shape[1]
        K2 = gt_center.shape[1]

        seed_obj_gt = tf.gather(end_points['point_obj_mask'], axis=1, indices=seed_inds, batch_dims=1)  # B,num_seed
        query_points_obj_gt = tf.gather(seed_obj_gt, axis=1, indices=query_points_sample_inds, batch_dims=1)  # B, query_points

        point_instance_label = end_points['point_instance_label']  # B, num_points
        seed_instance_label = tf.gather(point_instance_label, axis=1, indices=seed_inds, batch_dims=1)  # B,num_seed
        query_points_instance_label = tf.gather(seed_instance_label, axis=1, indices=query_points_sample_inds, batch_dims=1)  # B,query_points

        objectness_mask = tf.ones((B, K))

        # Set assignment
        object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
        object_assignment = tf.where(object_assignment < 0, tf.constant(K2-1, dtype=tf.int64), object_assignment) # set background points to the last gt bbox        

        end_points[f'{prefix}objectness_label'] = query_points_obj_gt
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment
        total_num_proposal = tf.shape(query_points_obj_gt)[0] * query_points_obj_gt.shape[1]
        end_points[f'{prefix}pos_ratio'] = \
            tf.reduce_sum(tf.cast(query_points_obj_gt, tf.float32)) / float(total_num_proposal)
        end_points[f'{prefix}neg_ratio'] = \
            tf.reduce_sum(tf.cast(objectness_mask, tf.float32)) / float(total_num_proposal) - end_points[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = tf.cast(objectness_mask, tf.float32)
        cls_normalizer = tf.reduce_sum(cls_weights, axis=1, keepdims=True)
        cls_weights /= tf.clip_by_value(cls_normalizer, clip_value_min=1.0, clip_value_max=tf.float32.max)        
        cls_loss_src = criterion.forward(tf.reshape(tf.transpose(objectness_scores, perm=[0, 2, 1]), [B, K, 1]),
                                 tf.expand_dims(tf.cast(query_points_obj_gt, tf.float32), -1),
                                 weights=cls_weights)
        objectness_loss = tf.reduce_sum(cls_loss_src) / float(B)

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points

def compute_box_and_sem_cls_loss(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin, num_size_cluster, num_class, mean_size_arr = config

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0

    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment'] #(B, K)
        K = object_assignment.shape[1]    

        # Compute center loss, smoothl1
        pred_center = end_points[f'{prefix}center'] # (batch_size, num_proposal, 3)    
        gt_center = end_points['center_label'][:,:,0:3] # (batch_size, MAX_NUM_OBJ, 3)
        objectness_label = tf.cast(end_points[f'{prefix}objectness_label'], dtype=tf.float32) #(batch_size, MAX_NUM_OBJ)
        assigned_gt_center = tf.gather(gt_center, axis=1, indices=object_assignment, batch_dims=1) #(batch_size, num_proposal, 3)
        center_loss = huber_loss(assigned_gt_center - pred_center, delta=center_delta)
        center_loss = tf.reduce_sum(center_loss*tf.expand_dims(objectness_label,-1))/(tf.reduce_sum(objectness_label)+1e-6)

        # Compute heading loss
        # Change object_assignment to be compatible with tf.gather_nd
    
        heading_class_label = tf.gather(end_points['heading_class_label'], axis=1, indices=object_assignment, batch_dims=1) #(B, K2) -> (B, K)    
        if end_points[f'{prefix}heading_scores'].shape[-1] == 1:
            heading_class_loss = 0        
        else:
            criterion_heading_class = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #SparseCategoricalCrossentropy is used because heading_class_label is NOT one-hot    
            heading_class_loss = criterion_heading_class(heading_class_label, end_points[f'{prefix}heading_scores']) # (B,K)    
            heading_class_loss = tf.reduce_sum(heading_class_loss * objectness_label) / (tf.reduce_sum(objectness_label)+1e-6)

    
        heading_residual_label = tf.gather(end_points['heading_residual_label'], axis=1, indices=object_assignment, batch_dims=1)
        pi = tf.constant(3.14159265359, dtype=tf.float32)
        heading_residual_normalized_label = tf.divide(heading_residual_label, (pi/tf.cast(num_heading_bin,tf.float32)))

        heading_label_one_hot = tf.one_hot(heading_class_label, depth=num_heading_bin, axis=-1) #(B, K, 12)    
        heading_residual_normalized_error = \
            tf.reduce_sum(end_points[f'{prefix}heading_residuals_normalized']*heading_label_one_hot, axis=-1) - heading_residual_normalized_label
        heading_residual_normalized_loss = heading_delta * huber_loss(heading_residual_normalized_error, delta=heading_delta)
        heading_residual_normalized_loss = tf.reduce_sum(objectness_label * heading_residual_normalized_loss) / (tf.reduce_sum(objectness_label)+1e-6)

        # Compute size loss   
        if size_cls_agnostic: 
            pred_size = end_points[f'{prefix}pred_size']
            size_label = tf.gather(
                end_points['size_gts'], axis=1, indices=object_assignment, batch_dims=1)  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            
            size_loss = size_delta * huber_loss(size_error, delta=size_delta)  # (B,K,3) 
            size_loss = tf.reduce_sum(size_loss * tf.expand_dims(objectness_label, -1)) / (
                    tf.reduce_sum(objectness_label) + 1e-6)
        else:
            size_class_label = tf.gather(end_points['size_class_label'], axis=1, indices=object_assignment, batch_dims=1) # select (B,K) from (B,K2)
            criterion_size_class = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)    
            size_class_loss = criterion_size_class(size_class_label, end_points[f'{prefix}size_scores']) # (B,K)
            size_class_loss = tf.reduce_sum(size_class_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

            #Create index used for tensorflow gather_nd... not very convenient compared to pytorch    
            size_residual_label = tf.gather(end_points['size_residual_label'], axis=1, indices=object_assignment, batch_dims=1) # select (B,K,3) from (B,K2,3)   
            size_label_one_hot = tf.one_hot(size_class_label, depth=num_size_cluster) #(B, K, num_size_cluster)    
            size_label_one_hot_tiled = tf.tile(tf.expand_dims(size_label_one_hot, axis=-1), multiples=(1,1,1,3)) # (B,K,num_size_cluster,3)    
            predicted_size_residual_normalized = tf.reduce_sum(end_points[f'{prefix}size_residuals_normalized']*size_label_one_hot_tiled, axis=2) # (B,K,3)
        
            mean_size_arr_expanded = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(mean_size_arr), axis=0), axis=0) # (1,1,num_size_cluster,3) 
            mean_size_label = tf.reduce_sum(size_label_one_hot_tiled * mean_size_arr_expanded, axis=2) # (B,K,3)
        
            size_residual_label_normalized = tf.divide(size_residual_label, mean_size_label) # (B,K,3)
            size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized
            # size_residual_normalized_loss = tf.reduce_mean(size_delta * huber_loss(size_residual_normalized_error, delta=size_delta), axis=-1) # (B,K,3) -> (B,K)
            size_residual_normalized_loss = size_delta * huber_loss(size_residual_normalized_error, delta=size_delta) # (B,K,3) -> (B,K)
            size_residual_normalized_loss = tf.reduce_sum(size_residual_normalized_loss*tf.expand_dims(objectness_label,-1))/(tf.reduce_sum(objectness_label)+1e-6)

        # 3.4 Semantic cls loss    
        sem_cls_label = tf.gather(end_points['sem_cls_label'], axis=1, indices=object_assignment, batch_dims=1)
        criterion_sem_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)    
        sem_cls_loss = criterion_sem_cls(sem_cls_label, end_points[f'{prefix}sem_cls_scores']) # (B,K)
        sem_cls_loss = tf.reduce_sum(sem_cls_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            end_points[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + size_loss
        else:
            end_points[f'{prefix}size_cls_loss'] = size_class_loss
            end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss

    return box_loss_sum, sem_cls_loss_sum, end_points

def get_loss(end_points, config, num_decoder_layers,
             query_points_generator_loss_coef=0.8, obj_loss_coef=0.1, box_loss_coef=1.0, sem_cls_loss_coef=0.1,
             query_points_obj_topk=4,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0,
             size_cls_agnostic=False):
    """ Loss functions
    """

    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(end_points, query_points_obj_topk)        
        end_points['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0

    # Obj loss
    objectness_loss_sum, end_points = \
        compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers)    

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta,
        size_cls_agnostic=size_cls_agnostic)
    
    end_points['sum_heads_box_loss'] = box_loss_sum
    end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

    # means average proposal with prediction loss
    loss = query_points_generator_loss_coef * query_points_generation_loss + \
           1.0 / (num_decoder_layers + 1) * (
                   obj_loss_coef * objectness_loss_sum + box_loss_coef * box_loss_sum + sem_cls_loss_coef * sem_cls_loss_sum)
    loss *= 10

    end_points['loss'] = loss
    # for k,v in end_points.items():
    #     if 'loss' in k:
    #         print(k, v)




    return loss, end_points
