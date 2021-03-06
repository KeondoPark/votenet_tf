# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

import tensorflow as tf

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].numpy()
        vote_xyz = end_points['vote_xyz'].numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].numpy()
    objectness_scores = end_points['objectness_scores'].numpy() # (B,K,2)
    pred_center = end_points['center'].numpy() # (B,K,3)
    B, K, _ = end_points['heading_scores'].shape  # K = num_proposal
    #Find maximum heading scores and get the corresponding heading residuals
    pred_heading_class = tf.math.argmax(end_points['heading_scores'], axis=-1) # B,K
    pred_heading_residual = tf.gather(end_points['heading_residuals'], axis=2, 
                                    indices=tf.expand_dims(pred_heading_class, axis=-1), batch_dims=2) #(B, K, num_heading_bin) -> (B, K, 1)
    
    #coords = tf.concat([tf.expand_dims(tf.stack(tf.meshgrid(tf.range(0,B), tf.range(0,K), indexing='ij'), axis=-1), axis=2), \
    #                   tf.expand_dims(tf.expand_dims(pred_heading_class, axis=-1), axis=-1)], axis=-1) # (B, K, 1, 2) + (B, K, 1, 1) -> (B, K, 1, 3)
    #heading_residuals: (B, K, num_heading_bin)
    #pred_heading_residual = tf.gather_nd(end_points['heading_residuals'], coords) # B,K,1   


    pred_heading_class = pred_heading_class.numpy() # B,K
    pred_heading_residual = tf.squeeze(pred_heading_residual, axis=[2]).numpy() # B,K

    #Find maximum size scores and get the corresponding size residuals
    #size_scores: (B, K, num_size_cluster)
    pred_size_class = tf.cast(tf.math.argmax(end_points['size_scores'], axis=-1), dtype=tf.int32) # B,K
    pred_size_residual = tf.gather(end_points['size_residuals'], axis=2, 
                                indices=tf.expand_dims(pred_size_class, axis=-1), batch_dims=2)        
    pred_size_residual = tf.squeeze(pred_size_residual, axis=[2]) # B,num_proposal,3  
    #coords = tf.concat([tf.expand_dims(tf.stack(tf.meshgrid(tf.range(0,B), tf.range(0,K), indexing='ij'), axis=-1), axis=2), \
    #                   tf.expand_dims(tf.expand_dims(pred_size_class, axis=-1), axis=-1)], axis=-1) # (B, K, 1, 2) + (B, K, 1, 1) -> (B, K, 1, 3)
    
    #size_residuals: (B, K, num_size_cluster, 3)
    #pred_size_residual = tf.gatehr_nd(end_points['size_residuals'], coords) # (B, K, 1, 3)
    #pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_xyz' in end_points:
            pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                pred_size_class[i,j], pred_size_residual[i,j])
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points['center_label'].numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].numpy() # B,K2
    gt_heading_class = end_points['heading_class_label'].numpy() # B,K2
    gt_heading_residual = end_points['heading_residual_label'].numpy() # B,K2
    gt_size_class = end_points['size_class_label'].numpy() # B,K2
    gt_size_residual = end_points['size_residual_label'].numpy() # B,K2,3
    objectness_label = end_points['objectness_label'].numpy() # (B,K,)
    objectness_mask = end_points['objectness_mask'].numpy() # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_mask[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        pc_util.write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply'%(idx_beg+i)))

    # OPTIONALL, also dump prediction and gt details
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()
