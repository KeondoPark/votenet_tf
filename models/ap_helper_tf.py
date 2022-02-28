# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_cls, eval_det_multiprocessing
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd_utils import extract_pc_in_box3d

#import torch

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """    

    pred_center = end_points['center'] # B,num_proposal,3
    K = tf.shape(end_points['heading_scores'])[1] #num_proposal
    
    pred_heading_class = tf.math.argmax(end_points['heading_scores'], axis = -1) #(B, K)        
    pred_heading_residual = tf.gather(end_points['heading_residuals'], axis=2, 
                                    indices=tf.expand_dims(pred_heading_class, axis=-1), batch_dims=2) #(B, K, num_heading_bin) -> (B, K, 1)    
    pred_heading_residual = tf.squeeze(pred_heading_residual, axis=[2])       

    pred_size_class = tf.math.argmax(end_points['size_scores'], axis=-1) # B,num_proposal
    
    # (B, K, 10, 3) -> (B, K, 1, 3)
    pred_size_residual = tf.gather(end_points['size_residuals'], axis=2, 
                                indices=tf.expand_dims(pred_size_class, axis=-1), batch_dims=2)        
    pred_size_residual = tf.squeeze(pred_size_residual, axis=[2]) # B,num_proposal,3            
    
    pred_sem_cls = tf.math.argmax(end_points['sem_cls_scores'], axis=-1) # B,num_proposal
    sem_cls_probs = softmax(end_points['sem_cls_scores'].numpy()) # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs,-1) # B,num_proposal


    """
    Pytorch validation    

    pred_center_torch = torch.tensor(end_points['center'].numpy()) # B,num_proposal,3
    heading_scores_torch = torch.tensor(end_points['heading_scores'].numpy())
    pred_heading_class_torch = torch.argmax(heading_scores_torch, -1) # B,num_proposal
    heading_residuals_torch = torch.tensor(end_points['heading_residuals'].numpy())
    pred_heading_residual_torch = torch.gather(heading_residuals_torch, 2,
        pred_heading_class_torch.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_residual_torch.squeeze_(2)
    print("Heading scores:", end_points['heading_scores'][0])
    print("(Tensorflow)Heading class:", pred_heading_class[0])
    print("(Torch)Heading class:", pred_heading_class_torch[0])
    
    print("(Tensorflow)Heading residual:", pred_heading_residual[0])
    print("(Torch)Heading residual:", pred_heading_residual_torch[0])

    size_scores_torch  = torch.tensor(end_points['size_scores'].numpy())
    pred_size_class_torch = torch.argmax(size_scores_torch, -1) # B,num_proposal
    print("Size scores:", end_points['size_scores'][0])
    print("(Tensorflow)Size class:", pred_size_class[0])
    print("(Torch)Size class:", pred_size_class_torch[0])
    print("(Tensorflow)Size residual before gather:", end_points['size_residuals'][0])    
    size_residuals_torch = torch.tensor(end_points['size_residuals'].numpy())
    pred_size_residual_torch = torch.gather(size_residuals_torch, 2,
        pred_size_class_torch.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual_torch.squeeze_(2)    
    print("(Tensorflow)Size residual:", pred_size_residual[0])
    print("(Torch)Size residual:", pred_size_residual_torch[0])

    sem_cls_scores_torch  = torch.tensor(end_points['sem_cls_scores'].numpy())
    pred_sem_cls_torch = torch.argmax(sem_cls_scores_torch, -1) # B,num_proposal
    sem_cls_probs_torch = softmax(sem_cls_scores_torch.detach().cpu().numpy()) # B,num_proposal,10
    pred_sem_cls_prob_torch = np.max(sem_cls_probs_torch,-1) # B,num_proposal
    print("(Tensorflow)pred_sem_cls_prob:", pred_sem_cls_prob[0])
    print("(Torch)pred_sem_cls_prob:", pred_sem_cls_prob_torch[0])
    """

    num_proposal = tf.shape(pred_center)[1] 
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = tf.shape(pred_center)[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_heading_angle = np.zeros((bsize, num_proposal))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle(\
                pred_heading_class[i,j].numpy(), pred_heading_residual[i,j].numpy())
            box_size = config_dict['dataset_config'].class2size(\
                int(pred_size_class[i,j].numpy()), pred_size_residual[i,j].numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i,j,:])
            pred_corners_3d_upright_camera[i,j] = corners_3d_upright_camera
            pred_heading_angle[i,j] = heading_angle

    K = tf.shape(pred_center)[1] # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].numpy()[:,:,0:3] # B,N,3
        for i in range(bsize):
            pc = batch_pc[i,:,:] # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i,j,:,:] # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i,j] = 0
        # -------------------------------------

    # detach() returns a new tensor, detached from the current graph / shares memory storage with original tensor  
    # #============= Validation =======================
    #print("end_points['objectness_scores'][5]:", end_points['objectness_scores'][5])
    #============= Validation =======================  
    obj_logits = tf.stop_gradient(end_points['objectness_scores']).numpy()
    obj_prob = softmax(obj_logits)[:,:,1] # (B,K)
    #============= Validation =======================
    #print("obj_prob[5]:", obj_prob[5])
    #============= Validation =======================
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K,5))
            for j in range(K):
                boxes_2d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,2] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,4] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,7))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1

        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,8))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
                boxes_3d_with_prob[j,7] = pred_sem_cls[i,j] # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['conf_thresh'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)       
            pred_mask[i, nonempty_box_inds[pick]] = 1     
            #============= Validation =======================
            #if i == 5: print("Pred mask:", pred_mask[i])
            #============= Validation =======================
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                cur_list += [(ii, pred_corners_3d_upright_camera[i,j], sem_cls_probs[i,j,ii]*obj_prob[i,j]) \
                    for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i,j].numpy(), pred_corners_3d_upright_camera[i,j], obj_prob[i,j], pred_heading_angle[i,j]) \
                for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']])
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls, pred_mask

def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """    

    bsize = end_points['center_label'].shape[0]
    K2 = end_points['center_label'].shape[1] # K2==MAX_NUM_OBJ
    box_label_mask = end_points['box_label_mask']

    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(end_points['center_label'][:,:,0:3].numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i,j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(end_points['heading_class_label'][i,j].numpy(), end_points['heading_residual_label'][i,j].numpy())
            box_size = config_dict['dataset_config'].class2size(int(end_points['size_class_label'][i,j].numpy()), end_points['size_residual_label'][i,j].numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i,j,:])
            gt_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(end_points['sem_cls_label'][i,j].numpy(), gt_corners_3d_upright_camera[i,j]) for j in range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i,j]==1])
    #end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls

class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
