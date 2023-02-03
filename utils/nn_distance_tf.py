# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Chamfer distance in Pytorch -> Tensorflow
Author: Charles R. Qi
Modified by Keondo Park
"""

import tensorflow as tf
#import torch
import numpy as np


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = tf.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    # quadratic = tf.clip_by_value(abs_error, clip_value_min = 0, clip_value_max=delta)
    # linear = (abs_error - quadratic)
    # loss = 0.5 * quadratic**2 + delta * linear
    inv_delta = tf.divide(1.0, float(delta))
    loss = tf.where(abs_error < delta, 0.5 * abs_error * abs_error * inv_delta, abs_error - 0.5*delta)
    return loss

def huber_loss_torch(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1, axis=2), multiples=[1,1,M,1])
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2, axis=1), multiples=[1,N,1,1])
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = tf.reduce_sum(huber_loss(pc_diff, delta), axis=-1) # (B,N,M)
    elif l1:
        pc_dist = tf.reduce_sum(tf.abs(pc_diff), axis=-1) # (B,N,M)
    else:
        pc_dist = tf.reduce_sum(pc_diff**2, axis=-1) # (B,N,M)
    idx1 = tf.argmin(pc_dist, axis=2) # (B,N)    
    dist1 = tf.reduce_min(pc_dist, axis=2)
    idx2 = tf.argmin(pc_dist, axis=1) #(B,M)
    dist2 =  tf.reduce_min(pc_dist, axis=1)    #(B,M)
    return dist1, idx1, dist2, idx2

def demo_nn_distance():
    np.random.seed(0)
    pc1arr = np.random.random((1,5,3))
    pc2arr = np.random.random((1,6,3))
    pc1 = tf.convert_to_tensor(pc1arr.astype(np.float32))
    pc2 = tf.convert_to_tensor(pc2arr.astype(np.float32))
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            dist[i,j] = np.sum((pc1arr[0,i,:] - pc2arr[0,j,:]) ** 2)
    print(dist)
    print('-'*30)
    print('L1smooth dists:')
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2, True)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            error = np.abs(pc1arr[0,i,:] - pc2arr[0,j,:])
            quad = np.minimum(error, 1.0)
            linear = error - quad
            loss = 0.5*quad**2 + 1.0*linear
            dist[i,j] = np.sum(loss)
    print(dist)


# class SigmoidFocalClassificationLoss(tf.keras.losses.Loss):
class SigmoidFocalClassificationLoss():
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input_tensor, target):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = tf.clip_by_value(input_tensor, clip_value_min=0, clip_value_max=tf.float32.max) - input_tensor * target + \
               tf.math.log1p(tf.math.exp(-tf.abs(input_tensor)))
        return loss

    def forward(self, input_tensor, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """


        pred_sigmoid = tf.math.sigmoid(input_tensor)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * tf.math.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input_tensor, target)

        loss = focal_weight * bce_loss

        weights = tf.expand_dims(weights, -1)
        assert weights.shape.__len__() == loss.shape.__len__()

        # if tf.reduce_sum(loss * weights) < 0:
        #     print("input tensor", input_tensor)
        #     print("target", target)
        #     print("weights", weights)
        #     print("bce loss", tf.reduce_sum(bce_loss))
        #     print("focal_weight", tf.reduce_sum(focal_weight))
        #     print("weights", tf.reduce_sum(weights))

        return loss * weights

if __name__ == '__main__':
    demo_nn_distance()
