import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
from tf_ops.interpolation import tf_interpolate



# def sample_and_group(npoint, nsample, xyz, normal, feature, return_polar=False):
#     # npoint: How many points are sampled from Furthest point sampling
#     # nsample: How many neighbors are queried from grouping
#     # xyz: (B, N, 3)
#     # normal: (B, ??)

#     # sample
#     if stride > 1:
#         # Guess when stride=4
#         # offset: [1000,2000,3000,...,50000]
#         # new_offset: [250,500,750,...,12500]
#         # 250 points are sampled from first 1000 points
#         # next 250 points are from second 1000 points ...
#         new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
#         for i in range(1, offset.shape[0]):
#             sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
#             new_offset.append(sample_idx)
#         new_offset = torch.cuda.IntTensor(new_offset)
#         if num_sector > 1 and training:
#             fps_idx = pointops.sectorized_fps(center, offset, new_offset, num_sector)  # [M]
#         else:
#             fps_idx = pointops.furthestsampling(center, offset, new_offset)  # [M]
#         new_center = center[fps_idx.long(), :]  # [M, 3]
#         new_normal = normal[fps_idx.long(), :]  # [M, 3]
#     else:
#         new_center = center
#         new_normal = normal
#         new_offset = offset

#     inds = tf_sampling.farthest_point_sample(npoint, xyz) 
#     new_center = tf_sampling.gather_point(xyz, inds) #(B, M, 3)

#     # group
#     N, M, D = xyz.shape[1], new_center.shape[1], normal.shape[1]
#     group_idx, _ = knn_point(xyz, new_center, nsample)  # [B, M, nsample]
#     group_center = xyz[group_idx.view(-1).long(), :].view(M, nsample, 3)  # [M, nsample, 3]
#     group_normal = normal[group_idx.view(-1).long(), :].view(M, nsample, D)  # [M, nsample, 10]
#     group_center_norm = group_center - new_center.unsqueeze(1)
#     if return_polar:
#         group_polar = xyz2sphere(group_center_norm)
#         group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)

#     if feature is not None:
#         C = feature.shape[1]
#         group_feature = feature[group_idx.view(-1).long(), :].view(M, nsample, C)
#         new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1)   # [npoint, nsample, C+D]
#     else:
#         new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

#     return new_center, new_normal, new_feature, new_offset



def knn_point(xyz1, xyz2, k=9):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0]
    n = xyz1.get_shape()[1]
    c = xyz1.get_shape()[2]
    m = xyz2.get_shape()[1]
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)    
    out, outi = tf.math.top_k(-dist, k)
    # idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    # val = tf.slice(out, [0,0,0], [-1,-1,k])
    # print(idx, val)
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return outi, out

def _fixed_rotate(xyz):
    # xyz: (B, N, 3)
    # output: (B, N, 3), rotated xyz
    # y-axis:45deg -> z-axis:45deg
    rot_mat = tf.convert_to_tensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]], dtype=tf.float32)
    # rot_mat = torch.FloatTensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]]).to(xyz.device)
    return tf.matmul(xyz, rot_mat)

def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] 
    :return: (rho, theta, phi) [B, N, 3] 
    """
    rho = tf.math.sqrt(tf.reduce_sum(tf.math.pow(xyz, 2), axis=-1, keepdims=True)) #(B, N, 1)
    rho = tf.clip_by_value(rho, clip_value_min=0, clip_value_max=tf.float32.max)  # range: [0, inf]
    theta = tf.math.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = tf.math.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    theta = tf.where(rho == 0, 0.0, theta)

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = tf.concat([rho, theta, phi], axis=-1)
    return out


def group_by_umbrella_v2(xyz, new_xyz, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [B, N, 3]
    :param new_xyz: [B, M, 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    M = new_xyz.shape[1]
    group_idx, _ = knn_point(xyz, new_xyz, k)  # (B, M, K)    
    xyz_tile = tf.tile(tf.expand_dims(xyz, -3), [1,M,1,1])
    
    
    #axis: xyz_tile's shape, batch_dims: indices's shape
    group_xyz = tf.gather(xyz_tile, axis=-2, indices=group_idx, batch_dims=-1) #(B, M, K, 3)     
    
    group_xyz_norm = group_xyz - tf.expand_dims(new_xyz, axis=-2) #(B, M, K, 3)
    group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]  # (B, M, K)
    sort_idx = tf.argsort(group_phi, axis=-1)  # (B, M, K)
    
    # [M, K-1, 1, 3]    
    sorted_group_xyz = tf.gather(group_xyz_norm, axis=-2, indices=sort_idx, batch_dims=-1) # (B, M, K, 3)    
    sorted_group_xyz = tf.expand_dims(group_xyz_norm, axis=-2) # (B, M, K, 1, 3)    
    sorted_group_xyz_roll = tf.roll(sorted_group_xyz, shift=-1, axis=-3) # (B, M, K, 1, 3)
    
    group_centriod = tf.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = tf.concat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], axis=-2)

    return umbrella_group_xyz


def group_by_umbrella(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    # group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    # group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    # group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2) # [M, K, 3]
    # group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [M, K-1]
    # sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # # [M, K-1, 1, 3]
    # sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    # sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    # group_centriod = torch.zeros_like(sorted_group_xyz)
    # umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)
    
    group_idx, _ = knn_point(xyz, new_xyz, k)  # (B, M, K)
    xyz_tile = tf.tile(tf.expand_dims(xyz, 1), [1,M,1,1])
    group_xyz = tf.gather(xyz_tile, axis=2, indices=group_idx, batch_dims=2) #(B, M, K, 3)    
    group_xyz_norm = group_xyz - tf.expand_dims(new_xyz, axis=-2) #(B, M, K, 3)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # (B, M, K)
    sort_idx = tf.argsort(group_phi, axis=-1)  # (B, M, K)

    # [M, K-1, 1, 3]    
    sorted_group_xyz = tf.gather(group_xyz_norm, indices=sort_idx, batch_dims=-1) # (B, M, K, 3)
    sorted_group_xyz = tf.expand_dims(group_xyz_norm, axis=-2) # (B, M, K, 1, 3)
    sorted_group_xyz_roll = tf.roll(sorted_group_xyz, shift=-1, axis=-3) # (B, M, K, 1, 3)
    group_centriod = tf.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = tf.concat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def sort_factory(s_type):
    if s_type is None:
        return group_by_umbrella
    elif s_type == 'fix':
        return group_by_umbrella_v2
    else:
        raise Exception('No such sorting method')

def cal_normal(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    input: group_xyz: [B, N, 3, 3]
    output: unit normal vector: [B, N, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = tf.linalg.cross(edge_vec1, edge_vec2) # [B, N, 3]
    unit_nor = nor / tf.norm(nor, axis=-1, keepdims=True)  # [B, N, 3] 
    if not is_group:
        pos_mask = tf.where(unit_nor[..., 0] > 0, 1.0, -1.0) # keep x_n positive        
    else:
        pos_mask = tf.where(unit_nor[..., 0:1, 0] > 0, 1.0, -1.0)        
    unit_nor = unit_nor * tf.expand_dims(pos_mask, -1)

    random_prob = tf.random.uniform(unit_nor.shape[:-1], minval=0, maxval=2, dtype=tf.int32) # 0 or 1
    random_mask = tf.cast(tf.expand_dims(random_prob,-1), tf.float32) * 2.0 - 1.0
    unit_nor = random_mask * unit_nor

    return unit_nor


def cal_center(group_xyz):
    """
    Calculate Global Coordinates of the Center of Triangle

    :input group_xyz: [B, N, 3, 3]
    :output: [B, N, 3] 
    """
    center = tf.reduce_mean(group_xyz, axis=-2)
    return center

def cal_const(normal, center, is_normalize=True):
    """
    Calculate Constant Term (Standard Version, with x_normal to be 1)

    math::
        const = x_nor * x_0 + y_nor * y_0 + z_nor * z_0

    :param is_normalize:
    :param normal: [B, N, 3]
    :param center: [B, N, 3]
    :return: [B, N, 1] 
    """
    const = tf.reduce_sum(normal * center, axis=-1, keepdims=True)
    # factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    factor = 3.0 ** 0.5
    const = const / factor if is_normalize else const

    return const


def check_nan_umb(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [B, N, K, 1]
    :param center: [B, N, K, 3]
    :param normal: [B, N, K, 3]
    """
    B = tf.shape(normal)[0]
    N = normal.shape[1]
    
    mask = tf.reduce_sum(tf.cast(tf.math.is_nan(normal),tf.int32), axis=-1) > 0 # boolean (B, N, K) if NaN 1, else 0    
    mask_first = tf.math.argmax(tf.cast(~mask,tf.int32), axis=-1) # index of not NaN, (B, N)

    normal_first = tf.gather(normal, indices=tf.expand_dims(mask_first,-1), axis=-2, batch_dims=-2)
    normal_first = tf.tile(normal_first, [1, 1, K, 1])
    
    center_first = tf.gather(center, indices=tf.expand_dims(mask_first,-1), axis=-2, batch_dims=-2)
    center_first = tf.tile(center_first, [1, 1, K, 1]) 
    
    mask = tf.expand_dims(mask,-1)
    mask_tile = tf.tile(mask, [1,1,1,3])
    normal = tf.where(mask_tile, normal_first, normal)
    center = tf.where(mask_tile, center_first, center)

    if pos is not None:        
        pos_first = tf.gather(pos, indices=tf.expand_dims(mask_first,-1), axis=1, batch_dims=1)
        pos_first = tf.tile(pos_first, [1, 1, K, 1]) 
        pos = tf.where(mask, pos_first, pos)
        return normal, center, pos
    return normal, center


class UmbrellaSurfaceConstructor(layers.Layer):
    """
    Umbrella Surface Representation Constructor
    """

    def __init__(self, k, in_channel, out_channel, random_inv=True, sort='fix', act='relu'):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.random_inv = random_inv

        self.conv1 = layers.Conv2D(filters=out_channel, kernel_size=1) 
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=out_channel, kernel_size=1) 
         
         if act == 'relu6':
            maxval = 6
        else:
            maxval = None
        self.relu1 = layers.ReLU(maxval)

        self.sort_func = sort_factory(sort)

    def call(self, center, offset):
        # umbrella surface reconstruction
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]
        # (B, M, K, 3, 3)

        # normal
        group_normal = cal_normal(group_xyz, offset, random_inv=self.random_inv, is_group=True)
        # (B, M, K, 3)

        # coordinate
        group_center = cal_center(group_xyz)
        # (B, M, K, 3)

        # polar
        group_polar = xyz2sphere(group_center) 
        # (B, M, K, 3)

        # surface position
        group_pos = cal_const(group_normal, group_center)
        # (B, M, K, 1)

        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        # (B, M, K, 3), (B, M, K, 3), (B, M, K, 1)

        new_feature = tf.concat([group_polar, group_normal, group_pos, group_center], dim=-1)  # P+N+SP+C: 10
        # (B, M, K, 10)
        # new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, G]

        # mapping
        new_feature = self.conv2(self.relu1(self.bn1(self.conv1(new_feature))))        

        # aggregation
        new_feature = tf.reduce_sum(new_feature, axis=1)

        return new_feature