import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
environ_file = os.path.join(ROOT_DIR,'configs','environ.json')
environ = json.load(open(environ_file))['environ']

if 'server' in environ:    
    grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so_server.so')) #For server
elif environ == 'jetson':
    grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so')) #For Jetson Nano


def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPoint')


def query_ball_point_cpu(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return grouping_module.query_ball_point_cpu(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPointCpu')

def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')

def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)

@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

def group_point_cpu(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point_cpu(points, idx)

@tf.RegisterGradient('GroupPointCpu')
def _group_point_cpu_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_cpu_grad(points, idx, grad_out), None]

def knn_point(k, xyz1, xyz2):
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
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print(b, n, c, m)
    print(xyz1, (b,1,n,c))
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    print(dist, k)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    print(idx, val)
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return idx, val


def knn_with_attention(k, xyz1, xyz2, attn):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
        attn: (batch_size, npoint, ndataset, c) float32 array, Attention score
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = tf.shape(xyz1)[0]
    n = tf.shape(xyz1)[1]
    c = tf.shape(xyz1)[2]
    m = tf.shape(xyz2)[1]
    
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1) #(b, m, n)
    dist = dist / (attn + 1e-6)    
    
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    #idx = tf.cast(idx, dtype=tf.int32)
    val = tf.slice(out, [0,0,0], [-1,-1,k])    
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return idx, val

if __name__=='__main__':
    knn=False
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((8,16+4,64)).astype('float32')
    tmp1 = np.random.random((8,16,3)).astype('float32')
    tmp2 = np.random.random((8,4,3)).astype('float32')
    
    points = tf.constant(pts)
    xyz1 = tf.constant(tmp1)
    xyz2 = tf.constant(tmp2)
    xyz2 = tf.concat([xyz1, xyz2], axis=1)
    radius = 0.5 
    nsample = 4
    if knn:
        _, idx = knn_point(nsample, xyz1, xyz2)
        grouped_points = group_point(points, idx)
    else:
        idx_gpu, _ = query_ball_point(radius, nsample, xyz1, xyz2)
        idx_cpu, _ = query_ball_point_cpu(radius, nsample, xyz1, xyz2)
        # print(tf.reduce_sum(idx_gpu - idx_cpu, axis=1))
        print(tf.reduce_sum(idx_gpu - idx_cpu))
        

        grouped_points = group_point(points, idx_gpu)
        grouped_points_cpu = group_point_cpu(points, idx_cpu)
        print(tf.reduce_sum(grouped_points.numpy() - grouped_points_cpu.numpy()))
            #grouped_points_grad = tf.ones_like(grouped_points)
            #points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
    # with tf.Session('') as sess:
    #     now = time.time() 
    #     for _ in range(100):
    #         ret = sess.run(grouped_points)
    #     print (time.time() - now)
    #     print (ret.shape, ret.dtype)
    #     print (ret)
