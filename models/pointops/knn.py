import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'knn_so_server.so')) #For server

def knn_sample(xyz, new_xyz, offset, new_offset, k):
    '''
input:
    k: int32
    xyz: B, N, 3        float32
    new_xyz: B, M, 3    float32
    offset: B, S         int32
    new_offset: B, S     int32
   
returns:
    idx: B, M, k         int32
    dist: B, M, k        float32
    '''
    B = tf.shape(xyz)[0]
    # for i in range(B):
    #     assert tf.math.equal(offset[i,-1], xyz.shape[1]), "The number of points in xyz should match the last element of offset"
    #     assert tf.math.equal(new_offset[i,-1], new_xyz.shape[1]), "The number of points in new_xyz should match the last element of new_offset"
    return sampling_module.knn_sample(xyz, new_xyz, offset, new_offset, k)
ops.NoGradient('KnnSample')