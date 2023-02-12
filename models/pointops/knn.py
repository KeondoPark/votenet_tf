import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'knn_so_server.so')) #For server

def knn_sample(k, xyz, new_xyz, offset, new_offset):
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
    return sampling_module.knn_sample(xyz, new_xyz, offset, new_offset, k)
ops.NoGradient('KnnSample')