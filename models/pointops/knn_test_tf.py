import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
import time

sampling_module=tf.load_op_library('knn_so_server.so') #For server


# xyz = np.load(os.path.join('..','..','..','RepSurf/segmentation/modules/pointops','random numbers.npy'))
xyz = np.random.random([50000,3])
xyz_tf = tf.convert_to_tensor(xyz, dtype=tf.float32)
xyz_tf = tf.tile(tf.expand_dims(xyz,0), [8,1,1])
xyz_tf = tf.cast(xyz_tf, tf.float32)

offset = tf.convert_to_tensor(np.arange(50)*1000 + 1000, dtype=tf.int32)
offset = tf.tile(tf.expand_dims(offset,0), [8,1])

start = time.time()
idx, dist = sampling_module.knn_sample(xyz_tf, xyz_tf, offset, offset, 9)
print(time.time() - start)
print(idx)
print(tf.math.sqrt(dist))