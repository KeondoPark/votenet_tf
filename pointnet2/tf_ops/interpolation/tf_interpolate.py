import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

import json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

# environ_file = os.path.join(ROOT_DIR,'configs','environ.json')
# environ = json.load(open(environ_file))['environ']

# if 'server' in environ:       
if True:
    interpolate_module=tf.load_op_library('/home/keondopark/votenet_tf/pointnet2/tf_ops/interpolation/tf_interpolate_so_server.so') #For server
    # interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so_server.so')) #For server
elif environ == 'jetson':
    interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so')) #For Jetson Nano

def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz1, xyz2)
ops.NoGradient('ThreeNN')

def three_nn_gpu(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn_gpu(xyz1, xyz2)
ops.NoGradient('ThreeNnGpu')

def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate(points, idx, weight)

def three_interpolate_gpu(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate_gpu(points, idx, weight)

@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]

@tf.RegisterGradient('ThreeInterpolateGpu')
def _three_interpolate_gpu_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_gpu_grad(points, idx, weight, grad_out), None, None]


if __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)

    class testModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer1 = tf.keras.layers.Dense(64)
            self.layer2 = tf.keras.layers.Dense(1)

            xyz1 = np.random.random((8,1024,3)).astype('float32')
            xyz2 = np.random.random((8,512,3)).astype('float32')
            self.xyz1 = tf.constant(xyz1)
            self.xyz2 = tf.constant(xyz2)
                    
        def call(self, inputs):           

            # dist, idx = three_nn(xyz1, xyz2)
            # weight = tf.ones_like(dist)/3.0            
            # interpolated_points = three_interpolate(inputs, idx, weight)    

            x = self.layer1(inputs)
            dist_gpu, idx_gpu = three_nn_gpu(self.xyz1, self.xyz2)            
            weight_gpu = tf.ones_like(dist_gpu)/3.0
            interpolated_points_gpu = three_interpolate_gpu(x, idx_gpu, weight_gpu)
            x = tf.keras.layers.Flatten()(interpolated_points_gpu)
            out = self.layer2(x)
            print(out.shape)
            return out

    model = testModel()

    with tf.GradientTape() as tape:
        pts = np.random.random((8,512,64)).astype('float32')        
        points = tf.constant(pts)
        out = model(points)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss_fn(tf.convert_to_tensor(np.expand_dims(np.array([1,0,1,0,1,1,2,1], dtype=np.int64),-1)), out)

    grads = tape.gradient(loss, model.trainable_weights)
    print("grads", grads)

    # print("interpolation", '*'*20)
    # print(tf.reduce_sum(interpolated_points[0] - interpolated_points_gpu[0]))
    # print(tf.reduce_sum(interpolated_points[1] - interpolated_points_gpu[1]))
    # print(tf.reduce_sum(interpolated_points[-1] - interpolated_points_gpu[-1]))
    # print(tf.reduce_sum(interpolated_points - interpolated_points_gpu))
    # print(interpolated_points - interpolated_points_gpu)





    # with tf.Session('') as sess:
    #     now = time.time() 
    #     for _ in range(100):
    #         ret = sess.run(interpolated_points)
    #     print(time.time() - now)
    #     print(ret.shape, ret.dtype)
        #print ret
