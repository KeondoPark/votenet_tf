''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))
#sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so_server.so'))


def prob_sample(inp,inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
# TF1.0 API requires set shape in C++
#@tf.RegisterShape('ProbSample')
#def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return sampling_module.gather_point(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_module.gather_point_grad(inp,idx,out_g),None]

def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')

def farthest_point_sample_bg(npoint,inp,weight=1, isFront=0):
    '''
input:
    int32
    batch_size * ndataset * 4   float32
    wegiht: weight for painted points' distance
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample_bg(inp, npoint,weight,isFront)
ops.NoGradient('FarthestPointSampleBg')


def farthest_point_sample_bg2(npoint,inp,weight1=1, weight2=1, isFront1=0, isFront2=0):
    '''
input:
    int32
    batch_size * ndataset * 4   float32
    wegiht: weight for painted points' distance
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample_bg2(inp, npoint, weight1, weight2, isFront1, isFront2)
ops.NoGradient('FarthestPointSampleBg2')



if __name__=='__main__':
    import numpy as np
    np.random.seed(100)
    npoint = 4
    N = 16
    #inputs = np.random.random((1,8,4))
    
    idxs = np.array(range(N))
    cos_x = np.cos(np.pi/N * 2 * idxs)
    sin_x = np.sin(np.pi/N * 2 * idxs)
    z_coord = np.array([0]*N)
    isBg = np.array([1] * (N//2) + [0] * (N//2))
    #isBg = np.array([0] * 16)

    inputs = np.vstack([cos_x, sin_x, z_coord, isBg])
    inputs = inputs.transpose()    
    print(inputs.shape)
    inputs = np.expand_dims(inputs, axis=0)
    #res = farthest_point_sample_bg(npoint, inputs, 0.01, 0)
    res = farthest_point_sample_bg2(npoint, inputs, 0.01, 100, -1, -1)
    print(res)
    

    """
    triangles=np.random.rand(1,5,3,3).astype('float32')
    with tf.device('/device:GPU:0'):
        inp=tf.constant(triangles)
        tria=inp[:,:,0,:]
        trib=inp[:,:,1,:]
        tric=inp[:,:,2,:]
        areas=tf.sqrt(tf.reduce_sum(tf.linalg.cross(trib-tria,tric-tria)**2,2)+1e-9)
        randomnumbers=tf.random.uniform((1,8192))

        print("Areas", areas)
        print("Randomnumbers", randomnumbers)

        triids=prob_sample(areas, randomnumbers)        
        print("triids", triids)
        
        tria_sample=gather_point(tria,triids)
        trib_sample=gather_point(trib,triids)
        tric_sample=gather_point(tric,triids)
        us=tf.random.uniform((1,8192))
        vs=tf.random.uniform((1,8192))
        uplusv=1-tf.abs(us+vs-1)
        uminusv=us-vs
        us=(uplusv+uminusv)*0.5
        vs=(uplusv-uminusv)*0.5
        pt_sample=tria_sample+(trib_sample-tria_sample)*tf.expand_dims(us,-1)+(tric_sample-tria_sample)*tf.expand_dims(vs,-1)
        print('pt_sample: ', pt_sample)
        reduced_sample=gather_point(pt_sample,farthest_point_sample(1024,pt_sample))
        print(reduced_sample)
    """


    #with tf.compat.v1.Session('') as sess:
    #    ret=sess.run(reduced_sample)
    #print(ret.shape,ret.dtype)
    #import cPickle as pickle
    #pickle.dump(ret,open('1.pkl','wb'),-1)
