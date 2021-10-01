import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class MultiheadAttention(tf.keras.layers.Layer):
  def __init__(self, n_heads, embed_dim):
    super().__init__()
    self.n_heads = n_heads
    self.embed_dim = embed_dim
    self.emb = layers.Dense(embed_dim * n_heads * 3, use_bias=False) 
    d = tf.cast(embed_dim, dtype=tf.float32)    
    self.scaling = tf.math.sqrt(d)

  def call(self, inputs):    
    """
    inputs: (B, num_seed, features)
    """
    num_seed = tf.shape(inputs)[1]
    embedding = self.emb(inputs) # (B, n, d * h * 3)    
    heads = layers.Reshape((num_seed, self.embed_dim, self.n_heads, 3))(embedding) #(B, n, d, h, 3)

    heads = tf.transpose(heads, perm=[0,4,3,1,2]) # (B, 3, h, n, d)
    q = heads[:,0,:,:,:] #(B, h, n, d)
    k = heads[:,1,:,:,:] #(B, h, n, d)
    v = heads[:,2,:,:,:] #(B, h, n, d)
    
    qk = tf.matmul(q, k, transpose_b=True) # (B, h, n, n)    
    qk = tf.keras.backend.softmax(qk) * self.scaling # (B, h, n, n)    
    
    output = tf.matmul(qk, v) # (B, h, n, d)
    output = tf.transpose(output, perm=[0,2,1,3]) #(B, n, h, d)
    output = layers.Reshape((num_seed, -1))(output)
    return output