import tensorflow as tf
from tensorflow.keras import layers

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

from typing import List
import time
import numpy as np
from modules_tf import PositionEmbeddingLearned

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim=288, nheads=8, dropout=0.0):
        super().__init__()        
        self.nheads = nheads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nheads
        self.emb_q = layers.Dense(embed_dim, use_bias=True, kernel_initializer=tf.keras.initializers.he_uniform())
        self.emb_k = layers.Dense(embed_dim, use_bias=True, kernel_initializer=tf.keras.initializers.he_uniform())
        self.emb_v = layers.Dense(embed_dim, use_bias=True, kernel_initializer=tf.keras.initializers.he_uniform())   
        self.scaling = float(self.head_dim) ** -0.5
        self.out_proj = layers.Dense(embed_dim, use_bias=True, kernel_initializer=tf.keras.initializers.he_uniform())
        self.dropout = dropout

    def call(self, query, key, value):    
        """
        query, key, value dim: (B, N, d*nhead)
        """
        #num_seed = tf.shape(inputs)[1]
        embedding_q = self.emb_q(query) * self.scaling # (B, nq, d*h)
        embedding_k = self.emb_k(key)  # (B, nk, d*h)
        embedding_v = self.emb_v(value)  # (B, nv, d*h)
        # attn = [tf.keras.backend.softmax(tf.matmul(q, k, transpose_b=True)) for q, k in zip(embedding_q, embedding_k)]
        attn = [tf.matmul(embedding_q[:,:,i*self.head_dim:(i+1)*self.head_dim], 
                          embedding_k[:,:,i*self.head_dim:(i+1)*self.head_dim], transpose_b=True)
                for i in range(self.nheads)] # h * (B, nq, nk)

        attn = [tf.keras.backend.softmax(a) for a in attn]

        # qkv_list = [layers.Dropout(self.dropout)(tf.matmul(qk, v)) for qk, v in zip(qk_list, embedding_v)] #(B, n, d) * h
        qkv_list = [tf.matmul(layers.Dropout(self.dropout)(attn[i]), embedding_v[:,:,i*self.head_dim:(i+1)*self.head_dim]) for i in range(self.nheads)] # h * (B, nq, d)

        output = layers.Concatenate(axis=-1)(qkv_list)        #(B, n, d * h)
        
        output = self.out_proj(output)
        return output


class MultiheadAttention2(tf.keras.layers.Layer):
    def __init__(self, embed_dim=288, nheads=8, dropout=0.1, qkv_same=False):
        super().__init__()        
        self.nheads = nheads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nheads
        self.scaling = float(self.head_dim) ** -0.5
        self.out_proj = layers.Dense(embed_dim, use_bias=True, 
                                     kernel_initializer=tf.keras.initializers.he_uniform(), 
                                     bias_initializer=tf.keras.initializers.constant(0.0))
        self.dropout = dropout
        self.qkv_same = qkv_same
    def build(self, input_shape):        
        self.in_proj_weight =  self.add_weight(name='in_proj_weight', 
                                  shape=[self.embed_dim, self.embed_dim * 3],                                  
                                  initializer=tf.keras.initializers.he_uniform(),
                                  trainable=True)
        self.in_proj_bias =  self.add_weight(name='in_proj_bias', 
                                  shape=[1,1,3 * self.embed_dim],
                                  initializer=tf.keras.initializers.constant(0.0),
                                  trainable=True)

    def call(self, query, key, value):    
        """
        query, key, value dim: (B, N, C)
        """      
        bsz = tf.shape(query)[0]
        tgt_len = query.shape[1]
        embed_dim = query.shape[-1]
        query = tf.transpose(query, [1,0,2]) #(N, B, C)
        key = tf.transpose(key, [1,0,2]) #(M, B, C)
        value = tf.transpose(value, [1,0,2]) #(M, B, C)
        
        if self.qkv_same:
            q, k, v = tf.split(tf.matmul(query, self.in_proj_weight) + self.in_proj_bias, num_or_size_splits=3, axis=-1) #(N, B, C) -> (N, B, 3C) -> (N, B, C) * 3
        else:
            _b = self.in_proj_bias
            _start = 0
            _end = self.embed_dim
            _w = self.in_proj_weight[:,_start:_end]
            _b_q = _b[:,:,_start:_end]
            q = tf.matmul(query, _w) + _b_q

            # This is inline in_proj function with in_proj_weight and in_proj_bias            
            _start = self.embed_dim            
            _w = self.in_proj_weight[:,_start:,]
            _b_kv = _b[:,:,_start:]
            k, v = tf.split(tf.matmul(key, _w) +  _b_kv, num_or_size_splits=2, axis=-1)
    
        q = q * self.scaling
        q = tf.reshape(q, [tgt_len, bsz * self.nheads, self.head_dim])
        q = tf.transpose(q, perm=[1,0,2])
        
        k = tf.reshape(k, [-1, bsz * self.nheads, self.head_dim])
        k = tf.transpose(k, perm=[1,0,2])
        v = tf.reshape(v, [-1, bsz * self.nheads, self.head_dim])
        v = tf.transpose(v, perm=[1,0,2])

        attn_output_weights = tf.matmul(q, k, transpose_b=True) # (B*nh, N, H) * (B*nh, H, N) -> (B*nh, N, N)
        attn_output_weights = tf.keras.backend.softmax(attn_output_weights)
        attn_output_weights = layers.Dropout(self.dropout)(attn_output_weights)
        
        attn_output = tf.matmul(attn_output_weights, v) # (B*nh, N, N) * (B*nh, N, H) -> (B*nh, N, H)
        attn_output = tf.transpose(attn_output, [1,0,2]) # (B*nh, N, H) -> (N, B*nh, H)
        attn_output = tf.reshape(attn_output, [tgt_len, bsz, embed_dim]) # (N, B*nh, H) -> (N, B, C)
        
        attn_output = self.out_proj(attn_output) #(N, B, C)        
        attn_output = tf.transpose(attn_output, [1,0,2]) #(B, N, C)
        
        return attn_output

class InducedSetAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim=288, nheads=8, dropout=0.0):
        super().__init__()
        
        self.mab0 = MultiheadAttention(embed_dim, nheads, dropout)
        self.mab1 = MultiheadAttention(embed_dim, nheads, dropout)

    def build(self, input_shape):
        self.I0 =  self.add_weight(name='induce0', 
                                  shape=[1,64,288],
                                  initializer=tf.keras.initializers.he_normal(),
                                  trainable=True)

        # self.I1 =  self.add_weight(name='induce1', 
        #                           shape=[1,128,288],
        #                           initializer=tf.keras.initializers.he_normal(),
        #                           trainable=True)

        super(InducedSetAttention, self).build(input_shape)

    def call(self, query, key, value):    
        """
        query, key, value dim: (B, N, d, nhead)
        """        
        H0 = self.mab0(self.I0, key, key)
        out = self.mab1(query, H0, H0)
        return out



if __name__ == '__main__':
    Nq = 128
    Nk = 256
    D = 288
    nheads = 8

    class testModel(tf.keras.Model):
        def __init__(self, embed_dim=D, nheads=8, dropout=0.1):
            super().__init__()
            self.self_posembed = PositionEmbeddingLearned(embed_dim)
            self.cross_posembed = PositionEmbeddingLearned(embed_dim)
            self.self_attention = MultiheadAttention(embed_dim, nheads, dropout)
            self.cross_attention = MultiheadAttention(embed_dim, nheads, dropout)
            self.norm = layers.LayerNormalization(axis=-1)
            # self.norm = layers.BatchNormalization(axis=-1)
        
        def call(self, inputs):
            query, query_pos, key, key_pos = inputs
            K = query.shape[1]
            C = query.shape[-1]
            N = key.shape[1]
            
            posC = query_pos.shape[-1]
            query_pos = layers.Reshape((K, 1, posC))(query_pos)     
            query_pos_embed = self.self_posembed(query_pos)                        
            query_pos_embed = layers.Reshape((K, C))(query_pos_embed)

            query = layers.Reshape((K, C))(query)
            query += query_pos_embed            
            
            query2 = self.self_attention(query, query, query)
            query = query2 + query

            posC = key_pos.shape[-1]
            key_pos = layers.Reshape((N, 1, posC))(key_pos)     
            key_pos_embed = self.cross_posembed(key_pos)
            key_pos_embed = layers.Reshape((N, C))(key_pos_embed)

            key = layers.Reshape((N, C))(key)

            key += key_pos_embed
            query += query_pos_embed
            query2 = self.cross_attention(query, key, key)
            output = query2 + query

            return output
            

    # class testModel(tf.keras.Model):
    #     def __init__(self, embed_dim=288, nheads=8, dropout=0.1):
    #         super().__init__()
    #         self.attention = InducedSetAttention(embed_dim, nheads, dropout)
    #         self.norm = layers.LayerNormalization(axis=-1)
    #         # self.norm = layers.BatchNormalization(axis=-1)
        
    #     def call(self, inputs):
    #         query, key, value = inputs
    #         output = self.attention(query, key, value)        
    #         # output = self.norm(output)
    #         return output

    model = testModel(D, nheads)

    query_shape = (1,Nq,D)
    query_pos_shape = (1,Nq,6)
    key_shape = (1,Nk,D)
    key_pos_shape = (1,Nk,3)


    query = tf.convert_to_tensor(np.random.random(query_shape))
    query_pos = tf.convert_to_tensor(np.random.random(query_pos_shape))
    key = tf.convert_to_tensor(np.random.random(key_shape))
    key_pos = tf.convert_to_tensor(np.random.random(key_pos_shape))
    inputs = query, query_pos, key, key_pos
    dummy_out = model(inputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization        

    def representative_data_gen():
        # inputs = tf.convert_to_tensor(np.random.random((1,256,128)), dtype=tf.float32)
        query = tf.convert_to_tensor(np.random.random(query_shape), dtype=tf.float32)
        query_pos = tf.convert_to_tensor(np.random.random(query_pos_shape), dtype=tf.float32)
        key = tf.convert_to_tensor(np.random.random(key_shape), dtype=tf.float32)
        key_pos = tf.convert_to_tensor(np.random.random(key_pos_shape), dtype=tf.float32)
        
        inputs = [query, query_pos, key, key_pos]
        yield inputs

    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open('attention_quant_groupfree.tflite', 'wb') as f:
        f.write(tflite_model)
