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
