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
from attention_tf import MultiheadAttention

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=288, nhead=8, dim_feedforward=2048, dropout=0.1,
                self_posembed=None, cross_posembed=None, model_config=None):
    
        super().__init__()
        maxval = None
        if model_config is not None:
            self.use_tflite = model_config['use_tflite']        
            self.q_gran = model_config['q_gran']
            act = model_config['activation'] if 'activation' in model_config else 'relu6'
            if act == 'relu6':
                maxval = 6            


        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = layers.Dense(dim_feedforward, kernel_initializer=tf.keras.initializers.he_uniform())
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model, kernel_initializer=tf.keras.initializers.he_uniform())

        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.norm3 = layers.LayerNormalization(axis=-1)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
       
        self.activation = layers.ReLU(maxval)        

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed


    def with_pos_embed(self, tensor, pos_embed=None):
        return tensor if pos_embed is None else tensor + pos_embed

    def call(self, query, key, query_pos, key_pos):
        '''
            query: (B, K, 1, C) K: number of object candidates
            key: (B, N, 1, C) N: number of seed points
            query_pos: (B, K, 1, 3 or 6)
            key_pos: (B, N, 1, 3 or 6)
        '''

        K = query.shape[1]
        C = query.shape[-1]
        N = key.shape[1]

        # Self attention
        if self.self_posembed is not None:      
            posC = query_pos.shape[-1]
            query_pos = layers.Reshape((K, 1, posC))(query_pos)     
            query_pos_embed = self.self_posembed(query_pos)                        
            query_pos_embed = layers.Reshape((K, C))(query_pos_embed)
        else:
            query_pos_embed = None

        query = layers.Reshape((K, C))(query)

        q = k = v = self.with_pos_embed(query, query_pos_embed) # Simple sum

        query2 = self.self_attn(q, k, value=v)
        query = query + self.dropout1(query2)
        
        query = self.norm1(query)

        # Cross atention

        if self.cross_posembed is not None:
            posC = key_pos.shape[-1]
            key_pos = layers.Reshape((N, 1, posC))(key_pos)     
            key_pos_embed = self.cross_posembed(key_pos)
            key_pos_embed = layers.Reshape((N, C))(key_pos_embed)
        else:
            key_pos_embed = None

        key = layers.Reshape((N, C))(key)

        query2 = self.cross_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        
        return query




if __name__ == '__main__':

    from modules_tf import PositionEmbeddingLearned
    class testModel(tf.keras.Model):
        def __init__(self, d_model=288, nhead=8, dim_feedforward=2048, dropout=0.1):
            super().__init__()
            model_config = {'use_tflite':False, 'q_gran':'semantic', 'activation':'relu'}
            self_posembed = PositionEmbeddingLearned(288)
            cross_posembed = PositionEmbeddingLearned(288)
            self.decoder = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, self_posembed, cross_posembed, model_config)
        
        def call(self, inputs):   
            query, key, query_pos, key_pos = inputs
            output = self.decoder(query, key, query_pos, key_pos)            
            return output

    model = testModel()    

    query = tf.convert_to_tensor(np.random.random((1,256,1,288)))
    key = tf.convert_to_tensor(np.random.random((1,1024,1,288)))
    query_pos = tf.convert_to_tensor(np.random.random((1,256,1,3)))
    key_pos = tf.convert_to_tensor(np.random.random((1,1024,1,3)))

    inputs = query, key, query_pos, key_pos
    print(model(inputs).shape)

    
    
    
