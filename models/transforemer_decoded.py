''' Group Free module: Generate proposals using Transformer(Attention) architecture

Date: July, 2019
Author: Keondo Park
Reference: Group-Free 3D Object Detection via Transformers, By Ze Liu, Zheng Zhang, Yue Cao, Han Hu, Xin Tong.
https://github.com/zeliu98/Group-Free-3D/tree/ef8b7bb5c3bf5b49b957624595dc6a642b6d0036
'''

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from multihead_attention import MultiheadAttention

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 self_posembed=None, cross_posembed=None, 
                 use_tflite=False, tflite_name=None):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)

        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.norm3 = layers.LayerNormalization(axis=-1)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

        self.activation = layers.ReLU(6)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos)
        else:
            key_pos_embed = None

        #query = query.permute(2, 0, 1)
        #key = key.permute(2, 0, 1)
        
        # Self attention
        q = k = v = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # Cross attention
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        #query = query.permute(1, 2, 0)
        return query
