#!/bin/bash

# sunrgbd baseline
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_baseline.json 

# sunrgbd 1way
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_1way.json 

# sunrgbd 2way
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_2way.json 


# scannet baseline
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_scannet_baseline.json
                                             

# scannet 1way
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_scannet_1way.json 


# scannet 2way
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_scannet_2way.json 