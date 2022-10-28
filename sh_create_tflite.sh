#!/bin/bash

source /home/${USER}/.bashrc # Initiate your shell environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate votenet_tf # Activate your conda environment
CUDA_VISIBLE_DEVICES=3 python /home/gundo0102/votenet_tf/create_tflite.py  --checkpoint_path /home/gundo0102/votenet_tf/tf_ckpt/210817 
