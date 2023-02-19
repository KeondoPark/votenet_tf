#!/bin/bash
#SBATCH --job-name=votenet_eval # Submit a job named "example"
#SBATCH --nodes=1 # Using 1 node
#SBATCH --gpus=1 # Using 1 GPU
#SBATCH --time=0-6:10:00 # 6 hours timelimit
#SBATCH --mem=24000MB # Using 24GB memory
#SBATCH --cpus-per-task=8 # Using 8 cpus per task (srun)

source /home/${USER}/.bashrc # Initiate your shell environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate votenet_tf # Activate your conda environment
#sunrgbd
CUDA_VISIBLE_DEVICES=0 python eval_tf.py  \
--use_3d_nms --use_cls_nms --per_class_proposal \
--batch_size 8 --ap_iou_thresholds 0.5 --faster_eval \
--config_path configs/config_rsgf_baseline.json --size_cls_agnostic
# --config_path configs/config_gf_baseline.json --size_cls_agnostic

#scannet
CUDA_VISIBLE_DEVICES=1 python eval_tf.py  \
--use_3d_nms --use_cls_nms --per_class_proposal \
--batch_size 16 --ap_iou_thresholds 0.25 \
--config_path configs/config_gf_scannet_baseline.json


