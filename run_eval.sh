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
srun python /home/keondopark/votenet_tf/eval_tf.py  --checkpoint_path /home/keondopark/votenet_tf/tf_ckpt/210924 --dump_dir eval14 --use_3d_nms --use_cls_nms --per_class_proposal --faster_eval --batch_size 1 --ap_iou_thresholds 0.25 --use_tflite