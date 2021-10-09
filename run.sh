#!/bin/bash
#SBATCH --job-name=example # Submit a job named "example"
#SBATCH --nodes=1 # Using 1 node
#SBATCH --gres=gpu:1 # Using 1 GPU
#SBATCH --time=0-12:00:00 # 6 hours timelimit
#SBATCH --mem=24000MB # Using 24GB memory
#SBATCH --cpus-per-task=16 #using 16 cpus per task (srun)

source /home/${USER}/.bashrc # Initiate your shell environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf_3090 # Activate your conda environment
srun python /home/keondopark/votenet_tf/train_tf.py  --checkpoint_path /home/keondopark/votenet_tf/tf_ckpt/211008 --max_epoch 180 --log_dir logs/log_211008
