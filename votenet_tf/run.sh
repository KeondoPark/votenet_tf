#!/bin/bash
#SBATCH --job-name=example # Submit a job named "example"
#SBATCH --nodes=1 # Using 1 node
#SBATCH --gpus=1 # Using 1 GPU
#SBATCH --time=0-6:00:00 # 6 hours timelimit
#SBATCH --mem=24000MB # Using 24GB memory
#SBATCH --cpus-per-task=8 # Using 8 cpus per task (srun)

source /home/${USER}/.bashrc # Initiate your shell environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate votenet_tf # Activate your conda environment
srun python /home/gundo0102/votenet_tf/train_tf.py  --checkpoint_path /home/gundo0102/votenet_tf/log_210817/tf_ckpt_210817 --max_epoch 180 --log_dir log_210817
