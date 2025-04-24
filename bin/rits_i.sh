#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-03:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB
#SBATCH --account=penner1
#SBATCH --output=/home/%u/slurm_logs/%x-%j.log

# set up job
module load python/3.9.12 cuda
pushd /home/rodrigov/clasp-src/
source env/bin/activate

# run job
cd ..
cd BRITS
# input
nvidia-smi


python main.py --model brits_i --epochs 30 --batch_size 64 --impute_weight 1 --hid_size 256