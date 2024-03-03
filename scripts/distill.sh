#!/bin/bash
#SBATCH -c 4  # Number of cpus per Task
#SBATCH --mem=20GB  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 0-08:00:00  # Job time limit   
#SBATCH -o out/B2.256-B-2M  # set experiment code here
#SBATCH --constraint 2080ti

module load miniconda/22.11.1-1
conda activate project4
cd /work/pi_dhruveshpate_umass_edu/project_4/distillation

python distillation.py 12 --exp_name bert-2.256-book-2M --batch_size 1024 --lr 5e-5 --epoch 5 --bert google/bert_uncased_L-2_H-256_A-4 --book --n_examples 2000000 --seed 0
