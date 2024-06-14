#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 2 -c 16                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4		# Specify number of tasks per node
#SBATCH --gpus-per-node=4		# Specify total number of GPUs
#SBATCH -t 1:00:00                      # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900502                     # Specify project name
#SBATCH -J Finetune_2N                  # Specify job name

module load Mamba/23.11.0-0
conda activate lightning-2.2.5

export PYTHONFAULTHANDLER=1
export HF_HOME=/project/lt900502-ck24tn/ck1055/hf/misc
export HF_DATASETS_CACHE=/project/lt900502-ck24tn/ck1055/hf/datasets
export TRANSFORMERS_CACHE=/project/lt900502-ck24tn/ck1055/hf/models
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

START=`date`
starttime=$(date +%s)

srun python Finetune.py

END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END

