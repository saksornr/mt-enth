#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 2 -c 16                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4		# Specify number of tasks per node
#SBATCH --gpus-per-node=4		# Specify total number of GPUs
#SBATCH -t 1:00:00                      # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900502                     # Specify project name
#SBATCH -J Finetune_MT_2N                  # Specify job name

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn 

START=`date`
starttime=$(date +%s)

export WANDB_MODE="offline"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

srun sh smultinode.sh