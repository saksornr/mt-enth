#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script


module restore
module load Mamba
module load cudatoolkit/22.7_11.7
module load gcc/10.3.0

conda deactivate
conda activate ./env

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

export PYTHONFAULTHANDLER=1
export HF_HOME=/home/ck1055/mt-enth/hf/misc
export HF_DATASETS_CACHE=/home/ck1055/mt-enth/hf/datasets
export TRANSFORMERS_CACHE=/home/ck1055/mt-enth/hf/models
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend 'no' \
    --debug \
    src/nllb/training_nllb.py \
        --model_repo facebook/nllb-200-distilled-600M \
        --epoch 5 \
        --output_dir checkpoints_toy_1p \
        --model_name nllb-600m-en_th-exp1 \
        --dataset ./hf_dataset/nllb-scb+opus-hf-tokenized-en_th-toy-1p \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 128 \
        --gradient_accumulation_steps 128 \
        --save_steps 100 \
        --max_length_eval 64 \
        --src_lang en
