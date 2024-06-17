# Machine Translation (TH-EN)

# Clone with token
```bash
git clone https://<token>@github.com/huak95/mt-enth.git 
```

# Dataset
In this project, used 2 datasets including SCB-1M and OPUS which can be downloaded from [thai2nmt project](https://github.com/vistec-AI/thai2nmt/releases/tag/scb-mt-en-th-2020%2Bmt-opus_v1.0).

```bash
# Load SCB + Opus Datasets
wget -O scb+opus.zip https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020%2Bmt-opus_v1.0/scb-mt-en-th-2020+mt-opus.zip
unzip -d data/ scb+opus.zip

# Load Kaggle Dataset
kaggle competitions download -c ai-cooking-machine-translation
unzip -d data/kaggle ai-cooking-machine-translation.zip

# Extract Unmatch Dataset From Kaggle
python src/add_unmatch_kaggle.py

# Clean data
python src/clean_text.py data/scb-mt-en-th-2020+mt-opus \
    --unicode_norm NFKC \
    --out_dir data/scb-mt-en-th-2020+mt-opus-cleaned
```


# Tokenized
```bash
# for en-th
python src/nllb/tokenized_nllb.py \
    --model_repo facebook/nllb-200-distilled-600M \
    --max_len 64 \
    --num_proc 8 \
    --test_size 0.10 \
    --src_lang en 
```

# Finetune

```bash
# for en-th
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
        --per_device_train_batch_size 80 \
        --per_device_eval_batch_size 80 \
        --gradient_accumulation_steps 80 \
        --save_steps 50 \
        --max_length_eval 32 \
        --src_lang en \
        --report_to wandb
```

# Evaluation
```bash
# TODO
# download iwslt_2015 test data
mkdir data/iwslt_2015/ -p
wget -P data/iwslt_2015/ https://raw.githubusercontent.com/vistec-AI/thai2nmt/master/iwslt_2015/test/tst2010-2013_th-en.en
wget -P data/iwslt_2015/ https://raw.githubusercontent.com/vistec-AI/thai2nmt/master/iwslt_2015/test/tst2010-2013_th-en.th

# eval
python src/nllb/eval_nllb.py \
    --model_repo facebook/nllb-200-distilled-600M \
    --max_len 64 \
    --batch_size 128 \
    --src_lang en \
    --csv eval.csv
```

# Inference 
```bash
# inference
python src/nllb/inference_nllb.py \
    --model_repo facebook/nllb-200-distilled-600M \
    --test_csv data/kaggle/test.csv \
    --batch_size 128 \
    --max_len 128


# tokenize
python src/deep_tokenize.py \
    --csv result/nllb-200-distilled-600M/full_pred.csv
```

tokenize env
```bash
pip install git+https://github.com/rkcosmos/deepcut.git
pip install pandarallel==1.6.5
```

# Using Lanta
### Install using Conda

```bash
ml Mamba
conda create -p ./env python=3.10 -y
conda activate ./env
pip install -r requirements.txt
```

## Submit Train Model

```bash
sbatch submit_multinode.sh
```