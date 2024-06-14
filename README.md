# Machine Translation (TH-EN)

# Dataset
In this project, used 2 datasets including SCB-1M and OPUS which can be downloaded from [thai2nmt project](https://github.com/vistec-AI/thai2nmt/releases/tag/scb-mt-en-th-2020%2Bmt-opus_v1.0).

```bash
# Load Datasets
wget -O scb+opus.zip https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020%2Bmt-opus_v1.0/scb-mt-en-th-2020+mt-opus.zip
unzip -d data/ scb+opus.zip

# Clean data
python src/clean_text.py data/scb-mt-en-th-2020+mt-opus \
    --unicode_norm NFKC \
    --out_dir data/scb-mt-en-th-2020+mt-opus-cleaned
```

# Tokenized
```bash
# todo add argsphase
python src/nllb/tokenized_nllb.py 
```

# Finetune

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend 'no' \
    src/nllb/training_nllb.py \
        --epoch 1 \
        --output_dir output \
        --model_name nllb-600m-th-en-exp1 \
        --dataset ./hf_dataset/nllb-scb+opus-hf-tokenized-toy-1p \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8
```

# Evaluation
```bash

# download iwslt_2015 test data
mkdir data/iwslt_2015/ -p
wget -P data/iwslt_2015/ https://raw.githubusercontent.com/vistec-AI/thai2nmt/master/iwslt_2015/test/tst2010-2013_th-en.en
wget -P data/iwslt_2015/ https://raw.githubusercontent.com/vistec-AI/thai2nmt/master/iwslt_2015/test/tst2010-2013_th-en.th

python src/nllb/eval_nllb.py
```