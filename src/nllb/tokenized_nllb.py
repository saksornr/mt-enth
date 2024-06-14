from transformers import NllbTokenizerFast
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets
import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import os

import config_nllb

# load tokenizer
model_repo = config_nllb.based_model_repo
nllb_tokenizer = NllbTokenizerFast.from_pretrained(model_repo, src_lang="tha_Thai", tgt_lang="eng_Latn")

# get all available datasets
data_files = list(Path(config_nllb.dataset_dir).glob("*.csv"))

source_lang = "th"
target_lang = "en"

def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    # model_inputs = nllb_tokenizer(inputs, text_target=targets, max_length=64, truncation=True)
    model_inputs = nllb_tokenizer(inputs, text_target=targets)
    return model_inputs

tokenized_dataset = []
token_distribution_data = []
graph_title = []
for data_file in data_files:
    print(f"Processing {data_file.stem}")
    graph_title.append(data_file.stem)
    df = pd.read_csv(data_file)
    df.rename(columns={'en_text': 'en', 'th_text': 'th'}, inplace=True)
    df.dropna(inplace=True)
    data = {"translation": df.to_dict(orient="records")}
    dataset = Dataset.from_dict(data)
    tokenized_sentence = dataset.map(
        preprocess_function, batched=True, num_proc=8, remove_columns=dataset.features)
    tokenized_dataset.append(tokenized_sentence)

dataset_concat = concatenate_datasets(tokenized_dataset)


dataset_concat = dataset_concat.shuffle(seed=42)

# save full dataset
dataset = dataset_concat.train_test_split(test_size=0.15)
os.makedirs("./hf_dataset/", exist_ok=True)
dataset.save_to_disk("./hf_dataset/nllb-scb+opus-hf-tokenized")

# save toy dataset
sample_dataset = dataset_concat.train_test_split(test_size=0.01)['test']
sample_dataset = sample_dataset.train_test_split(test_size=0.15)
sample_dataset.save_to_disk("./hf_dataset/nllb-scb+opus-hf-tokenized-toy-1p")