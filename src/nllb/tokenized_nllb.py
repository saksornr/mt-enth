from transformers import NllbTokenizerFast
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets
import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--model_repo", default='facebook/nllb-200-distilled-600M', type=str)
parser.add_argument("--dataset_dir", default='data/scb-mt-en-th-2020+mt-opus-cleaned/', type=str)
parser.add_argument("--max_len", default=64, type=int)
parser.add_argument("--num_proc", default=8, type=int)
parser.add_argument("--test_size", default=0.1, type=float)
parser.add_argument("--src_lang", default='en', type=str)
args = parser.parse_args()

# load tokenizer
if args.src_lang == 'en':
    nllb_src_lang = 'eng_Latn'
    nllb_tgt_lang = 'tha_Thai'
    translate_direction = "en_th"
elif args.src_lang == 'th':
    nllb_src_lang = 'tha_Thai'
    nllb_tgt_lang = 'eng_Latn'
    translate_direction = "th_en"

nllb_tokenizer = NllbTokenizerFast.from_pretrained(args.model_repo, src_lang=nllb_src_lang, tgt_lang=nllb_tgt_lang)

# get all available datasets
data_files = list(Path(args.dataset_dir).glob("*.csv"))

def preprocess_function(examples):
    inputs = [example[nllb_src_lang] for example in examples["translation"]]
    targets = [example[nllb_tgt_lang] for example in examples["translation"]]
    model_inputs = nllb_tokenizer(inputs, text_target=targets, max_length=args.max_len, truncation=True)
    # model_inputs = nllb_tokenizer(inputs, text_target=targets)
    return model_inputs

tokenized_dataset = []
token_distribution_data = []
graph_title = []
for data_file in data_files:
    print(f"Processing {data_file.stem}")
    graph_title.append(data_file.stem)
    df = pd.read_csv(data_file)
    df.rename(columns={'en_text': 'eng_Latn', 'th_text': 'tha_Thai'}, inplace=True)
    df.dropna(inplace=True)
    data = {"translation": df.to_dict(orient="records")}
    dataset = Dataset.from_dict(data)
    tokenized_sentence = dataset.map(
        preprocess_function, batched=True, num_proc=args.num_proc, remove_columns=dataset.features)
    tokenized_dataset.append(tokenized_sentence)

dataset_concat = concatenate_datasets(tokenized_dataset)


dataset_concat = dataset_concat.shuffle(seed=42)

# save full dataset
dataset = dataset_concat.train_test_split(test_size=args.test_size)
os.makedirs("./hf_dataset/", exist_ok=True)
dataset.save_to_disk(f"./hf_dataset/nllb-scb+opus-hf-tokenized-{translate_direction}")

# save toy 10p dataset
sample_dataset = dataset_concat.train_test_split(test_size=0.10)['test']
sample_dataset = sample_dataset.train_test_split(test_size=args.test_size)
sample_dataset.save_to_disk(f"./hf_dataset/nllb-scb+opus-hf-tokenized-{translate_direction}-toy-10p")

# save toy 1p dataset
sample_dataset = dataset_concat.train_test_split(test_size=0.01)['test']
sample_dataset = sample_dataset.train_test_split(test_size=args.test_size)
sample_dataset.save_to_disk(f"./hf_dataset/nllb-scb+opus-hf-tokenized-{translate_direction}-toy-1p")