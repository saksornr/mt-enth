import glob
import os
import pandas as pd
import json
from pathlib import Path
import argparse
from datasets import Dataset

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)
from transformers import AutoTokenizer

def process_dataset(ds, num_proc):
    
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def get_len_tokenize_dataset(row):
        new_row = {}
        tokenizer.src_lang = "eng_Latn"
        tokenized_eng = tokenizer(text=row["en_text"])
        new_row["len_en"] = len(tokenized_eng["input_ids"])
        
        tokenized_th = tokenizer(text=row["th_text"])
        tokenizer.src_lang = "tha_Thai"
        new_row["len_th"] = len(tokenized_th["input_ids"])
        return new_row
        
    ds = ds.map(get_len_tokenize_dataset, num_proc=num_proc,)
    ds = ds.map(lambda row : {"diff":abs(len(row["en_text"])-len(row["th_text"]))}, num_proc=num_proc)
    ds = ds.filter(lambda row : row["len_en"] >= 10 and row["len_en"] <= 100, num_proc=num_proc)
    ds = ds.filter(lambda row : row["diff"]<=100, num_proc=num_proc)

    ds = ds.filter(lambda row :  row["en_text"][0].isalnum() or row["en_text"][0]=="(" or row["en_text"][0]=='"' or row["en_text"][0]=="'", num_proc=num_proc) 
    return ds

def main(args):

    csv_file_paths = glob.glob(os.path.join(args.csv_dir, '*.csv'))

    n_before_total = 0
    n_after_total = 0
    for csv_file_path in csv_file_paths:

        df = pd.read_csv(csv_file_path, encoding='utf-8')
        csv_filename = Path(csv_file_path).stem

        print(f'Begin cleaning, filtering from sub-dataset: {csv_filename}')
        print(f'\nNumber of segment pairs (before): {df.shape[0]}')

        n_before = df.shape[0]
        n_before_total += n_before
        
        # 
        ds = Dataset.from_pandas(df)
        ds = process_dataset(ds, int(args.numproc))
        df = ds.to_pandas()

        n_after = df.shape[0]
        n_after_total += n_after

        print(
            f'Number of segment pairs (after): {n_after} (filtered out {n_before - n_after})')
        print('\nDone cleaning and fitering.')

        if not os.path.exists(args.out_dir):
            print(f'\nCreate a directory at: `{args.out_dir}`')
            os.makedirs(args.out_dir, exist_ok=True)

        out_path = os.path.join(args.out_dir, f'{csv_filename}.csv')

        print(f'\nBegin writing file to: {out_path}\n')

        df = df[["en_text", "th_text"]]
        df.to_csv(out_path, index=False, encoding='utf-8')

        print('-'*30)   
    print(f'Total number of segment pairs')
    print(f'(before): {n_before_total}')
    print(f'(filtered out): {n_before_total - n_after_total}')
    print(f'(after): {n_after_total}')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'csv_dir', help='Directory that stored the dataset in .csv format')
    parser.add_argument('--out_dir', default='./dataset/filtered')
    parser.add_argument('--numproc', default=8)

    args = parser.parse_args()

    main(args)
