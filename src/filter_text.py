import glob
import os
import pandas as pd
import json
from pathlib import Path
from const.word import specific_domain_word
import argparse

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

def any_word_in_sentence(sentence, topw):
    # Convert the sentence to lowercase and split it into a set of words
    sentence_words = set(sentence.lower().split())
    
    # Convert the word list to lowercase set for efficient lookup
    word_set = set(map(str.lower, specific_domain_word[:int(topw)]))
    
    # Use set intersection to check if there's any common word
    return not sentence_words.isdisjoint(word_set)

def filter_df(df, randomp, topw):

    sentence_include_word = df['en_text'].parallel_map(
        lambda x: any_word_in_sentence(str(x), topw)
        )
    
    df = pd.concat([
        df[sentence_include_word],
        df[~sentence_include_word].sample(frac=float(randomp))
    ])
    return df 

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
        df = filter_df(df, args.randomp, args.topw)

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
    parser.add_argument('--randomp', default=0.1)
    parser.add_argument('--topw', default=500)

    args = parser.parse_args()

    main(args)
