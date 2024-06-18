import deepcut
import os, glob
import pandas as pd
import argparse

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

# set args
parser = argparse.ArgumentParser()

# parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--csv", type=str)
parser.add_argument("--csv_col", default='translation', type=str)

args = parser.parse_args()


def tok(x):
    x = str(x).strip()
    x = deepcut.tokenize(x)
    x = " ".join(x)
    x = " ".join(x.split())
    return x 

def main():
    # Save
    df = pd.read_csv(args.csv)
    df['translation'] = df['translation'].parallel_map(tok)
    
    new_csv = args.csv.replace(".csv", "_tok.csv")

    df.to_csv(new_csv, index=False, encoding='utf-8')
    
if __name__ == "__main__":
    main()