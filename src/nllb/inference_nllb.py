from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import evaluate
import pandas as pd
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# set args
parser = argparse.ArgumentParser()

parser.add_argument("--model_repo", default='facebook/nllb-200-distilled-600M', type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_len", type=int, default=64)
parser.add_argument("--test_csv", default='test.csv', type=str)
parser.add_argument("--test_csv_col", default='source', type=str)

args = parser.parse_args()

def main_pred():
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_repo).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_repo)

    en_text = []

    df = pd.read_csv(args.test_csv)
    en_text = [line.strip() for line in df[args.test_csv_col].to_list()]

    print(f"en_text: {len(en_text)}")

    # prediction
    predictions = []
    for i in tqdm(range(0, len(en_text), args.batch_size)):
        batch = (en_text[i:i+args.batch_size])

        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["tha_Thai"], max_length=args.max_len)
        predictions += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    # Check
    print('len_orig:', len(en_text))
    print('len_pred:', len(predictions))
    torch.cuda.empty_cache()
    
    return predictions

def main():
    # Save
    df = pd.read_csv(args.test_csv)
    df['translation'] = main_pred()

    model_name = args.model_repo.split('/')[-1]

    os.makedirs("result", exist_ok=True)
    os.makedirs(f"result/{model_name}", exist_ok=True)
    df.to_csv(f'result/{model_name}/full_pred.csv', index=False)
    df[['id', 'translation']].to_csv(f'result/{model_name}/submission.csv', index=False)
    
if __name__ == "__main__":
    main()