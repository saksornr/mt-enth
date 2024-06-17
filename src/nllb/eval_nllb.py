from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import evaluate
import pandas as pd
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--model_repo", default='facebook/nllb-200-distilled-600M', type=str)
parser.add_argument("--dataset_dir", default='data/scb-mt-en-th-2020+mt-opus-cleaned/', type=str)
parser.add_argument("--max_len", default=64, type=int)
parser.add_argument("--num_proc", default=8, type=int)
parser.add_argument("--batch_size", default=128, type=float)
parser.add_argument("--src_lang", default='en', type=str)
args = parser.parse_args()

# Load Original Model / Full finetuned model
model_repo = args.model_repo

# load tokenizer
if args.src_lang == 'en':
    nllb_src_lang = 'eng_Latn'
    nllb_tgt_lang = 'tha_Thai'
    translate_direction = "en_th"
elif args.src_lang == 'th':
    nllb_src_lang = 'tha_Thai'
    nllb_tgt_lang = 'eng_Latn'
    translate_direction = "th_en"

model = AutoModelForSeq2SeqLM.from_pretrained(model_repo).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_repo)

# load test data
th_text = []
en_text = []

with open("data/iwslt_2015/tst2010-2013_th-en.en", "r") as f:
    en_text = [line.strip() for line in f]
with open("data/iwslt_2015/tst2010-2013_th-en.th", "r") as f:
    th_text = [line.strip() for line in f]

print(f"th_text: {len(th_text)}, en_text: {len(en_text)}")

# todo
# input_text = 

# prediction
predictions = []
batch_size = args.batch_size
for i in tqdm(range(0, len(input_text), batch_size)):
    batch = (input_text[i:i+batch_size])

    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[nllb_tgt_lang], max_length=args.max_len)
    predictions += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# calculate bleu score
metric = evaluate.load("sacrebleu")
en_ref = [[line] for line in en_text]
eval_result = metric.compute(predictions=predictions, references=en_ref)
print(eval_result)