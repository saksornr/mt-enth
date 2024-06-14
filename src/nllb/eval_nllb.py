from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import evaluate
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Original Model / Full finetuned model
model_repo = "output/final_checkpoint"

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

# prediction
predictions = []
batch_size = 128
for i in tqdm(range(0, len(th_text), batch_size)):
    batch = (th_text[i:i+batch_size])

    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=64)
    predictions += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# calculate bleu score
metric = evaluate.load("sacrebleu")
en_ref = [[line] for line in en_text]
eval_result = metric.compute(predictions=predictions, references=en_ref)
print(eval_result)