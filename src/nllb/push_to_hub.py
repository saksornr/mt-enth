import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizerFast,
)
import numpy as np
import os

# os.environ['HF_TOKEN_PATH'] = '/home/ck1055/.cache/huggingface/token'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

checkpoint = "/home/saksorn/code/aicooking/mt-enth/output/checkpoint-1700"
tokenizer = NllbTokenizerFast.from_pretrained(
    checkpoint, src_lang="tha_Thai", tgt_lang="eng_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# model.save_pretrained('saksornr/nllb-enth')
# help(model.save_pretrained)
model.push_to_hub('nllb-enth')
tokenizer.push_to_hub('nllb-enth')
# model.save_pretrained('nllb-enth', push_to_hub=True)
