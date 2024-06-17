from datasets import load_from_disk
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
import numpy as np
import evaluate
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_repo", default='facebook/nllb-200-distilled-600M', type=str)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--report_to", type=str, default="none")
parser.add_argument("--dataset", type=str)
parser.add_argument("--max_length_eval", type=int, default=64)
parser.add_argument("--save_steps", type=int, default=10000)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--src_lang", default='en', type=str)

args, _ = parser.parse_known_args()

if args.src_lang == 'en':
    nllb_src_lang = 'eng_Latn'
    nllb_tgt_lang = 'tha_Thai'
    translate_direction = "en_th"
elif args.src_lang == 'th':
    nllb_src_lang = 'tha_Thai'
    nllb_tgt_lang = 'eng_Latn'
    translate_direction = "th_en"

checkpoint = args.model_repo
tokenizer = NllbTokenizerFast.from_pretrained(
    checkpoint, src_lang=nllb_src_lang, tgt_lang=nllb_tgt_lang, 
)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
generation_config = GenerationConfig.from_pretrained(checkpoint, max_length=args.max_length_eval)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
metric = evaluate.load("sacrebleu")
tokenized_sentence = load_from_disk(args.dataset, keep_in_memory=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True,)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


transformers.logging.set_verbosity_info()

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=False,
    bf16=True,
    bf16_full_eval=False,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=args.epochs,
    predict_with_generate=True,
    save_steps=args.save_steps,
    optim='adamw_bnb_8bit',
    push_to_hub=False,
    report_to=args.report_to,
    run_name=args.model_name,
    generation_config=generation_config
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_sentence["train"],
    eval_dataset=tokenized_sentence["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{args.output_dir}/final_checkpoint")