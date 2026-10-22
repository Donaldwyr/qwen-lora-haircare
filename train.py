import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACC = 4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./deepseek-lora-output"

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

def format_prompt(sample):
    messages = [
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def tokenize_fn(sample):
    text = format_prompt(sample)
    tok = tokenizer(
        text,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length"
    )
    # 把 padding 位置（attention_mask=0）的 label 设为 -100，让 loss 只计算真实内容
    tok["labels"] = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(tok["input_ids"], tok["attention_mask"])
    ]
    return tok

tokenized_data = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=10,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_data
)

if __name__ == "__main__":
    trainer.train()
