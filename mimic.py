import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
# from peft import LoraConfig, get_peft_model, TaskType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# -----------------------------------------------------------------

from huggingface_hub import login
import os

login(token="") #put your huggingface-token here
# login(new_session=False)

# -----------------------------------------------------------------
# Load model directly

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
)


model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
path = "/home/faakash/code/data/physionet.org/files/mimic-iv-note/2.2/note/"
# train_dataset = trainer.prepare_dataset(data)
# discharge.csv.gz
import pandas as pd
discharge_df = pd.read_csv(path+"discharge.csv.gz", compression="gzip", nrows=100000)

def split_keyword(text, keyword="Instructions:"):
	if keyword in text:
		parts = text.split(keyword, maxsplit=1)
		input_text = parts[0].strip()
		target_text = parts[1].strip()
		return pd.Series([input_text, target_text])
	else:
		return pd.Series([text.strip(), ""])

discharge_df[["input", "target"]] = discharge_df["text"].apply(lambda x: split_keyword(x, "Instructions:"))

train_dataset = Dataset.from_pandas(discharge_df[["input", "target"]])

#discharge = load_dataset("csv", data_files=path+"discharge.csv.gz", split="train[:4%]")
#radiology = load_dataset("csv", data_files=path+"radiology.csv.gz", split="train[:4%]")

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
	tokenized = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=1024)
	with tokenizer.as_target_tokenizer():
		labels = tokenizer(examples["target"], truncation=True, padding="max_length", max_length=1024)
	tokenized["labels"] = labels["input_ids"]
	return tokenized

#tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
# -----------------------------------------------------------------

torch.cuda.empty_cache()

# -----------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
trainer.train()

trainer.save_model("./model/llama-finetuned_v2")
tokenizer.save_pretrained("./model/llama-finetuned_v2")
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# from transformers import pipeline

# pipe = pipeline("text-generation", model="./llama3-multi-finetuned", tokenizer=tokenizer)
# pipe("<s>[INST] [/INST]")
# -----------------------------------------------------------------
