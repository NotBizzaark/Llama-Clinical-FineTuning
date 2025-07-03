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


# -----------------------------------------------------------------
# Load model directly

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     load_in_4bit=True,  # for efficient finetuning
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # or torch.float32 if needed
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
path = "./data/mimic-iv-note-deidentified-free-text-clinical-notes-2/note/"
# train_dataset = trainer.prepare_dataset(data)
# discharge.csv.gz

discharge = load_dataset("csv", data_files=path+"discharge.csv.gz", split="train[:5%]")
radiology = load_dataset("csv", data_files=path+"radiology.csv.gz", split="train[:5%]")

from datasets import Value
discharge = discharge.cast_column("hadm_id", Value("int64"))
radiology = radiology.cast_column("hadm_id", Value("int64"))

from datasets import concatenate_datasets
train_dataset = concatenate_datasets([discharge, radiology])

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# -----------------------------------------------------------------


# -----------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    optim="paged_adamw_32bit"
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

trainer.save_model("./llama3-multi-finetuned")
tokenizer.save_pretrained("./llama3-multi-finetuned")
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# from transformers import pipeline

# pipe = pipeline("text-generation", model="./llama3-multi-finetuned", tokenizer=tokenizer)
# pipe("<s>[INST] [/INST]")
# -----------------------------------------------------------------