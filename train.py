import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 1024
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# Login to Hugging Face (required for Llama models)
from huggingface_hub import login
login(token="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")

class ProblemListDataset(Dataset):
    """Custom dataset for Problem List Summarization task"""
    
    def __init__(self, data, tokenizer, max_length=1024, max_target_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Construct input text following the paper's format
        input_text = self._construct_input(row)
        target_text = str(row['Summary']).strip()
        
        # Create prompt format for instruction following
        prompt = f"""You are a medical AI assistant. Based on the following hospital progress note sections, generate a concise problem list summarizing the patient's main diagnoses and problems.

### Input:
{input_text}

### Problem List:
{target_text}"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        labels = inputs['input_ids'].clone()
        
        # Mask the input portion for loss calculation (only train on the response)
        input_only = f"""You are a medical AI assistant. Based on the following hospital progress note sections, generate a concise problem list summarizing the patient's main diagnoses and problems.

### Input:
{input_text}

### Problem List:
"""
        
        input_tokens = self.tokenizer(
            input_only,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Mask input tokens in labels
        input_length = input_tokens['input_ids'].shape[1]
        labels[:, :input_length] = -100
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels.flatten()
        }
    
    def _construct_input(self, row):
        """Construct input text from different sections"""
        sections = []
        
        # Add Subjective section
        if pd.notna(row.get('Subjective', '')) and str(row['Subjective']).strip():
            sections.append(f"<Subjective>\n{str(row['Subjective']).strip()}")
        
        # Add Objective section  
        if pd.notna(row.get('Objective', '')) and str(row['Objective']).strip():
            sections.append(f"<Objective>\n{str(row['Objective']).strip()}")
            
        # Add Assessment section
        if pd.notna(row.get('Assessment', '')) and str(row['Assessment']).strip():
            sections.append(f"<Assessment>\n{str(row['Assessment']).strip()}")
        
        return '\n\n'.join(sections)

def load_and_preprocess_data(csv_file_path):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(csv_file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove rows with missing summaries
    df = df.dropna(subset=['Summary'])
    
    # Clean and preprocess text
    for col in ['Subjective', 'Objective', 'Assessment', 'Summary']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df

def setup_model_and_tokenizer():
    """Setup model and tokenizer with quantization for memory efficiency"""
    
    # Quantization config for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def compute_rouge_metrics(predictions, references):
    """Compute ROUGE-L metrics"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(score['rougeL'].fmeasure)
    
    return {
        'rouge_l': np.mean(rouge_scores)
    }

def train_model(model, tokenizer, train_dataset, val_dataset=None):
    """Train the model"""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        max_steps=1000,  # Limit steps for Colab
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        report_to=None,  # Disable wandb
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model('./fine-tuned-llama-medical')
    tokenizer.save_pretrained('./fine-tuned-llama-medical')
    
    return trainer

def generate_predictions(model, tokenizer, test_data, max_samples=10):
    """Generate predictions for evaluation"""
    model.eval()
    predictions = []
    
    for idx in range(min(len(test_data), max_samples)):
        row = test_data.iloc[idx]
        
        # Construct input
        input_text = construct_input_for_inference(row)
        
        prompt = f"""You are a medical AI assistant. Based on the following hospital progress note sections, generate a concise problem list summarizing the patient's main diagnoses and problems.

### Input:
{input_text}

### Problem List:
"""
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        prediction = generated_text[len(prompt):].strip()
        predictions.append(prediction)
        
        print(f"Sample {idx + 1}:")
        print(f"Ground Truth: {row['Summary']}")
        print(f"Prediction: {prediction}")
        print("-" * 50)
    
    return predictions

def construct_input_for_inference(row):
    """Construct input text for inference"""
    sections = []
    
    if pd.notna(row.get('Subjective', '')) and str(row['Subjective']).strip():
        sections.append(f"<Subjective>\n{str(row['Subjective']).strip()}")
    
    if pd.notna(row.get('Objective', '')) and str(row['Objective']).strip():
        sections.append(f"<Objective>\n{str(row['Objective']).strip()}")
        
    if pd.notna(row.get('Assessment', '')) and str(row['Assessment']).strip():
        sections.append(f"<Assessment>\n{str(row['Assessment']).strip()}")
    
    return '\n\n'.join(sections)

def main():
    """Main training pipeline"""
    
    # Load data
    # Replace with your actual CSV file path
    csv_file_path = 'home/faakash/code/data/physionet.org/files/bionlp-workshop-2023-task-1a/2.0.0/BioNLP2023-1A-Train.csv' 
    
    print("Loading data...")
    df = load_and_preprocess_data(csv_file_path)
    
    # Split data (80% train, 20% validation)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create datasets
    train_dataset = ProblemListDataset(train_df, tokenizer, MAX_LENGTH, MAX_TARGET_LENGTH)
    val_dataset = ProblemListDataset(val_df, tokenizer, MAX_LENGTH, MAX_TARGET_LENGTH)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Generate some predictions
    print("\nGenerating sample predictions...")
    predictions = generate_predictions(model, tokenizer, val_df[:5])
    
    # Compute ROUGE scores
    references = val_df['Summary'].head(5).tolist()
    metrics = compute_rouge_metrics(predictions, references)
    print(f"\nROUGE-L Score: {metrics['rouge_l']:.4f}")
    
    print("\nTraining completed!")
    print("Model saved to './fine-tuned-llama-medical'")

# Example usage for loading your own data
def load_your_data(file_path):
    """
    Load your CSV file with the expected format:
    - FILE ID: Unique identifier
    - Subjective: Subjective section of progress note
    - Objective: Objective section of progress note  
    - Assessment: Assessment section of progress note
    - Summary: Ground truth problem list (target)
    """
    return pd.read_csv(file_path)

if __name__ == "__main__":
    main()

# Additional utility functions for inference after training
def load_trained_model(model_path):
    """Load the fine-tuned model for inference"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def inference_single_sample(model, tokenizer, subjective, objective, assessment):
    """Run inference on a single sample"""
    sections = []
    
    if subjective and subjective.strip():
        sections.append(f"<Subjective>\n{subjective.strip()}")
    
    if objective and objective.strip():
        sections.append(f"<Objective>\n{objective.strip()}")
        
    if assessment and assessment.strip():
        sections.append(f"<Assessment>\n{assessment.strip()}")
    
    input_text = '\n\n'.join(sections)
    
    prompt = f"""You are a medical AI assistant. Based on the following hospital progress note sections, generate a concise problem list summarizing the patient's main diagnoses and problems.

### Input:
{input_text}

### Problem List:
"""
    
    inputs = tokenizer(prompt, return_tensors='pt', max_length=MAX_LENGTH, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = generated_text[len(prompt):].strip()
    
    return prediction
