import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import bitsandbytes as bnb
import random
from collections import defaultdict
import wandb

# # Defined in the secrets tab in Google Colab
# hf_token = userdata.get('HF_TOKEN')
# wb_token = userdata.get('wandb')
# wandb.login(key=wb_token)
wandb.init(project='experiment_gaia_llama2-7b_tests', config={
    "learning_rate":5e-5,
    "architecture": "lora_llama2-7b",
    "epochs": 3,
    "batch_size":2,
    "weight_decay":0.05,
    "lr_scheduler_type":"linear",
})
model_name = "/gaia_data/pretrained_models/llama2-7b-hf/"


# Load dataset
def load_and_prepare_dataset(data_dir, data_file_pattern, seed=42):
    dataset_filename = data_file_pattern  # get_latest_filename(data_dir, data_file_pattern)
    dataset_file = os.path.join(data_dir, dataset_filename)
    print(f"Loading Dataset File: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # Use a fixed random seed for reproducibility in dataset splitting
    # Split the dataset into train (60%), validation (20%), and test (20%)
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    train_val_split = split['train'].train_test_split(test_size=0.25, seed=seed)  # 0.25 * 0.8 = 0.2

    return DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': split['test']
    })


def load_and_prepare_dataset_no_leaks(data_dir, data_file_pattern, seed=42):
    dataset_filename = data_file_pattern
    dataset_file = os.path.join(data_dir, dataset_filename)
    print(f"Loading Dataset File: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # Group records by prompt
    prompt_groups = defaultdict(list)
    for idx, record in enumerate(dataset):
        prompt_groups[record['prompt']].append(idx)

    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle the groups
    prompt_keys = list(prompt_groups.keys())
    random.shuffle(prompt_keys)

    # Calculate split sizes
    total_groups = len(prompt_keys)
    train_size = int(0.6 * total_groups)
    val_size = int(0.2 * total_groups)
    # test_size will be the remaining groups

    # Split the groups
    train_groups = prompt_keys[:train_size]
    val_groups = prompt_keys[train_size:train_size+val_size]
    test_groups = prompt_keys[train_size+val_size:]

    # Create index lists for each split
    train_indices = [idx for group in train_groups for idx in prompt_groups[group]]
    val_indices = [idx for group in val_groups for idx in prompt_groups[group]]
    test_indices = [idx for group in test_groups for idx in prompt_groups[group]]

    # Create the splits
    return DatasetDict({
        'train': dataset.select(train_indices),
        'validation': dataset.select(val_indices),
        'test': dataset.select(test_indices)
    })


data_dir = "/data/araghavan/persuasion/data/"
data_file_pattern = "dpo_random_neg_op_comment_v001.json"
datasets = load_and_prepare_dataset(data_dir, data_file_pattern, seed=42)
print(datasets)

no_leaks_datasets = load_and_prepare_dataset_no_leaks(data_dir, data_file_pattern, seed=42)
print(no_leaks_datasets)

train_dataset = no_leaks_datasets['train'].select(range(150))
val_dataset = no_leaks_datasets['validation'].select(range(100))
test_dataset = no_leaks_datasets['test']


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./")
tokenizer.pad_token = tokenizer.unk_token


# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['v_proj', 'q_proj']
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.float16,
    # load_in_4bit=True,
    cache_dir="./",
    device_map="auto",
)
model.config.use_cache = False
model.enable_input_require_grads()

# import pdb; pdb.set_trace()
# Training arguments
training_args = DPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_strategy="no",
    logging_steps=15,
    output_dir="./first_try_hf",
    optim="paged_adamw_32bit",
    warmup_steps=100,
    num_train_epochs=3,
    bf16=True,
    seed=42,
    report_to="wandb",
    evaluation_strategy="epoch",
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)
# Fine-tune model with DPO
dpo_trainer.train()
wandb.finish()
dpo_trainer.model.save_pretrained("./first_try_hf_fin")
print("Model Fine-Tuning Completed")