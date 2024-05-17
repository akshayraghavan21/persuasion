import os
import sys
import shutil
import datetime
import wandb
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from trl import DPOTrainer
import json
import glob

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print(f"Failed to create directory: {path}")
        raise SystemExit("Script aborted due to inability to create necessary directories.")

def setup_directories(base_path, sweep_id):
    directory_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{sweep_id}"
    runs_path = os.path.join(base_path, "runs", directory_name)
    create_directory(runs_path)
    paths = {'models': os.path.join(runs_path, 'models'), 'logs': os.path.join(runs_path, 'logs'), 'wandb': os.path.join(runs_path, 'wandb')}
    for path in paths.values():
        create_directory(path)
    return runs_path, paths

def get_latest_filename(directory, pattern):
    file_list = glob.glob(os.path.join(directory, pattern + "*.json"))
    file_list.sort(reverse=True)
    return os.path.basename(file_list[0]) if file_list else pattern + "001.json"

def load_and_prepare_dataset(data_dir, data_file_pattern, seed=42):
    dataset_filename = get_latest_filename(data_dir, data_file_pattern)
    dataset_file = os.path.join(data_dir, dataset_filename)
    print(f"Loading Dataset File: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # Use a fixed random seed for reproducibility in dataset splitting
    # Split the dataset into train (64%), validation (16%), and test (20%)
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    train_val_split = split['train'].train_test_split(test_size=0.25, seed=seed)  # 0.25 * 0.8 = 0.2

    return DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': split['test']
    })

def load_model_and_tokenizer(model_type, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_type, truncation=True, model_max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.int8, device_map="auto")
    return tokenizer, model

import json

def log_run_details(runs_path, run, oppaths, config):
    log_file = os.path.join(runs_path, "run_details.log")
    with open(log_file, "a") as file:
        file.write(f"Sweep ID: {run.sweep_id}, Run ID: {run.id}\n")
        file.write(f"Model Directory: {oppaths['models']}\n")
        file.write(f"Tokenizer Directory: {oppaths['tokenizer']}\n")
        file.write(f"Checkpoints Directory: {oppaths['checkpoints']}\n")
        file.write(f"Logs Directory: {oppaths['logs']}\n")
        file.write(f"WandB Directory: {oppaths['wandb']}\n")
        file.write("Configuration:\n")
        file.write(f"{config}\n")
        # json.dump(config, file, indent=4)
        file.write("\n\n")

def run_model(paths, train_dataset, val_dataset, test_dataset, runs_path):
    with wandb.init() as run:
        config = run.config
        tokenizer, model = load_model_and_tokenizer(config['model_type'], config['max_length'])
        tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        print(f"Running for config: {config} With Run: {run.id}")

        model_op_dir = os.path.join(paths['models'], f"{run.id}_model")
        tokenizer_op_dir = os.path.join(paths['models'], f"{run.id}_token")
        model_checkpoints_dir = os.path.join(paths['logs'], f"{run.id}_checkpoints")

        training_args = TrainingArguments(
            per_device_train_batch_size=config['batch_size'],
            num_train_epochs=config['num_epochs'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            lr_scheduler_type=config['lr_scheduler_type'],
            logging_steps=config.get('logging_steps', 10),
            warmup_ratio=config.get('warmup_ratio', 0.1),
            weight_decay=config['weight_decay'],
            evaluation_strategy="epoch",
            save_strategy="no",
            output_dir=model_checkpoints_dir,
            report_to='wandb',
            remove_unused_columns=False
        )

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            beta=config['beta'],
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_prompt_length=config['max_length'],
            max_length=config['max_length'],
            max_target_length=config['max_length']
        )

        try:
            dpo_trainer.train()
            dpo_trainer.model.save_pretrained(model_op_dir)
            dpo_trainer.tokenizer.save_pretrained(tokenizer_op_dir)

            # final_test_loss = dpo_trainer.evaluate(test_dataset)['loss']
            # print(f"Final Test Loss: {final_test_loss}")
            
            # wandb.log({"final_test_loss": final_test_loss})  # Log final test loss for sweep optimization

            # Log the details of this run
            oppaths = {
                "models": model_op_dir,
                "tokenizer": tokenizer_op_dir,
                "checkpoints": model_checkpoints_dir,
                "wandb": paths["wandb"],
                "logs": paths["logs"]
            }
            log_run_details(runs_path, run, oppaths, config)
        except Exception as e:
            print(f"An error occurred: {e}")

def read_sweep_config(file_path):
    with open(file_path, 'r') as file:
        sweep_config = json.load(file)
    return sweep_config

import re

def replace_chars_regex(text):
    chars_to_replace = r'[\/\\#?%:]'  # Regular expression to match all special characters
    return re.sub(chars_to_replace, '_', text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    sweep_config = read_sweep_config(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #str(sweep_config.get('parameters', {}).get('gpu', {}).get('values', ['0']))
    base_path = '/cronus_data/araghavan/persuasion'
    data_dir = os.path.join(base_path, "data")
    data_file_pattern = "dpo_src_"

    datasets = load_and_prepare_dataset(data_dir, data_file_pattern, seed=42)
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    test_dataset = datasets['test']

    
    # Example usage
    model_type_ = sweep_config['parameters']['model_type']['values'][0]
    model_type_ = replace_chars_regex(model_type_)
    sweep_id = wandb.sweep(sweep_config, project="experiment_" + model_type_)
    runs_path, paths = setup_directories(base_path, sweep_id)
    os.environ['WANDB_DIR'] = runs_path
    wandb.agent(sweep_id, function=lambda: run_model(paths, train_dataset, val_dataset, test_dataset, runs_path))

if __name__ == "__main__":
    main()