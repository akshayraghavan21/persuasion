## To run the script:
# nohup python dpo_v4.py > ../logs/train_output_$(date +%Y_%m_%d_%H%M%S).log 2>&1 &
import os
import gc
import torch
import glob
import random
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer, GPT2LMHeadModel, AutoModel
import pdb
import datetime

# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
from datasets import Dataset, load_dataset
from transformers import set_seed
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(1)
print("Using device:", device)


def get_elapsed_time_formatted(start_time, end_time):
    elapsed_time = end_time - start_time
    # Convert elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return f"Elapsed time: {hours}h:{minutes}m:{seconds}s"

def get_latest_filename(directory, pattern):
    # Get the list of filenames matching the pattern
    file_list = glob.glob(os.path.join(directory, pattern + "*.json"))

    # # Sort the filenames by modification time to get the latest one
    # latest_file = max(file_list, key=os.path.getmtime)
    # return os.path.basename(latest_file) if latest_file else None
    file_list.sort(reverse=True)

    return os.path.basename(file_list[0]) if file_list else pattern+"001.json"

def get_incremented_filename(directory, pattern):
    # Get the list of filenames matching the pattern
    file_list = glob.glob(os.path.join(directory, pattern + "*.json"))
    if len(file_list) > 0:

        # Sort the filenames by modification time to get the latest one
        # latest_file = max(file_list, key=os.path.getmtime)
        file_list.sort(reverse=True)
        latest_file = file_list[0]

        # Extract the version number from the filename
        match = re.search(rf'{pattern}(\d+)\.json', latest_file)

        if match:
            # current_version = int(match.group(1))
            current_version = int(re.findall(rf'{pattern}(\d+)\.json', latest_file)[0])
        else:
            current_version = 0  # Set default version if no match found

        # Increment the version number
        new_version = current_version + 1
        new_version_str = str(new_version).zfill(3)

        # Construct the new filename with the incremented version
        new_filename = f"{pattern}{new_version_str}.json"
    else:
        new_filename = f"{pattern}001.json"

    # Set the full path for the new filename
    new_file_path = os.path.join(directory, new_filename)

    # # Rename the latest file with the new filename
    # os.rename(latest_file, new_file_path)

    return new_file_path #new_filename

project_dir = "/cronus_data/araghavan/persuasion"
model_save_dir = os.path.join(project_dir, "model")
model_logs_dir = os.path.join(project_dir, "logs")
# model_save_dir = "/kaggle/working"
data_dir = os.path.join(project_dir, "data")
wandb_dir = os.path.join(project_dir)
# data_dir = "/kaggle/input/data-persuasion-v2"


data_file_pattern = "dpo_src_"  # Adjust this pattern as per your filenames
dataset_filename = get_latest_filename(data_dir, data_file_pattern)
dataset_file = os.path.join(data_dir, dataset_filename)
print(f"Loading Dataset File: {dataset_file}")
dataset = load_dataset("json", data_files=dataset_file, split="train")
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(
    batched=True,
    batch_size=32
)


max_length = 1024
print(dataset)
train_dataset = dataset["train"].shuffle(seed=42)
test_dataset = dataset["test"]
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 


def train():
    fail_flag = False
    try:
        with wandb.init() as run:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")    
            config = run.config
            set_seed(42)
            run_id = f"{timestamp}_{wandb.run.id}"
            print(f"Running for config: {config} With Run: {run_id}")

            model_op_dir = os.path.join(
                model_save_dir, 
                f"dpo_final_model_{run_id}"
            )
            tokenizer_op_dir = os.path.join(
                model_save_dir, 
                f"dpo_final_token_{run_id}"
            )
            model_checkpoints_dir = os.path.join(
                model_logs_dir, 
                f"dpo_model_checkpoints_{run_id}"
            )

            tokenizer = AutoTokenizer.from_pretrained("gpt2", truncation=True, model_max_length=max_length)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)

            print("Setting Training Arguments")
            # Training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=config.batch_size,
                num_train_epochs=config.num_train_epochs,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=True,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                save_strategy="no",
                evaluation_strategy="epoch",
                logging_steps=10,
                output_dir=model_checkpoints_dir,
                optim="adamw_torch",
                warmup_ratio=0.1,
                remove_unused_columns=False,
                report_to="wandb",
                weight_decay=config.weight_decay
            )

            print("Initializing DPOTrainer")
            max_prompt_length = max_length
            dpo_trainer = DPOTrainer(
                model,
                ref_model=None,
                args=training_args,
                beta=0.1,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                max_prompt_length=max_prompt_length,
                max_length=max_length,
                max_target_length = max_prompt_length,
            )

            # Fine-tune model with DPO
            print("Begin Model Fine-Tuning")
            try:
                dpo_trainer.train()
                dpo_trainer.model.save_pretrained(model_op_dir)
                tokenizer.save_pretrained(tokenizer_op_dir)
                print("Model Fine-Tuning Completed")
            except Exception as e:
                print(f"An error occurred: {e}")

    except RuntimeError as e:
        fail_flag = True
        print(f"\tAn error occurred: {e}")
        if 'CUDA out of memory' in str(e):
            print(f"\tCUDA out of memory for config: {config}")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(e)  # Re-raise the exception if it is not a handled GPU error
    if fail_flag:
        print(f"Finished for config: FAIL : {config}\n\n")
    else:
        print(f"Finished for config: SUCC : {config}\n\n")

os.environ['WANDB_DIR'] = wandb_dir
wandb.login(key="1f61250bd905a3c517611326864509e3969aa3da")
# wandb.init(project='cronus_persuasion_dpo', config={
#     "learning_rate":lr,
#     "architecture": "DPO",
#     "dataset": dataset_filename,
#     "epochs": epochs,
#     "batch_size":batch_size
# })

sweep_config = {
    'method': 'bayes',  # Choose the search strategy: grid, random, or bayesian
    'metric': {
        'name': 'eval_loss',  # Define the metric to optimize
        'goal': 'minimize'   # Set the optimization goal: minimize or maximize
    },
    'parameters': {
        'learning_rate': {
            'values': [5e-4, 5e-5] # Learning rates to try: 4e-4, 5e-3
        },
        'batch_size': {
            'values': [4]  # Batch sizes to try: 8, 16
        },
        'num_train_epochs': {
            'values': [3, 5]  # Number of epochs to try: 10
        },
        'gradient_accumulation_steps': {
            'values': [8]  # Number of grad steps to try: 4, 16
        },
        'weight_decay': {
            'values': [0.0, 0.01, 0.05, 0.1]  # Weight decay values to try
        },
        'lr_scheduler_type': {
            'values': [ 'linear' ]
                # 'linear', 'cosine', 
                # 'cosine_with_restarts', 'polynomial', 
                # 'constant', 'exponential']  # LR scheduler strategies to try
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='cronus_persuasion_dpo')
wandb.agent(sweep_id, train)
print("Finished Running Whole Script")
