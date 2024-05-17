import torch
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

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

from unsloth import PatchDPOTrainer
PatchDPOTrainer()


max_seq_length = 1024 # Supports automatic RoPE Scaling, so choose any number.

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # torch.cuda.set_device(1)
# print("Using device:", device)

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

# Load model
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/zephyr-sft",
#     max_seq_length = max_seq_length,
#     device_map="auto",
#     dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     # load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
# max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/tinyllama-bnb-4bit", # "unsloth/tinyllama" for 16bit loading
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
from unsloth import FastLanguageModel
import torch
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama-bnb-4bit", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = "auto",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# # tokenizer.pad_token = tokenizer.eos_token
# # Do model patching and add fast LoRA weights
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Dropout = 0 is currently optimized
#     bias = "none",    # Bias = "none" is currently optimized
#     use_gradient_checkpointing = True,
#     random_state = 42,
# )
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    use_gradient_checkpointing = False, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# print(model)
# # model.to(device)
# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")
# max_memory = get_balanced_memory(
#     model,
#     max_memory=None,
#     no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
#     dtype='float16',
#     low_zero=False,
# )

# device_map = infer_auto_device_map(
#     model,
#     max_memory=max_memory,
#     no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
#     dtype='float16'
# )

# model = dispatch_model(model, device_map=device_map)

print("Finished Loading Model")
# training_args = TrainingArguments(
#     per_device_train_batch_size=4,
#     num_train_epochs=2,
#     gradient_accumulation_steps=8,
#     gradient_checkpointing=True,
#     learning_rate=2e-6,
#     lr_scheduler_type="cosine",
#     save_strategy="no",
#     evaluation_strategy="epoch",
#     logging_steps=10,
#     output_dir="./output",
#     optim="adamw_torch",
#     warmup_ratio=0.1,
#     remove_unused_columns=False,
#     weight_decay=0.1,
#     report_to="wandb"
# )
# training_args = DPOConfig(
#     per_device_train_batch_size=4,
#     num_train_epochs=2,
#     gradient_accumulation_steps=8,
#     gradient_checkpointing=True,
#     learning_rate=2e-6,
#     lr_scheduler_type="cosine",
#     save_strategy="no",
#     evaluation_strategy="epoch",
#     logging_steps=10,
#     output_dir="./output",
#     optim="adamw_8bit",
#     warmup_ratio=0.1,
#     remove_unused_columns=False,
#     weight_decay=0.1,
#     report_to="wandb",
#     fp16 = not torch.cuda.is_bf16_supported(),
#     bf16 = torch.cuda.is_bf16_supported(),
# )
training_args = DPOConfig(
    per_device_train_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-6,
    lr_scheduler_type="cosine",
    save_strategy="no",
    evaluation_strategy="epoch",
    logging_steps=10,
    output_dir="./output",
    optim="adamw_8bit",
    warmup_ratio=0.1,
    remove_unused_columns=False,
    weight_decay=0.1,
    seed = 42,
    report_to="wandb",
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
)
print("Finished Initializing Training Args")

wandb.init(project='experiment_unsloth_tinyllama-bnb-4bit', config={
    "learning_rate":2e-6,
    "architecture": "unsloth_tinyllama-bnb-4bit",
    "dataset": dataset_filename,
    "epochs": 2,
    "batch_size":4
})
# print("Finished Initializing Training Args")
# dpo_trainer = DPOTrainer(
#     model,
#     ref_model=None,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     beta=0.1,
#     tokenizer=tokenizer,
#     max_prompt_length=max_length,
#     max_length=max_length,
#     max_target_length = max_length,
# )
# print("Finished initializing DPO Trainer")
# dpo_trainer.train()
# dpo_trainer.model.save_pretrained("./unsloth_zephyr")
# print("Model Fine-Tuning Completed")
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    beta=0.1,
    tokenizer=tokenizer,
    max_prompt_length=max_length,
    max_length=max_length,
    max_target_length = max_length,
)
print("Finished initializing DPO Trainer")
dpo_trainer.train()
dpo_trainer.model.save_pretrained("./unsloth_zephyr")
print("Model Fine-Tuning Completed")