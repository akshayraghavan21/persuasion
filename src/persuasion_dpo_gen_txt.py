import argparse
import os
import random
from typing import Dict, Any
import json

import torch
from datasets import load_dataset, DatasetDict, Dataset 
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

import wandb
import pandas as pd
import gc
from tqdm import tqdm
import csv
import datetime

def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments for Text Generation using Checkpointed DPO Fine-tuned model.
        
        Returns:
            argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Persuasion DPO Fine-Tuned Text Generation Code")

    ## Project Params
    parser.add_argument("--data_file", type=str, default="dpo_random_neg_op_comment_v002.json")
    parser.add_argument("--project_data_dir", type=str, default="../data/")
    parser.add_argument("--project_gen_txt_output_dir", type=str, default="../output/generated_text_output")

    ## Run & Checkpoint Params
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--wandb_run_id", type=str)
    parser.add_argument("--checkpoint_epoch", type=int)
    parser.add_argument("--checkpoint_model_path", type=str)
    parser.add_argument("--dataset_to_gen_txt", type=str, default="validation")

    ## Output File Params
    parser.add_argument("--metadata_file_suffix", type=str, default="metadata")
    parser.add_argument("--metadata_file_extension", type=str, default=".json")
    parser.add_argument("--gen_txt_file_suffix", type=str, default="gen_txt_output")
    parser.add_argument("--gen_txt_file_extension", type=str, default=".csv")

    ## Eval Sample Gen Txt Params
    parser.add_argument("--gen_txt_params", type=str, default='{}',
                        help="JSON string of parameters to pass to model.generate")
    parser.add_argument("--eval_sample_gen_txt_cols", type=str, nargs='+', default = [
        'wandb_run_name',
        'wandb_run_id',
        'checkpoint_model_path',
        'checkpoint_epoch',
        'dataset_to_gen_txt',
        'sample_index',
        'prompt',
        'generated_text',
        'chosen',
        'rejected',
    ])

    ## Tokenizer Params
    parser.add_argument("--max_prompt_length", type=int, default=1024)

    ## General Params
    parser.add_argument("--seed", type=int, default=42)

    ## Dataset Prep
    parser.add_argument("--preprocess_datasets", type=bool, default=True)
    parser.add_argument("--preprocess_datasets_fn", type=str, default="template_edit_dataset")
    parser.add_argument("--template_prefix", type=str, default="Instruction: When given a piece of text, your task is to craft a persuasive response that encourages the reader to reconsider or adjust their position. Employ strategies that include (but are not limited to) presenting well-supported arguments, appealing to both emotions and logic, and thoughtfully addressing counterarguments. Adapt your tone to the context, ensuring your response is assertive yet respectful. Your goal is to move the reader's stance, ideally toward your perspective, but complete reversal isn't always necessary — nudging their viewpoint or sparking doubt can be just as valuable.\n\nInput: ")
    parser.add_argument("--template_suffix", type=str, default="\n\nOutput: ")

    return parser.parse_args()


def get_generation_params(args: argparse.Namespace) -> dict:
    """
    Parses the JSON string of generation parameters from command-line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        dict: A dictionary of text generation parameters.
    """
    try:
        gen_params = json.loads(args.gen_txt_params)
        return gen_params
    except json.JSONDecodeError:
        print("Runtime Error: Invalid JSON string for gen_txt_params.")
        exit()
    

def load_and_prepare_dataset_no_leaks(args: argparse.Namespace) -> DatasetDict:
    """
        Load and prepare the dataset, ensuring no data leakage between splits.
        This function loads the dataset, groups examples by prompt, and then splits
            the groups into train, validation, and test sets to prevent data leakage.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        
        Returns:
            DatasetDict: A dictionary containing 'train', 'validation', and 'test' datasets.
    """
    dataset_file = os.path.abspath(os.path.join(args.project_data_dir, args.data_file))
    print(f"Loading Dataset File: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    ## Aggregate based on prompt
    prompt_groups = {record['prompt']: [] for record in dataset}
    for idx, record in enumerate(dataset):
        prompt_groups[record['prompt']].append(idx)
    
    ## Segregate based on prompt
    random.seed(args.seed)
    prompt_keys = list(prompt_groups.keys())
    random.shuffle(prompt_keys)
    
    ## Split to 0.6, 0.2, 0.2 for train, dev, test
    total_groups = len(prompt_keys)
    train_size = int(0.6 * total_groups)
    val_size = int(0.2 * total_groups)
    
    ## Group to respective splits
    train_groups, val_groups, test_groups = prompt_keys[:train_size], prompt_keys[train_size:train_size+val_size], prompt_keys[train_size+val_size:]
    
    def get_indices(groups):
        return [idx for group in groups for idx in prompt_groups[group]]
    
    ## Create dataset
    return DatasetDict({
        'train': dataset.select(get_indices(train_groups)),
        'validation': dataset.select(get_indices(val_groups)),
        'test': dataset.select(get_indices(test_groups)),
    })


def setup_checkpoint_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    """
    Set up the model and tokenizer for training.
    This function loads the pre-trained model and tokenizer,
    and configures them for the DPO training task, including LoRA configuration.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        tuple: A tuple containing the configured model and tokenizer.
    """
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint_model_path,
        device_map="auto",
        # low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_model_path)

    return model, tokenizer


def template_edit_dataset(dataset, prefix, suffix):
    def add_prefix_and_suffix(example):
        example["prompt"] = prefix + example["prompt"] + suffix
        return example
    return dataset.map(add_prefix_and_suffix)


def get_gen_txt_file_paths(args: argparse.Namespace , timestamp_suffix: str) -> tuple:
    """
        Returns paths of the two files that is created as part of this
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            timestamp_suffix (str): Suffix to append to file names.
        
        Returns:
            tuple: Paths for metadata file and generated text output file.
    """
    
    ## Metadata File
    eval_samples_gen_txt_metadata_file_name = "_".join([
        args.wandb_run_name,
        args.wandb_run_id,
        f"{args.checkpoint_epoch:02}",
        timestamp_suffix,
        args.metadata_file_suffix
    ]) + args.metadata_file_extension # "wandb_run_name_runid_checkpoint_epoch_metadata.json"

    eval_samples_gen_txt_metadata_file_path = os.path.join(
        args.project_gen_txt_output_dir,
        eval_samples_gen_txt_metadata_file_name
    )

    ## Generated Text Output File
    eval_samples_gen_txt_file_name = "_".join([
        args.wandb_run_name,
        args.wandb_run_id,
        f"{args.checkpoint_epoch:02}",
        timestamp_suffix,
        args.gen_txt_file_suffix
    ]) + args.gen_txt_file_extension  # "wandb_run_name_runid_checkpoint_epoch_gen_txt_output.csv"

    eval_samples_gen_txt_file_path = os.path.join(
        args.project_gen_txt_output_dir,
        eval_samples_gen_txt_file_name
    )

    return eval_samples_gen_txt_metadata_file_path, eval_samples_gen_txt_file_path 


def write_metadata_info(args: argparse.Namespace , op_file_path: str):
    """
        Creates and Logs Metadata info as part of the first output file created for this run
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            op_file_path (str): Output file path where metadata will be logged.
    """
    
    ## Initialize metadata info to log
    metadata_params = {
        'eval_gen_txt_file_path': op_file_path,
        **vars(args)
    }

    ## Log metadata info to file
    with open(op_file_path , 'w') as f:
        json.dump(metadata_params , f)

    print("Logged Metadata info of the run!!")


def write_gen_txt_file(args: argparse.Namespace , model , tokenizer , eval_dataset , op_file_path):
    """
        Generates text using the model and logs it to a CSV file.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            model: The pre-trained language model used for generating text.
            tokenizer: The tokenizer used with the language model.
            eval_dataset: The evaluation dataset used for generating text samples.
            op_file_path (str): Output file path where generated text will be logged.
    """

    ## Begin text generation of val/dev dataset using checkpoint model
    for sample_idx, sample in tqdm(enumerate(eval_dataset), total=len(eval_dataset), desc="Generating text"):
        inputs = tokenizer(sample['prompt'], return_tensors="pt", truncation=True, padding=True, max_length=args.max_prompt_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                **args.gen_txt_params_kwargs
            )

            generated_text = tokenizer.batch_decode(gen_output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

            file_exists = os.path.isfile(op_file_path)
            
            with open(op_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(args.eval_sample_gen_txt_cols)
                
                writer.writerow([
                    args.wandb_run_name,
                    args.wandb_run_id,
                    args.checkpoint_model_path,
                    f"{args.checkpoint_epoch:02}",
                    args.dataset_to_gen_txt,
                    sample_idx,
                    sample['prompt'],
                    generated_text,
                    sample.get('chosen', ''),
                    sample.get('rejected', ''),
                ])
    print("Logged Text Generation info of the run!!")


if __name__ == "__main__":
    ## Sample Command:
    # python persuasion_dpo_gen_txt.py --gen_txt_params '{"max_length": 150, "top_k": 50, "top_p": 0.95, "temperature": 0.7, "do_sample": true, "new_param": "value"}'
    
    ## Parse CLI Arguments
    args = parse_args()

    # Add parsed generation parameters to args as a new attribute
    args.gen_txt_params_kwargs = get_generation_params(args)
    print(f"Please find the kwargs for this run: {args.gen_txt_params_kwargs}")

    ## Load Dataset
    datasets = load_and_prepare_dataset_no_leaks(args)

    gen_txt_run_tmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    ## Preprocess dataset if required
    if args.preprocess_datasets:
        if args.preprocess_datasets_fn:
            preprocess_function = globals().get(args.preprocess_datasets_fn)
            if callable(preprocess_function):
                datasets = preprocess_function(
                    dataset=datasets,
                    prefix=args.template_prefix,
                    suffix=args.template_suffix
                )
            else:
                print(f"Warning: {args.preprocess_datasets_fn} is not a callable function.")
        else:
            print("No preprocessing function specified.")
    
    ## Establish Val/Dev Dataset
    eval_dataset_raw = datasets[args.dataset_to_gen_txt]#.select(range(5))

    # Remove duplicate prompts
    print(f"Removing Duplicate Prompts. Raw Dataset Size: {len(eval_dataset_raw)}")
    eval_df = eval_dataset_raw.to_pandas()
    eval_df_cleaned = eval_df.drop_duplicates(subset=['prompt'])
    eval_dataset = Dataset.from_pandas(eval_df_cleaned)
    print(f"Removed Duplicate Prompts if any. Cleaned Dataset Size: {len(eval_dataset)}")

    ## Load checkpoint model and tokenizer
    model, tokenizer = setup_checkpoint_model_and_tokenizer(args)
    model_device = model.device

    ## Get two output files paths
    eval_samples_gen_txt_metadata_file_path, eval_samples_gen_txt_file_path = get_gen_txt_file_paths(
        args=args,
        timestamp_suffix=gen_txt_run_tmp
    )

    ## Metadata Logging
    write_metadata_info(
        args=args,
        op_file_path=eval_samples_gen_txt_metadata_file_path
    )

    ## Text Generation Logging
    write_gen_txt_file(
       args=args,
       model=model,
       tokenizer=tokenizer,
       eval_dataset=eval_dataset,
       op_file_path=eval_samples_gen_txt_file_path 
    )

    print(f"Finished Run Successfully! {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Logged Metadata File: {eval_samples_gen_txt_metadata_file_path}\nLogged Text Generation File: {eval_samples_gen_txt_file_path}")