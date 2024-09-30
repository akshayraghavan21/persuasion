import argparse
import os
import random
from typing import Dict, Any

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig

from persuasion_dpo_trainer import PersuasionDPOTrainer
import wandb
import pandas as pd
import gc

def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments for the DPO training script.
        
        Returns:
            argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DPO training for persuasion")

    ## Project Params
    parser.add_argument("--wandb_project", type=str, default='experiment_gaia_llama2-7b_tests')
    parser.add_argument("--model_name", type=str, default="/gaia_data/pretrained_models/llama2-7b-hf/")
    parser.add_argument("--data_file", type=str, default="dpo_random_neg_op_comment_v001.json")
    parser.add_argument("--project_data_dir", type=str, default="../data/")
    parser.add_argument("--project_output_dir", type=str, default="../output")
    
    ## HF Trainer Params
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")

    ## LORA Params
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    ## DPO Params
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=1536)

    ## Eval Sample Gen Txt Params
    parser.add_argument("--eval_sample_gen_txt_idx", type=int, default=15)
    parser.add_argument("--eval_sample_gen_txt_max_len", type=int, default=1500)
    parser.add_argument("--eval_sample_gen_txt_top_k", type=int, default=50)
    parser.add_argument("--eval_sample_gen_txt_top_p", type=int, default=0.9)
    parser.add_argument("--eval_sample_gen_txt_temperature", type=int, default=0.7)
    parser.add_argument("--eval_sample_gen_txt_do_sample", type=bool, default=True)

    ## General Params
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


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
        'train': dataset.select(get_indices(train_groups)).select(range(50)),
        'validation': dataset.select(get_indices(val_groups)).select(range(15)),
        'test': dataset.select(get_indices(test_groups)).select(range(15)),
    })


def setup_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    """
        Set up the model and tokenizer for training.
    
        This function loads the pre-trained model and tokenizer, and configures
        them for the DPO training task.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        
        Returns:
            tuple: A tuple containing the configured model and tokenizer.
    """
    ## Load Tokenizer and set pad_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="./")
    tokenizer.pad_token = tokenizer.eos_token
    
    ## Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir="./",
        device_map="auto",
    )

    ## Configure Model
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    return model, tokenizer


def get_training_args(args: argparse.Namespace, output_dir: str) -> DPOConfig:
    """
        Create a DPOConfig object with training arguments.
    
        This function sets up the training configuration for the DPO trainer,
        using the parsed command-line arguments and the specified output directory.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            output_dir (str): Directory where training outputs will be saved.
        
        Returns:
            DPOConfig: Configuration object for DPO training.
    """
    return DPOConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        output_dir=output_dir,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
        evaluation_strategy=args.evaluation_strategy,
    )


def train_dpo(args: argparse.Namespace) -> float:
    """
        Train the model using Direct Preference Optimization (DPO).
    
        This function sets up the wandb run, prepares the dataset, model, and trainer,
        and then runs the DPO training process. It also handles logging and saving the model.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        
        Returns:
            float: The final evaluation loss after training.
    """
    try:
        with wandb.init(project=args.wandb_project, config=vars(args), dir=args.project_output_dir) as run:
            print(f"Started Experiment: {run.id} with Args: {args}")
            ## Fetch unique run dir
            run_timestamp_id = run.dir.split('/')[-2].replace("run-", "").replace("-", "_")
            run_output_dir = os.path.join(args.project_output_dir, f"run_{run_timestamp_id}")
            os.makedirs(run_output_dir, exist_ok=True)

            ## Load dataset and model, tokenizer
            datasets = load_and_prepare_dataset_no_leaks(args)
            model, tokenizer = setup_model_and_tokenizer(args)

            ## Configure LORA
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=['v_proj', 'q_proj']
            )
            
            ## Configure DPOConfig
            training_args = get_training_args(args, run_output_dir)
            
            ## Configure Trainer
            trainer = PersuasionDPOTrainer(
                model,
                args=training_args,
                train_dataset=datasets['train'],
                eval_dataset=datasets['validation'],
                tokenizer=tokenizer,
                peft_config=peft_config,
                beta=args.beta,
                max_prompt_length=args.max_prompt_length,
                max_length=args.max_length,
                wandb_run=run
            )
            
            ## Train model
            trainer.train()
            
            ## Write model to disk
            model_output_dir = os.path.join(run_output_dir, "final_model")
            trainer.model.save_pretrained(model_output_dir)
            
            ## Log Sample Level and Eval Level Metrics to Wandb
            run.log({"sample_level_metrics_table_data": trainer.sample_level_metrics_table_data})
            run.log({"eval_sample_gen_txt_table_data": trainer.eval_sample_gen_txt_table_data})
            print("Logged Metrics to Wandb")

            ## Return Eval Loss Metric for Hyperparam optimization
            loss_history = pd.DataFrame(trainer.state.log_history)
            final_loss = loss_history.eval_loss.dropna().iloc[-1]
            print(f"Finished Experiment: {run.id} With Final Eval Loss: {final_loss}")
            return final_loss, run.id 
    finally:
        # Cleanup
        del datasets, model, tokenizer, trainer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    ## Parse CLI Arguments
    args = parse_args()

    ## Train DPO
    final_eval_loss = train_dpo(args)

    ## Print Metric Performance
    print(f"Final Eval Loss: {final_eval_loss}")
