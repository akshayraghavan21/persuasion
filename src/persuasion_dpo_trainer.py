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

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.amp as amp
import torch.nn as nn
from contextlib import contextmanager, nullcontext
from torch.utils.data import DataLoader
from accelerate.utils import is_deepspeed_available, tqdm
from transformers.trainer_utils import EvalLoopOutput
import csv
import datetime
from datasets import Dataset
from torchmetrics.text import Perplexity

class PersuasionDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        self.wandb_run = kwargs.pop('wandb_run', None)
        self.file_suffix = self.generate_file_suffix(self.wandb_run)
        self.sample_level_metrics_cols = ['type', 'epoch', 'step', 'prompt', 'loss', 'chosen_reward', 'rejected_reward', 'policy_logps_chosen', 'policy_logps_rejected', 'chosen', 'rejected', 'policy_chosen_pred_text', 'reference_chosen_pred_text', 'policy_rejected_pred_text', 'reference_rejected_pred_text']
        self.eval_sample_gen_txt_cols = ['eval_sample_index', 'epoch', 'prompt', 'generated_text', 'chosen', 'rejected']

        # Extract custom arguments
        self.sample_level_metrics_file_name = kwargs.pop('sample_level_metrics_file_name', f"batch_level_dpo_logs_{self.file_suffix}.csv")
        self.sample_level_metrics_table_data = kwargs.pop('sample_level_metrics_table_data', None)

        self.eval_sample_gen_txt_file_name = kwargs.pop('eval_sample_gen_txt_file_name', f"generated_samples_{self.file_suffix}.csv")
        self.eval_sample_gen_txt_table_data = kwargs.pop('eval_sample_gen_txt_table_data', None)
        self.eval_sample_gen_txt_idx = kwargs.pop('eval_sample_gen_txt_idx', 12)
        self.eval_sample_gen_txt_max_len = kwargs.pop('eval_sample_gen_txt_max_len', 1500)
        self.eval_sample_gen_txt_top_k = kwargs.pop('eval_sample_gen_txt_top_k', 50)
        self.eval_sample_gen_txt_top_p = kwargs.pop('eval_sample_gen_txt_top_p', 0.9)
        self.eval_sample_gen_txt_temperature = kwargs.pop('eval_sample_gen_txt_temperature', 0.7)
        self.eval_sample_gen_txt_do_sample = kwargs.pop('eval_sample_gen_txt_do_sample', True)
        
        
        self.generate_and_log_custom_sample_during_eval = kwargs.pop('generate_and_log_custom_sample_during_eval', True)

        super().__init__(*args, **kwargs)
        self.perplexity_metric = Perplexity(ignore_index=-100)

        # Initialize wandb tables if not provided
        if self.sample_level_metrics_table_data is None and self.wandb_run is not None:
            self.sample_level_metrics_table_data = wandb.Table(columns=self.sample_level_metrics_cols)
        
        if self.eval_sample_gen_txt_table_data is None and self.wandb_run is not None:
            self.eval_sample_gen_txt_table_data = wandb.Table(columns=self.eval_sample_gen_txt_cols)
    
    def generate_file_suffix(self, wandb_run):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if wandb_run is not None:
            return wandb_run.dir.split('/')[-2].replace("run-", "").replace("-", "_")
        else:
            return timestamp
        
    def calculate_perplexity(self, logits, labels, is_encoder_decoder, len_chosen):

        ppl_logits_chosen = logits[:len_chosen]
        ppl_labels_chosen = labels[:len_chosen]
        
        ppl_logits_rejected = logits[len_chosen:]
        ppl_labels_rejected = labels[len_chosen:] 

        if not is_encoder_decoder:
            ppl_logits_chosen = ppl_logits_chosen[:, :-1, :]
            ppl_labels_chosen = ppl_labels_chosen[:, 1:]

            ppl_logits_rejected = ppl_logits_rejected[:, :-1, :]
            ppl_labels_rejected = ppl_labels_rejected[:, 1:] 
        
    
        chosen_ppl = self.perplexity_metric(
            ppl_logits_chosen.to("cpu"),
            ppl_labels_chosen.to("cpu")
        )
        rejected_ppl = self.perplexity_metric(
            ppl_logits_rejected.to("cpu"),
            ppl_labels_rejected.to("cpu")
        )
        chosen_ppl = torch.nan_to_num(chosen_ppl, nan=0.0)
        rejected_ppl = torch.nan_to_num(rejected_ppl, nan=0.0)

        return chosen_ppl, rejected_ppl
    
    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        ## Added/Edited as part of Persuasion-Perplexity
        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    reference_chosen_logps, reference_rejected_logps, _, _, _, (reference_chosen_ppls, reference_rejected_ppls), (reference_chosen_predictions_text, reference_rejected_predictions_text) = self.concatenated_forward(
                        self.model, padded_batch
                    )[:7]
            else:
                reference_chosen_logps, reference_rejected_logps, _, _, _, (reference_chosen_ppls, reference_rejected_ppls), (reference_chosen_predictions_text, reference_rejected_predictions_text) = self.concatenated_forward(
                    self.ref_model, padded_batch
                )[:7]

        return reference_chosen_logps, reference_rejected_logps, (reference_chosen_ppls, reference_rejected_ppls), (reference_chosen_predictions_text, reference_rejected_predictions_text)
        ## End

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            reference_chosen_ppls = []
            reference_rejected_ppls = []
            reference_chosen_predictions_texts = []
            reference_rejected_predictions_texts= []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp, (reference_chosen_ppl, reference_rejected_ppl), (reference_chosen_predictions_text, reference_rejected_predictions_text)  = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp, reference_chosen_ppl, reference_rejected_ppl = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp, reference_chosen_ppl, reference_rejected_ppl)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())
                reference_chosen_ppls.append(reference_chosen_ppl.cpu())
                reference_rejected_ppls.append(reference_rejected_ppl.cpu())
                reference_chosen_predictions_texts.append(reference_chosen_predictions_text)
                reference_rejected_predictions_texts.append(reference_rejected_predictions_text)

                # Unnecessary cache clearing to avoid OOM
                torch.cuda.empty_cache()
                self.accelerator.free_memory()

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()
            all_reference_chosen_ppls = torch.cat(reference_chosen_ppls).float().numpy()
            all_reference_rejected_ppls = torch.cat(reference_rejected_ppls).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_ppls", column=all_reference_chosen_ppls
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_ppls", column=all_reference_rejected_ppls
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_predictions_texts", column=reference_chosen_predictions_texts
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_predictions_texts", column=reference_rejected_predictions_texts
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            reference_chosen_ppls = []
            reference_rejected_ppls = []
            reference_chosen_predictions_texts = []
            reference_rejected_predictions_texts= []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp, (reference_chosen_ppl, reference_rejected_ppl), (reference_chosen_predictions_text, reference_rejected_predictions_text) = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp, reference_chosen_ppl, reference_rejected_ppl = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp, reference_chosen_ppl, reference_rejected_ppl)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())
                reference_chosen_ppls.append(reference_chosen_ppl.cpu())
                reference_rejected_ppls.append(reference_rejected_ppl.cpu())
                reference_chosen_predictions_texts.append(reference_chosen_predictions_text)
                reference_rejected_predictions_texts.append(reference_rejected_predictions_text)


            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()
            all_reference_chosen_ppls = torch.cat(reference_chosen_ppls).float().numpy()
            all_reference_rejected_ppls = torch.cat(reference_rejected_ppls).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )
            eval_dataset = eval_dataset.add_column(name="reference_chosen_ppls", column=all_reference_chosen_ppls)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_ppls", column=all_reference_rejected_ppls
            )
            eval_dataset = eval_dataset.add_column(name="reference_chosen_predictions_texts", column=reference_chosen_predictions_texts)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_predictions_texts", column=reference_rejected_predictions_texts
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.get("concatenated_decoder_input_ids")

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        ## Added as part of getting model prediction as text
        model_text_logits = all_logits[:, :-1, :]
        model_logits_softmax = torch.softmax(model_text_logits, axis=-1)
        model_logits_argmax = torch.argmax(model_logits_softmax, axis=-1)
        batch_model_predictions_text = self.tokenizer.batch_decode(model_logits_argmax)
        chosen_predictions_text = batch_model_predictions_text[:len_chosen]
        rejected_predictions_text = batch_model_predictions_text[len_chosen:]
        ## End

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        ## Added/Edited as part of Persuasion-Perplexity
        chosen_ppl, rejected_ppl = self.calculate_perplexity(
            logits=all_logits,
            labels=labels,
            is_encoder_decoder=self.is_encoder_decoder,
            len_chosen=len_chosen
        )
        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, (chosen_ppl, rejected_ppl), (chosen_predictions_text, rejected_predictions_text), outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, (chosen_ppl, rejected_ppl), (chosen_predictions_text, rejected_predictions_text))
        ## End

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        concat_forward_return_size = 7
        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            (policy_chosen_ppls, policy_rejected_ppls),
            (policy_chosen_predictions_texts, policy_rejected_predictions_texts)
        ) = forward_output[:concat_forward_return_size]
        if self.aux_loss_enabled:
            aux_loss = forward_output[concat_forward_return_size]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and (self.precompute_ref_log_probs or self.args.rpo_alpha is not None)
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
            reference_chosen_ppls = batch["reference_chosen_ppls"]
            reference_rejected_ppls = batch["reference_rejected_ppls"]
            reference_chosen_predictions_texts = batch["reference_chosen_ppls"]
            reference_rejected_predictions_texts = batch["reference_rejected_ppls"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_chosen_logps, reference_rejected_logps, _, _, _, (reference_chosen_ppls, reference_rejected_ppls), (reference_chosen_predictions_texts, reference_rejected_predictions_texts) = self.concatenated_forward(
                            self.model, batch
                        )[:concat_forward_return_size]
                else:
                    reference_chosen_logps, reference_rejected_logps, _, _, _, (reference_chosen_ppls, reference_rejected_ppls), (reference_chosen_predictions_texts, reference_rejected_predictions_texts) = self.concatenated_forward(
                        self.ref_model, batch
                    )[:concat_forward_return_size]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}perplexity_policy/chosen"] = policy_chosen_ppls.detach().mean().cpu()
        metrics[f"{prefix}perplexity_policy/rejected"] = policy_rejected_ppls.detach().mean().cpu()
        metrics[f"{prefix}perplexity_reference/chosen"] = reference_chosen_ppls.detach().mean().cpu()
        metrics[f"{prefix}perplexity_reference/rejected"] = reference_rejected_ppls.detach().mean().cpu()

        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        # Append to CSV file
        sample_level_metrics_file_path = os.path.join(self.args.output_dir, self.sample_level_metrics_file_name)
        # print(sample_level_metrics_file_path) 
        # Create the CSV file with headers if it doesn't exist
        if not os.path.exists(sample_level_metrics_file_path):
            with open(sample_level_metrics_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.sample_level_metrics_cols)

        # Write data for each sample in the batch
        with open(sample_level_metrics_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(batch['prompt'])):
                writer.writerow([
                    train_eval,
                    self.state.epoch,
                    self.state.global_step,
                    batch['prompt'][i],
                    losses[i].item(),
                    chosen_rewards[i].item(),
                    rejected_rewards[i].item(),
                    policy_chosen_logps[i].item(),
                    policy_rejected_logps[i].item(),
                    batch['chosen'][i],
                    batch['rejected'][i],
                    policy_chosen_predictions_texts[i],
                    reference_chosen_predictions_texts[i],
                    policy_rejected_predictions_texts[i],
                    reference_rejected_predictions_texts[i],
                ])
                if self.sample_level_metrics_table_data is not None:
                    self.sample_level_metrics_table_data.add_data(
                        train_eval,
                        self.state.epoch,
                        self.state.global_step,
                        batch['prompt'][i],
                        losses[i].item(),
                        chosen_rewards[i].item(),
                        rejected_rewards[i].item(),
                        policy_chosen_logps[i].item(),
                        policy_rejected_logps[i].item(),
                        batch['chosen'][i],
                        batch['rejected'][i],
                        policy_chosen_predictions_texts[i],
                        reference_chosen_predictions_texts[i],
                        policy_rejected_predictions_texts[i],
                        reference_rejected_predictions_texts[i], 
                    )

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # Call the original evaluation loop
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # Generate text for a random sample
        if self.generate_and_log_custom_sample_during_eval:
            # Select a random sample
            eval_dataset = dataloader.dataset
            random_index = self.eval_sample_gen_txt_idx
            sample = eval_dataset[random_index]

            model_device = self.model.device 
            # Prepare the input
            inputs = self.tokenizer(sample['prompt'], return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                gen_output = self.model.generate(
                    **inputs,
                    max_length=self.eval_sample_gen_txt_max_len,
                    top_k=self.eval_sample_gen_txt_top_k,
                    top_p=self.eval_sample_gen_txt_top_p,
                    temperature=self.eval_sample_gen_txt_temperature,
                    do_sample=self.eval_sample_gen_txt_do_sample
                )
                generated_text = self.tokenizer.batch_decode(gen_output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

            print("Printing Eval Text:",generated_text)
            
            # Get the current epoch
            current_epoch = self.state.epoch
                   
            # Print to screen
            print(f"Generated text for sample {random_index} at epoch {current_epoch:.2f}:")
            print(generated_text)
            
            # Write to file
            eval_samples_file_path = os.path.join(self.args.output_dir, self.eval_sample_gen_txt_file_name)

        
            # Create the CSV file with headers if it doesn't exist
            if not os.path.exists(eval_samples_file_path):
                with open(eval_samples_file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.eval_sample_gen_txt_cols)
            
            with open(eval_samples_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    random_index,
                    current_epoch,
                    sample['prompt'],
                    generated_text,
                    sample['chosen'],
                    sample['rejected'],
                ])
            
            if self.eval_sample_gen_txt_table_data is not None:
                self.eval_sample_gen_txt_table_data.add_data(
                    random_index,
                    current_epoch,
                    sample['prompt'],
                    generated_text,
                    sample['chosen'],
                    sample['rejected'], 
                )
        
        return output
 