import optuna
from persuasion_experiment import train_dpo, parse_args
import wandb
import torch
import gc
import json
import os
from datetime import datetime

def objective(trial):
    args = parse_args()
    
    # Define the hyperparameters to tune
    args.learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    args.beta = trial.suggest_categorical('beta', [0.10, 0.15, 0.20, 0.25])
    args.lr_scheduler_type = trial.suggest_categorical('lr_scheduler_type', ['cosine', 'linear'])
    # args.num_train_epochs = trial.suggest_int('num_train_epochs', 2, 8) 

    # Run the training
    try:
        # Run the training
        best_metric, wandb_run_id = train_dpo(args)
        trial.set_user_attr('wandb_run_id', wandb_run_id)
        print(trial.user_attrs.get('wandb_run_id', None))
        return best_metric
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return float('inf')  # Return a large value to indicate failure
    finally:
        # Additional cleanup if necessary
        torch.cuda.empty_cache()
        gc.collect()

def optimize_hyperparameters(n_trials=10):
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Create a unique filename for the study results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"study_results_{timestamp}.json"
    filepath = os.path.abspath(os.path.join("../output", filename))
    
    # Ensure the output directory exists
    os.makedirs("../output", exist_ok=True)
    
    # Save study results as JSON
    study_results = {
        "best_trial": {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "wandb_run_id": None  # We'll try to fetch this from the trial user attrs
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "wandb_run_id": t.user_attrs.get('wandb_run_id', None)
            } for t in study.trials
        ]
    }
    # Try to fetch the wandb run ID for the best trial
    best_trial_wandb_id = trial.user_attrs.get('wandb_run_id', None)
    if best_trial_wandb_id:
        study_results["best_trial"]["wandb_run_id"] = best_trial_wandb_id
    
    with open(filepath, "w") as f:
        json.dump(study_results, f, indent=2)
    
    print(f"Study results saved to: {filepath}")
    
if __name__ == "__main__":
    optimize_hyperparameters(n_trials=3)