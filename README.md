# Persuasion
```
#######################################################
Project Name: Persuasion
Dev: Akshay Raghavan
Description: Modeling Persuasion for Mental Health using Large Language Models
#######################################################
```

## Introduction
Persuasion is an attempt made by a person to change anothers belief, attitude, or emptions associated with some issue, person, concept, or object. It is a key component in mental health practices such as Cognitive Behavioral Therapy (CBT) and counseling. The growing interest in digital tools for mental health prompts us to explore whether large language models can serve as effective persuaders and whether their presence can enhance interpersonal processes directly.

## Dataset
Our dataset is derived from the subreddit r/changemyview, which serves as a platform for discussions about persuasion. In this community, a user (OP) posts their opinion and invites others to challenge and potentially change their viewpoint through comments. If a comment successfully convinces the OP, they award a delta symbol to signify its persuasiveness.

## Method
The goal of modeling persuasion can be framed as a reinforcement learning task: learning from delta 1 comments (those deemed persuasive) and unlearning from delta 0 comments (those not persuasive). We utilize Direct Preference Optimization to fine-tune our model through reinforcement learning with human feedback.

## Directory Structure
```
~Persuasion
├── data
│   ├── dpo_random_neg_op_comment_v001.json
│   └── dpo_random_neg_op_comment_v002.json
├── documentation
│   └── next_steps.txt
├── input
├── lcache
├── README.md
├── requirements
│   └── dpo_noneric_optimized.yaml
└── src
    ├── llama2_7b_dpo_modified.py
    ├── llama2_7b_dpo_modified_v2.py
    ├── llama2_7b_dpo.py
    ├── persuasion_dpo_trainer.py
    ├── persuasion_experiment.py
    └── persuasion_hp_search_optuna.py
```
## Run Instructions
Pre-requisites: Setup Conda environment and load the env.yml file in ~/persuasion/requirements dir to setup the environment required for the codebase.
```
conda env create -f ./requirements/dpo_noneric_optimized.yaml
conda activate dpo_noneric_optimized
```

The experiments and logging is done to wandb, hence would require to setup wandb login and more. If not setup, follow further, else please skip this step.
```
wandb login
```
This will prompt you to enter your API key. You can find your API key in your W&B account by going to https://wandb.ai/authorize.

### Memory Requirement
Hyperparam search or running an experiment for DPO + LoRA Training requires 2+ A6000 GPUs.

### Training/Hyperparam tuning
For default setup, please refer to ~/persuasion/documentation/default_config.txt

Note: It's best to run these experiments in a tmux session.
```
tmux new -s dpo_persuasion_exp
tmux a -t dpo_persuasion_exp
```

### Run an Experiment
To run an experiment with default hyperparams [refer to default_config.txt file], run the below command:
```
[CUDA_VISIBLE_DEVICES=0,1] python ./src/persuasion_experiment.py"
```

### Hyperparam Search
We perform hyperparam search using optuna. To run one, please run the below command:
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/persuasion_hp_search_optuna.py --gradient_accumulation_steps=16 --logging_steps=50
```

Post running an hyperparam search, information of the trials along with the best trial is provided in: ```~/persuasion/output/study_results_YYYYMMDD_HHMMSS.json"

### Custom Hyperparam Search
Edit the objective function in ```~/persuasion/src/persuasion_hp_search_optuna.py" to tune the parameters of your choice.

## Interpreting the Wandb Dashboard
Once an experiment has successfully completed, it's visible in your wandb dashboard under the project specified as part of ```wandb_project``` cli argument [default: experiment_gaia_llama2-7b_tests]

#### Eval & Train
Policy: model to be aligned
Reference: base model

1. loss - DPO Loss [lower the better]
2. perplexity - measures how well a model predicts a sequence of words and provides an indication of how "surprised" the model is by the actual data
3. rewards/chosen: the mean difference between the log probabilities of the policy model and the reference model for the chosen responses scaled by beta
4. rewards/rejected: the mean difference between the log probabilities of the policy model and the reference model for the rejected responses scaled by beta
5. rewards/accuracies: mean of how often the chosen rewards are > than the corresponding rejected rewards
6. rewards/margins: the mean difference between the chosen and corresponding rejected rewards

#### Tables
1. sample_level_metrics_table_data - train example level qualitative and quantitative metrics
2. eval_sample_gen_txt_table_data - test custom sample text generation across epochs