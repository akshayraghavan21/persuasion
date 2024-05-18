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
│   ├── dpo_src_002.json
├── documentation
│   └── dpo_gpt2_reg_beta_v1.png
├── input
│   └── config.text
├── requirements
│   └── env.yml
├── op_logs
│   └── op_logs.log
├── runs
│   ├── YYYYMMDD_HHMMSS_WANDBRUNID
│       ├── logs
│       ├── model
│       └── wandb
└── src
    ├── custom_run_code_wandb.py
    ├── dpo_v4.py
    ├── generate_sample_text.py
    └── zephyr_test.py
```
## Run Instructions
Pre-requisites: Setup Conda environment and load the env.yml file in ~/Persuasion/requirements dir to setup the environment required for the codebase.

### Config For Hyperparameter Tuning
The script is modularized to receive config values to hyperparameter tune the model using wandb. As per the format on how sweep config has to be set, the config file is created and passed as an argument to the training run file.


### Training
As per the config file above, models, tokenizer, hyperparameters are initialized/decided.

Run the "```custom_run_code_wandb.py```" file with the below command to train the DPO model:
```nohup python ~/persuasion/src/custom_run_code_wandb.py "~/persuasion/input/config.txt" > ~/persuasion/op_logs/op_logs.log 2>&1 &```

This training process creates files in ```op_logs``` and ```runs``` dir which consists of wandb logs, model weights, and tokenizer checkpoints. This can be utilized further for evaluation.


### Evaluation
Post model training, the satisfactory model can be used to generate sample text using the script ```~/persuasion/src/generate_sample_text.py```. Edit the prompt in the script to generate output for it.

** Edit the path to the satisfactory model to generate samples using that model trained.

## Results
Can view the run results here: ```https://api.wandb.ai/links/sbu-hlab-persuasion/xk0r538f```
