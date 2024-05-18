from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_model_and_tokenizer(model_path, tokenizer_path, use_vanilla_gpt2=False):
    if use_vanilla_gpt2:
        # Load the vanilla GPT-2 model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    else:
        # Load a custom model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode

        # Set the padding token if not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

# Define model paths
custom_model_path = "/cronus_data/araghavan/persuasion/runs/20240511_173459_uzfrorki/models/9ok0dlsx_model"
custom_tokenizer_path = '/cronus_data/araghavan/persuasion/runs/20240511_173459_uzfrorki/models/9ok0dlsx_token'

# Load models and tokenizers
custom_tokenizer, custom_model = load_model_and_tokenizer(custom_model_path, custom_tokenizer_path)
vanilla_tokenizer, vanilla_model = load_model_and_tokenizer(None, None, use_vanilla_gpt2=True)

# Define models and tokenizers for easy management
models = {'Custom Model': custom_model, 'Vanilla GPT-2': vanilla_model}
tokenizers = {'Custom Model': custom_tokenizer, 'Vanilla GPT-2': vanilla_tokenizer}

# List of prompts
prompts = [
    {"id": "124", "text": "I believe that computers are detrimental to kids. Convince me that it is not detrimental to kids"},
    {"id": "456", "text": "CMV: a person making an accusation should be referred to as 'the complainant; and not the victim. In legal matters this is important: The term victim assumes that the person making a complaint is correct. That creates bias at every stage. If you are a suspect being interviewed by the police, hearing the word victim being used to describe the person making an accusation against you is unfair. It makes you feel that the police are biased against you when they are interviewing you. If the matter goes to trial, the jury is more likely to convict someone unfairly if the language used during a trial by the media and police etc assumes guilt. A neutral term such as complainant will result in much fairer outcomes."},
    {"id": "1aa38w", "text": "I think single people should get the most tax breaks in America. CMV! Why should I pay higher taxes when I use less government services than people with children? I challenge you to change my view."},
    {"id": "16ralh", "text": "I think that the feel good military posts on reddit are propaganda CMV The US military uses psychological operations. The military is known to use fake online accounts for gathering information. I think that Reddit is being manipulated by these emotional posts."},
    {"id": "123", "text": "Indian Premier League is destrying the essence of Cricket. CMV"},
    {"id": "234", "text": "Artifical sugar is better than natural sugar. CMV"},
]

def generate_text(model, tokenizer, prompt, max_length=3096, num_return_sequences=1):
    device = "cpu"#'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    encoded_prompt = tokenizer(prompt['text'], return_tensors='pt', padding=True).to(device)
    
    output_sequences = model.generate(
        **encoded_prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

# Generate text for each prompt and model
for prompt in prompts:
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        generated_text = generate_text(model, tokenizer, prompt)
        
        # Print model outputs
        print(f"Model Type/Name: {model_name}")
        print(f"\nPrompt ID: {prompt['id']}")
        print(f"\nPrompt Text: {prompt['text']}")
        print(f"\nGenerated Text: {generated_text}\n")
        print(f"\n\n")
