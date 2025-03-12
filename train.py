import torch
from transformers import AutoModelForCausalLM, AutoProcessor,AutoConfig  

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #number of parameters in b
    return total_params/1e9, trainable_params/1e9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_moe = "lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta"

processor = AutoProcessor.from_pretrained(model_name_moe, trust_remote_code=True) 
moe_model = AutoModelForCausalLM.from_pretrained(
    model_name_moe,
    trust_remote_code=True,  torch_dtype=torch.bfloat16,    
).to(device)
count_parameters(moe_model)

from huggingface_hub import HfApi, hf_hub_download
from tqdm.notebook import tqdm
import os
import shutil

# Repository details
repo_id = "lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta"
api = HfApi()

# List all files in the repository
files_in_repo = api.list_repo_files(repo_id)

# Filter for .py files
py_files = [file for file in files_in_repo if file.endswith('.py')]

# Directory to save the downloaded files
save_dir = "./Phi_3V_MoE/"
os.makedirs(save_dir, exist_ok=True)

# Download each .py file
for file_name in tqdm(py_files):
    file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
    new_path = os.path.join(save_dir, file_name)
    shutil.move(file_path, new_path)
    print(f"Downloaded: {file_name}")

print("Download completed.")

from Phi_3V_MoE.moe_phi3_v import Phi3VForCausalLMMoE, Phi3VForCausalLMMoEConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model specialized in bio-inspired/mechanics and materials
model_name_1 = f"lamm-mit/Cephalo-Phi-3-vision-128k-4b-beta"
model_1 = AutoModelForCausalLM.from_pretrained(
    model_name_1,
    trust_remote_code=True,  torch_dtype=torch.bfloat16, 
    
).to(device)

#Original model
model_name_2 = f"microsoft/Phi-3-vision-128k-instruct"
model_2 = AutoModelForCausalLM.from_pretrained(
    model_name_2,
    trust_remote_code=True,  torch_dtype=torch.bfloat16, 
    
).to(device)

#Model trained on conversion of images to LaTeX formulas
model_name_3 = f"lamm-mit/Cephalo-LaTeX-Phi-3-vision-128k-4b-beta"
model_3 = AutoModelForCausalLM.from_pretrained(
    model_name_3,
    trust_remote_code=True,  torch_dtype=torch.bfloat16, 
    
).to(device)

dtype = torch.bfloat16  # Desired dtype for new layers in MoE model

# Initialize the models
base_model = copy.deepcopy(model_2)  # Your base model
expert_models = [model_1, model_2,  model_3  ]  # List of expert models
 
# Load a processor (e.g. from base model)
processor = AutoProcessor.from_pretrained(model_name_2, trust_remote_code=True) 

# Create the config
config =  AutoConfig.from_pretrained(model_name_2, trust_remote_code=True)

# Create the MoE model
moe_config = Phi3VForCausalLMMoEConfig(config=config, k=1, num_expert_models=len (expert_models))
moe_model = Phi3VForCausalLMMoE(moe_config, base_model, expert_models,  layer_dtype = dtype).to(device)

count_parameters(expert_models[0]),count_parameters(moe_model)

messages = [ {"role": "user", "content": "<|image_1|>\nWhat is shown in this image, and what is the relevance for materials design?"}, ]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt

from PIL import Image
import requests

image_1 = Image.open(requests.get("https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg", stream=True).raw) 
image_2 = Image.open(requests.get("https://https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg", stream=True).raw) 
image_3 = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/a/a0/Euplectella_aspergillum_Okeanos.jpg", stream=True).raw) 

prompts_per_expert = [
    [{"text": "<|user|>\n<|image_1|>\nPrompt 1 for expert 1<|end|>\n<|assistant|>\n", "image": [image_1]}, 
     {"text": "<|user|>\n<|image_1|>\nPrompt 2 for expert 1<|end|>\n<|assistant|>\n", "image": [image_1]}],

    [{"text": "<|user|>\n<|image_1|>\nPrompt 1 for expert 2<|end|>\n<|assistant|>\n", "image": [image_2]}, 
     {"text": "<|user|>\n<|image_1|>\nPrompt 2 for expert 2<|end|>\n<|assistant|>\n", "image": [image_2]}],

    [{"text": "<|user|>\n<|image_1|>\nPrompt 1 for expert 3<|end|>\n<|assistant|>\n", "image": [image_3]}, 
     {"text": "<|user|>\n<|image_1|>\nPrompt 2 for expert 3<|end|>\n<|assistant|>\n", "image": [image_3]}],
]

# Train gating layers using the provided prompts
gating_layer_params = moe_model.train_gating_layer_params_from_hidden_states(processor, prompts_per_expert,
                                              epochs=1000,
                                              loss_steps=100,
                                              lr=5e-5,
                                          )

# Set parameters
moe_model.set_gating_layer_params(gating_layer_params)

freeze_except_gating_layers(moe_model)
count_parameters(moe_model)
un_freeze_all(moe_model)

from datasets import load_dataset

train_dataset = load_dataset("lamm-mit/Cephalo-Wikipedia-Materials", split="train")

import random

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["query"] 
            answer = example["answer"]            
            messages = [ {
                            "role": "user",  "content": '<|image_1|>\n'+question},
                           {"role": "assistant", "content": f"{answer}"}, ]
                
            text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
            images.append(image)
             
        batch = processor(text=text, images=[image], return_tensors="pt", padding=True
            
        labels = batch["input_ids"].clone() 
        labels[labels <0] = -100 

        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)

from transformers import TrainingArguments, Trainer

optim = "paged_adamw_8bit"

training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=250,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=25,
    output_dir="output_training",
    optim=optim,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=16,
    #fp16=True,
    bf16=True,  
    push_to_hub_model_id=FT_repo_id,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=moe_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

merged_name='Cephalo-Phi-3-MoE-vision-128k-3x4b'
repo_id= '...'
processor.push_to_hub (repo_id+'/'+merged_name, safe_serialization=False)
moe_model.push_to_hub (repo_id+'/'+merged_name, safe_serialization=False)

merged_name='Cephalo-Phi-3-MoE-vision-128k-3x4b'
processor.save_pretrained(merged_name,safe_serialization=False)
moe_model.save_pretrained(merged_name,safe_serialization=False )
