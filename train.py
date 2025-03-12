import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoProcessor, AutoConfig,
    Trainer, TrainingArguments
)
from Phi_3V_MoE.moe_phi3_v import Phi3VForCausalLMMoE, Phi3VForCausalLMMoEConfig
import argparse
import os
from huggingface_hub import HfApi, hf_hub_download
import shutil

# Parsing hyperparameters from SageMaker
parser = argparse.ArgumentParser()
parser.add_argument("--num_train_epochs", type=int, default=2)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model names
expert_model_names = [
    "lamm-mit/Cephalo-Phi-3-vision-128k-4b-beta",
    "microsoft/Phi-3-vision-128k-instruct",
    "lamm-mit/Cephalo-LaTeX-Phi-3-vision-128k-4b-beta"
]

# Load processor from base model
processor = AutoProcessor.from_pretrained(expert_model_names[1], trust_remote_code=True)

# Load expert models
expert_models = [
    AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    for name in expert_model_names
]

# Initialize MoE Model configuration
base_model = expert_models[1]
config = AutoConfig.from_pretrained(expert_model_names[1], trust_remote_code=True)
moe_config = Phi3VForCausalLMMoEConfig(config=config, k=1, num_expert_models=len(expert_models))
moe_model = Phi3VForCausalLMMoE(moe_config, base_model, expert_models, layer_dtype=torch.bfloat16).to(device)

# Load training dataset
train_dataset = load_dataset("lamm-mit/Cephalo-Wikipedia-Materials", split="train")

# Custom Data Collator
class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts, images = [], []
        for example in examples:
            image = example["image"]
            question, answer = example["query"], example["answer"]
            messages = [
                {"role": "user", "content": '<|image_1|>\n'+question},
                {"role": "assistant", "content": answer},
            ]
            text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images.append(image)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels < 0] = -100
        batch["labels"] = labels
        return batch

data_collator = MyDataCollator(processor)

# Define TrainingArguments (using SageMaker hyperparameters)
training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=4,
    warmup_steps=250,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=16,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
)

# SageMaker-compatible Trainer
trainer = Trainer(
    model=moe_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the trained model (for SageMaker deployment)
processor.save_pretrained("/opt/ml/model")
moe_model.save_pretrained("/opt/ml/model")
