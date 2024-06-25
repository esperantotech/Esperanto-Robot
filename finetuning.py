import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import accelerate
from accelerate import Accelerator
import peft
import transformers
import json


def single_GPU_FT():

    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    print(f"PEFT version: {peft.__version__}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model and tokenizer
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
    max_length = 1024


    # Function to preprocess the dataset
    def process_dataset(data):
        for entry in data:
            input_text = entry['input']
            output_text = entry['output']
            
            if not input_text.startswith("<|user|>"):
                entry['input'] = "<|user|>" + input_text
            if not input_text.endswith("<|end|>"):
                entry['input'] = entry['input'] + "<|end|>"
            if not output_text.startswith("<|assistant|>"):
                entry['output'] = "<|assistant|>" + output_text
        
        return data

    # # Load and process datasets
    # with open('/home/xilun/ET_robot/two_stack_dataset.json', 'r') as f:
    #     two_stack_dataset = json.load(f)

    # with open('/home/xilun/ET_robot/three_stack_dataset.json', 'r') as f:
    #     three_stack_dataset = json.load(f)

    # # Process the datasets
    # two_stack_dataset = process_dataset(two_stack_dataset)
    # three_stack_dataset = process_dataset(three_stack_dataset)


    # Load your dataset
    # dataset = load_dataset('json', data_files={'train': '/home/xilun/ET_robot/updated_two_stack_dataset.json', 'validation': '/home/xilun/ET_robot/updated_three_stack_dataset.json'})
    dataset = load_dataset('json', data_files={'train': '/home/xilun/ET_robot/dataset/all_data.json', 'validation': '/home/xilun/ET_robot/updated_three_stack_dataset.json'})
    
    

    # Preprocess the dataset
    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['output']
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length", add_special_tokens=True).input_ids
        model_inputs['labels'] = labels
        return model_inputs

    # Apply preprocessing
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Filter and ensure the dataset is of the correct format
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Setup LoRA configuration
    target_modules = ['self_attn.qkv_proj', 'mlp.gate_up_proj']
    # target_modules = ['self_attn.qkv_proj']
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank of the update matrices
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=target_modules
    )

    # Get the model with LoRA
    model = get_peft_model(model, config).to(device)

    output_dir = "./phi-3-mini-LoRA"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=3,  # Reduced batch size to save memory
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,  # Reduced batch size to save memory
        log_level="debug",
        save_strategy="epoch",
        logging_steps=30,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_steps=20,
        num_train_epochs=60,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        report_to="wandb",
        seed=42,
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        peft_config=config,
        dataset_text_field="input",
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    
def multi_GPU_FT():
    print ("Distributed Training in Progress")
    
    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    print(f"PEFT version: {peft.__version__}")

    # Initialize the accelerator
    accelerator = Accelerator()

    # Device configuration
    device = accelerator.device

    # Load pre-trained model and tokenizer
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
    max_length = 512

    # Function to preprocess the dataset
    def process_dataset(data):
        for entry in data:
            input_text = entry['input']
            output_text = entry['output']
            
            if not input_text.startswith(""):
                entry['input'] = "" + input_text
            if not input_text.endswith(""):
                entry['input'] = entry['input'] + ""
            if not output_text.startswith(""):
                entry['output'] = "" + output_text
        
        return data

    # Load and process datasets
    with open('/home/xilun/ET_robot/two_stack_dataset.json', 'r') as f:
        two_stack_dataset = json.load(f)

    with open('/home/xilun/ET_robot/three_stack_dataset.json', 'r') as f:
        three_stack_dataset = json.load(f)

    # Process the datasets
    two_stack_dataset = process_dataset(two_stack_dataset)
    three_stack_dataset = process_dataset(three_stack_dataset)

    # Save the processed datasets
    with open('/home/xilun/ET_robot/updated_two_stack_dataset.json', 'w') as f:
        json.dump(two_stack_dataset, f)
        
    with open('/home/xilun/ET_robot/updated_three_stack_dataset.json', 'w') as f:
        json.dump(three_stack_dataset, f)

    # Load your dataset
    dataset = load_dataset('json', data_files={'train': '/home/xilun/ET_robot/updated_two_stack_dataset.json', 'validation': '/home/xilun/ET_robot/updated_three_stack_dataset.json'})
 
    # Preprocess the dataset
    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['output']
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length").input_ids
        model_inputs['labels'] = labels
        return model_inputs

    # Apply preprocessing
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Filter and ensure the dataset is of the correct format
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Setup LoRA configuration
    target_modules = ['self_attn.qkv_proj']
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank of the update matrices
        lora_dropout=0.2,
        target_modules=target_modules
    )

    # Get the model with LoRA
    model = get_peft_model(model, config).to(device)

    output_dir = "./phi-3-mini-LoRA"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=8,  # Reduced batch size to save memory
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,  # Reduced batch size to save memory
        log_level="debug",
        save_strategy="epoch",
        logging_steps=30,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_steps=20,
        num_train_epochs=20,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        report_to="wandb",
        seed=42,
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        peft_config=config,
        dataset_text_field="input",
        tokenizer=tokenizer,
        accelerator=accelerator  # Pass the accelerator to the trainer
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    

if __name__ == "__main__":
    single_GPU_FT()
    # multi_GPU_FT()


