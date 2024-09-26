import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel
from trl import SFTTrainer
import accelerate
from accelerate import Accelerator
import peft
import transformers
import json
import argparse
import time


def single_GPU_FT():

    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    print(f"PEFT version: {peft.__version__}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model and tokenizer
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
    max_length = 1024
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

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


    # Load your dataset
    dataset = load_dataset('json', data_files={'train': '/all_data.json', 'validation': '/172_dataset.json'}) ## please make sure the validation data is the one you set during data generation
    
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
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank of the update matrices
        lora_alpha=32,
        lora_dropout=0.05,
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
        per_device_train_batch_size=6,  # Reduced batch size to save memory
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=3,  # Reduced batch size to save memory
        log_level="debug",
        save_strategy="epoch",
        logging_steps=30,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_steps=20,
        num_train_epochs=25,
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
        max_seq_length = max_length,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
def Evaluation(device):
    output_dir = "./phi-3-mini-LoRA/checkpoint-1000"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True, add_eos_token=True, use_fast=True)

    # Apply the LoRA adapter to the base model
    model = PeftModel.from_pretrained(model, output_dir, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model = model.to(device)
    
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    def generate_response(input_text):
        print ("generating response...")
        start_time = time.time()
        output = pipe(input_text, max_new_tokens=2048, do_sample=True, temperature=0.2, num_beams=1, top_k=50, top_p=0.95)
        end_time = time.time()
        elapsed_time = end_time - start_time
        num_tokens = len(tokenizer.encode(output[0]['generated_text']))
        time_per_token = elapsed_time / num_tokens if num_tokens > 0 else 0
        return output[0]['generated_text'].strip(), elapsed_time, time_per_token

    eval_times = 1
    ## loading evaluation dataset
    eval_dataset_path = 'updated_val_data.json'
    eval_dataset = json.loads(open(eval_dataset_path).read())
    
    for i in range (eval_times):
        prompt = eval_dataset[i+10]['text']
        ## get the text prompt until the text "<|end|>" appears
        prompt = prompt.split("<|assistant|>")[0] + "<|assistant|>"
        # Evaluate the model
        result = generate_response(prompt)
        ## only save the python code part from the completion
        with open(f'box/Phi-3-FT/output_{i}.txt', 'w') as f:
            f.write(result)
        try: 
            completion_code = result.split('**\npython')[1].split('<|end|>')[0]
        except:
            completion_code = result.split('```python')[1].split('```')[0]
        with open(f'box/Phi-3-FT/python_output_{i}.txt', 'w') as f:
            f.write(completion_code)
        print(f'Python code saved to python_output_{i}.txt') 
    

if __name__ == "__main__":
    ## initiliaze arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--FT", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.FT:
        single_GPU_FT() 
    if args.eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Evaluation(device)


