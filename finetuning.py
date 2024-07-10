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
    # dataset = load_dataset('json', data_files={'train': '/home/xilun/ET_robot/updated_two_stack_dataset.json', 'validation': '/home/xilun/ET_robot/updated_three_stack_dataset.json'})
    dataset = load_dataset('json', data_files={'train': '/home/xilun/ET_robot/dataset/all_data.json', 'validation': '/home/xilun/ET_robot/dataset/172_dataset.json'})
    # dataset = load_dataset('json', data_files={'train': '/home/xilun/ET_robot/dataset/172_dataset.json', 'validation': '/home/xilun/ET_robot/dataset/172_dataset.json'})
    
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
    # target_modules = ['self_attn.qkv_proj', 'mlp.gate_up_proj']
    # target_modules = ['self_attn.qkv_proj']
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
    print (tokenized_datasets)
    input()
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
    
    
def FT_Wen(device):
    training_config = {
        "bf16": True,
        "do_eval": True,
        "learning_rate": 2e-05,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 40,
        "max_steps": -1,
        "eval_steps":20,
        "output_dir": "./phi-3-mini-LoRA",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 12,
        "per_device_train_batch_size": 12,
        "remove_unused_columns": True,
        "save_steps": 100,
        "save_total_limit": 5,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
        "report_to": "wandb",
        }

    peft_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ]
    }

    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)
    
    checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # tokenizer.model_max_length = 2125
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    
    model = get_peft_model(model, peft_conf)
    model = model.to(device)
    model.print_trainable_parameters()
    
    dataset_path = '/home/xilun/ET_robot/dataset/all_data.json'
    
    dataset = json.loads(open(dataset_path).read())
    
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):]
    
    # train_data = [
    #     {
    #     "text":"<s><|user|>\n" + row_dict['input'].replace('\n\n','\n').replace('<|user|>\n', '').replace('\n<|end|>', '<|end|>') + '\n<|assistant|>\n' + row_dict['output'].replace('\n\n\n','\n').replace('\n\n','\n').replace('<|assistant|>','') + '<|end|>'
    #     }
    #     for row_dict in train_dataset
    # ]
    # updated_train_data = [
    #     {
    #     "text":"<s><|user|>\n" + row_dict['input'].split('You are a robotic arm')[0].replace('\n\n','\n').replace('<|user|>\n', '') + '<|end|>' + \
    #         '\n<|system|>\nYou are a robotic arm' + row_dict['input'].split('You are a robotic arm')[1].replace('\n\n','\n').replace('<|user|>\n', '').replace('graspable point', 'position').replace('graspable_point', 'position').replace('\n<|end|>', '').replace('<|end|>', '') + '<|end|>' + \
    #         '\n<|assistant|>\n' + row_dict['output'].replace('graspable point', 'position').replace('graspable_point', 'position').replace('\n\n\n','\n').replace('\n\n','\n').replace('<|assistant|>','').replace('\n<|end|>', '').replace('<|end|>', '') + '<|end|>'
    #     }
    #     for row_dict in train_dataset
    # ]
    # with open('updated_train_data.json', 'w') as f:
    #     json.dump(updated_train_data, f)
    # with open('train_data.json', 'w') as f:
    #     json.dump(train_data, f)

    # val_data = [
    #     {
    #     "text":"<s><|user|>\n" + row_dict['input'].replace('\n\n','\n').replace('Starts<', '').replace('>Ends', '').replace('<|user|>\n', '').replace('\n<|end|>', '<|end|>') + '\n<|assistant|>\n' + row_dict['output'].replace('Starts<', '').replace('>Ends', '').replace('\n\n\n','\n').replace('\n\n','\n').replace('<|assistant|>','') + '<|end|>'
    #     }
    #     for row_dict in val_dataset
    # ]
    # updated_val_data = [
    #     {
    #     "text":"<s><|user|>\n" + row_dict['input'].split('You are a robotic arm')[0].replace('\n\n','\n').replace('<|user|>\n', '') + '<|end|>' + \
    #         '\n<|system|>\nYou are a robotic arm' + row_dict['input'].split('You are a robotic arm')[1].replace('\n\n','\n').replace('<|user|>\n', '').replace('graspable point', 'position').replace('graspable_point', 'position').replace('\n<|end|>', '').replace('<|end|>', '') + '<|end|>' + \
    #         '\n<|assistant|>\n' + row_dict['output'].replace('graspable point', 'position').replace('graspable_point', 'position').replace('\n\n\n','\n').replace('\n\n','\n').replace('<|assistant|>','').replace('\n<|end|>', '').replace('<|end|>', '') + '<|end|>'
    #     }
    #     for row_dict in val_dataset
    # ]
    # with open('val_data.json', 'w') as f:
    #     json.dump(val_data, f)
    # with open('updated_val_data.json', 'w') as f:
    #     json.dump(updated_val_data, f)
        
    train_data = load_dataset('json', data_files='updated_train_data.json')
    val_data = load_dataset('json', data_files='updated_val_data.json')
    print (train_data)
    print (val_data)
    
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=train_data['train'],
        eval_dataset=val_data['train'],
        max_seq_length=1500,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    trainer.train()
    trainer.save_model(train_conf.output_dir)
    
def Evaluation(device):
    output_dir = "./phi-3-mini-LoRA/checkpoint-1320"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True, add_eos_token=True, use_fast=True)


    # Load the LoRA adapter configuration
    # peft_config = PeftConfig.from_pretrained(output_dir)

    # Apply the LoRA adapter to the base model
    model = PeftModel.from_pretrained(model, output_dir, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model = model.to(device)
    
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    def generate_response(input_text):
        print ("generating response...")
        # prompt = pipe.tokenizer.apply_chat_template([{"role": "system", "content": "you are a helpful robot planner"},{"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
        output = pipe(input_text, max_new_tokens=1024, do_sample=True, temperature=0.2, num_beams=1, top_k=50, top_p=0.95)
        return output[0]['generated_text'].strip()

    eval_times = 10
    ## loading evaluation dataset
    eval_dataset_path = 'updated_val_data.json'
    eval_dataset = json.loads(open(eval_dataset_path).read())
    
    for i in range (eval_times):
        prompt = eval_dataset[i]['text']
        ## get the text prompt until the text "<|end|>" appears
        prompt = prompt.split("<|assistant|>")[0] + "<|assistant|>"
        print (prompt)
        # Evaluate the model
        result = generate_response(prompt)
        ## only save the python code part from the completion
        with open(f'box/Phi-3-FT/output_{i}.txt', 'w') as f:
            f.write(result)
        # try: 
        #     completion_code = result.split('**\npython')[1].split('<|end|>')[0]
        # except:
        #     completion_code = result.split('```python')[1].split('```')[0]
        # with open(f'box/Phi-3-FT/python_output_{i}.txt', 'w') as f:
        #     f.write(completion_code)
        print(f'Python code saved to python_output_{i}.txt') 
        
        # print (result)
    

if __name__ == "__main__":
    ## initiliaze arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--FT", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.FT:
        single_GPU_FT()
    elif args.training:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FT_Wen(device)
    
    if args.eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Evaluation(device)


