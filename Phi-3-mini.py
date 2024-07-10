import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os 
import argparse
from peft import PeftModel


def raw_inference(args):


    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    
    if args.objects == "2":
        prompt_path = 'prompts_folder/prompt_stack_two.txt'
    elif args.objects == "3":
        prompt_path = 'prompts_folder/prompt_stack_three.txt'
    elif args.objects == "3_free":
        prompt_path = 'prompts_folder/prompt_stack_three_free.txt'
    ## load prompt
    with open(prompt_path, 'r') as f:
        prompt = f.read()
        
    testing_iterations = args.iterations
    save_path = f'box/Phi-3/{args.objects}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    system_prompt_path = 'prompts_folder/system_plan.txt'
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    code_prompt_path = 'prompts_folder/system_code.txt'
    with open(code_prompt_path, 'r') as f:
        code_prompt = f.read()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Could you generate a plan for me to" + prompt},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 2000,
        "return_full_text": False,
        "temperature": 0.1,
        "do_sample": True,
    }


    for i in range(testing_iterations):

        plan_output = pipe(messages, **generation_args)
        plan_result = plan_output[0]['generated_text']
        print(plan_result)
        
        print ("############################################################")
        ## coder model 
        coder_output = pipe([{"role": "system", "content": code_prompt}, {"role": "user", "content": plan_result}], **generation_args)
        coder_result = coder_output[0]['generated_text']
        print(coder_result)
        
        ## only save the python code part from the completion
        # try: 
        #     completion_code = result.split('**\npython')[1].split('<|end|>')[0]
        # except:
        #     completion_code = result.split('```python')[1].split('```')[0]
        # with open(f'box/Phi-3/{args.objects}/python_output_{i}.txt', 'w') as f:
        #     f.write(completion_code)
        # with open(f'box/Phi-3/{args.objects}/output_{i}.txt', 'w') as f:
        #     f.write(output[0]['generated_text'])
        print(f'Python code saved to python_output_{i}.txt') 


def finetune_inference(args):
    output_dir = "./phi-3-mini-LoRA/checkpoint-1040"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply the LoRA adapter to the base model
    # model = PeftModel.from_pretrained(model, output_dir, torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()
    model = model.to(device)
    
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device, return_full_text=False)
    def generate_response(input_text, system_prompt, code_prompt):
        print ("generating response...")
        plan_prompt = pipe.tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
        plan_output = pipe(plan_prompt, max_new_tokens=1024, do_sample=True, temperature=0.2, num_beams=1, top_k=50, top_p=0.95)
        plan_text = plan_output[0]['generated_text'].strip()
        ## based on the plan generate code 
        print (plan_text)
        # input()
        # new_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
        code_prompt = pipe.tokenizer.apply_chat_template([{"role": "system", "content": code_prompt}, {"role": "user", "content": plan_text}], tokenize=False, add_generation_prompt=True)
        code_output = pipe(code_prompt, max_new_tokens=1024, do_sample=True, temperature=0.2, num_beams=1, top_k=50, top_p=0.95)
        code_text = code_output[0]['generated_text'].strip()
        return code_text

    eval_times = 1
    
    if args.objects == "2":
        prompt_path = 'prompts_folder/prompt_stack_two.txt'
    elif args.objects == "3":
        prompt_path = 'prompts_folder/prompt_stack_three.txt'
    elif args.objects == "3_free":
        prompt_path = 'prompts_folder/prompt_stack_three_free.txt'
    ## load prompt
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    system_prompt_path = 'prompts_folder/system_plan.txt'
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    code_prompt_path = 'prompts_folder/system_code.txt'
    with open(code_prompt_path, 'r') as f:
        code_prompt = f.read()
    

    for i in range (eval_times):
        result = generate_response(prompt, system_prompt, code_prompt)
        # print(f'Python code saved to python_output_{i}.txt') 
        
        print (result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=str, default="2", help="Number of objects in the scene")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    args = parser.parse_args()
    
    finetune_inference(args)
    # raw_inference(args)
