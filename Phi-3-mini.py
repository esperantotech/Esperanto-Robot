import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os 
import argparse

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
parser = argparse.ArgumentParser()
parser.add_argument('--objects', type=str, help="file_name", required=True)
parser.add_argument('--iterations', type=int, help="number of iterations", default=1)
args = parser.parse_args()
if args.objects == "2":
    prompt_path = 'prompt_stack_two.txt'
elif args.objects == "3":
    prompt_path = 'prompt_stack_three.txt'
elif args.objects == "3_free":
    prompt_path = 'prompt_stack_three_free.txt'
## load prompt
with open(prompt_path, 'r') as f:
    prompt = f.read()
    
testing_iterations = args.iterations
save_path = f'box/Phi-3/{args.objects}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": prompt},
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

    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])
    
    ## only save the python code part from the completion
    completion_code = output[0]['generated_text'].split('```python')[1].split('```')[0]
    with open(f'box/Phi-3/{args.objects}/python_output_{i}.txt', 'w') as f:
        f.write(completion_code)
    with open(f'box/Phi-3/{args.objects}/output_{i}.txt', 'w') as f:
        f.write(output[0]['generated_text'])
    print(f'Python code saved to python_output_{i}.txt') 
