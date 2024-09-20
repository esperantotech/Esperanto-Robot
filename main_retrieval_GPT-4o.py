import openai
import os
import time
import argparse
import tiktoken


def count_tokens(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model("gpt-4")
    # encoding = tiktoken.get_encoding(model)
    num_tokens = 0
    for message in messages:
        # num_tokens += len(encoding.encode(message['role']))
        num_tokens += len(encoding.encode(message['content']))
    return num_tokens

def open_ai_call_with_retry(model_name, messages, temperature=0.0, seed=1000, max_tokens=300):
    reset_trigger_phrases = ["CHOICE", "NUM"]
    success = False
    while not success:
        try:
            # Initialize the OpenAI client
            openai.api_key = os.getenv("OPENAI_API_KEY")
            input_time = time.time()
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # seed=seed,
            )
            output_time = time.time()
            success = True
            for reset_trigger_phrase in reset_trigger_phrases:
                if reset_trigger_phrase in completion.choices[0].message['content']:
                    success = False
        except Exception as e:
            print("OpenAI API call issue, re-trying..." + str(e) + "\n")
            time.sleep(5)
    return completion, output_time - input_time

def load_prompt(prompt_file_path):
    with open(prompt_file_path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    model = "gpt-4o"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', type=str, help="file_name", required=True)
    parser.add_argument('--iterations', type=int, help="number of iterations", default=1)
    args = parser.parse_args()
    if args.objects == "2":
        prompt_file_path = 'prompts_folder/prompt_stack_two.txt'
    elif args.objects == "3":
        prompt_file_path = 'prompts_folder/prompt_stack_three.txt'
    elif args.objects == "3_free":
        prompt_file_path = 'prompts_folder/prompt_stack_three_free.txt'
        
    testing_iterations = args.iterations
    save_path = f'box/GPT-4o/{args.objects}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Load the prompt from the file
    prompt_text = load_prompt(prompt_file_path)
    inference_time = 0
    time_per_token = 0
    for i in range(testing_iterations):
        
        messages = [
            {
                "role": "user",
                "content": prompt_text
            },

        ]

        temperature = 0.3

        completion, check_time = open_ai_call_with_retry(model, messages, temperature, max_tokens=2000)
        inference_time += check_time
        print ('inference_time', inference_time, 'check_time', check_time)
        
        token_length = count_tokens(messages, model)
        print(f'Token length: {token_length}')
        print (f'Inference time per token: {check_time / token_length} seconds')
        time_per_token += (check_time / token_length)
        # if completion:
        #     print(completion.choices[0].message['content'])
            
        ## only save the python code part from the completion
        # completion_code = completion.choices[0].message['content'].split('```python')[1].split('```')[0]
        # with open(f'box/GPT-4o/{args.objects}/python_output_{i}.txt', 'w') as f:
        #     f.write(completion_code)
        # with open(f'box/GPT-4o/{args.objects}/output_{i}.txt', 'w') as f:
        #     f.write(completion.choices[0].message['content'])
        # print(f'Python code saved to python_output_{i}.txt') 

    print(f'Inference time: {inference_time} seconds')
    print(f'Average inference time: {inference_time / testing_iterations} seconds')
    print (f'Average Inference time per token: {time_per_token / testing_iterations} seconds')