import openai
import os
import time
import json
import tqdm

################# This file needs OPENAI_API_KEY to be set in the environment variables #################

def open_ai_call_with_retry(model_name, messages, temperature=0.0, seed=1000, max_tokens=300):
    reset_trigger_phrases = ["CHOICE", "NUM"]
    success = False
    while not success:
        try:
            # Initialize the OpenAI client
            openai.api_key = os.getenv("OPENAI_API_KEY")
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                # seed=seed,
            )
            success = True
            for reset_trigger_phrase in reset_trigger_phrases:
                if reset_trigger_phrase in completion.choices[0].message['content']:
                    success = False
        except Exception as e:
            print("OpenAI API call issue, re-trying..." + str(e) + "\n")
            time.sleep(5)
    return completion

def data_generation():

    model = "gpt-4o"

    task_prompt_path = "data_prep_stack_three.txt" # Path to the task prompt file, if two objects, use "data_prep_stack_two.txt"
    input_prompt = open(task_prompt_path, 'r').read()

    human_prompt = 'Good Job! This is the dataset for my phi-3-mini model fientuning tasks. Could you generate 3 more datapoints for me following the content format and content? Focus on planning and code generation abilities. Please give me the results that are in the same format as the input data. \
                    The objects should be sampled from cubeA, cubeB, cubeC, CubeD, ballA. Please generate stacking or related tasks and solutions with any two or three objects.\
                    More than successful cases, you could also generate some failure cases due to geometry such as you cannot stack box on a ball. \
                    Make sure generate your code with comments. \
                    Please make sure your generated output is correct, dont need extra explaination, just generate the json part is enough. \
                    If it helps, you can try to generate three objects stacking or interations. Thank you!'

    input_prompt = input_prompt + human_prompt

    messages = [
        {"role": "user", "content": input_prompt},
    ]

    all_output = []
    for i in tqdm.tqdm(range(50)):
        output = open_ai_call_with_retry(model, messages, temperature=0.8).choices[0].message['content']

        # Save the combined data into a JSON file
        with open(f"{i}_dataset.json", 'w') as f:
            json.dump(output, f, indent=4)

    # Print the generated questions and answers
    # for i in range(len(all_output)):
    #     print("question:", all_output[i])
    #     print("answer:", all_message[i])
    #     print("-------------------")


def data_combination():

    # Get the current working directory
    path = os.getcwd()

    # List all files in the directory
    files = os.listdir(path)

    # Filter out only the JSON files
    json_files = [file for file in files if file.endswith('dataset.json')]
    ## create an empty json file    
    all_data = []
    # Loop through each JSON file and read its content
    for index, file in enumerate(json_files):
        # print (index, file)
        with open(json_files[index], 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = data[data.find('```json'):] 
                data = data.replace('```json', '').replace('```', '')
                data = json.loads(data)
        if file == "172_dataset.json":
            print ("skipping",file,"for evaluation")
        else:
            for item in data:
                item['input'] = item['input'].replace("Starts<", "")
                item['input'] = item['input'].replace(">Ends", "")
                item['output'] = item['output'].replace("Starts<", "")
                item['output'] = item['output'].replace(">Ends", "")
                all_data.append(item)
    # Print the combined data
    print(len(all_data))

    ## if you manually add some data, you can load the data from manual_dataset
        
    # with open('manual_dataset1.json', 'r') as f:
    #     manual_data = json.load(f)
    # all_data.extend(manual_data)
    # print(len(all_data))
    ##########

    with open('all_data.json', 'w') as f:
        json.dump(all_data, f, indent=4)
        
        
        
if __name__ == "__main__":
    ## first generate data using GPT-4o
    data_generation()
    ## Then combine the generated data all together
    data_combination()
