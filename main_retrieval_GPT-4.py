import openai
import os
import time
import argparse

from robosuite.environments.manipulation.stack import Stack

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
                max_tokens=max_tokens,
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

def load_prompt(prompt_file_path):
    with open(prompt_file_path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    model = "gpt-4"
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', type=str, help="file_name", required=True)
    parser.add_argument('--iterations', type=int, help="number of iterations", default=1)
    args = parser.parse_args()
    if args.objects == "2":
        prompt_file_path = 'prompt_stack_two.txt'
    elif args.objects == "3":
        prompt_file_path = 'prompt_stack_three.txt'
    elif args.objects == "3_free":
        prompt_file_path = 'prompt_stack_three_free.txt'
    testing_iterations = args.iterations
    save_path = f'box/GPT-4/{args.objects}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the prompt from the file
    prompt_text = load_prompt(prompt_file_path)
    
    for i in range(testing_iterations):
        # cubeA_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        # cubeB_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        # while np.linalg.norm(cubeA_pos - cubeB_pos) < 0.04:
        #     cubeB_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        # print('Generated CubeA position:', cubeA_pos)
        # print('Generated CubeB position:', cubeB_pos)
        
        # env = Stack(
        #     robots="Kinova3",
        #     has_renderer=False,
        #     has_offscreen_renderer=True,
        #     use_camera_obs=True,
        #     camera_names="frontview",
        #     horizon=1000,
        #     control_freq=20,
        #     initialization_noise=None,
        #     cube_A_pos=cubeA_pos,
        #     cube_B_pos=cubeB_pos
        # )

        # obs = env.reset()
        # env.close()
        
        messages = [
            {
                "role": "user",
                "content": prompt_text
            },

        ]

        temperature = 0.05

        completion = open_ai_call_with_retry(model, messages, temperature, max_tokens=2000)
        if completion:
            print(completion.choices[0].message['content'])
            
        ## only save the python code part from the completion
        completion_code = completion.choices[0].message['content'].split('```python')[1].split('```')[0]
        with open(f'box/GPT-4/{args.objects}/python_output_{i}.txt', 'w') as f:
            f.write(completion_code)
        with open(f'box/GPT-4/{args.objects}/output_{i}.txt', 'w') as f:
            f.write(completion.choices[0].message['content'])
        print(f'Python code saved to python_output_{i}.txt') 

