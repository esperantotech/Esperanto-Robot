import openai
import os
import time
import base64
import numpy as np
from PIL import Image
import io
import imageio

# from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.stack_multiple import Stack

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

def encode_image_path(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def load_prompt(prompt_file_path):
    with open(prompt_file_path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    model = "gpt-4o"
    prompt_file_path = 'prompt.txt'  # Path to your prompt file
    testing_iterations = 10

    # Load the prompt from the file
    prompt_text = load_prompt(prompt_file_path)
    for i in range(testing_iterations):
        cubeA_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        cubeB_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        while np.linalg.norm(cubeA_pos - cubeB_pos) < 0.04:
            cubeB_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        print('Generated CubeA position:', cubeA_pos)
        print('Generated CubeB position:', cubeB_pos)
        
        env = Stack(
            robots="Kinova3",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="birdview",
            horizon=1000,
            control_freq=20,
            initialization_noise=None,
            # cube_A_pos=cubeA_pos,
            # cube_B_pos=cubeB_pos
        )

        obs = env.reset()
        for _ in range(50):
            obs, reward, done, info = env.step([-1, 0, 0, 0, 0, 0, 0, 0])
            
        image = obs["birdview_image"]
        painted_image = env.process_image(image, obs)
        env.close()        
        image_path = 'code_results/box/GPT/image.png'
        
        if testing_iterations > 1:
            if not os.path.exists(image_path):
                imageio.imwrite(image_path, painted_image)
        else:
            imageio.imwrite(image_path, painted_image)
        base64_image = encode_image_path(image_path)
        
        messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        
        messages.append(
                {
                    "role": "system",
                    "content": f"data:image/png;base64,{base64_image}"
                }
            )

        temperature = 0.05

        completion = open_ai_call_with_retry(model, messages, temperature, max_tokens=1000)
        if completion:
            print(completion.choices[0].message['content'])
            
        ## only save the python code part from the completion
        completion_code = completion.choices[0].message['content'].split('```python')[1].split('```')[0]
        with open(f'code_results/box/GPT/python_output_{i}.txt', 'w') as f:
            f.write(completion_code)
        with open(f'code_results/box/GPT/output_{i}.txt', 'w') as f:
            f.write(completion.choices[0].message['content'])
        print(f'Python code saved to python_output_{i}.txt') 

