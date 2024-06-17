
import os
import time
import numpy as np
from PIL import Image
import io
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.eval.run_llava import load_model_parameters
from llava.eval.run_llava import evaluate_without_loading_model

# from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.stack_multiple import Stack

import imageio


def LLaVA_call_with_retry(tokenizer, model, image_processor, context_len, prompts, images, temperature=0.0, seed=1000, max_tokens=300):
    success = False
    while not success:
        try:
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompts,
                "conv_mode": None,
                "image_file": images,
                "sep": ",",
                "temperature": temperature,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": max_tokens,
            })()

            completion = evaluate_without_loading_model(args, model, tokenizer, image_processor, context_len)
            success = True
        except Exception as e:
            print("LLaVA call issue, re-trying..." + str(e) + "\n")
            time.sleep(5)
    return completion

  
def load_prompt(prompt_file_path):
    with open(prompt_file_path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    prompt_file_path = 'prompt.txt'  # Path to your prompt file
    testing_iterations = 10
    model_path = "liuhaotian/llava-v1.5-7b"
    
    ## load model parameters
    tokenizer, model, image_processor, context_len = load_model_parameters(None, model_path, None)

    # Load the prompt from the file
    prompt_text = load_prompt(prompt_file_path)
    
    for i in range(testing_iterations):
        time.sleep(1)
        cubeA_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        cubeB_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
        while np.linalg.norm(cubeA_pos[:-1] - cubeB_pos[:-1]) < 0.04 or cubeA_pos[0] < 0 or cubeB_pos[0] < 0 or cubeA_pos[0] > cubeB_pos[0]:
            cubeB_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
            cubeA_pos = np.random.uniform(low=-0.08, high=0.08, size=3)
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
            # cube_B_pos=cubeB_pos,
        )

        obs = env.reset()
        for _ in range(50):
            obs, reward, done, info = env.step([-1, 0, 0, 0, 0, 0, 0, 0])
            
        image = obs["birdview_image"]
        painted_image = env.process_image(image, obs)
        env.close()       
        # save image as png 
        image_path = 'code_results/box/LLaVA/image.png'
        if testing_iterations > 1:
            if not os.path.exists(image_path):
                imageio.imwrite(image_path, painted_image)
        else:
            imageio.imwrite(image_path, painted_image)
            
        temperature = 0.05

        completion = LLaVA_call_with_retry(tokenizer, model, image_processor, context_len, prompt_text, image_path, temperature, max_tokens=2000)
            
        ## only save the python code part from the completion
        completion_code = completion.split('```python')[1].split('```')[0]
        with open(f'code_results/box/LLaVA/python_output_{i}.txt', 'w') as f:
            f.write(completion_code)
        with open(f'code_results/box/LLaVA/output_{i}.txt', 'w') as f:
            f.write(completion)
        print(f'Python code saved to python_output_{i}.txt') 

