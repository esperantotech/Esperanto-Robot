
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

from robosuite.environments.manipulation.stack import Stack

def LLaVA_call_with_retry(model_path, prompts, images, temperature=0.0, seed=1000, max_tokens=300):
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

            completion = eval_model(args)
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
    testing_iterations = 1
    model_path = "liuhaotian/llava-v1.5-7b"

    # Load the prompt from the file
    prompt_text = load_prompt(prompt_file_path)
    
    for i in range(testing_iterations):
        time.sleep(1)
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
            cube_A_pos=cubeA_pos,
            cube_B_pos=cubeB_pos
        )

        obs = env.reset()
        env.close()       
        image = Image.fromarray(cv2.rotate(obs[f'birdview_image'], cv2.ROTATE_180))
        # save image as png 
        image.save(f'code_results/box/LLaVA/image.png')
        image = 'code_results/box/LLaVA/image.png'
        
        temperature = 0.05

        completion = LLaVA_call_with_retry(model_path, prompt_text, image, temperature, max_tokens=700)
            
        ## only save the python code part from the completion
        completion_code = completion.split('```python')[1].split('```')[0]
        with open(f'code_results/box/LLaVA/python_output_{i}.txt', 'w') as f:
            f.write(completion_code)
        print(f'Python code saved to python_output_{i}.txt') 

