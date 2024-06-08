import robosuite
import numpy as np
import gym 
import imageio
from robosuite.environments.manipulation.stack import Stack
from robosuite import load_controller_config
import time
import cv2
    
class BoxStack(gym.Env):
    def __init__(self, render=False):
        self.render = render
        
        config = load_controller_config(
                     custom_fpath="controllers/config/osc_position_custom.json")
        
        self.camera_name = "birdview"
        self.env = Stack(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            camera_names = self.camera_name,
            horizon=1000,
            control_freq=20,
            initialization_noise=None,
            
        )
        
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, 0.9]),
                    high=np.array([1, 1, 1, 1]),
                    dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self._seed = None
        self.context = {}
        
    def step(self, action, writer=None):
        done = False
        front_view = [] 
        success = 0
        obs, reward, done, info = self.env.step(action)
        for k, v in obs.items():
            if "image" not in k:
                print(k, v)
        if self.render:
            front_view.append(obs[self.camera_name + "_image"].copy())
            if writer is not None:
                writer.append_data(cv2.rotate(obs[self.camera_name + "_image"], cv2.ROTATE_180))
            # self.env.render()
            time.sleep(0.01)
        if self.env._check_success():
            success = 1

        # obs = self.env.get_obs()
        
        info = {'reward': reward, 
                'success': success,
                'render_image': front_view}
        
        
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.seed(0)

    def get_goal(self):
        # get the goal from the env
        return self.env.target_pos
    
def main():

    env = BoxStack(render=True)
 
    writer = imageio.get_writer('./render/dummyDemo_video.mp4', fps=env.env.control_freq)
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action, writer)
    
    save_frame = obs[env.camera_name + "_image"]
    ## cut the middle part of the image as 256 x 256 from 512 x 512
    save_frame = save_frame[70:70+256, 126:126+256]
    imageio.imwrite('./render/dummyDemo_image.png', save_frame)
        
    env.close()
    writer.close()

if __name__ == '__main__':
    main()