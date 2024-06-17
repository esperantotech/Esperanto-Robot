import robosuite
import numpy as np
import gym
import imageio
# from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.push_roll_lift import Push_Roll_Lift
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite import load_controller_config
import time
import cv2

class Wipecube(gym.Env):
    def __init__(self, render=False):
        self.render = render
        
        config = load_controller_config(
            custom_fpath="controllers/config/osc_position_custom.json")
        
        self.camera_name = "frontview"
        self.env = PickPlace(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            camera_names=self.camera_name,
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
        if self.render:
            front_view.append(obs[self.camera_name + "_image"].copy())
            if writer is not None:
                writer.append_data(cv2.rotate(obs[self.camera_name + "_image"], cv2.ROTATE_180))
            time.sleep(0.01)
        if self.env._check_success():
            success = 1

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
        return self.env.target_pos

def main():
    
    env = Wipecube(render=True)
    
    writer = imageio.get_writer('./render/dummyDemo_video.mp4', fps=env.env.control_freq)
    
    env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs,reward,done,info = env.step(action, writer)
    print (obs)
    
    env.close()
if __name__ == "__main__":
    main()