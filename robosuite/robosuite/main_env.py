import robosuite
import numpy as np
import gym
import imageio
from robosuite.environments.manipulation.stack_multiple import Stack
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
    
def convert_to_pixel_coords(world_pos, table_region, table_size=0.8):
    table_x_min, table_y_min, table_x_max, table_y_max = table_region
    table_width = table_x_max - table_x_min
    table_height = table_y_max - table_y_min
    scale_factor_x = table_width / table_size
    scale_factor_y = table_height / table_size
    pixel_y = int((world_pos[0] + table_size / 2) * scale_factor_x + table_x_min)
    pixel_x = int((world_pos[1] + table_size / 2) * scale_factor_y + table_y_min)
    return (pixel_x, pixel_y)

def convert_to_pixel_size(world_size, table_region, table_size=0.8):
    table_x_min, table_y_min, table_x_max, table_y_max = table_region
    table_width = table_x_max - table_x_min
    table_height = table_y_max - table_y_min
    scale_factor_x = table_width / table_size
    scale_factor_y = table_height / table_size
    pixel_width = int(world_size[0] * scale_factor_x)
    pixel_height = int(world_size[1] * scale_factor_y)
    return (pixel_width, pixel_height)

def detect_table_edges(image):
    # Convert to grayscale and apply a binary threshold to detect the white table
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the table
    table_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box for the table
    x, y, w, h = cv2.boundingRect(table_contour)
    return (x, y, x + w, y + h)


def process_image(save_frame, obs):
    
    save_frame = save_frame[70:70+256, 126:126+256]
    ## mirror the image
    save_frame = cv2.flip(save_frame, 0)

    # Detect table edges in the image
    table_region = detect_table_edges(save_frame)
    
    object_info = {}
    
    for key, value in obs.items():
        if "cube" in key or "ball" in key or "bottle" in key:
            if "pos" in key:
                object_info[key] = value[:2]  # Use only x, y positions
            else:
                object_info[key] = value
                
    print (object_info)
    
 # Draw circles for each object
    for obj, pos in object_info.items():
        if "pos" in obj:
            pixel_pos = convert_to_pixel_coords(pos, table_region)
            if "cube" in obj:
                object_size = object_info[obj.replace("pos", "size")]
                pixel_size = convert_to_pixel_size(object_size, table_region)
                half_width = pixel_size[0]
                half_height = pixel_size[1]
                cv2.rectangle(save_frame, (pixel_pos[0] - half_width, pixel_pos[1] - half_height),
                              (pixel_pos[0] + half_width, pixel_pos[1] + half_height), (55, 55, 55), 1)
            elif "ball" in obj:
                object_size = object_info[obj.replace("pos", "size")]
                pixel_size = convert_to_pixel_size([object_size, object_size], table_region)
                cv2.circle(save_frame, pixel_pos, pixel_size[0], (55, 55, 55), 1)
            else:
                cv2.circle(save_frame, pixel_pos, 5, (255, 50, 0), -1)
                object_size = object_info[obj.replace("pos", "size")]
            pos = [round(p, 2) for p in pos]
            cv2.putText(save_frame, obj.split('_')[0], (pixel_pos[0], pixel_pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            cv2.putText(save_frame, f"pos: {pos}", (pixel_pos[0]-30, pixel_pos[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            cv2.putText(save_frame, f"{object_size}", (pixel_pos[0]-30, pixel_pos[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            
    # Draw table edges in the image
    cv2.rectangle(save_frame, (table_region[0], table_region[1]), (table_region[2], table_region[3]), (255, 0, 0), 2)
    
    return save_frame
    
    
def main():

    env = BoxStack(render=True)
 
    writer = imageio.get_writer('./render/dummyDemo_video.mp4', fps=env.env.control_freq)
    
    for _ in range(50):
        action = env.action_space.sample()
        action = np.array([-1, 0, 0, 0])
        obs, reward, done, info = env.step(action, writer)
    
    save_frame = obs[env.camera_name + "_image"]
    
    updated_image = process_image(save_frame, obs)

    imageio.imwrite('./render/dummyDemo_image.png', updated_image)
        
    env.close()
    writer.close()

if __name__ == '__main__':
    main()
