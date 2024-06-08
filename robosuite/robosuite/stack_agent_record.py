from typing import Optional
import cv2
import imageio
import numpy as np
import copy
import termcolor
import re
import time
import os
import argparse

from robosuite.controllers import load_controller_config

# from robosuite.utils.input_utils import *
# from robosuite.environments.manipulation.pick_place_ycb import PickPlace
# from robosuite.environments.manipulation.magnetic import Magnetic
# from robosuite.environments.manipulation.pull_paper import pull_paper
from robosuite.environments.manipulation.stack import Stack


class ArmClient:

    def __init__(self, vis=False):
        self.task = "Magnetic"
        self.vis = vis
        self.recording = True
        self.print_vals = False
        self.create_env()

    def reset(self):
        # TODO: need to reset the whole environment, instead of the super object
        pass

    def execute_code(self, code):
        if "__" in code:
            raise ValueError("Invalid code contains access to private members.")

        print("ABOUT TO EXECUTE\n", code)

        empty_fn = lambda *args, **kwargs: None
        exec(
            code,
            self.function_mapping(),
            {"exec": empty_fn, "eval": empty_fn},
        )

    def get_workspace_range(self):
        # x_min, y_min, z_min, x_max, y_max, z_max
        return (
            self.workspace_min[0],
            self.workspace_min[1],
            self.workspace_min[2],
            self.workspace_max[0],
            self.workspace_max[1],
            self.workspace_max[2],
        )

    def function_mapping(self):
        return {
            "move_to_position": self.move_to_position,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_center": self.get_center,
            "get_graspable_point": self.get_graspable_point,
            "get_size": self.get_size,
            "workspace_clip": self.workspace_clip,
            "get_workspace_range": self.get_workspace_range,
        }

    def create_env(self):
        log_video = False

        config = load_controller_config(default_controller="OSC_POSE")

        self.env = Stack(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            controller_configs=config,
            camera_names="sideview",
            control_freq=20,
        )
        self.obs = self.env.reset()
        if self.vis:
            # 0: front, 1: top, 2: zoom in front far, 3: side robot left, 4: zoom in front close, 5: inhand,
            self.env.viewer.set_camera(camera_id=3)
        if self.recording:
            self.writer = imageio.get_writer(
                "./dummyDemo_video.mp4", fps=self.env.control_freq
            )

        self.robot_ee_pos = copy.deepcopy(
            self.obs["robot0_eef_pos"]
        )  # [-0.72199001 -0.04194331  0.83100355]
        self.robot_ee_quat = copy.deepcopy(
            self.obs["robot0_eef_quat"]
        )  # [-0.99981456  0.00618901  0.016277   -0.00822216]
        self.additional_scale = 0.2
        self.world_to_base = np.array([-0.9, 0.0, 0.8])
        self.filter_obj_list = ["cubeA", "cubeB"]
        self.workspace_min = [-0.5, -0.25, 0.02]
        self.workspace_max = [0.50, 0.25, 0.9]

    def parse_scene_info(self):
        obj_list = "["
        for obj in self.env.model.mujoco_objects:
            obj_list += obj.name + ", "
        obj_list += "]"

        filtered_obj_list = "["
        for i, obj in enumerate(self.filter_obj_list):
            filtered_obj_list += obj
            if i != len(self.filter_obj_list) - 1:
                filtered_obj_list += ", "
        filtered_obj_list += "]"

        obj_info = ""
        for obj in self.env.model.mujoco_objects:
            if obj.name in self.filter_obj_list:
                pos_key = obj.name + "_pos"
                print("obj name", obj.name)
                if obj.name == "cubeB" or obj.name == "cubeA":
                    obj_size = np.around(self.get_size(obj.name), 3)
                    obj_size_string = self.change_array_to_string(obj_size)
                    obj_position = self.get_center(obj.name)
                    obj_pos_string = self.change_array_to_string(obj_position)
                    obj_graspable_point = self.get_graspable_point(obj.name)
                    obj_graspable_point_string = self.change_array_to_string(obj_graspable_point)
                    obj_info += f"- [{obj.name}]: <center>: {obj_pos_string}; <size>: {obj_size_string}; <graspable point>: {obj_graspable_point_string}. \n"
                else:
                    raise ValueError("Invalid object name")

        # robot_workspace_info="The robot arm's workspace is [0.0-0.30, -0.25-0.25, 0.0-0.35] along the x, y and z axis."
        robot_workspace_info = f"The robot arm's workspace's x-range is {self.workspace_min[0]}m to {self.workspace_max[0]}m, y-range is {self.workspace_min[1]}m to {self.workspace_max[1]}m, z-range is {self.workspace_min[2]}m to {self.workspace_max[2]}m."
        common_info = f"You are in a 3D world. You are a robot arm mounted on a table. You can control the end effector's position and gripper. Object list = ['CubeA', 'CubeB']. \nNumerical scene information:\nThe position is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other.\nThe base of the robot arm is at [0.0, 0.0, 0.0].\n{obj_info}\nConstraints you must follow:\n- {robot_workspace_info}"
        print(termcolor.colored(common_info + "\n", "cyan", attrs=["bold"]))
        # \n - One magnetic_cube will attract and attach to another one if the distance between the centers of them is less or equal to 0.02m.

        return common_info
    
    def change_array_to_string(self, array):
        list_array = np.around(array, 3).astype(str).tolist()
        string_array = "["
        for i, val in enumerate(list_array):
            if i != 0:
                string_array += " "
            string_array += val
            if i != len(list_array) - 1:
                string_array += ","
        string_array += "]"
        return string_array
    
    def create_env_demo(self):
        log_video = True
        vis = False

        config = load_controller_config(default_controller="OSC_POSE")

        self.env = Stack(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            controller_configs=config,
            camera_names="sideview",
            control_freq=20,
        )
        self.env.reset()
        if vis:
            self.env.viewer.set_camera(camera_id=0)

        # Get action limits
        low, high = self.env.action_spec
        writer = None
        if log_video:
            writer = imageio.get_writer(
                "./dummyDemo_video.mp4", fps=self.env.control_freq
            )
        # do visualization
        self.env.reset()
        for i in range(100):
            # action = np.random.uniform(low, high)
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            self.obs, reward, done, info = self.env.step(action)
            if self.print_vals:
                print("action", action)
            if writer is not None:
                writer.append_data(
                    cv2.rotate(self.obs["sideview_image"], cv2.ROTATE_180)
                )
            if vis:
                self.env.render()

        if writer is not None:
            writer.close()

        self.env.close()

    def get_gripper_openness(self):
        gripper_state = self.obs["robot0_gripper_qpos"][0]
        if self.print_vals:
            print("gripper state", gripper_state)
        clip_range = np.array([0.0, 0.5])
        gripper_openness = np.clip(gripper_state, clip_range[0], clip_range[1])
        if gripper_openness < 0.35:
            return "open"
        else:
            return "closed"

    def workspace_clip(self, target_position_robot):
        clip_value = np.clip(
            target_position_robot, self.workspace_min, self.workspace_max
        )
        return clip_value

    def move_to_position(self, target_position, target_rotation=None):
        print("target position in move to position", target_position)
        target_position = self.workspace_clip(target_position)

        # target_position = self.transform_to_world(target_position)
        # print("target position in world frame", target_position)

        target_position = np.array(target_position)
        if self.print_vals:
            print("target_position", target_position, target_position.shape)
            print("offset", self.env.table_offset)

        # z axis offset for not colliding with the table
        if target_position[2] <= 0.82:
            target_position[2] = 0.82

        if len(target_position) == 3:
            target_position = np.concatenate(
                (target_position, self.obs["robot0_eef_quat"])
            )

        current_robot_pos = self.obs["robot0_eef_pos"]
        current_robot_rot = self.obs["robot0_eef_quat"]
        target_robot_pos = target_position[:3]
        target_robot_rot = target_position[3:]
        old_pos = copy.deepcopy(current_robot_pos)
        if self.print_vals:
            print("current pos", current_robot_pos, "rot:", current_robot_rot)
            print("target pos", target_robot_pos, "rot:", target_robot_rot)

        # if input is provided as target position in the global frame
        # the action space scale is [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        pos_scale = 0.05
        pos_max_steps = np.int32(
            np.floor(
                np.max(
                    np.abs(
                        (target_robot_pos - current_robot_pos)
                        / (pos_scale * self.additional_scale)
                    )
                )
            )
        )
        rot_scale = 0.5
        rot_max_steps = np.int32(
            np.floor(
                np.max(
                    np.abs(
                        (target_robot_rot - current_robot_rot)
                        / (rot_scale * self.additional_scale)
                    )
                )
            )
        )
        max_steps = np.max((pos_max_steps, rot_max_steps))

        for i in range(max_steps * 10):
            # prepare the signed action
            current_robot_pos = self.obs["robot0_eef_pos"]
            current_robot_rot = self.obs["robot0_eef_quat"]
            target_robot_pos = target_position[:3]
            # target_robot_rot = target_position[3:]
            target_robot_rot = self.obs["robot0_eef_quat"]
            act_pos = np.sign(target_robot_pos - current_robot_pos)
            act_rot = np.sign(target_robot_rot - current_robot_rot)
            action = np.concatenate((act_pos, act_rot)) * 1.0 * self.additional_scale

            # smooth the action
            if self.print_vals:
                print("action", action, np.abs(target_robot_pos - current_robot_pos))
            action[:3][np.abs(target_robot_pos - current_robot_pos) < 0.001] = 0.0
            if self.print_vals:
                print("after smooth action", action)

            self.obs, reward, done, info = self.env.step(action)
            if self.print_vals:
                print("i", i, "out of", max_steps)
                print("action", action)
                print("robots", self.obs["robot0_eef_pos"])
            if self.vis:
                self.env.render()
            if self.writer is not None:
                self.writer.append_data(cv2.flip(self.obs["sideview_image"], 0))
        if self.print_vals:
            print("finish move to position")
            print(
                "current pos", self.obs["robot0_eef_pos"], self.obs["robot0_eef_quat"]
            )

            print("target poisition", target_position[:3])
            print("robot current position", self.obs["robot0_eef_pos"])

        # print('====================success check', self.success_check())

    def open_gripper(self):
        for i in range(20):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            self.obs, reward, done, info = self.env.step(action)
            if self.vis:
                self.env.render()
            if self.writer is not None:
                self.writer.append_data(cv2.flip(self.obs["sideview_image"], 0))
        if self.print_vals:
            print("finish open gripper")
            print("robot position", self.obs["robot0_eef_pos"])
            print("robot quaternion", self.obs["robot0_eef_quat"])

    def close_gripper(self):
        for i in range(100):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            self.obs, reward, done, info = self.env.step(action)
            # print('obs keys', self.obs.keys())
            if self.vis:
                self.env.render()
            if self.writer is not None:
                self.writer.append_data(cv2.flip(self.obs["sideview_image"], 0))
        if self.print_vals:
            print("finish close gripper")

    def get_graspable_point(self, obj_name):
        obj_name = obj_name.lower()
        if obj_name == "end_effector" or obj_name == "gripper":
            obj_name = "robot0_eef"
        if (
            obj_name == "cubeA"
            or obj_name == "CubeA"
            or obj_name == "cubea"
        ):
            obj_name = "cubeA"
        if (
            obj_name == "cubeB"
            or obj_name == "CubeB"
            or obj_name == "cubeb"
        ):
            obj_name = "cubeB"
        obs_key = obj_name + "_pos"
        pos = copy.deepcopy(self.obs[obs_key])
        pos_to_base = self.transform_to_robot_base(pos)
        val = np.around(pos_to_base, 3)
        print("object", obj_name, "position", val)
        return val

    def get_center(self, obj_name):
        obj_name = obj_name.lower()
        if obj_name == "end_effector" or obj_name == "gripper":
            obj_name = "robot0_eef"
        if (
            obj_name == "cubeA"
            or obj_name == "CubeA"
            or obj_name == "cubea"
        ):
            obj_name = "cubeA"
        if (
            obj_name == "cubeB"
            or obj_name == "CubeB"
            or obj_name == "cubeb"
        ):
            obj_name = "cubeB"
        obs_key = obj_name + "_pos"
        pos = copy.deepcopy(self.obs[obs_key])
        pos_to_base = self.transform_to_robot_base(pos)
        val = np.around(pos_to_base, 3)
        orientation = None
        print("object", obj_name, "position", val)
        return val

    def get_size(self, obj_name):
        obj_name = obj_name.lower()
        if (
            obj_name == "cubeA"
            or obj_name == "CubeA"
            or obj_name == "cubea"
        ):
            obj_name = "cubeA"
        if (
            obj_name == "cubeB"
            or obj_name == "CubeB"
            or obj_name == "cubeb"
        ):
            obj_name = "cubeB"
            
        for obj in self.env.model.mujoco_objects:
            if obj.name == obj_name:
                obj_size = np.around(np.array(obj.size) * 2, 3)
                print("object name", obj_name, obj_size)
        return obj_size

    def transform_to_world(self, position):
        # position is actually from base to some desired position
        # base_to_desired + world_to_based = world_to_desired
        return position + self.world_to_base

    def transform_to_robot_base(self, position):
        # position is actually from world to some desired position
        # return position - self.world_to_base
        return position

    def success_check(self):
        success_inline = 0.0
        threshold = 0.02
        # get can position
        cubeA_pos = self.transform_to_robot_base(self.obs["cubeA_pos"])
        cubeB_pos = self.transform_to_robot_base(self.obs["cubeB_pos"])
        ## check if cubeA and cubeB are inline
        dist_12_x = np.abs(cubeA_pos[0] - cubeB_pos[0])
        dist_12_y = np.abs(cubeA_pos[1] - cubeB_pos[1])

        if dist_12_y < threshold and dist_12_x < threshold:
            success_inline = 1.0

        print("================================")
        print("dist_12_x", dist_12_x)
        print ("dist_12_y", dist_12_y)
        print("success_inline_12", success_inline)
        
        success_dict = {
            "success_inline": success_inline,
        }
        return success_dict


def reevaluate_codes():
    save_path = (
        os.environ["folder_path"]
        + "/results/box/testing"
    )

    for i in range(20):  # for all the saved files

        index = int(309 + i)
        file_path = os.path.join(save_path, "coder_result", f"gpt-4_trial_{index}.txt")
        with open(file_path, "r") as f:
            code = f.read()
        code = code.replace("python", "# python")

        for _ in range(5):
            agent = ArmClient()
            agent.execute_code(code)
            success_dict = agent.success_check()
            agent.env.close()
            if success_dict["final_success"] > 0.0:
                print("success_dict", success_dict)
                break
        # save the new success dict to the path
        success_path = os.path.join(
            save_path, "success_dict", f"gpt-4_trial_{index}.txt"
        )
        with open(success_path, "w") as f:
            f.write(str(success_dict))
            print("======================== success_path updated", success_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help="file_name", required=True)
    args = parser.parse_args()

    agent = ArmClient(vis=False)
    save_path = (
        os.getcwd()
        + f"/code_results/box/"
    )
    # file_path = "./test_script.txt"
    file_path = os.path.join(save_path, args.file_name)
    print('file path')
    print(file_path)
    with open(file_path, "r") as f:
        code = f.read()
    print("code", code)
    code = code.replace("python", "# python")
    agent.execute_code(code)
    print("agent.success_check()", agent.success_check())
    agent.env.close()

    # reevaluate_codes()