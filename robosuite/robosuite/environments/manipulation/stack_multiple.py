from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BallObject
from robosuite.models.objects import BottleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
import cv2 


class Stack(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=4000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        cube_A_pos = [-0.2, 0.2, 0.1],
        cube_B_pos = [-0.2, -0.2, 0.1],
        cube_C_pos = [0.15, 0.2, 0.1],
        cube_D_pos = [0.15, 0, 0.1],
        bottle_A_pos = [0, -0.15, 0.1],
        ball_A_pos = [0, 0.3, 0.1],
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        
        ## set cube position 
        self.cube_A_pos = cube_A_pos
        self.cube_B_pos = cube_B_pos
        self.cube_C_pos = cube_C_pos
        self.cube_D_pos = cube_D_pos
        self.bottle_A_pos = bottle_A_pos
        ## TODO: get bottle A size 
        self.bottle_A_size = [0.02, 0.02, 0.1]
        self.ball_A_pos = ball_A_pos

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to the center of the cube
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if grasping_cubeA:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
            r_stack = 2.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size = [0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size = [0.03, 0.03, 0.03],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size = [0.02, 0.02, 0.02],
            rgba=[0, 0, 1, 1],
        )
        self.cubeD = BoxObject(
            name="cubeD",
            size = [0.02, 0.02, 0.02],
            rgba=[1, 1, 0, 1],
        )
        
        self.bottleA = BottleObject(
            name="bottleA",
        )
        
        self.ballA = BallObject(
            name="ballA",
            size=[0.02],
            rgba=[1, 0, 0, 1],
        )
        # cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(cubes)
        # else:
        
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(
        name="ObjectSampler")
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="ObjectSamplerA",
            mujoco_objects=self.cubeA,
            x_range=[self.cube_A_pos[0], self.cube_A_pos[0]],
            y_range=[self.cube_A_pos[1], self.cube_A_pos[1]],
            rotation=0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="ObjectSamplerB",
            mujoco_objects=self.cubeB,
            x_range=[self.cube_B_pos[0], self.cube_B_pos[0]],
            y_range=[self.cube_B_pos[1], self.cube_B_pos[1]],
            rotation=0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="ObjectSamplerC",
            mujoco_objects=self.cubeC,
            x_range=[self.cube_C_pos[0], self.cube_C_pos[0]],
            y_range=[self.cube_C_pos[1], self.cube_C_pos[1]],
            rotation=0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))
        
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="ObjectSamplerD",
            mujoco_objects=self.cubeD,
            x_range=[self.cube_D_pos[0], self.cube_D_pos[0]],
            y_range=[self.cube_D_pos[1], self.cube_D_pos[1]],
            rotation=0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))
        
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="ObjectSamplerE",
            mujoco_objects=self.bottleA,
            x_range=[self.bottle_A_pos[0], self.bottle_A_pos[0]],
            y_range=[self.bottle_A_pos[1], self.bottle_A_pos[1]],
            rotation=0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.1,
        ))
        
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="ObjectSamplerF",
            mujoco_objects=self.ballA,
            x_range=[self.ball_A_pos[0], self.ball_A_pos[0]],
            y_range=[self.ball_A_pos[1], self.ball_A_pos[1]],
            rotation=0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cubeA, self.cubeB, self.cubeC, self.cubeD, self.bottleA, self.ballA],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)
        self.cubeD_body_id = self.sim.model.body_name2id(self.cubeD.root_body)
        self.bottleA_body_id = self.sim.model.body_name2id(self.bottleA.root_body)
        self.ballA_body_id = self.sim.model.body_name2id(self.ballA.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeB_body_id])

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def cubeC_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeC_body_id])
            
            @sensor(modality=modality)
            def cubeD_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeD_body_id])
            
            @sensor(modality=modality)
            def bottleA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bottleA_body_id])
            
            @sensor(modality=modality)
            def ballA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ballA_body_id])

            @sensor(modality=modality)
            def gripper_to_cubeA(obs_cache):
                return (
                    obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache
                    else np.zeros(3)
                )
                
            @sensor(modality=modality)
            def cubeA_size(obs_cache):
                return np.array(self.cubeA.size)
            
            @sensor(modality=modality)
            def cubeB_size(obs_cache):
                return np.array(self.cubeB.size)
            
            @sensor(modality=modality)
            def cubeC_size(obs_cache):
                return np.array(self.cubeC.size)
            
            @sensor(modality=modality)
            def cubeD_size(obs_cache):
                return np.array(self.cubeD.size)
            
            @sensor(modality=modality)
            def bottleA_size(obs_cache):
                return np.array(self.bottle_A_size)
            
            @sensor(modality=modality)
            def ballA_size(obs_cache):
                return np.array(self.ballA.size)

            # sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, gripper_to_cubeA, gripper_to_cubeB, cubeA_to_cubeB, cubeC_pos, cubeD_pos, bottleA_pos]
            sensors = [cubeA_pos, cubeB_pos, cubeC_pos, cubeD_pos, bottleA_pos, cubeA_size, cubeB_size, cubeC_size, cubeD_size, bottleA_size, ballA_pos, ballA_size]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)
            
            
    def convert_to_pixel_coords(self, world_pos, table_region, table_size=0.8):
        table_x_min, table_y_min, table_x_max, table_y_max = table_region
        table_width = table_x_max - table_x_min
        table_height = table_y_max - table_y_min
        scale_factor_x = table_width / table_size
        scale_factor_y = table_height / table_size
        pixel_y = int((world_pos[0] + table_size / 2) * scale_factor_x + table_x_min)
        pixel_x = int((world_pos[1] + table_size / 2) * scale_factor_y + table_y_min)
        return (pixel_x, pixel_y)

    def convert_to_pixel_size(self, world_size, table_region, table_size=0.8):
        table_x_min, table_y_min, table_x_max, table_y_max = table_region
        table_width = table_x_max - table_x_min
        table_height = table_y_max - table_y_min
        scale_factor_x = table_width / table_size
        scale_factor_y = table_height / table_size
        pixel_width = int(world_size[0] * scale_factor_x)
        pixel_height = int(world_size[1] * scale_factor_y)
        return (pixel_width, pixel_height)

    def detect_table_edges(self, image):
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
    
    def process_image(self, save_frame, obs):
    
        save_frame = save_frame[70:70+256, 126:126+256]
        ## mirror the image
        save_frame = cv2.flip(save_frame, 0)

        # Detect table edges in the image
        table_region = self.detect_table_edges(save_frame)
        
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
                pixel_pos = self.convert_to_pixel_coords(pos, table_region)
                if "cube" in obj:
                    object_size = object_info[obj.replace("pos", "size")]
                    pixel_size = self.convert_to_pixel_size(object_size, table_region)
                    half_width = pixel_size[0]
                    half_height = pixel_size[1]
                    # cv2.rectangle(save_frame, (pixel_pos[0] - half_width, pixel_pos[1] - half_height),
                    #             (pixel_pos[0] + half_width, pixel_pos[1] + half_height), (55, 55, 55), 1)
                elif "ball" in obj:
                    object_size = object_info[obj.replace("pos", "size")]
                    pixel_size = self.convert_to_pixel_size([object_size, object_size], table_region)
                    # cv2.circle(save_frame, pixel_pos, pixel_size[0], (55, 55, 55), 1)
                else:
                    # cv2.circle(save_frame, pixel_pos, 5, (255, 50, 0), -1)
                    object_size = object_info[obj.replace("pos", "size")]
                pos = [round(p, 2) for p in pos]
                cv2.putText(save_frame, obj.split('_')[0], (pixel_pos[0], pixel_pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                cv2.putText(save_frame, f"pos:{pos}", (pixel_pos[0]-43, pixel_pos[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                cv2.putText(save_frame, f"{object_size}", (pixel_pos[0]-43, pixel_pos[1]+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                
        # Draw table edges in the image
        # cv2.rectangle(save_frame, (table_region[0], table_region[1]), (table_region[2], table_region[3]), (255, 0, 0), 2)
        
        return save_frame