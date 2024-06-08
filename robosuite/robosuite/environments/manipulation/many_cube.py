from collections import OrderedDict

import numpy as np
import numbers
import copy
import gym
import robosuite as suite
import cv2
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, ObjectPositionSampler
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat, get_bounding_box
from robosuite.models.objects import BoxObject
from robosuite.models.objects.composite.Button import Button
from robosuite.controllers import load_controller_config
from stl import mesh
from robosuite.utils.mjmod import DynamicsModder



# Lift --> Pusher
class Magnetic(SingleArmEnv):
    """
    This class corresponds to the pushing task for a single robot arm.

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
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(0.1, 5e-2, 1e-4),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=True,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            # frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            # camera_widths=368,
            camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,  # {None, instance, class, element}
            renderer="mujoco",
            renderer_config=None,
            random_target_generation=False,  # Below Added by Compass Team
            threshold=0.05,
            default_table_size=False,
            deterministic_reset=True,
    ):

        self.camera_name = camera_names
        # settings for table top
        if default_table_size is True:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (2, 2, 0.05)
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # Define default object position and range
        self.default_target_pos = np.asarray([-0.45, 0, 0.8])
        # self.default_cube2_pos = np.asarray([-0.0, 0.0, 0.8])
        # self.default_cube1_pos = np.asarray([-0.1, -0.0, 0.8])
        # self.default_cube3_pos = np.asarray([0.1, 0.0, 0.8])
        self.default_cube2_pos = np.asarray([-0.1, 0.0, 0.8])
        self.default_cube1_pos = np.asarray([-0.11 - 0.035, -0.0, 0.8])
        self.default_cube3_pos = np.asarray([-0.09 + 0.035, 0.0, 0.8])
        self.default_button_pos = np.asarray([0.15, 0.0, 0.8])
    
        self.default_target_pos_range = np.asarray([0.0, 0.])
        self.default_velocity_delay_range = np.asarray([0.0, 0.0])
        self.default_cube2_pos_range = np.asarray([0.00, 0.00])
        self.default_cube1_pos_range = np.asarray([0.00, 0.00])
        self.default_cube3_pos_range = np.asarray([0.00, 0.00])
        self.default_button_pos_range = np.asarray([0.00, 0.00])
        
        # Define object position and range
        self.target_rotation = 0.8   
        self.target_pos = self.generate_positions(self.default_target_pos, self.default_target_pos_range)
        self.cube2_init_pos = self.generate_positions(self.default_cube2_pos, self.default_cube2_pos_range)
        self.cube1_init_pos = self.generate_positions(self.default_cube1_pos, self.default_cube1_pos_range)
        self.cube3_init_pos = self.generate_positions(self.default_cube3_pos, self.default_cube3_pos_range)
        self.button_init_pos = self.generate_positions(self.default_button_pos, self.default_button_pos_range)
        
        # Define reaching target threshold
        self.threshold = threshold


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
        
    def reset_obj_init_positions(self):
        self.target_pos = self.generate_positions(self.default_target_pos, self.default_target_pos_range)
        self.cube2_init_pos = self.generate_positions(self.default_cube2_pos, self.default_cube2_pos_range)
        self.cube1_init_pos = self.generate_positions(self.default_cube1_pos, self.default_cube1_pos_range)
        self.cube3_init_pos = self.generate_positions(self.default_cube3_pos, self.default_cube3_pos_range)
        self.button_init_pos = self.generate_positions(self.default_button_pos, self.default_button_pos_range)
    
    @staticmethod
    def generate_positions(pose_mean, pose_range):
        
        assert len(pose_range) == 2
        # print ("pose_mean", pose_mean, "type", type(pose_mean))
        if isinstance(pose_mean, numbers.Number):
            new_pose = pose_mean 
            new_pose += np.random.uniform(pose_range[0], pose_range[1])
            return new_pose
        elif type(pose_mean) is np.ndarray and len(pose_mean) == 3:
            pose = pose_mean.copy()
            pose[:2] += np.random.uniform(pose_range[0], pose_range[1], 2)
            return pose
        else:
            raise ValueError("Pose Mean Needs to be either 1 or 3 dimensional")
        
    def compute_magnetic_force(self, object1_pos, object2_pos, strength):
        distance = np.linalg.norm(object1_pos - object2_pos)
        
        # Assuming force is proportional to 1/distance^2 for simplicity
        force_magnitude = strength / (distance*2)
        
        # Calculate the direction of the force
        force_direction = (object2_pos - object1_pos) / distance
        
        return force_direction * force_magnitude

    def reward(self, action):
        return 0.0
    

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        self.robots[0].init_qpos = [0.00662393, 0.97028886,
                                    0.00229584, 1.82422673, -0.0059355, 0.35976223, -1.55589321]
        
        # print ("setting robot initial position:", self.robots[0].init_qpos)
        
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=[0,0,0],
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.cube1 = BoxObject(
                    name="cube1",
                    size=(0.01525, 0.01525, 0.01525),
                    rgba=[1, 0, 0, 0.7],
                    obj_type="all",
                    solref = [-10000, -40],
                    duplicate_collision_geoms=True,
        )
        
        self.cube2 = BoxObject(
                    name="cube2",
                    size=(0.01525, 0.01525, 0.01525),
                    rgba=[0, 1, 0, 0.7],
                    obj_type="all",
                    solref = [-10000, -40],
                    duplicate_collision_geoms=True,
        )
        
        self.cube3 = BoxObject(
                    name="cube3",
                    size=(0.01525, 0.01525, 0.01525),
                    rgba=[0, 0, 1, 0.7],
                    obj_type="all",
                    solref = [-10000, -40],
                    duplicate_collision_geoms=True,
        )
        
        self.button = Button(name="button",size=(0.1, 0.1, 0.1),base_thickness=0.1)
        

        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler")
        
        
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler4",
                mujoco_objects=self.cube1,
                x_range=[self.cube1_init_pos[0],self.cube1_init_pos[0]],
                y_range=[self.cube1_init_pos[1],self.cube1_init_pos[1]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rotation = 0.0,
            ))  
        
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler5",
                mujoco_objects=self.cube2,
                x_range=[self.cube2_init_pos[0],self.cube2_init_pos[0]],
                y_range=[self.cube2_init_pos[1],self.cube2_init_pos[1]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rotation=0.0,
            ))
        
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler6",
                mujoco_objects=self.cube3,
                x_range=[self.cube3_init_pos[0],self.cube3_init_pos[0]],
                y_range=[self.cube3_init_pos[1],self.cube3_init_pos[1]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rotation=0.0,
            ))
        
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionSampler1",
                mujoco_objects=self.button,
                x_range=[self.button_init_pos[0],self.button_init_pos[0]],
                y_range=[self.button_init_pos[1],self.button_init_pos[1]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rotation=0.0,
            ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube1, self.cube2, self.cube3, self.button],
        )
    
    def get_3D_pose(self, geom_name):
        
        geom_name = geom_name.split('_')[0]
        # Get the quaternion of the object
        object_attr = getattr(self, geom_name, None)

        if object_attr and hasattr(object_attr, 'root_body'):
            # Get the quaternion of the object
            object_name = self.sim.model.body_name2id(object_attr.root_body)
            object_quat = self.sim.data.body_xquat[object_name]
        else:
            raise ValueError(f"No object named '{geom_name}' with a 'root_body' attribute found.")
        
        # Load the STL file
        # Check if the object is box or cylinder
        if geom_name == 'table' or geom_name == 'cube1' or geom_name == 'cube2' or geom_name == 'cube3':
            dim1, dim2, dim3 = 2 * object_attr.size
        elif geom_name == 'cylinder':
            radius, half_height = object_attr.size
            dim1 = dim2 = 2 * radius
            dim3 = 2 * half_height
        elif geom_name == 'paperroll' or geom_name == 'button':
            dim1, dim2, dim3 = get_bounding_box(object_quat=object_quat, object_attr=object_attr)
        else:
            if geom_name == 'drill':
                geom_name = 'power_drill'
            mesh_obj = mesh.Mesh.from_file(f'./robosuite/models/assets/objects/meshes/{geom_name}.stl')
            dim1, dim2, dim3 = get_bounding_box(mesh_obj=mesh_obj, object_quat=object_quat)
        
        return dim1, dim2, dim3
        

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        
        self.cube3_body_id = self.sim.model.body_name2id(
            self.cube3.root_body)
        self.cube2_body_id = self.sim.model.body_name2id(
            self.cube2.root_body)
        self.cube1_body_id = self.sim.model.body_name2id(
            self.cube1.root_body)
        self.button_body_id = self.sim.model.body_name2id(
            self.button.root_body)
        

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

            # cube2 observables
            @sensor(modality=modality)
            def cube2_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube2_body_id])

            @sensor(modality=modality)
            def cube2_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube2_body_id]), to="xyzw")


            @sensor(modality=modality)
            def gripper_to_cube2_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube2_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube2_pos" in obs_cache
                    else np.zeros(3)
                )
                 
            # cube1 observables
            @sensor(modality=modality)
            def cube1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube1_body_id])

            @sensor(modality=modality)
            def cube1_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube1_body_id]), to="xyzw")


            @sensor(modality=modality)
            def gripper_to_cube1_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube1_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube1_pos" in obs_cache
                    else np.zeros(3)
                )  
                
            # cube3 observables
            @sensor(modality=modality)
            def cube3_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube3_body_id])

            @sensor(modality=modality)
            def cube3_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube3_body_id]), to="xyzw")


            @sensor(modality=modality)
            def gripper_to_cube3_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube3_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube3_pos" in obs_cache
                    else np.zeros(3)
                )  
                
            # button observables
            @sensor(modality=modality)
            def button_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.button_body_id])
            
            @sensor(modality=modality)
            def button_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.button_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def gripper_to_button_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["button_pos"]
                    if f"{pf}eef_pos" in obs_cache and "button_pos" in obs_cache
                    else np.zeros(3)
                )
                           
            sensors = [cube2_pos, cube2_quat, gripper_to_cube2_pos,
                       cube3_pos, cube3_quat, gripper_to_cube3_pos,
                       cube1_pos, cube1_quat, gripper_to_cube1_pos,
                       button_pos, button_quat, gripper_to_button_pos
                       ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

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
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.drill_body_id] = obj_pos
                    self.sim.model.body_quat[self.drill_body_id] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate(
                        [np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drill.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the drill
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.drill)
    
    def get_obs(self):
        goal = self.target_pos.flatten()
        
        observation = super()._get_observations()
        print (observation.keys())
        eef_pos = observation["robot0_eef_pos"].flatten()
        # print ("eef_pos", eef_pos)
        # obs = np.concatenate([
        #     drill_pos, drill_quat, gripper_to_drill_pos
        # ])
        
        return True
    
    def step(self, action, writer = None):
        obs, reward, done, info = super().step(action)
        info.update(obs)
        
        
        # self.cube2_body_id = self.sim.model.body_name2id(
        #     self.cube2.root_body)
        # self.cube1_body_id = self.sim.model.body_name2id(
        #     self.cube1.root_body)
        force = self.compute_magnetic_force(self.sim.data.body_xpos[self.cube1_body_id], self.sim.data.body_xpos[self.cube2_body_id], 0.012)
        self.sim.data.xfrc_applied[self.cube1_body_id, :3] = force
        self.sim.data.xfrc_applied[self.cube2_body_id, :3] = -0.3 * force
        self.sim.data.xfrc_applied[self.cube3_body_id, :3] = -0.4 * force

        return obs, reward, done, info

    def reset(self):
        
        # generate init positions
        self.reset_obj_init_positions()
        # restore to default parameter
        obs = super().reset()
        
        modder = DynamicsModder(self.sim)
        modder.mod("gripper0_left_inner_finger_collision", "friction", (5,1e-5,1e-5))
        modder.mod("gripper0_left_inner_knuckle_collision", "friction", (5,1e-5,1e-5))
        modder.mod("gripper0_right_inner_finger_collision", "friction", (5,1e-5,1e-5))
        modder.mod("gripper0_right_inner_knuckle_collision", "friction", (5,1e-5,1e-5))
        modder.update()
            
        self.sim.model.opt.integrator = 1

        return obs

    def set_seed(self, seed):
        if seed is not None:
            if isinstance(seed, int):
                self._seed = seed
            else:
                self._seed = None