"""
Habitat Lab utilities for the Topological Navigation Task.
"""
from typing import Any, Optional, List

import attr
import os
import numpy as np
import quaternion
import habitat_sim
import json
import random
import gym
gym.logger.set_level(40) # suppress warnings
import magnum as mn

from copy import copy, deepcopy
from tqdm import tqdm
from gym import spaces
from habitat_sim.physics import VelocityControl
from habitat_sim import RigidState
from habitat import Dataset
from habitat import logger

from habitat.datasets.pointnav.pointnav_dataset import (
    PointNavDatasetV1,
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
)
from habitat.utils.visualizations import maps
from habitat.gym.gym_wrapper import HabGymWrapper
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.core.dataset import ALL_SCENES_MASK
from habitat.core.environments import RLTaskEnv
from habitat.config import read_write
from habitat.core.simulator import (
    Sensor,
    Simulator,
    SensorTypes,
    DepthSensor,
    RGBSensor,
    AgentState,
    )
from habitat.core.embodied_task import (
    EmbodiedTask,
    SimulatorTaskAction,
)
from habitat.tasks.nav.nav import (
    NavigationTask,
    NavigationEpisode,
    NavigationGoal,
    TopDownMap,
    ShortestPathPoint,
)
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size
cv2 = try_cv2_import()

@attr.s(auto_attribs=True, kw_only=True)
class OrientedNavigationGoal(NavigationGoal):
    r"""Navigation goal with specified viewing direction.
    
    Args:
    rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
    """

    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)

@attr.s(auto_attribs=True, kw_only=True)
class TopologicalNavigationEpisode(NavigationEpisode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal, optional shortest paths and a list 
    of topological node coordinates. An episode is a description of one
    task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
        subgoals: list of episode subgoals
        current_subgoal: current subgoal
    """
        
    subgoals: List[OrientedNavigationGoal] = None
    current_subgoal: OrientedNavigationGoal = None
    current_nav_target: np.ndarray = None
    path = None

@registry.register_measure
class TopDownTopoMap(TopDownMap):
    r"""
    Top down map measure for visualization.
    """

    def _draw_shortest_path(
        self, episode: TopologicalNavigationEpisode, agent_position: AgentState
    ):
        if self._config.draw_shortest_path:
            _shortest_path_points = episode.path
            _subgoals = episode.subgoals

            if _shortest_path_points is None:
                # Likely cause: pathfinder failed to find a path
                return

            self._shortest_path_points = [
                maps.to_grid(
                    p[2],
                    p[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

            # Also draw the subgoal positions as red dots
            self._subgoals = [
                maps.to_grid(
                    p.position[2],
                    p.position[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in _subgoals
            ]

            for p in self._subgoals:
                cv2.circle(
                    self._top_down_map,
                    p[::-1],
                    radius=10,
                    color=(255, 0, 255),
                    thickness=5,
                )

            # Also draw the orientation of the subgoals
            # p.rotation is quaternion with (x, y, z, w) elements
            for p_pos, p in zip(self._subgoals, _subgoals):
                quat = quaternion.from_float_array(np.roll(p.rotation, 1))
                dir_vec = habitat_sim.utils.quat_rotate_vector(quat, habitat_sim.geo.FRONT)
                dir_vec = np.array([dir_vec[0], dir_vec[2]])
                dir_vec = dir_vec / np.linalg.norm(dir_vec)

                end = (p_pos[::-1] + 25*dir_vec).round().astype(int)
 

                cv2.line(
                    self._top_down_map,
                    p_pos[::-1],
                    end,
                    (255, 0, 255),
                    5,
                )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        super().update_metric(episode, action, *args, **kwargs)

        subgoal: OrientedNavigationGoal = episode.current_subgoal
        nav_target: np.ndarray = episode.current_nav_target

        if subgoal is not None:
            # The current subgoal coordinates
            sg_x, sg_y = maps.to_grid(
                subgoal.position[2],
                subgoal.position[0],
                (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                sim=self._sim,
            )
            self._metric['subgoal_position'] = [sg_x, sg_y]

            # The current nav target coordinates
            nt_x, nt_y = maps.to_grid(
                nav_target[2],
                nav_target[0],
                (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                sim=self._sim,
            )
            self._metric['nav_target_position'] = [nt_x, nt_y]

            lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
            self._metric['map_bounds'] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
            self._metric['agent_state'] = self._sim.get_agent_state()

# Import here to avoid circular import
from faint.train.habitat.velocity_integration import integrate_agent_state
from faint.train.habitat.continuous_path_follower import ContinuousPathFollower, track_waypoint

@registry.register_sensor
class SubgoalSensor(Sensor):
    r"""Sensor for Subgoal observations which are used in Topological Navigation.
    This sensor returns the subgoal position and rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the Subgoal sensor.
    """
    cls_uuid: str = "subgoal_tracker"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id: Optional[str] = None
        self._path_follower: Optional[ContinuousPathFollower] = None
        self._controller_lookahead = config.controller_lookahead
        self._align_agent = config.align_agent
        self._action_pred_horizon = config.action_pred_horizon
        assert self._action_pred_horizon > 0, "Action prediction horizon must be greater than 0"

        self._linear_velocity_cmd = np.zeros(3, dtype=np.float32)
        self._angular_velocity_cmd = np.zeros(3, dtype=np.float32)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    @classmethod
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        linear_vel_cmd_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, 3),
            dtype=np.float32,
        )
        angular_vel_cmd_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, 3),
            dtype=np.float32,
        )
        pos_cmd_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, 3),
            dtype=np.float32,
        )
        rot_cmd_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, 4),
            dtype=np.float32,
        )

        subgoal_position_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )
        subgoal_rotation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )
        goal_position_diff_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.float32,
        )
        goal_rotation_diff_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.float32,
        )
        return spaces.Dict({
            "agent_position": subgoal_position_space,
            "agent_rotation": subgoal_rotation_space,
            "subgoal_position": subgoal_position_space,
            "subgoal_rotation": subgoal_rotation_space,
            "linear_vel_cmds": linear_vel_cmd_space,
            "angular_vel_cmds": angular_vel_cmd_space,
            "pos_cmds": pos_cmd_space,
            "rot_cmds": rot_cmd_space,
            "route_goal_position_diff": goal_position_diff_space,
            "route_goal_rotation_diff": goal_rotation_diff_space,
            }
        )

    def _reset(self, episode: NavigationEpisode):
        '''Recompute the navigation path at the start of each episode'''

        self._linear_velocity_cmd = np.zeros(3, dtype=np.float32)
        self._angular_velocity_cmd = np.zeros(3, dtype=np.float32)

        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        self.allow_sliding = self._sim.config.sim_cfg.allow_sliding

        # Get the navmesh settings from the simulator config
        default_nav_mesh_settings = self._sim.sim_config.sim_cfg.navmesh_settings

        # Inflated navmesh
        # NavMeshSettings object doesn't have a copy constructor, so we have to copy each field
        inflated_nav_mesh_settings = habitat_sim.NavMeshSettings()
        inflated_nav_mesh_settings.set_defaults()

        # Inflate the navmesh by the safety margin
        inflated_nav_mesh_settings.agent_radius = default_nav_mesh_settings.agent_radius + self.config.planner_safety_margin
        inflated_nav_mesh_settings.agent_height = default_nav_mesh_settings.agent_height
        inflated_nav_mesh_settings.agent_max_climb = default_nav_mesh_settings.agent_max_climb
        inflated_nav_mesh_settings.agent_max_slope = default_nav_mesh_settings.agent_max_slope
        inflated_nav_mesh_settings.include_static_objects = default_nav_mesh_settings.include_static_objects

        # Compute the navmesh inflated by a safety margin
        recompute_successful = self._sim.recompute_navmesh(self._sim.pathfinder, inflated_nav_mesh_settings)
        if not recompute_successful:
            # raise ValueError("Failed to compute inflated navmesh!")
            logger.info(
                    f"Failed to compute inflated navmesh!"
            )
            return False

        agent_pos = self._sim.get_agent_state().position
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = episode.goals[0].position

        found_path = self._sim.pathfinder.find_path(path)

        if (
            not(found_path)
        ) or (
            path.geodesic_distance < 1e-1
        ) or (
            np.linalg.norm(path.points[0] - path.requested_start) > 1e-1
        ) or (
            np.allclose(agent_pos, path.points[-1])
        ):
            # Avoid situations where the measurement reset produces
            # division by zero in the SPL calculation if the agent is already at the goal
            while self._sim.geodesic_distance(agent_pos, episode.goals[0].position) < 1e-1:
                episode.goals[0].position = self._sim.pathfinder.get_random_navigable_point()
            return False
        
        # This one weird trick that the scientists don't want you to know about:
        # the episode goal point might not be navigable for an agent of certain size
        # and the pathfinder will return a path that ends at the closest navigable point.
        # In order for the success measurements etc to work, we need to make sure that
        # the path ends at the actual goal point.
        episode.goals[0].position = path.points[-1]

        # Recompute the default navmesh (determines when agent is in collision)
        recompute_successful = self._sim.recompute_navmesh(self._sim.pathfinder, default_nav_mesh_settings)
        if not recompute_successful:
            raise ValueError("Failed to recompute navmesh!")

        self._path_follower = ContinuousPathFollower(
            sim=self._sim,
            path=path,
            lookahead=self._controller_lookahead,
            subgoal_sampling_strategy=self.config.subgoal_sampling_strategy,
            subgoal_spacing=self.config.subgoal_spacing,
            subgoal_min_spacing=self.config.subgoal_min_spacing,
            subgoal_max_spacing=self.config.subgoal_max_spacing,
            align_agent=self._align_agent,
        )

        # Set the episode path to the pathfinder path
        episode.subgoals = self._path_follower.subgoals
        episode.path = self._path_follower.path
        return True

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        """
        Get the subgoal observation for the current agent position,
        and generate 'oracle' to navigate towards it.
        """

        """     
        Unlike the Measurements, sensors don't define a reset() method
        that would b called by env.reset()
        so we have to manually reset the sensor at the start of each episode.
    
        It is possible to determine that the episode has changed by checking
        _shortest_path_cache, since it is set to None when the env is reset.
        (see https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/core/env.py#L247).
        This is a bit hacky, but the method used by the Habitat team in the ImageGoal sensor actually doesn't correctly handle
        the case where the episode iterator cycles back to the first episode and shuffles so that the last and first episodes
        are the same. This can actually happen!
        -> Habitat team should add a reset() method to the Sensor class to properly support stateful sensors.
        """
        if episode._shortest_path_cache is None:
            reset_successful = self._reset(episode)
            self._current_episode_id = f"{episode.scene_id} {episode.episode_id}"
            if not reset_successful:
                return None
            
        # 1. Update waypoint
        # 2. Compute the linear and angular velocity commands
        # 3. Integrate the agent state from the velocity commands
        # 4. Repeat until n iterations
        # 5. Reset the path follower waypoint to the result of the first iteration
        # 6. Convert the integrated poses to local coordinates
        linear_vel_cmds = []
        angular_vel_cmds = []
        pos_cmds = []
        rot_cmds = []
        current_pose = self._sim.get_agent_state()
        linear_velocity, angular_velocity = self._linear_velocity_cmd, self._angular_velocity_cmd

        # Roll out waypoint tracking to get self._action_pred_horizon steps of oracle actions
        for i in range(self._action_pred_horizon):
            self._path_follower.update_waypoint(current_pose.position)
            linear_velocity, angular_velocity = track_waypoint(
                self._path_follower.nav_target,
                current_pose.position,
                current_pose.rotation,
                linear_velocity,
                angular_velocity,
                self.config.time_step,
                angular_error_threshold=self.config.controller_angular_error_threshold,
                max_linear_speed=self.config.max_linear_speed,
                max_turn_speed=self.config.max_turn_speed,
                smooth_acceleration=self.config.smooth_acceleration,
            )

            if i == 0:
                stash_wp, stash_progress = self._path_follower.nav_target, self._path_follower.route_progress
                self._linear_velocity_cmd, self._angular_velocity_cmd = deepcopy(linear_velocity), deepcopy(angular_velocity)

            self.vel_control.linear_velocity = linear_velocity
            self.vel_control.angular_velocity = angular_velocity

            # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
            normalized_quaternion = current_pose.rotation.normalized()
            agent_mn_quat = mn.Quaternion(
                normalized_quaternion.imag, normalized_quaternion.real
            )
            current_rigid_state = RigidState(
                agent_mn_quat,
                current_pose.position,
            )

            # snap rigid state to navmesh and set state to object/agent
            if self.allow_sliding:
                step_fn = self._sim.pathfinder.try_step  # type: ignore
            else:
                step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

            current_pose.position, current_pose.rotation, collided = integrate_agent_state(
                current_rigid_state=current_rigid_state,
                vel_control=self.vel_control,
                time_step=self.config.time_step,
                step_fn=step_fn,
            )
            linear_vel_cmds.append(linear_velocity)
            angular_vel_cmds.append(angular_velocity)
            pos_cmds.append(current_pose.position)
            rot_cmds.append(current_pose.rotation)

        self._path_follower.set_waypoint(stash_wp, stash_progress)
        current_pose = self._sim.get_agent_state()

        linear_vel_cmds = np.array(linear_vel_cmds)
        angular_vel_cmds = np.array(angular_vel_cmds)
        pos_cmds = np.array(pos_cmds)

        # Make the future positions and rotations relative to the agent current position
        pos_cmds = pos_cmds - current_pose.position
        pos_cmds = quaternion.rotate_vectors(current_pose.rotation.inverse(), pos_cmds)
        rot_cmds = np.array([quaternion.as_float_array(current_pose.rotation.inverse() * q) for q in rot_cmds])

        assert not any(np.isnan(self._linear_velocity_cmd)), f"Linear velocity command is NaN {self._linear_velocity_cmd}"
        assert not any(np.isnan(self._angular_velocity_cmd)), f"Angular velocity command is NaN {self._angular_velocity_cmd}"

        # Set the current subgoal in the episode
        # Utilized by the TopDownTopoMap measurement
        episode.current_subgoal = self._path_follower.current_subgoal
        episode.current_nav_target = self._path_follower.nav_target
        goal_position_diff, goal_rotation_diff = self._path_follower.distance_to_goal(episode)

        # Return copies as the original arrays are unwritable
        return {
            "subgoal_position": copy( self._path_follower.current_subgoal.position ),
            "subgoal_rotation": copy( self._path_follower.current_subgoal.rotation ),
            "agent_position":   copy( current_pose.position ),
            "agent_rotation":   copy( quaternion.as_float_array(current_pose.rotation.normalized())),
            "time_step":        self.config.time_step,
            "linear_vel_cmds":   copy( linear_vel_cmds ),
            "angular_vel_cmds":  copy( angular_vel_cmds ),
            "pos_cmds":          copy( pos_cmds ),
            "rot_cmds":          copy( rot_cmds ),
            "route_goal_position_diff": copy( goal_position_diff ),
            "route_goal_rotation_diff": copy( goal_rotation_diff ),
            }

@registry.register_sensor
class ImageSubgoalSensor(SubgoalSensor):
    r"""
    Sensor for ImageGoal observations which are used in Topological Image Navigation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagesubgoal"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors

        if config.obs_type == "rgb":
            sensor_type = RGBSensor
        elif config.obs_type == "depth":
            sensor_type = DepthSensor
        else:
            raise ValueError(f"Unrecognized observation type {config.obs_type}")

        image_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, sensor_type)
        ]
        if len(image_sensor_uuids) != 1:
            raise ValueError(
                f"ImageSubgoalNav requires one {config.obs_type} sensor, {len(image_sensor_uuids)} detected"
            )

        (self._image_sensor_uuid,) = image_sensor_uuids
        self._current_subgoal_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(sim = sim, config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        
        subgoal_space = SubgoalSensor._get_observation_space()
        subgoal_space['subgoal_image'] = self._sim.sensor_suite.observation_spaces.spaces[
            self._image_sensor_uuid
        ]
        return subgoal_space

    def _get_subgoal_image(self, subgoal_position: np.ndarray, subgoal_rotation: np.ndarray):

        # This only fetches the habitat_sim sensor observations
        # ie. depth and rgb images, not the habitat-lab sensors
        goal_observation = self._sim.get_observations_at(
            position=subgoal_position.tolist(), rotation=subgoal_rotation.tolist()
        )
        return goal_observation[self._image_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):   
        """
        Get the observations of the SubgoalSensor, and the image observation at the subgoal pose.
        """     
        subgoal_obs_dict = super().get_observation(observations=observations, episode=episode, **kwargs)

        if subgoal_obs_dict is None:
            return subgoal_obs_dict

        subgoal_uniq_id =  self._current_episode_id + f"{subgoal_obs_dict['subgoal_position']}"
        if subgoal_uniq_id == self._current_subgoal_id:
            subgoal_obs_dict["subgoal_image"] = self._current_image_goal
            return subgoal_obs_dict

        self._current_image_goal = self._get_subgoal_image(
            subgoal_obs_dict["subgoal_position"], subgoal_obs_dict["subgoal_rotation"]
        )
        self._current_subgoal_id = subgoal_uniq_id
        subgoal_obs_dict["subgoal_image"] = self._current_image_goal

        return subgoal_obs_dict 

@registry.register_task_action
class SE2VelocityAction(SimulatorTaskAction):
    name: str = "se2_velocity_control"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        config = kwargs["config"]
        self.min_lin_vel, self.max_lin_vel = config.lin_vel_range
        self.min_ang_vel, self.max_ang_vel = config.ang_vel_range
        self.time_step = config.time_step
        self.time_step_noise_multiplier = config.timestep_noise_multiplier
        self._allow_sliding = self._sim.config.sim_cfg.allow_sliding  # type: ignore

        if self._allow_sliding:
            self.step_fn = self._sim.pathfinder.try_step  # type: ignore
        else:
            self.step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

    @property
    def action_space(self):
        # return ActionSpace(
        return spaces.Dict(
            {
                "expert_linear_velocity": spaces.Box(
                    low=np.float32(self.min_lin_vel),
                    high=np.float32(self.max_lin_vel),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "expert_angular_velocity": spaces.Box(
                    low=np.float32(self.min_ang_vel),
                    high=np.float32(self.max_ang_vel),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "student_linear_velocity": spaces.Box(
                    low=np.float32(self.min_lin_vel),
                    high=np.float32(self.max_lin_vel),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "student_angular_velocity": spaces.Box(
                    low=np.float32(self.min_ang_vel),
                    high=np.float32(self.max_ang_vel),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "apply_student_action": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(1),
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        )

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        expert_linear_velocity: np.ndarray,
        expert_angular_velocity: np.ndarray,
        student_linear_velocity: np.ndarray,
        student_angular_velocity: np.ndarray,
        apply_student_action: np.ndarray,
        time_step: Optional[float] = None,
        allow_sliding: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            task: The task instance
            expert_linear_velocity: The linear velocity to move the agent with
            expert_angular_velocity: The angular velocity to move the agent with
            student_linear_velocity: The linear velocity to move the agent with (if apply_student_action is True)
            student_angular_velocity: The angular velocity to move the agent with (if apply_student_action is True)
            apply_student_action: Whether to apply the student action or the expert action
            time_step: amount of time to move the agent for
            allow_sliding: whether the agent will slide on collision
        """

        if time_step is None:
            time_step = self.time_step

        apply_student_action = bool(apply_student_action.item())
        if apply_student_action:
            linear_velocity = student_linear_velocity
            angular_velocity = student_angular_velocity
        else:
            linear_velocity = expert_linear_velocity
            angular_velocity = expert_angular_velocity

        # Stop is called if linear speed is positive (since habitat uses -x for forward)
        if (
            np.any(linear_velocity > 0)
        ):
            obs = self._sim.step(None)  # type: ignore
            task.is_stop_called = True  # type: ignore
            return obs

        self.vel_control.linear_velocity = linear_velocity
        self.vel_control.angular_velocity = angular_velocity

        agent_state = self._sim.get_agent_state()

        # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
        normalized_quaternion = agent_state.rotation.normalized()
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )

        # Add noise to the time step duration
        if self.time_step_noise_multiplier > 0:
            time_step_multiplier = 1 + np.random.uniform(low=-self.time_step_noise_multiplier,
                                               high=self.time_step_noise_multiplier,
                                               size=None)
            time_step = time_step * time_step_multiplier


        # snap rigid state to navmesh and set state to object/agent
        if allow_sliding:
            step_fn = self._sim.pathfinder.try_step  # type: ignore
        else:
            step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

        final_position, final_rotation, collided = integrate_agent_state(
            current_rigid_state=current_rigid_state,
            vel_control=self.vel_control,
            time_step=time_step,
            step_fn=step_fn,
        )

        # If agent collided and apply_student_action is True, try the expert action
        if collided and apply_student_action:
            self.vel_control.linear_velocity = expert_linear_velocity
            self.vel_control.angular_velocity = expert_angular_velocity
            final_position, final_rotation, collided = integrate_agent_state(
                current_rigid_state=current_rigid_state,
                vel_control=self.vel_control,
                time_step=time_step,
                step_fn=self.step_fn,
            )

        self._sim.set_agent_state(  # type:ignore
            final_position, final_rotation, reset_sensors=False
        )
        # Step the sim here with no actions since the agent state
        # was set kinematically
        observations = self._sim.step(None)  # type: ignore

        # TODO: Make a better way to flag collisions
        self._sim._prev_sim_obs["collided"] = collided  # type: ignore
        return observations

class RLTaskEnvWithEmptyStep(RLTaskEnv):
    """
    Define an empty step method to enable getting agent observations after aligning the agent.
    """
    def step_empty(self):
        # Step the environment with no actions
        # This only returns the observations from
        # habitat-sim sensors (== no habitat-lab sensors)
        return self.habitat_env.sim.step(None)

@registry.register_env(name="GymHabitatEnvWithEmptyStep")
class GymHabitatEnvWithEmptyStep(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = RLTaskEnvWithEmptyStep(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)


@registry.register_task(name="TopoNav-v1")
class TopologicalNavigationTask(NavigationTask):
    """
    Bugfix for the collision handling.
    """

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        collision = self._sim.previous_step_collided
        stop_called = getattr(self, "is_stop_called", False)

        return not(stop_called) and not(collision)
    

@registry.register_dataset(name="TopoNav-v1")
class TopoNavDataset(PointNavDatasetV1):

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        """
        Redefined here to switch episode type to TopologicalNavigationEpisode.
        """
        self.episodes = []
        self.config = config

        if config is None:
            return

        datasetfile_path = config.data_path.format(split=config.split)

        self._load_from_file(datasetfile_path, config.scenes_dir)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            _, world_rank, _ = get_distrib_size()
            scene_it = tqdm(scenes, desc="Loading dataset scenes") if (world_rank == 0) else scenes
            for scene in scene_it:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                self._load_from_file(scene_filename, config.scenes_dir)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

        # Change the episode type to TopologicalNavigationEpisode
        self.episodes = [
            TopologicalNavigationEpisode(**{k:v for k,v in episode.__dict__.items() if k != '_shortest_path_cache'})
              for episode in self.episodes
        ]


    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        r"""
        Return list of scene ids for which dataset has separate files with
        episodes.

        Redefined here to allow distibution of scenes across ranks.
        """

        dataset_dir = os.path.dirname(
            config.data_path.format(split=config.split)
        )
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset file `{dataset_dir}`"
            )

        cfg = config.copy()
        with read_write(cfg):
            cfg.content_scenes = []
            dataset = cls(cfg)
            has_individual_scene_files = os.path.exists(
                dataset.content_scenes_path.split("{scene}")[0].format(
                    data_path=dataset_dir
                )
            )
            if has_individual_scene_files:
                scenes = cls._get_scenes_from_folder(
                    content_scenes_path=dataset.content_scenes_path,
                    dataset_dir=dataset_dir,
                )
            else:
                # Load the full dataset, things are not split into separate files
                cfg.content_scenes = [ALL_SCENES_MASK]
                dataset = cls(cfg)
                scenes = list(map(cls.scene_from_scene_path, dataset.scene_ids))

        local_rank, world_rank, world_size = get_distrib_size()
        if world_size > 1:
            if world_rank == 0:
                logger.info(
                    f"Distributing scenes across {world_size} processes"
                )
            scenes = scenes[world_rank::world_size]

        return scenes

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        """
        Redefiniton of the original method to allow sampling subset of episodes per scene.
        """
        deserialized = json.loads(json_str)
        if deserialized['episodes']:
            if (
                (self.config.episodes_per_scene > 0)
                and 
                (len(deserialized['episodes']) > self.config.episodes_per_scene)
            ):
                deserialized['episodes'] = random.sample(
                    deserialized['episodes'],
                    self.config.episodes_per_scene
                )
        
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)


