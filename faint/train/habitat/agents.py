"""
Simulated Habitat agents for the topological image navigation task.
"""
from typing import Union, Tuple, List

import magnum as mn
import torch

import habitat_sim
from habitat_sim._ext.habitat_sim_bindings import RigidState
from habitat_baselines.utils.common import inference_mode
from habitat import logger

from faint.train.config.toponav_registry import toponav_registry
from faint.train.habitat import velocity_integration
from faint.common.utils import clamp_raw_velocity

class BaseAgent:
    """
    Base class for agents in the TopoNav task.
    """
    def __init__(self, agent_config, env_spec, device, num_envs, agent_name):
        self._config = agent_config
        self._env_spec = env_spec
        self._device = device
        self._num_envs = num_envs
        self.agent_name = agent_name
        if self.agent_name in ['expert_agent', 'eval_agent']:
            self.role_name = 'expert'
        elif self.agent_name == 'student_agent':
            self.role_name = 'student'
        else:
            raise ValueError(f"Unknown agent name: {self.agent_name}")
        self.policy = None

        self._success_pos_threshold = agent_config.success_distance
        self._success_rot_threshold = agent_config.success_rotation

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
    
    def _integrate_velocities(
            self,
            linear_velocity,
            angular_velocity,
            agent_position,
            agent_rotation,
            time_step,
        ):
        """
        Integrate the commanded velocities to get the new positions of the agents
        """
        goal_states = []
        for i in range(self._num_envs):

            self.vel_control.linear_velocity = mn.Vector3(linear_velocity[i])
            self.vel_control.angular_velocity = mn.Vector3(angular_velocity[i])

            # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
            normalized_quaternion = agent_rotation[i]
            agent_mn_quat = mn.Quaternion(
                normalized_quaternion[1:],
                normalized_quaternion[0],
            )
            current_rigid_state = RigidState(
                agent_mn_quat,
                agent_position[i],
            )
            # manually integrate the rigid state
            goal_rigid_state = velocity_integration.rigid_state_SE2_update(
                time_step[i],
                current_rigid_state,
                self.vel_control,
            )
            goal_states.append(goal_rigid_state)

        return goal_states

    def reset(self):
        pass

    def _init_policy(self, policy_config):
        pass

    def set_policy(self, policy):
        pass

    def on_init(self, num_envs):
        pass

    def on_envs_change(self, done_envs):
        pass

    def on_envs_pause(self, envs_to_continue):
        pass

    def act(self, observations, get_infos=False):
        pass

    def train(self):
        pass

    def eval(self):
        pass

class ImitationBaseAgent(BaseAgent):
    """
    Base class for imitation learning agents in the TopoNav task.
    """
    def __init__(
            self,
            agent_config,
            env_spec,
            device,
            num_envs,
            agent_name,
        ):
        super().__init__(
            agent_config,
            env_spec,
            device,
            num_envs,
            agent_name,
        )
        self.goal_key = agent_config.goal_sensor_key
        self.obs_key = agent_config.obs_key
        self.policy = None

    def _init_policy(self, policy_config):
        """
        Initialize a policy from a config file.
        """
        if not policy_config.checkpoint:
            raise ValueError("Checkpoint not found in policy config")

        policy_type = toponav_registry.get_policy(policy_config.type)            
        self.policy = policy_type.load_from_checkpoint(policy_config.checkpoint).to(self._device)

    def set_policy(self, policy):
        """
        Set the policy to be used by the agent.
        """
        self.policy = policy
        self.policy.to(self._device)

    def reset(self):
        return
    
    def forward(self, obs, goal):
        raise NotImplementedError("Implement in subclass")

    def act(self, observations, get_infos=False):
        """
        Use the policy to get agent control commands from the observations.
        """
        if self.policy is None:
            raise ValueError("Policy not initialized. Call self._init_policy() or self.set_policy() first")

        goal_obs = observations[self.goal_key]

        # Permute to (B, C, H, W)
        obs = observations[self.obs_key].permute(0, 3, 1, 2).contiguous()
        goal = goal_obs['subgoal_image'].permute(0, 3, 1, 2).contiguous()

        model_out = self.forward(obs, goal)

        if self.policy.output_type == 'continuous':
            v, w = clamp_raw_velocity(
                model_out[:, 0],
                model_out[:, 1],
                self._config.max_linear_vel,
                self._config.max_angular_vel,
            )
        elif self.policy.output_type == 'waypoints':
            # The model outputs waypoints relative to the agent's current position and orientation
            v, w = pd_control(
                waypoint=model_out[:, self._config.target_waypoint_idx],
                DT=self._config.time_step,
                MAX_V=self._config.max_linear_vel,
                MAX_W=self._config.max_angular_vel,
            )

            assert torch.all(v <= self._config.max_linear_vel)
            assert torch.all(v >= 0)
            assert torch.all(torch.abs(w) <= self._config.max_angular_vel)

        linear_vel_cmds = torch.zeros_like( goal_obs['linear_vel_cmds'][:, 0])
        angular_vel_cmds = torch.zeros_like( goal_obs['angular_vel_cmds'][:, 0])

        # -x is forward in habitat
        linear_vel_cmds[:,2] = -v
        angular_vel_cmds[:,1] = w

        should_stop = (
            (goal_obs['route_goal_position_diff'] < self._success_pos_threshold)
            *
            (goal_obs['route_goal_rotation_diff'] < self._success_rot_threshold )
        )

        # Stop the agent if it is close enough to the goal
        # The toponav_task.SE2VelocityAction will stop the agent if linear velocity is positive
        # (meaning backwards)
        linear_vel_cmds[ should_stop ] = torch.ones_like(linear_vel_cmds[0,:])*0.1

        # Compose the action data
        action_data = {
            self.role_name + '_linear_velocity': linear_vel_cmds.cpu().numpy(),
            self.role_name + '_angular_velocity': angular_vel_cmds.cpu().numpy(),
        }

        if not get_infos:
            return action_data, None
        else:
            # Ugly solution to get the commanded positions for visualization
            new_positions = self._integrate_velocities(
                linear_vel_cmds.cpu().numpy(),
                angular_vel_cmds.cpu().numpy(),
                goal_obs['agent_position'].cpu().numpy(),
                goal_obs['agent_rotation'].cpu().numpy(),
                goal_obs['time_step'].cpu().numpy()
            )
            return action_data, new_positions

    def train(self):
        logger.info(f"Called .train() on ImitationPolicy, but not implemented")
        pass

    def eval(self):
        self.policy.eval()


@toponav_registry.register_agent
class NonRecurrentAgent(ImitationBaseAgent):
    """
    Non-recurrent imitation learning agent.
    """
    def __init__(self, agent_config, env_spec, device, num_envs, agent_name):
        super().__init__(agent_config, env_spec, device, num_envs, agent_name)

        self.obs:  List[torch.Tensor] = []
        self.obs_seq_len = None # Get this from model hparams when initialized
        self._env_changes = [False]*num_envs
        self.previous_shape = None

    def _init_policy(self, policy_config):
        super()._init_policy(policy_config)
        self.obs_seq_len = self.policy.hparams.config.sequence_length

    def set_policy(self, policy):
        super().set_policy(policy)
        self.obs_seq_len = self.policy.hparams.config.sequence_length

    def reset(self):
        self.obs = []

    def on_envs_change(self, dones):
        """
        Flag the environments that have had an episode change.
        Modifying the observation queue will have to be made in
        the forward pass of the model.

        Args:
            dones: List[bool]
        """
        super().on_envs_change(dones)
        self._env_changes = dones

    def on_envs_pause(self, envs_to_continue):
        """
        Remove the observations in self.obs for the environments that are paused

        Args:
            envs_to_continue: List[int]
                List of environment indices that should continue
        """
        super().on_envs_pause(envs_to_continue)
        if self.obs:
            number_of_previous_envs = self.obs[0].shape[0]
            if len(envs_to_continue) != number_of_previous_envs:
                for obs_idx in range(len(self.obs)):
                    obs = self.obs[obs_idx]
                    batch_idxs = range(len(obs))
                    batch_idxs = [idx for idx in batch_idxs if idx in envs_to_continue]
                    self.obs[obs_idx] = obs[batch_idxs]


    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        """
        Args:
            obs: (B, C, H, W)
                Latest observation image
            goal: (B, C, H, W)
                Latest observation and goal images
                concatenated along the channel dimension
        """

        if len(self.obs) == 0:
            # In the beginning, fill the observation sequence with the first observation
            self.obs = [obs]*self.obs_seq_len
        else:
            self.obs.append(obs)

        # If a certain environment has had an episode change, set the observation
        # sequence for that environment to the current observation for that environment
        for env_idx, done in enumerate(self._env_changes):
            if done:
                for previous_obs in self.obs:
                    previous_obs[env_idx] = obs[env_idx]

        # Remove the oldest observation if the sequence is too long
        if len(self.obs) > self.obs_seq_len:
            self.obs.pop(0)

        # # obs: (B, seq_len * C, H, W)
        # obs = torch.cat(self.obs, dim=1)
        obs = torch.stack(self.obs, dim=1)
        out = self.policy(obs, goal)

        return out

@toponav_registry.register_agent
class OracleVelocityAgent(BaseAgent):
    """
    An agent that uses the ground truth velocity commands to navigate.
    The commands are generated in the SubgoalSensor in the toponav_task.py.
    """
    def reset(self):
        return

    def act(self, observations, get_infos=False):
        obs_key = 'imagesubgoal' if 'imagesubgoal' in observations else 'subgoal_tracker'
        goal_obs = observations[obs_key]
        linear_vel_cmds = goal_obs['linear_vel_cmds'][:, 0]
        angular_vel_cmds = goal_obs['angular_vel_cmds'][:, 0]

        should_stop = (
            (goal_obs['route_goal_position_diff'] < self._success_pos_threshold)
            *
            (goal_obs['route_goal_rotation_diff'] < self._success_rot_threshold )
        )

        # Stop the agent if it is close enough to the goal
        # The toponav_task.SE2VelocityAction will stop the agent if linear velocity is positive
        # (meaning backwards velocity)
        linear_vel_cmds[ should_stop ] = torch.ones_like(linear_vel_cmds[0,:])*0.1

        action_data = {
            self.role_name + '_linear_velocity': linear_vel_cmds.cpu().numpy(),
            self.role_name + '_angular_velocity': angular_vel_cmds.cpu().numpy(),
        }

        if not get_infos:
            return action_data, None
        else:
            # Ugly solution to get the commanded positions for visualization
            new_positions = self._integrate_velocities(
                linear_vel_cmds.cpu().numpy(),
                angular_vel_cmds.cpu().numpy(),
                goal_obs['agent_position'].cpu().numpy(),
                goal_obs['agent_rotation'].cpu().numpy(),
                goal_obs['time_step'].cpu().numpy()
            )
            return action_data, new_positions

    def train(self):
        logger.info(f"Called .train() on OracleVelocityPolicy, but it has no trainable parameters")
        pass

    def eval(self):
        logger.info(f"Called .eval() on OracleVelocityPolicy, but it has no trainable parameters")
        pass

def clip_angle(theta: torch.Tensor) -> torch.Tensor:
    """Clip angle to [-pi, pi] for batched torch tensors."""
    theta = torch.remainder(theta, 2 * torch.pi)
    return torch.where(theta > torch.pi, theta - 2 * torch.pi, theta)

def pd_control(
    waypoint: torch.Tensor,
    DT: float,
    MAX_V: float,
    MAX_W: float,
    EPS: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PD controller to turn batches of waypoints
    into velocity commands for the simulated Habitat agent.
    """
    assert waypoint.shape[-1] == 4, "waypoint must be a 4D vector"
    
    dx, dy, hx, hy = waypoint[..., 0], waypoint[..., 1], waypoint[..., 2], waypoint[..., 3]

    zero_tensor = torch.zeros_like(dx)
    
    # First condition: both dx and dy are small
    condition_both_small = (torch.abs(dx) < EPS) & (torch.abs(dy) < EPS)
    
    # Second condition: only dx is small
    condition_dx_small = (torch.abs(dx) < EPS) & (torch.abs(dy) >= EPS)
    
    v = torch.where(
        condition_both_small,
        zero_tensor, 
        torch.where(
            condition_dx_small,
            zero_tensor,
            dx / DT,
        )
    )
    # Handle the angular velocity 'w'
    w = torch.where(
        condition_both_small,  # Both dx and dy are small
        clip_angle(torch.atan2(hy, hx)) / DT,
        torch.where(
            condition_dx_small,  # Only dx is small
            torch.sign(dy) * torch.pi / (2 * DT),
            torch.atan(dy / dx) / DT  # Otherwise, calculate w normally
        )
    )
    v = torch.clamp(v, 0, MAX_V)
    w = torch.clamp(w, -MAX_W, MAX_W)

    return v, w
