"""
DAgger trainer main class that collects topological navigation demonstrations
from Habitat, trains a policy with the demonstrations, and evaluates the policy.
"""
from typing import Dict, List, Set, Any, Tuple
from omegaconf import DictConfig

import hydra
import torch
import torch.distributed as dist
import time
import tqdm
import shutil
import logging
import tqdm
import os
import numpy as np
import wandb

from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from copy import deepcopy

from habitat import logger
from habitat.config import read_write

from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.ppo.ppo_trainer import get_device
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.info_dict import extract_scalars_from_info, NON_SCALAR_METRICS
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    inference_mode,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
)

from faint.train.config.toponav_registry import toponav_registry
from faint.train.data_utils.demo_handler import DemoHandler
from faint.train.data_utils.data_module import HabitatImitationDataModule
from faint.train.data_utils.trajectory_accumulator import TrajectoryAccumulator
from faint.train.habitat.viz_utils import observations_to_image, add_frames, create_viz_frames
from faint.train.train_utils.lightning_trainer import LightningTrainer
from faint.train.config.default_structured_configs import TopDownTopoMapMeasurementConfig
from faint.train.habitat.agents import BaseAgent
from faint.train.habitat.distributed import init_distributed, aggregate_distributed_stats
from faint.train.habitat.utils import (
    flatten_batch,
    flatten_obs,
    flatten_obs_space,
    get_active_obs_transforms,
    ExponentialBetaSchedule,
    depth_to_normalized_openni,
    preprocess_rgb_tensor,
    squash_step_action_data,
    create_dummy_student_actions,

)

@baseline_registry.register_trainer(name="dagger")
class HabitatDaggerTrainer(BaseILTrainer):

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

        self.envs = None
        self.obs_transforms = []
        self._env_spec = None
        self.lightning_trainer = None
        self.demo_handler = None
        self.round_num = None
        self.device = None
        self.wandb_online_eval_log_keys = ['success', 'spl']

        # The sensors that can fail in a way that shoud cause an episode to end
        self._critical_sensors = []
        for sensor_name, sensor_params in config.habitat.task.lab_sensors.items():
            if hasattr(sensor_params, "can_fail"):
                if sensor_params.can_fail:
                    self._critical_sensors.append(sensor_name)

    def _init_envs(
            self,
            config: DictConfig = None,
            is_eval: bool = False,
        ):
        """
        Initialize the habitat environments for training or evaluation.
        """
        if config is None:
            config = self.config

        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=self.world_rank == 0,
        )
        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        # The measure keys that should only be logged on rank0 and nowhere
        # else. They will be excluded from all other workers and only reported
        # from the single worker.
        self._rank0_keys: Set[str] = set(
            list(self.config.habitat.task.rank0_env0_measure_names)
            + list(self.config.habitat.task.rank0_measure_names)
        )

        # Information on measures that declared in `self._rank0_keys` or
        # to be only reported on rank0. This is seperately logged from
        # `self.window_episode_stats`.
        self._single_proc_infos: Dict[str, List[float]] = {}

    def _create_obs_transforms(self):
        self.obs_transforms = get_active_obs_transforms(self.config.habitat_baselines.il)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _create_agent(
            self,
            agent_type: str,
            config: DictConfig = None,
            **kwargs,
        ) -> BaseAgent:
        """
        Sets up a simulated navigation agent.

        Args:
            agent_type: (expert_agent | student_agent | eval_agent)
        """

        if config is None:
            config = self.config

        return toponav_registry.get_agent(
            config.habitat_baselines.il[agent_type].type
        )(
            agent_config=config.habitat_baselines.il[agent_type],
            env_spec=self._env_spec,
            device=self.device,
            num_envs=self.envs.num_envs,
            agent_name=agent_type,
        )

    def _init_train(self):
        """
        Configure the simulator, policy, and data collection for training.
        """
        self.local_rank, self.world_rank, self.world_size = init_distributed(self.config)
        self.device = get_device(self.config)

        # Log file handler
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        # Get the student policy from the config
        policy_type = toponav_registry.get_policy(
            self.config.habitat_baselines.il.student_policy.type
        )
        # Check if the policy has a checkpoint to load
        if self.config.habitat_baselines.il.student_policy.checkpoint != Path():
            logger.info(f"Loading student policy from checkpoint: {self.config.habitat_baselines.il.student_policy.checkpoint}")
            policy = policy_type.load_from_checkpoint(
                self.config.habitat_baselines.il.student_policy.checkpoint
            )
        else:
            # Init the policy from scratch
            if self.config.toponav.image_sensor.obs_type == "depth":
                channels = 1
            elif self.config.toponav.image_sensor.obs_type == "rgb":
                channels = 3

            policy = policy_type(
                self.config.habitat_baselines.il.student_policy,
                self.world_size,
                channels = channels,
            )

        # Get a handler for the demonstration paths
        self.demo_handler = DemoHandler(
            self.config,
            self.local_rank,
            self.world_rank,
            self.world_size,
        )

        # Get a Lightning trainer for the policy
        self.lightning_trainer = LightningTrainer(
            policy,
            baselines_config=self.config.habitat_baselines,
        )

        # Beta decay schedule for DAgger
        self._beta_schedule = ExponentialBetaSchedule(
            self.config.habitat_baselines.il.dagger_beta_decay
        )

        if self.config.habitat_baselines.verbose:
            if (self.world_rank == 0) and (self.round_num == 0):
                logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation
        if len(self.config.habitat_baselines.eval.video_option) == 0:
            for non_scalar_metric in NON_SCALAR_METRICS:
                non_scalar_metric_root = non_scalar_metric.split(".")[0]
                if non_scalar_metric_root in self.config.habitat.task.measurements:
                    with read_write(self.config):
                        OmegaConf.set_struct(self.config, False)
                        self.config.habitat.task.measurements.pop(
                            non_scalar_metric_root
                        )
                        OmegaConf.set_struct(self.config, True)
                    if self.config.habitat_baselines.verbose:
                        logger.info(
                            f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                        )
        
        self.t_start = time.time()

    def _init_eval(self):
        """
        Configure the simulator and policy for evaluation.
        """
        self.local_rank, self.world_rank, self.world_size = init_distributed(self.config)

        self.device = get_device(self.config)
        logger.add_filehandler(self.config.habitat_baselines.log_file)

    def _gen_episode_key(self, episode_info):
        """
        Generate a unique key for the episode based on the scene_id and episode_id.
        """
        return f"{Path(episode_info.scene_id).parts[-2]}_{episode_info.episode_id}"


    def _reset_failed_episodes(
        self,
        observations: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[int]]:
        """
        Reset the environments in which the oracle failed to
        find a valid path to episode goal.
        """

        reset_observations = []
        envs_to_pause = []
        skipped_episodes: Dict[str, int] = defaultdict(lambda: 0)
        for i, env_obs in enumerate(observations):

            # Check if env_obs has None values for any of the sensors that can fail
            if any([env_obs[key] is None for key in self._critical_sensors]):

                reset_obs = {key: None for key in self._critical_sensors}

                while any([reset_obs[key] is None for key in self._critical_sensors]):
                    current_episode_info = self.envs.current_episodes()[i]
                    episode_key = self._gen_episode_key(current_episode_info)

                    # The env has cycled though without finding scenes with valid paths
                    if skipped_episodes[episode_key] > 2:
                        reset_obs = None
                        envs_to_pause.append(i)
                        break

                    # Increase the skip count for the episode
                    skipped_episodes[episode_key] += 1
                    logger.debug(f"Episode {episode_key} failed, skipping")

                    # Reset the environment, stepping the episode iterator
                    reset_obs, reset_info = self.envs.call_at(i, "reset", {'return_info': True})

                reset_observations.append((i, reset_obs))

        for i, (reset_obs_i, reset_obs) in enumerate(reset_observations):
            observations[reset_obs_i] = reset_obs

        # Return the updated versions of the function arguments
        return observations, skipped_episodes, envs_to_pause
    
    def _get_envs_to_pause(
        self,
        dones: List[bool],
        ep_eval_count: Dict[str, int],
        pre_step_episodes_info: List[Any],
        post_step_episodes_info: List[Any],
        skipped_episodes: Dict[str, int],
        num_episode_repeats: int,
    ) -> Tuple[List[int], Dict[str, int]]:
        """
        Check if any of the environments have cycled through and should be paused.
        """
        if not(pre_step_episodes_info is None):
            assert len(dones) == len(pre_step_episodes_info), f"len(dones) {len(dones)}, len(pre_step_episodes_info) {len(pre_step_episodes_info)}"

        envs_to_pause = []
        # Increment the episode evaluation count for finished trajectories
        for i in range(len(dones)):
            if dones[i]:
                ep_eval_count[
                    self._gen_episode_key(pre_step_episodes_info[i])
                ] += 1

        # Add the skipped episodes to the evaluation count
        for ep_key, skip_count in skipped_episodes.items():
            ep_eval_count[ep_key] += skip_count

        # Check if any of the environments have cycled through 
        # num_episode_repeats times, and pause them if they have
        for i in range(self.envs.num_envs):
            if (
                ep_eval_count[
                    self._gen_episode_key(post_step_episodes_info[i])
                ] >= num_episode_repeats
            ):
                envs_to_pause.append(i)

        return envs_to_pause, ep_eval_count
    
    def _align_agent_view(
        self,
        observations,
        env_idxs,
    ) -> List[Dict[str, Any]]:
        """
        Do an empty step in the environment to update the agent's view
        to correspond to the aligned agent pose set in SubgoalSensor init.
        """
        for i in env_idxs:
            align_obs = self.envs.call_at(i, "step_empty", {})
            for key in align_obs.keys():
                observations[i][key] = align_obs[key]
        return observations

    def _process_obs(
        self,
        config: DictConfig,
        observations,
        flat_obs_space=None,
        dones=None,
    ):
        """
        Process the observations from the environment to prepare them for the policy.
        """
        # Skip the episodes where the pathfinder couldn't find a valid path
        observations, skipped_episodes, envs_to_pause = self._reset_failed_episodes(observations)

        # Only get episode info AFTER calling reset_failed_episodes
        post_step_episodes_info = self.envs.current_episodes()

        assert len(observations) == len(post_step_episodes_info), f"len(observations) {len(observations)}, len(post_step_episodes_info) {len(post_step_episodes_info)}"
        if envs_to_pause:
            for i in envs_to_pause:
                key = self._gen_episode_key(post_step_episodes_info[i])
                assert key in skipped_episodes, f"Episode key {key} not in skipped episodes"
                assert skipped_episodes[key] > 0, f"Episode key {key} has not been skipped"

        # If the agent is aligned to the path direction at the start of an episode
        align=False
        if hasattr(config.habitat.task.lab_sensors, 'imagesubgoal'):
            if config.habitat.task.lab_sensors.imagesubgoal.align_agent:
                align = True
        elif hasattr(config.habitat.task.lab_sensors, 'subgoal_tracker'):
            if config.habitat.task.lab_sensors.subgoal_tracker.align_agent:
                align = True

        if align:
            # Step all envs with empty action to update sensor observations
            # to correspond to the aligned agent pose
            align_idxs = range(self.envs.num_envs) if dones is None else [k for k, v in enumerate(dones) if v]
            align_idxs = [i for i in align_idxs if i not in envs_to_pause]
            observations = self._align_agent_view(
                observations,
                align_idxs,
            )
        observations = self.envs.post_step(observations)

        # Remove the envs to pause from the observations so they don't get processed
        # in the batch creation
        observations = [obs for i, obs in enumerate(observations) if i not in envs_to_pause]
        batch = batch_obs(observations, device=self.device)


        # Assume the image is float tensor with values in [0, inf] in meters
        # Transform the data as if it were created by normalizing from
        # OpenNI format data: uint16, mm, in [0, 2**16 - 1]
        if self.config.habitat.task.lab_sensors.imagesubgoal.obs_type == "depth":
            batch['imagesubgoal']['subgoal_image'] = depth_to_normalized_openni(batch['imagesubgoal']['subgoal_image'])
            batch['depth'] = depth_to_normalized_openni(batch['depth'])

        elif self.config.habitat.task.lab_sensors.imagesubgoal.obs_type == "rgb":
            model_input_size = [
                config.habitat_baselines.il.student_policy.input_height,
                config.habitat_baselines.il.student_policy.input_width,
            ]
            batch['imagesubgoal']['subgoal_image'] = preprocess_rgb_tensor(
                rgb_img = batch['imagesubgoal']['subgoal_image'],
                normalize_mean=HabitatImitationDataModule.rgb_normalize_mean,
                normalize_std=HabitatImitationDataModule.rgb_normalize_std,
                crop_aspect_ratio=None, # Let's not crop for now
                image_size = model_input_size,
            )
            batch['rgb'] = preprocess_rgb_tensor(
                rgb_img = batch['rgb'],
                normalize_mean=HabitatImitationDataModule.rgb_normalize_mean,
                normalize_std=HabitatImitationDataModule.rgb_normalize_std,
                crop_aspect_ratio=None,
                image_size = model_input_size,
            )

        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        # Modify the observation dict hierarchy
        observations = [flatten_batch(obs) for obs in observations]
        if flat_obs_space is not None:
            observations = flatten_obs(observations, flat_obs_space)

        return observations, batch, skipped_episodes, post_step_episodes_info

    def _pause_envs(
        self,
        envs_to_pause: List[int],
        batch: Dict[str, torch.Tensor] = None,
        actives: np.ndarray = None,
        actions: List[np.ndarray] = None,
        expert_actions: List[np.ndarray] = None,
        agents: List[BaseAgent] = None,
        current_episode_reward: torch.Tensor = None,
        post_step_episodes_info: List[Any] = None,
        trajectory_accumulator: TrajectoryAccumulator = None,
    ):  
        """
        Remove entries from the batch and other data structures for the environments to pause.
        """
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(self.envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                self.envs.pause_at(idx)

            if agents is not None:
                for agent in agents:
                    agent.on_envs_pause(state_index)

            if actives is not None:
                actives = actives[state_index]

            if actions is not None:
                actions = [actions[i] for i in state_index]

            if expert_actions is not None:
                expert_actions = [expert_actions[i] for i in state_index]

            if current_episode_reward is not None:
                current_episode_reward = current_episode_reward[state_index]

            if batch is not None:
                for k, v in batch.items():
                    batch[k] = v[state_index]

            if post_step_episodes_info is not None:
                post_step_episodes_info = [post_step_episodes_info[i] for i in state_index]

            if trajectory_accumulator is not None:
                trajectory_accumulator.on_envs_pause(state_index)

        return (
            batch,
            actives,
            actions,
            expert_actions,
            current_episode_reward,
            post_step_episodes_info,
        )

    def _update_dataset(self, split="train"):
        """
        Collect new train demonstrations from the simulation using DAgger. The agent collecting the
        demonstrations executes the expert policy or the student policy with a
        probability dependent on the DAgger beta value and round number.
        """
        config = deepcopy(self.config)
        
        with read_write(config):
            if "top_down_topomap" in config.habitat.task.measurements:
                config.habitat.task.measurements.pop("top_down_topomap")

            if config.habitat_baselines.evaluate or split == "val":
                # Always collect the whole val dataset
                config.habitat.dataset.split = config.habitat_baselines.eval.split
                config.habitat.dataset.episodes_per_scene = -1
                config.habitat_baselines.il.data_collection.num_episodes_per_round = config.habitat_baselines.test_episode_count

            # Remove SPL since it's not needed and can cause some errors
            if 'spl' in config.habitat.task.measurements:
                config.habitat.task.measurements.pop('spl')

            # We might want to use different success distance during data collection and evaluation
            # for example if heavy noise is added to the actions during data collection
            config.habitat.task.measurements.success.success_distance = config.habitat_baselines.il.data_collection.success_distance
                
        il_config = config.habitat_baselines.il

        # Initialize the environments and the expert agent
        self._init_envs(config, is_eval=True)
        expert_agent = self._create_agent(agent_type='expert_agent', config=config)
        expert_agent._init_policy(il_config.expert_policy)
        expert_agent.eval()
        expert_agent.on_init(self.envs.num_envs)

        # Init the student agent and get the policy maintained by the BC trainer
        student_agent = self._create_agent(agent_type='student_agent', config=config)
        student_agent.set_policy(self.lightning_trainer.policy)
        student_agent.eval()
        student_agent.on_init(self.envs.num_envs)

        # Get the DAgger beta value for round
        if self.round_num == -1:
            # Pretraining round
            beta = 1.0
        else:
            beta = self._beta_schedule(self.round_num)

        # Get the directory to save the demonstrations
        demo_save_dir = self.demo_handler.demo_dir_path(
            self.round_num,
            config.habitat.dataset.split,
        )

        # Create an accumulator to manage and save the simulated trajectories
        trajectory_accumulator = TrajectoryAccumulator(
            save=il_config.data_collection.save_demonstrations,
            save_dir=demo_save_dir,
            trajectory_min_lenght=il_config.data_collection.trajectory_min_length,
            num_workers=il_config.data_collection.num_workers,
            rank=self.world_rank,
            world_size=self.world_size,
            chunk_size=il_config.data_collection.chunk_size,
            round_num=self.round_num if split == "train" else None,
        )

        # The subgoal sensor returns a dict of observations,
        # which are nested. We need to flatten them to a single
        # observation space for the trajectory accumulator
        flat_obs_space = flatten_obs_space(self.envs.observation_spaces[0])

        # Track eval counts for episodes
        ep_eval_count: Dict[str, int] = defaultdict(lambda: 0)

        ### Start the main loop
        update_start_time = time.time()

        # Get initial observations
        observations = self.envs.reset()
        observations, batch, skipped_episodes, post_step_episodes_info = self._process_obs(
            config,
            observations,
            flat_obs_space,
            dones=None
        )

        # Pause the environments that didn't have any episodes with
        # valid paths
        envs_to_pause, ep_eval_count = self._get_envs_to_pause(
            [],
            ep_eval_count,
            None,
            post_step_episodes_info,
            skipped_episodes,
            il_config.data_collection.num_episode_repeats,
        )
        _, _, _, _, _, post_step_episodes_info = self._pause_envs(
            envs_to_pause=envs_to_pause,
            agents=[expert_agent, student_agent],
            post_step_episodes_info=post_step_episodes_info,
        )

        # Check post_step_episodes_info that no two envs have the same scene_id
        scene_ids = [ep.scene_id for ep in post_step_episodes_info]
        assert len(set(scene_ids)) == len(scene_ids), f"Two envs have the same scene_id: {scene_ids}"

        # Add the initial observations to the trajectory accumulator
        trajectory_accumulator.init_trajectories(
            observations,
            post_step_episodes_info,
        )

        # Get the expert and student policy actions
        with inference_mode():
            expert_action_data, _ = expert_agent.act(batch)
            student_action_data, _ = student_agent.act(batch)

        # Join the actions into single dict
        actions = {**expert_action_data, **student_action_data}
        apply_student_action = (np.random.rand(self.envs.num_envs, 1) > beta).astype(np.float32)
        actions['apply_student_action'] = apply_student_action
        actions = squash_step_action_data(actions)
        expert_action_data = squash_step_action_data(expert_action_data)

        step_count = 0
        ep_count = 0
        total_episodes = ( 
            sum(self.envs.count_episodes()) 
            if il_config.data_collection.num_episodes_per_round == -1 
            else round(il_config.data_collection.num_episodes_per_round / self.world_size)
        )

        # Check there's enough episodes to collect
        assert total_episodes <= sum(self.envs.count_episodes()), f"Total episodes {total_episodes} is greater than available episodes {sum(self.envs.count_episodes())}"

        # Track which environments are active
        actives = np.ones(self.envs.num_envs , dtype=bool)

        progress_bar = tqdm.tqdm(
            total=total_episodes,
            dynamic_ncols=True,
            desc=f"Collecting {config.habitat.dataset.split} trajectories (rank {self.world_rank})",
            unit="episodes",
            bar_format= "{l_bar}{bar}{r_bar}",
            position=self.world_rank,
            leave=None,
        )

        # Main loop for collecting trajectories
        while np.any(actives):
            # Store the episode info before stepping the environments
            pre_step_episodes_info = deepcopy(post_step_episodes_info)

            # Step the environments with the expert policy actions
            outputs = self.envs.step(actions)
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            observations, batch, skipped_episodes, post_step_episodes_info = self._process_obs(
                config,
                observations,
                flat_obs_space,
                dones,
            )

            # Add the new observations to the trajectory accumulator
            trajectories = trajectory_accumulator.add_step_and_finish(
                observations,
                expert_action_data,
                rewards_l,
                dones,
                infos,
                post_step_episodes_info,
            )

            # Reset the recurrent hidden states for the environments that had a scene change
            expert_agent.on_envs_change(dones)
            student_agent.on_envs_change(dones)

            # Get the expert and student policy actions
            with inference_mode():
                expert_action_data, _ = expert_agent.act(batch)

                if beta < 1.0:
                    student_action_data, _ = student_agent.act(batch)

                    # Execute student policy action with probability 1-beta
                    apply_student_action = (np.random.rand(self.envs.num_envs, 1) > beta).astype(np.float32)

                    # Join the actions into single dict
                    actions = {**expert_action_data, **student_action_data}
                    actions['apply_student_action'] = apply_student_action

                else:
                    # In the case of beta=1, the student policy is never used so we can
                    # just create some dummy data
                    actions = create_dummy_student_actions(expert_action_data)

            actions = squash_step_action_data(actions)
            expert_action_data = squash_step_action_data(expert_action_data)

            # Increment the step and episode counts
            ep_count += len(trajectories)
            progress_bar.update(len(trajectories)) # + sum(skipped_episodes.values()))
            for trajectory in trajectories:
                step_count += len(trajectory.obs)

            # Check if we have reached the desired number of steps
            if (
                (step_count >= il_config.data_collection.num_steps_per_round) and 
                (il_config.data_collection.num_steps_per_round > 0)
                ):
                actives *= False

            # Check if we have reached the desired number of episodes
            if (
                (ep_count >= il_config.data_collection.num_episodes_per_round) and 
                (il_config.data_collection.num_episodes_per_round > 0)
                ) or (
                ep_count >= total_episodes
                ):
                actives *= False

            envs_to_pause, ep_eval_count = self._get_envs_to_pause(
                dones,
                ep_eval_count,
                pre_step_episodes_info,
                post_step_episodes_info,
                skipped_episodes,
                il_config.data_collection.num_episode_repeats,
            )

            # Pause the environments that have cycled through num_episode_repeats times
            (
                batch,
                actives,
                actions,
                expert_action_data,
                _,
                post_step_episodes_info,
            ) = self._pause_envs(
                envs_to_pause=envs_to_pause,
                batch=batch,
                actives=actives,
                actions=actions,
                expert_actions=expert_action_data,
                agents=[expert_agent, student_agent],
                trajectory_accumulator=trajectory_accumulator,
                post_step_episodes_info=post_step_episodes_info,
            )

        progress_bar.close()
        self.envs.close()
        logger.info(f"Waiting for trajectory accumulator to finish..")
        del trajectory_accumulator
        logger.info(f"Dataset handler on rank {self.world_rank} has {step_count} steps from {ep_count} episodes, stopping, Average FPS is {step_count / (time.time() - update_start_time)}")


    def train(self,
              ) -> None:
        """
        Main loop for training a policy with BC / DAgger.

        High-level overview of training:
        1. Collect demonstrations for the current round
        2. Train the policy on the collected demonstrations
        3. Evaluate the policy online
        4. Repeat for the specified number of rounds
        """
        config = self.config
        self.round_num = 0
        self._last_loaded_round = -1
        self._all_demos = []

        self._init_train()

        # Delete previous validation demos
        if config.habitat_baselines.il.data_collection.recollect_val_demos:
            val_demo_dir = self.demo_handler.demo_dir_path(self.round_num, "val")
            shutil.rmtree(val_demo_dir, ignore_errors=True)

        # Redirect logger output to tqdm.write in order to not mess up the pbar
        with logging_redirect_tqdm(loggers=[logger, logging.root]):

            # Check if the validation demos are available
            val_demo_dir = self.demo_handler.demo_dir_path(self.round_num, "val")
            if not self.demo_handler.has_demos(val_demo_dir):
                logger.info(f"No validation demos found in '{val_demo_dir}', starting collection.")
                self._update_dataset(split="val")
                if self.world_size > 1:
                    logger.info(f"Rank {self.world_rank} waiting on validation demo barrier")
                    torch.distributed.barrier()
                logger.info('Done collecting validation demos')

            # The main Dagger loop
            while self.round_num < config.habitat_baselines.il.num_rounds:
                train_demo_dir = self.demo_handler.demo_dir_path(
                    self.round_num,
                    "train",
                )
                # Delete previous training demos
                if config.habitat_baselines.il.data_collection.recollect_train_demos:
                    shutil.rmtree(train_demo_dir, ignore_errors=True)

                # Collect new training demos for the round
                if not self.demo_handler.has_demos(train_demo_dir):
                    logger.info(f"No train demos found in '{train_demo_dir}', starting collection.")
                    self._update_dataset()
                    logger.info(f"Rank {self.world_rank} finished collecting demos")
                    if self.world_size > 1:
                        torch.distributed.barrier()
                        logger.info(f"Rank {self.world_rank} passed barrier")

                # Load the demonstrations for the round
                (
                    demonstrations,
                    self._last_loaded_round,
                ) = self.demo_handler.try_load_demos(
                    self.round_num,
                    self._last_loaded_round,
                )
                self.lightning_trainer.set_demonstrations(demonstrations)

                # Train the policy
                if self.round_num >= config.habitat_baselines.il.trainer.skip_train_until:
                    self.lightning_trainer.train(self.round_num)

                # Wait for all ranks to finish training
                if self.world_size > 1:
                    torch.distributed.barrier()

                # Run eval
                if (config.habitat_baselines.il.trainer.closed_loop_eval_every_round or
                    self.round_num == config.habitat_baselines.il.num_rounds - 1
                    ):
                    self.eval_online()
                self.round_num += 1

                if self.world_size > 1:
                    torch.distributed.barrier()

            if self.world_rank == 0:
                num_train_samples, raw_dataset_size = self.lightning_trainer.train_data_size()
                logger.info(f"Finished training {self.round_num} rounds of DAgger")
                logger.info(f"Total number of training samples: {num_train_samples}")
                logger.info(f"Total number of collected steps in the dataset: {raw_dataset_size}")
                logger.info(f"Total time taken: {(time.time() - self.t_start)/(60**2):.2f} hours")
                
                logger.info("Exporting the checkpoints to TorchScript")
                self.lightning_trainer.export_checkpoints()
                logger.info("Finished exporting checkpoints")

                # Let wandb finish uploading
                self.lightning_trainer.logger.experiment.finish()

            if self.world_size > 1:
                dist.destroy_process_group()

    def eval_online(
        self,
    ) -> None:
        """
        Evaluate a navigation policy in the simulator.
        """
        if self.lightning_trainer is None:
            self._init_eval()

        if self.world_rank == 0:
            logger.info(f"Starting online evaluation for round {self.round_num}\n")

        config = deepcopy(self.config)
        
        # Set some test time configuration options
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            config.habitat.dataset.episodes_per_scene = -1

            if not config.toponav.augmentation.test_time_augment:
                # This is ugly but best we can do since config variable interpolation is currently resolved
                # at the start of training
                if self.world_rank == 0:
                    logger.info("Disabling augmentations during test time")
                    
                if config.habitat.task.lab_sensors.imagesubgoal.subgoal_sampling_strategy == 'random':
                    config.habitat.task.lab_sensors.imagesubgoal.subgoal_sampling_strategy = 'uniform'
                
                config.habitat.task.actions.se2_velocity_action.timestep_noise_multiplier = 0.0

            if len(config.habitat_baselines.eval.video_option) > 0:
                # If visualizing, add a top down mp sensor to the config
                config.habitat.task.measurements.top_down_topomap = TopDownTopoMapMeasurementConfig() 

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        if self.device is None:
            self.device = (
                torch.device("cuda", config.habitat_baselines.torch_gpu_id)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # Count the number of evaluations per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # Initialize the environments and the agent
        self._init_envs(config, is_eval=True)
        eval_agent = self._create_agent('eval_agent', config)
        if self.lightning_trainer is not None:
            eval_agent.set_policy(self.lightning_trainer.policy)
        else:
            # Get the policy from the config
            eval_agent._init_policy(config.habitat_baselines.il.eval_policy)

        if eval_agent.policy is not None:
            eval_agent.policy.to(self.device)

        eval_agent.eval()
        eval_agent.on_init(self.envs.num_envs)

        # Get initial observations
        observations = self.envs.reset()
        (
            observations,
            batch,
            skipped_episodes,
            post_step_episodes_info,
        ) = self._process_obs(
            config,
            observations,
        )

        # Pause the environments that didn't have any episodes with
        # valid paths
        envs_to_pause, ep_eval_count = self._get_envs_to_pause(
            [],
            ep_eval_count,
            None,
            post_step_episodes_info,
            skipped_episodes,
            config.habitat_baselines.eval.evals_per_ep,
        )
        _, _, _, _, _, post_step_episodes_info = self._pause_envs(
            envs_to_pause=envs_to_pause,
            agents=[eval_agent],
            post_step_episodes_info=post_step_episodes_info,
        )

        # Init visualizations
        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            viz_batch = create_viz_frames(self.config, batch)
            rgb_frames: Dict[List[np.ndarray]] = {
                self._gen_episode_key(post_step_episodes_info[env_idx]):
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in viz_batch.items()}, {}, image_keys=['rgb', 'depth', 'subgoal_image', 'top_down_map']
                    )
                ]
                for env_idx in range(self.envs.num_envs)
            }
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)
        else:
            rgb_frames = None

        # Calculate the number of test episodes
        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.count_episodes())
        else:
            total_num_eps = sum(self.envs.count_episodes())
            # Divide the number of total episodes by number of DDP processes
            number_of_eval_episodes = round(number_of_eval_episodes / self.world_size)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        # Create an empty dict to count the steps taken in each episode
        scene_steps: Dict[Any, int] = defaultdict(lambda: 0)

        # Track the reward for each episode
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        # dict of dicts that stores stats per episode
        stats_episodes: Dict[
            Any, Any
        ] = {}

        # Init a tqdm progress bar
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)

        # Get the initial actions
        with inference_mode():
            action_data, _ = eval_agent.act(batch)
            action_data = create_dummy_student_actions(action_data)
            step_data = squash_step_action_data(action_data)

        # MAIN LOOP
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            pre_step_episodes_info = deepcopy(post_step_episodes_info)
            
            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            (
                observations,
                batch,
                skipped_episodes,
                post_step_episodes_info,
            ) = self._process_obs(
                config,
                observations,
                dones=dones,
            )
            eval_agent.on_envs_change(dones)
            
            # Predict actions for next step
            with inference_mode():      
                # Only get the integrated states from policy if we need them for visualization
                get_infos = 'top_down_map' in infos[0]
                action_data, commanded_states = eval_agent.act(
                    batch,
                    get_infos=get_infos,
                )
                # Simulator step interface expects student actions to be in the action_data
                action_data = create_dummy_student_actions(action_data)
                step_data = squash_step_action_data(action_data)

            # Increase the step count for each episode that is not done
            for i in range(len(dones)):
                pre_step_ep_id = self._gen_episode_key(pre_step_episodes_info[i])
                if not dones[i]:
                    scene_steps[pre_step_ep_id] += 1
                else:
                    logger.info(
                        f"Rank {self.world_rank}: Episode {pre_step_ep_id} is done, took {scene_steps[pre_step_ep_id]} steps")

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            n_envs = self.envs.num_envs

            if len(config.habitat_baselines.eval.video_option) > 0:
                viz_batch = create_viz_frames(self.config, batch)

            ( # Update episode eval counts and get the environments
              # that have cycled through evals_per_ep times      
                envs_to_pause,
                ep_eval_count,
            ) = self._get_envs_to_pause(
                dones,
                ep_eval_count,
                pre_step_episodes_info,
                post_step_episodes_info,
                skipped_episodes,
                config.habitat_baselines.eval.evals_per_ep,
            )

            for i in range(n_envs):

                pre_step_ep_id = self._gen_episode_key(pre_step_episodes_info[i])
                post_step_ep_id = self._gen_episode_key(post_step_episodes_info[i])

                # Visualize the episode
                if len(config.habitat_baselines.eval.video_option) > 0:
                    # Exclude the keys from `_rank0_keys` from displaying in the video
                    disp_info = {
                        k: v for k, v in infos[i].items() if (
                            (k not in self._rank0_keys)
                        )
                    }
                    # action_data also contains the dummy student actions so
                    # we need to remove them before adding to the video
                    viz_step_data = {k: v for k, v in action_data.items()
                                     if k in ['expert_angular_velocity', 'expert_linear_velocity']
                                    }
                    viz_step_data = squash_step_action_data(viz_step_data)
                    add_frames(
                        rgb_frames,
                        viz_batch,
                        disp_info,
                        dones,
                        viz_step_data,
                        commanded_states,
                        i,
                        pre_step_ep_id,
                        post_step_ep_id,
                        image_keys=['rgb', 'depth', 'subgoal_image', 'top_down_map'],
                    )

                # episode ended
                if dones[i]:
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    stats_episodes[(pre_step_ep_id, ep_eval_count[pre_step_ep_id])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}", #config.habitat_baselines.video_dir, 
                            images=rgb_frames[pre_step_ep_id],
                            episode_id=f"{pre_step_episodes_info[i].episode_id}_{ep_eval_count[pre_step_ep_id]}",
                            checkpoint_idx=0,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=None,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )
                        del rgb_frames[pre_step_ep_id]
            
            (
                batch,
                _,
                step_data,
                _,
                current_episode_reward,
                post_step_episodes_info,
            ) = self._pause_envs(
                envs_to_pause=envs_to_pause,
                batch=batch,
                actions=step_data,
                agents=[eval_agent],
                current_episode_reward=current_episode_reward,
                post_step_episodes_info=post_step_episodes_info
            )

        pbar.close()

        # Log the stats
        local_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            local_stats[stat_key] = [v[stat_key] for v in stats_episodes.values() if stat_key in v]

        # Aggregate the stats across all ranks
        aggregated_stats = aggregate_distributed_stats(local_stats, self.world_size, self.device)

        if self.world_rank == 0:
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.4f}")
        
        # Log the stats to wandb
        # Eval as part of training run
        if self.lightning_trainer is not None:
            wandb_stats = {k: aggregated_stats[k] for k in self.wandb_online_eval_log_keys if k in aggregated_stats}

            # Upload the stats to wandb
            if self.world_rank == 0:
                wandb.log(wandb_stats)
                
        self.envs.close()
