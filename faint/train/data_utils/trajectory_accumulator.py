'''
Accumulates trajectories from observations collected from simulated environments.
'''
from typing import List, Any, Union, Mapping, Hashable

import collections
import numpy as np
import PIL
import time
import datasets
datasets.disable_progress_bars()
import torch.multiprocessing as mp

from pathlib import Path
from habitat import logger

import faint.train.data_utils.types as types
from faint.train.data_utils.types import Trajectory
from faint.train.data_utils.huggingface_utils import trajectory_to_dict, write_dataset_checksum
from faint.train.data_utils.utils import depth_image_to_openni

# Initialize disk writing queues
ctx = mp.get_context('spawn')
trajectory_write_queue = ctx.Queue()
chunk_write_queue = ctx.Queue()


def chunk_writer_loop(_chunk_write_queue: mp.Queue):
    """
    Loop that writes dataset chunks to disk.
    """
    try:
        while True:
            save_data = _chunk_write_queue.get()
            if save_data is None:
                break
            chunk_write_idx, save_dir, current_chunk = save_data
            write_chunk(chunk_write_idx, save_dir, current_chunk)

    except Exception as e:
        logger.info(f"TrajectoryAccumulator disk writer got an exception {e}")
        # Write the error to a file
        error_file = save_dir / f"chunk_writer_error.txt"
        with open(error_file, "w") as f:
            f.write(str(e))
        raise e

def write_chunk(
        chunk_write_idx: int,
        save_dir: Path,
        current_chunk: datasets.Dataset,
    ):
    """
    Write a single chunk to disk.
    """
    filename = f"chunk-{chunk_write_idx:04d}.npz"
    npz_path = save_dir / filename
    current_chunk.save_to_disk(
        str(npz_path),
        num_shards=1,
    )
    # Compute dataset checksum
    write_dataset_checksum(npz_path)

def buffering_disk_writer_loop(
        trajectory_queue: mp.Queue,
        chunk_queue: mp.Queue,
        save_dir: Path,
        chunk_write_size: int,
        round_num: int,
        rank: int,
        world_size: int,
        worker_idx: int,
        num_workers: int,
        image_encoding_format: str,
    ):
    """
    Aggregate trajectories into larger chunks
    and write them to disk.
    """
    current_chunk_size = 0
    current_chunk = []
    chunk_write_idx = rank + worker_idx * world_size

    while True:
        save_data = trajectory_queue.get()

        # A shutdown has been called in __del__
        if save_data is None:
            if current_chunk:
                current_chunk = datasets.concatenate_datasets(current_chunk)
                chunk_queue.put((chunk_write_idx, save_dir, current_chunk))
            break

        traj, traj_key = save_data        
        traj = trajectory_to_dict(traj)

        # Manually set format doesn't seem to survive pickling
        # for multiprocessing, so set it here 
        if 'rgb' in traj:
            for img in traj['rgb']:
                img.format = image_encoding_format
            for img in traj['subgoal_image']:
                img.format = image_encoding_format

        traj_save_key = str(traj_key)
        if round_num is not None:
            traj_save_key += f"-round_{round_num}"
        traj['trajectory_id'] = [traj_save_key] * traj['acts'].shape[0]
        traj['idx'] = np.arange(traj['acts'].shape[0], dtype=np.int32)

        ds = datasets.Dataset.from_dict(traj)
        current_chunk.append(ds)
        current_chunk_size += ds._estimate_nbytes()
        if current_chunk_size >= chunk_write_size:
            current_chunk = datasets.concatenate_datasets(current_chunk)
            chunk_queue.put((chunk_write_idx, save_dir, current_chunk))

            current_chunk = []
            current_chunk_size = 0
            chunk_write_idx += world_size * num_workers


class TrajectoryAccumulator:
    """
    Accumulates trajectories from environments and saves them to disk as Huggingface datasets.
    """

    obs_keys_to_save = [
        'rgb',
        'depth',
        'subgoal_image',
        'agent_position',
        'agent_rotation',
        'linear_vel_cmds',
        'angular_vel_cmds',
        'pos_cmds',
        'rot_cmds',
    ]

    def __init__(
        self,
        save = False,
        save_dir = None,
        round_num = None,
        num_workers = 1,
        trajectory_min_lenght = 1,
        rank = 0,
        world_size = 1,
        chunk_size = "1GB",
        rgb_encoding_format = 'JPEG',
        ):
        """
        Initialise the trajectory accumulator.
        """
        self.partial_trajectories = collections.defaultdict(list)
        self._written_trajectory_keys: List[str] = []
        self._previous_traj_keys: List[str] = []

        self._num_workers = num_workers
        self._rank = rank
        self._world_size = world_size
        self._chunk_write_idx = self._rank
        self._worker_lock = mp.Lock()

        assert self._world_size > 0
        
        self._trajectory_min_lenght = trajectory_min_lenght
        self._round_num = round_num

        self._unsuccessfuls = 0

        if save:
            assert save_dir is not None
            self._save = save
            if not save_dir.is_absolute():
                save_dir = Path.cwd() / save_dir
            self._save_dir = save_dir
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._rng = np.random.default_rng()
            chunk_write_size = datasets.py_utils.convert_file_size_to_int(chunk_size)

            # Initialize workers for writing the trajectories to disk
            self._queue_warn_length = 20
            self._num_workers = num_workers
            self.traj_processes = []
            self.write_processes = []

            # One worker pool for trajectory aggregation into chunks
            for i in range(self._num_workers):
                p = ctx.Process(
                    target=buffering_disk_writer_loop,
                    args=(
                        trajectory_write_queue,
                        chunk_write_queue,
                        save_dir,
                        chunk_write_size,
                        round_num,
                        rank,
                        world_size,
                        i,
                        num_workers,
                        rgb_encoding_format,
                    ),
                )
                p.start()
                self.traj_processes.append(p)

            # One worker pool for writing the chunks to disk
            for i in range(1):
                p = ctx.Process(target=chunk_writer_loop, args=(chunk_write_queue,))
                p.start()
                self.write_processes.append(p)
        else:
            self._save = False


    def __del__(self):
        if self._save:
            for _ in range(len(self.traj_processes)):
                trajectory_write_queue.put(None)

            for p in self.traj_processes:
                p.join()

            for _ in range(len(self.write_processes)):
                chunk_write_queue.put(None)

            for p in self.write_processes:
                p.join()

            logger.info(f"""
                Rank {self._rank} TrajectoryAccumulator finished with
                {self._unsuccessfuls} unsuccessful trajectories"""
            )

    def init_trajectories(
            self, 
            obs,
            ep_infos,
    ) -> None:
        """
        Initialise a new trajectory identified by `key`.

        Args:
            obs: list of observations from each environment
            ep_infos: list of info objects from the environment
        """
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)
        assert len(wrapped_obs) == len(ep_infos), f"Length of obs and ep_infos must be the same. Obs: {len(obs)}, ep_infos: {len(ep_infos)}"

        zip_iter = zip(wrapped_obs, ep_infos)
        traj_keys = []

        for ob, ep_info in zip_iter:
        
            traj_key = self.gen_traj_key(ep_info)
            if traj_key in self.partial_trajectories.keys():
                raise ValueError(f"Trajectory key {traj_key} already exists in partial_trajectories. Rank {self.world_rank}, num_envs {self.envs.num_envs}")

            step_dict = dict(
                obs=ob,
            )
            self.add_step(step_dict, traj_key)
            traj_keys.append(traj_key)

        self._previous_traj_keys = traj_keys

    def add_step(
        self,
        step_dict: Mapping[str, Union[types.Observation, Mapping[str, Any]]],
        key: Hashable = None,
    ) -> None:
        """
        Add a single step to the partial trajectory identified by `key`.

        Generally a single step could correspond to, e.g., one environment managed
        by a VecEnv.

        Args:
            step_dict: dictionary containing information for the current step. Its
                keys could include any (or all) attributes of a `TrajectoryWithRew`
                (e.g. "obs", "acts", etc.).
            key: key to uniquely identify the trajectory to append to, if working
                with multiple partial trajectories.
        """

        # Convert the image observations to PIL images
        # so that they are recognized as images by Huggingface datasets
        if 'obs' in step_dict:

            if 'depth' in step_dict['obs'] and 'rgb' in step_dict['obs']:
                raise ValueError("Both depth and rgb images are present in the observation, which is not allowed")
            
            if 'depth' in step_dict['obs']:
                # Convert depth observation and subgoal image to OpenNI standard
                for obs_key in ['depth', 'subgoal_image']:
                    obs_val = step_dict['obs'].get(obs_key)
                    obs_val = depth_image_to_openni(obs_val)
                    img = PIL.Image.fromarray(obs_val, mode='I;16')
                    step_dict['obs']._d[obs_key] = img

            if 'rgb' in step_dict['obs']:
                # Convert RGB observation to PIL image
                for obs_key in ['rgb', 'subgoal_image']:
                    obs_val = step_dict['obs'].get(obs_key)
                    img = PIL.Image.fromarray(obs_val)
                    step_dict['obs']._d[obs_key] = img

            for obs_key in list(step_dict['obs'].keys()):
                if obs_key not in self.obs_keys_to_save:
                    del step_dict['obs']._d[obs_key]
                
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(
        self,
        key: Hashable,
        terminal: bool,
    ) -> types.TrajectoryWithRew:
        """
        Complete the trajectory labelled with `key`.

        Args:
            key: key uniquely identifying which in-progress trajectory to remove.
            terminal: trajectory has naturally finished (i.e. includes terminal state).

        Returns:
            traj: list of completed trajectories popped from
                `self.partial_trajectories`.
        """
        assert key not in self._written_trajectory_keys, (
            f"""Trajectory key {key} already exists in written_trajectory_keys.
            If you want to save the same episode multiple times, add additional identifier to the key.
            Rank {self._rank}."""
        )
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        self._written_trajectory_keys.append(key)

        if ('obs' not in part_dicts[0]) and len(part_dicts) == 1:
            raise ValueError("Trajectory must have at least one observation")
        
        # Only save successful trajectories to improve data quality
        if not part_dicts[-1]['infos']['success']:
            return None

        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for k, array in part_dict.items():
                out_dict_unstacked[k].append(array)

        out_dict_stacked = {
            k: types.stack_maybe_dictobs(arr_list)
            for k, arr_list in out_dict_unstacked.items()
        }

        if 'infos' in out_dict_stacked:
            del out_dict_stacked['infos']

        if 'rews' in out_dict_stacked:
            del out_dict_stacked['rews']

        traj = types.Trajectory(**out_dict_stacked, infos=None, terminal=terminal)

        assert traj.acts.shape[0] == len(traj.obs)
        return traj

    def gen_traj_key(self, ep_info):
        return f"{Path(ep_info.scene_id).parts[-2]}_{ep_info.episode_id}"
    
    def on_envs_pause(self, envs_to_continue):
        """
        Remove the environments that are paused from the list of previous trajectory keys.
        """
        self._previous_traj_keys = [self._previous_traj_keys[i] for i in envs_to_continue]

    def add_step_and_finish(self, obs, action, reward, done, info, episodes_info) -> List[Trajectory]:
        '''
        Add a observations and infos from single step to the trajectory
        and add it to write queue if it is done.

        Args:
            obs: list of observations from each environment after action
            action: list of actions taken in each environment
            reward: list of rewards received from the environment after action
            done: list of done booleans indicating whether the episode terminated after action
            info: list of info objects from the environment after action
            episodes_info: list of info objects from the environment after action
        '''
        trajs: List[types.TrajectoryWithRew] = []
        traj_keys: List[str] = []
        save_keys: List[str] = []
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # Check that everything has the same length
        assert len(action) == len(wrapped_obs) == len(reward) == len(done) == len(info) == len(episodes_info) == len(self._previous_traj_keys), (
            f"Length of action, obs, reward, done, info, episodes_info, and self._previous_traj_keys must be the same."
            f"""
            Action: {len(action)},
            obs: {len(wrapped_obs)},
            reward: {len(reward)},
            done: {len(done)},
            info: {len(info)},
            episodes_info: {len(episodes_info)},
            previous_traj_keys: {len(self._previous_traj_keys)}
        """
        )

        zip_iter = enumerate(zip(action, wrapped_obs, reward, done, info, episodes_info))
        for i, (act, ob, rew, done, info, ep_info) in zip_iter:

            traj_key = self.gen_traj_key(ep_info)
            if done:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                # Current habitat baselines implementation
                # seems to drop the terminal observation
                # TODO: get terminal observation similar to StableBaselines3
                step_dict = dict(
                    acts=act,
                    rews=rew,
                    infos=info,
                )
                self.add_step(step_dict, self._previous_traj_keys[i])

                new_traj = self.finish_trajectory(self._previous_traj_keys[i], terminal=True)

                if new_traj is not None:
                    trajs.append(new_traj)
                    save_keys.append(self._previous_traj_keys[i])
                else:
                    # Self.finish_trajectory returns None for unsuccessful trajectories
                    self._unsuccessfuls += 1
                    logger.warn(f"Rank {self._rank} unsuccessful trajectories: {self._unsuccessfuls}")

                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                assert traj_key not in self.partial_trajectories.keys(), (
                    f"Trajectory key {traj_key} already exists in partial_trajectories. Len {len(self.partial_trajectories[traj_key])}. Rank {self._rank}."
                )
                step_dict = dict(
                    obs=ob,
                )
            else:
                step_dict = dict(
                    acts=act,
                    rews=rew,
                    obs=ob,
                    infos=info,
                )

            self.add_step(step_dict, traj_key)
            traj_keys.append(traj_key)

        self._previous_traj_keys = traj_keys

        # Add the finished trajectories to the disk write queue
        if self._save:
            for traj, traj_key in zip(trajs, save_keys):
                qsize = chunk_write_queue.qsize()
                if qsize > self._queue_warn_length:
                    logger.warn(
                        f'''Rank {self._rank} trajectory disk writer queue length {qsize}.
                        '''
                        )
                    time.sleep(qsize/5)
                    # Check if all the workers are alive
                    for p in self.write_processes:
                        if not p.is_alive():
                            raise RuntimeError("One of the write workers died")
                    for p in self.traj_processes:
                        if not p.is_alive():
                            raise RuntimeError("One of the trajectory processing workers died")

                if len(traj) >= self._trajectory_min_lenght:
                    trajectory_write_queue.put((traj, traj_key))

        return trajs