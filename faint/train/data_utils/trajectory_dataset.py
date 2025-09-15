"""
A PyTorch dataset for loading trajectories saved as HuggingFace datasets.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import datasets
import pathlib
import albumentations as A
import tqdm

from itertools import chain
from collections import defaultdict
datasets.disable_caching()

def group_observation_fields(
        traj_row: Dict[str, np.ndarray],
        obs_keys: List,
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Group the observation fields under key `obs`.
    """
    traj_row['obs'] = {key: traj_row[key] for key in obs_keys}
    for key in obs_keys:
        del traj_row[key]
    return traj_row

def get_image_keys(traj: datasets.Dataset) -> Tuple[List[str], List[str]]:
    """Get the keys of the depth and RGB images in the trajectory."""
    # Check which db keys are images
    image_keys = [key for key, val in 
                        traj.features.items()
                        if isinstance(val, datasets.features.image.Image)]
    
    # Separate the depth and RGB keys
    depth_image_keys = []
    rgb_image_keys = []
    for key in image_keys:
        # 16-bit depth images
        if traj[0][key].mode == 'I;16':
            depth_image_keys.append(key)
        # 8-bit RGB images
        elif traj[0][key].mode == 'RGB':
            rgb_image_keys.append(key)
        else:
            raise Exception(f"Unexpected image mode {traj[0][key].mode} for key {key}")
        
    return depth_image_keys, rgb_image_keys

def selector(key, values, depth_image_keys, rgb_image_keys):
    """Convert the values from the Huggingface dataset to numpy arrays."""
    if key in depth_image_keys:
        # 1. Convert the PIL images to numpy arrays
        # 2. Add a channel dimension to depth images (H, W) -> (H, W, 1)
        return [np.array(im, dtype=np.int32)[..., np.newaxis] for im in values]
    elif key in rgb_image_keys:
        return [np.array(im) for im in values]
    else:
        return np.asarray(values)

def get_numpy_transform(depth_image_keys, rgb_image_keys):
    def numpy_transform(batch):
        batch_np = {
            key: selector(key, val, depth_image_keys, rgb_image_keys)
            for key, val in batch.items()
        }
        return batch_np
    return numpy_transform
    
class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Present a collection of trajectories saved
    as HuggingFace datasets as a PyTorch dataset.
    """

    def __init__(
            self,
            trajectory_paths: List[pathlib.Path],
            depth_transform: A.Compose,
            rgb_transform: A.Compose,
            seq_len: int = 1,
            seq_stride: int = 1,
            pred_horizon_len: int = 1,
            mode="train",
            load_full_goal_sequence=False,
            debug_mode=False,
            rgb_transform_sequence: A.Compose = None,
        ):
        """
        Construct a TrajectoryDataset.

        Args:
            trajectory_paths: Paths to the saved trajectory shards.
            depth_transform: Albumentations transform for depth images.
            rgb_transform: Albumentations transform for RGB images.
            seq_len: Length of the sequence to sample.
            seq_stride: Stride to step over when sampling the sequence.
            pred_horizon_len: Number of future actions to load.
            mode: Usage mode of the dataset (e.g. "train", "val").
            load_full_goal_sequence: Whether to load the entire goal image sequence or the last image.
            debug_mode: Enables debug-specific checks when True.
            rgb_transform_sequence: Optional Albumentations transform for sequential RGB.
        """
        with open('/proc/sys/vm/max_map_count', 'r') as f:
            max_mmap = int(f.read())
        if len(trajectory_paths) - 1000 > max_mmap:
            raise OSError(f"""Size of dataset {len(trajectory_paths)} is close to max_map_count {max_mmap}.
                          Increase /proc/sys/vm/max_map_count or decrease the number of mmap calls.""")
        
        self._seq_len = seq_len
        self._seq_stride = seq_stride
        self._pred_horizon_len = pred_horizon_len
        self._mode = mode
        self._load_full_goal_sequence = load_full_goal_sequence
        self._debug_mode = debug_mode
        self._is_rank_0=(
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        # Set the image transforms
        self._depth_image_transform = depth_transform
        self._rgb_image_transform = rgb_transform
        self._rgb_image_transform_sequence = rgb_transform_sequence
        if self._rgb_image_transform_sequence is not None:
            self._setup_seq_tf_target_keys()
        if (self._depth_image_transform is None) and (self._rgb_image_transform is None):
            raise ValueError("At least one image transform is required")
        
        # Get the keys of image observations from the first trajectory
        first_traj = datasets.load_from_disk(str(trajectory_paths[0]), keep_in_memory=False)
        self._depth_image_keys, self._rgb_image_keys = get_image_keys(first_traj)
        self._obs_keys = self._rgb_image_keys + self._depth_image_keys + ["pos_cmds", "rot_cmds"]

        # Load the trajectories
        trajectories = self._load_trajectories(trajectory_paths)

        # Build the index for sampling sequences
        self._build_index(trajectories, seq_len, seq_stride, pred_horizon_len)

        # Prune some columns not needed for training
        trajectories = trajectories.select_columns(
            self._rgb_image_keys + self._depth_image_keys + ["pos_cmds", "rot_cmds", "trajectory_id", "idx"],
        )

        # Avoid python lists because of the multiprocessing refcount issue
        self._rgb_image_keys = np.array(self._rgb_image_keys, dtype=str)
        self._depth_image_keys = np.array(self._depth_image_keys, dtype=str)
        self._obs_keys = np.array(self._obs_keys, dtype=str)

        if not load_full_goal_sequence:
            self._trajectories = trajectories.remove_columns(["subgoal_image"])
            self.goals = trajectories.select_columns(["subgoal_image",])
        else:
            self._trajectories = trajectories
            
    def __len__(self) -> int:
        """
        Returns the number of training examples.
        """
        return self._len
    
    def raw_data_size(self) -> int:
        """
        Returns the length of the underlying raw dataset.
        """
        return len(self._trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:

        # For use with the TrajectoryBatchSampler
        if idx is None:
            return None

        db_idx = self._global_to_db_idx[idx]
        if not self._mode == "train":
            episode_idx, _ = self._global_to_local_idx[idx]
        else:
            episode_idx = 0

        # Extract the sequence
        seq_len = self._seq_len * self._seq_stride
        seq = self._trajectories[db_idx: db_idx + seq_len: self._seq_stride]

        if not self._load_full_goal_sequence:
            # Load the goal image corresponding to the last element of the sequence
            subgoal_image = self.goals[int(db_idx + seq_len -1),]
            seq.update(subgoal_image)

        # Group the observation fields under `obs`
        seq = group_observation_fields(seq, self._obs_keys)

        # Convert the commands to torch tensors
        for key in ["pos_cmds", "rot_cmds"]:
            seq['obs'][key] = torch.tensor(seq['obs'][key], dtype=torch.float32)

        # Apply transforms / augmentations
        if self._depth_image_keys.size != 0:
            seq = self._transform_depth_images(seq)

        if self._rgb_image_keys.size != 0:
            seq = self._transform_rgb_images(seq)

        return episode_idx, seq

    def _load_trajectories(self, trajectory_paths: List[pathlib.Path]) -> datasets.Dataset:
        """
        Memory map the trajectories from disk.
        """
        if self._is_rank_0:
            pbar = tqdm.tqdm(total=len(trajectory_paths),
                             desc=f"Memory-mapping {self._mode} dataset files",
            )
        trajectory_shards = []
        for path in trajectory_paths:
            if self._is_rank_0:
                pbar.update(1)
            shard = datasets.load_from_disk(str(path), keep_in_memory=False)
            trajectory_shards.append( shard )

        # Trick to reduce memory footprint in pytorch Dataloder
        trajectories = datasets.concatenate_datasets(trajectory_shards)
        trajectories._data = datasets.table.MemoryMappedTable(trajectories.data.table, '', [])

        # Format the loaded data to numpy arrays
        numpy_transform = get_numpy_transform(
            np.array(self._depth_image_keys, dtype=str),
            np.array(self._rgb_image_keys, dtype=str),
        )
        trajectories = trajectories.with_transform(numpy_transform)
        return trajectories

    def _build_index(
            self,
            trajectories: datasets.Dataset,
            seq_len: int,
            seq_stride: int,
            pred_horizon_len: int,
        ):
        """
        Build the index for loading samples from the trajectories
        saved as Huggingface dataset. This enables skipping first and last
        few trajectory elements, and ensures that all the images in a sampled
        sequence are from the same trajectory.
        """
        if self._is_rank_0:
            pbar = tqdm.tqdm(total=len(trajectories), desc=f"Getting trajectory lengths")
        traj_lens = defaultdict(int)

        for traj_id in trajectories['trajectory_id']:
            traj_lens[traj_id] += 1
            if self._is_rank_0:
                pbar.update(1)

        # An index to map each trajectory element to a 'global idx' 
        # while skipping the last seg_len * seq_stride elements + pred_horizon_len
        # of each trajectory
        end_skip = seq_len * seq_stride + pred_horizon_len
        cheker_lens = {}
        self._global_to_db_idx: List[int] = []
        for local_idx, (trajectory_id, element_idx) in enumerate(zip(trajectories['trajectory_id'], trajectories['idx'])):
            if element_idx <= (traj_lens[trajectory_id] - end_skip):
                self._global_to_db_idx.append(local_idx)
                cheker_lens[trajectory_id] = cheker_lens.get(trajectory_id, 0) + 1
        self._global_to_db_idx = torch.tensor(self._global_to_db_idx, dtype=torch.int64)
        self._len = len(self._global_to_db_idx)

        # Map the global idx -> (traj_idx, obs_idx)
        traj_ids = trajectories.unique('trajectory_id')
        not_matching = {}
        self._global_to_local_idx: List[Tuple[int, int]] = []
        for traj_idx, traj_id in enumerate(traj_ids):
            traj_idxs = [(traj_idx, obs_idx) for obs_idx in range(traj_lens[traj_id] - end_skip + 1)]
            
            if len(traj_idxs) > 0:
                if not len(traj_idxs) == cheker_lens[traj_id]:
                    not_matching[traj_id] = (len(traj_idxs), cheker_lens[traj_id])
                self._global_to_local_idx.extend(traj_idxs)

        if not_matching:
            print(f"Trajectory lengths not matching: {not_matching}")

        if not_matching:
            for key in not_matching.keys():
                matches = trajectories['trajectory_id'] == key
                raise Exception(trajectories['idx'][matches])
            
        assert len(self._global_to_db_idx) == len(self._global_to_local_idx), f"""
            Mismatch in the number of global indices: {len(self._global_to_db_idx)}
            and {len(self._global_to_local_idx)}. {len(traj_ids)} trajectories.
            Total db size {len(trajectories)}
            """

    def _transform_depth_images(self, seq):
        """
        Apply the depth image transformations.
        """
        for key in self._depth_image_keys:
            # Augment the images
            seq['obs'][key] = [ self._depth_image_transform(image=img)['image'] for img in seq['obs'][key] ]
            # Stack the sequence of images along a new dimension
            seq['obs'][key] = torch.stack(seq['obs'][key], axis=0)
        return seq

    def _transform_rgb_images(self, seq):
        """
        Apply the RGB image transformations.
        """

        # Transform with same random seed for all images under rgb_image_keys
        if self._rgb_image_transform_sequence is not None:

            # Pack all the images under rgb_image_keys to a single sequence
            target_imgs = chain.from_iterable([seq['obs'][key] for key in self._rgb_image_keys])

            # Construct the targets in the Albumentations format
            targets = {target_key: target_img for target_key, target_img
                        in zip(self._seq_tf_target_keys, target_imgs)}
            
            # Apply the same augmentation to all the images
            augmented = self._rgb_image_transform_sequence(**targets)

            # Unpack the augmented images
            rec_keys = []
            for i, key in enumerate(self._rgb_image_keys):
                target_keys = self._seq_tf_target_keys[self._seq_tf_index[i, 0]:self._seq_tf_index[i, 1]]
                rec_keys.append((target_keys, key))
                seq['obs'][key] = [augmented[target_key] for target_key in target_keys]

        # Transform with own random seed for each image under rgb_image_keys
        for key in self._rgb_image_keys:
            # # Augment the images
            seq['obs'][key] = [ self._rgb_image_transform(image=img)['image'] for img in seq['obs'][key] ]
            # Stack the sequence of images along a new dimension
            seq['obs'][key] = torch.stack(seq['obs'][key], axis=0)

        return seq
    
    def _setup_seq_tf_target_keys(self):
        """
        Construct the target keys for the sequence transform.
        """
        # Construct the target keys for the sequence transform
        self._seq_tf_target_keys = ["image"] + list(self._rgb_image_transform_sequence.additional_targets.keys()) 
        self._seq_tf_target_keys = np.array(self._seq_tf_target_keys, dtype=str)

        # Record the start and end indice for each key in the packed target sequence
        key_indices = []
        start_idx = 0
        for key in self._rgb_image_keys:
            if key != "subgoal_image":
                key_seq_len = self._seq_len
            else:
                key_seq_len = self._seq_len if self._load_full_goal_sequence else 1

            end_idx = start_idx + key_seq_len
            key_indices.append([start_idx, end_idx])
            start_idx = end_idx
        self._seq_tf_index = np.array(key_indices, dtype=np.int64)