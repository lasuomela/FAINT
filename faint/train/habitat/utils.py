from typing import List, Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
from copy import deepcopy
from torch import Tensor
from gym import spaces

import torchvision.transforms.functional as TF

from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.common.baseline_registry import baseline_registry

def depth_to_normalized_openni(depth_img: Tensor):
    """
    Convert depth image in meters to OpenNI
    depth image format: [0, 2**16 - 1] in mm
    and normalize it to [0, 1].
    """
    # OpenNI format: [0, 2**16 - 1] in mm
    depth_img = depth_img * 1000

    depth_img = torch.clamp(depth_img, 0, 2**16 - 1)
    depth_img = depth_img.round(decimals=0)
    
    # Normalize to [0, 1]
    depth_img = depth_img / (2**16 - 1)
    return depth_img

def unnormalize_openni_depth(depth_img: Tensor):
    """
    Unnormalize OpenNI depth image to meters.

    Args:
        depth_img: OpenNI depth image tensor. shape: (B, H, W)
    """
    # Unnormalize to [0, 2**16 - 1]
    depth_img = depth_img * (2**16 - 1)

    # Convert to meters
    depth_img = depth_img / 1000

    # Normalize to [0, 1] by clamping to 10m
    depth_img = torch.clamp(depth_img, 0, 10)
    depth_img /= 10

    return depth_img

def crop_to_aspect_ratio(img: Tensor, aspect_ratio: float):
    """
    Crop the image to the desired aspect ratio.

    Args:
        img: Image tensor. shape: (B, C, H, W)
        aspect_ratio: Aspect ratio to crop the image to.
    """
    _, _, h, w = img.shape
    ratioed_h = int(w / aspect_ratio)
    ratioed_w = int(h * aspect_ratio)

    if w>=h:
        if ratioed_h <= h:
            size = (ratioed_h, w)
        else:
            size = (h, ratioed_w)
    else:
        if ratioed_w <= w:
            size = (h, ratioed_w)
        else:
            size = (ratioed_h, w)
    return TF.center_crop(img, size)

def preprocess_rgb_tensor(
        rgb_img: Tensor,
        normalize_mean: List[float],
        normalize_std: List[float],
        crop_aspect_ratio: float = None,
        image_size: Tuple[int, int] = None,
        ):
    """
    Preprocess RGB image tensor by cropping, resizing and normalizing.

    Args:
        rgb_img: RGB image tensor. shape: (B, H, W, 3)
        crop_aspect_ratio: Input image is cropped to this aspect ratio prior to resizing. (W / H)
        image_size: Target image size for resizing.
        normalize_mean: Mean values for normalization.
        normalize_std: Standard deviation values for normalization.
    """
    assert (len(rgb_img.shape) == 4) and (rgb_img.shape[-1] == 3) , "Input tensor must have shape (B, H, W, 3)"
    rgb_img = rgb_img.permute(0, 3, 1, 2)

    if crop_aspect_ratio is not None:
        rgb_img = crop_to_aspect_ratio(rgb_img, crop_aspect_ratio)

    if image_size is not None:
        rgb_img = TF.resize(rgb_img, image_size)

    rgb_img = rgb_img / 255.0
    rgb_img = TF.normalize(rgb_img, mean=normalize_mean, std=normalize_std)
    rgb_img = rgb_img.permute(0, 2, 3, 1)
    return rgb_img

def unnormalize_rgb_tensor(rgb_img: Tensor, normalize_mean: List[float], normalize_std: List[float]):
    """
    Unnormalize RGB image by reverse imagenet normalization.

    Args:
        rgb_img: RGB image tensor. shape: (B, H, W, 3)
    """
    assert (len(rgb_img.shape) == 4) and (rgb_img.shape[-1] == 3) , "Input tensor must have shape (B, H, W, 3)"

    # Switch to CHW format
    rgb_img = rgb_img.permute(0, 3, 1, 2)

    rgb_img = TF.normalize(rgb_img, mean=[0., 0., 0.], std=[1/std for std in normalize_std])
    rgb_img = TF.normalize(rgb_img, mean=[-mean for mean in normalize_mean], std=[1., 1., 1.])

    # Switch back to HWC format
    rgb_img = rgb_img.permute(0, 2, 3, 1)
    return rgb_img

class ExponentialBetaSchedule():
    """Exponentially decaying schedule for DAgger beta value."""

    def __init__(self, decay_probability: float):
        """Builds ExponentialBetaSchedule.

        Args:
            decay_probability: the decay factor for beta.

        Raises:
            ValueError: if `decay_probability` not within (0, 1].
        """
        if not (0 < decay_probability <= 1):
            raise ValueError("decay_probability lies outside the range (0, 1].")
        self.decay_probability = decay_probability

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta as `self.decay_probability ^ round_num`
        """
        assert round_num >= 0
        return self.decay_probability**round_num

def flatten_obs_space(obs_space):
    """
    Flatten a nested observation space to a flat space.
    """
    flat_space = spaces.Dict({})

    for k, v in obs_space.spaces.items():
        if isinstance(v, spaces.Dict):
            for k2, v2 in v.spaces.items():
                flat_space.spaces[k2] = v2
        else:
            flat_space.spaces[k] = v

    return flat_space

def flatten_batch(batch: Dict):
    '''
    Bring any nested dicts within batch to the top level of batch
    '''
    flat_batch = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat_batch[k2] = v2
        else:
            flat_batch[k] = v
    return flat_batch


def flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]

def get_active_obs_transforms(
    policy_config: "DictConfig", agent_name: str = None
) -> List[ObservationTransformer]:
    active_obs_transforms = []

    # When using observation transformations, we
    # assume for now that the observation space is shared among agents
    agent_name = list(policy_config)[0] #list(config.habitat_baselines.rl.policy.keys())[0]

    if hasattr(
        policy_config[agent_name], "obs_transforms"
    ):
        obs_trans_conf = policy_config[
        agent_name
        ].obs_transforms
        for obs_transform_config in obs_trans_conf.values():
            obs_trans_cls = baseline_registry.get_obs_transformer(
                obs_transform_config.type
            )
            if obs_trans_cls is None:
                raise ValueError(
                    f"Unkown ObservationTransform with name {obs_transform_config.type}."
                )
            obs_transform = obs_trans_cls.from_config(obs_transform_config)
            active_obs_transforms.append(obs_transform)
    return active_obs_transforms


def squash_step_action_data(
    step_data: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """
    Concatenate the ndarrays in the step_data dict
    (with keys corresponding to task action space components)
    to a list of ndarrays, with each list element corresponding to
    the action for a single environment.
    """
    # Habitat gym wrapper seems to rearrange the
    # action with keys in alphabetical order
    keys = list(step_data.keys())
    keys.sort()
    num_envs = len(step_data[list(step_data)[0]])

    # Concatenate the ndarrays element wise
    return [
        np.concatenate([step_data[k][i] for k in keys], axis=-1)
        for i in range(num_envs)
    ]

def create_dummy_student_actions(
        expert_actions: Dict[str, np.ndarray]):
    """
    Create dummy student actions by copying expert actions.
    """
    # Copy expert actions
    expert_actions = deepcopy(expert_actions)
    expert_actions['student_linear_velocity'] = np.zeros_like(expert_actions['expert_linear_velocity'])
    expert_actions['student_angular_velocity'] = np.zeros_like(expert_actions['expert_angular_velocity'])
    expert_actions['apply_student_action'] = np.zeros((len(expert_actions['expert_linear_velocity']),1), dtype=np.float32)
    return expert_actions