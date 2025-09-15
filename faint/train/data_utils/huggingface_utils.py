"""
Helpers to convert between Trajectories and HuggingFace's datasets library.
Copied from Imitation: https://github.com/HumanCompatibleAI/imitation
"""
from typing import Any, Dict, Iterable, Optional, cast
from faint.train.data_utils import types

from pathlib import Path

import datasets
datasets.disable_progress_bars()
import jsonpickle
import json

def trajectory_to_dict(traj: types.Trajectory) -> Dict[str, Any]:
    """Convert a trajectory to a dict."""
    # Convert to dict
    trajectory_dict: Dict[str, Any] = dict(
        acts=traj.acts,
        infos=traj.infos if traj.infos is not None else [{}] * len(traj),
        terminal=[traj.terminal]*len(traj),
    )

    # If the observations are DictObs, bring them to the top level of
    # trajectory_dict
    if isinstance(traj.obs, types.DictObs):
        keys = traj.obs.keys()
        for key in keys:
            val = traj.obs.get(key)
            trajectory_dict[key] = val

    # Encode infos as jsonpickled strings
    trajectory_dict["infos"] = [
        jsonpickle.encode(info) for info in cast([Iterable[Dict]], trajectory_dict["infos"])
    ]
    # Add rewards if applicable
    if isinstance(traj, types.TrajectoryWithRew):
        trajectory_dict["rews"] = cast(types.TrajectoryWithRew, traj).rews

    # Check that all the values in trajectory_dict are iterable
    # (otherwise HuggingFace will complain)
    for key, val in trajectory_dict.items():
        if not isinstance(val, Iterable):
            raise ValueError(f"Value for key '{key}' is not iterable")

    return trajectory_dict

def trajectory_to_dataset(
    trajectory: types.Trajectory,
    info: Optional[datasets.DatasetInfo] = None,
) -> datasets.Dataset:
    """Convert a trajectory to a HuggingFace dataset."""
    ds = datasets.Dataset.from_dict( trajectory_to_dict(trajectory), info=info)
    return ds

def write_dataset_checksum(dataset_path: Path):
    """Compute Huggingface dataset checksum
    and write it to the state file.
    """
    with (dataset_path / 'state.json').open('r') as f:
        state = json.load(f)
    
    sizes = {}
    checksums = {}
    for file in state['_data_files']:
        size_cksum_dict = datasets.utils.info_utils.get_size_checksum_dict(
            str(dataset_path / file['filename'])
        )
        sizes[file['filename']] = size_cksum_dict['num_bytes']
        checksums[file['filename']] = size_cksum_dict['checksum']

    # Delete the old state file
    (dataset_path / 'state.json').unlink()

    state['num_bytes'] = sizes
    state['checksums'] = checksums
    with (dataset_path / 'state.json').open('w', encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)