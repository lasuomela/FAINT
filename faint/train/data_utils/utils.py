from typing import Dict, Sequence, Union
from faint.train.data_utils.types import AnyTensor

import numpy as np
import datasets
import json
from pathlib import Path
from torch.utils import data as th_data
import sysrsync

def check_chunk_correctness(
    original: Path,
    copy: Path,
) -> bool:
    """
    Check if the copy of a chunk is correct by comparing the checksums and sizes of the files in the original and copy.

    Args:
        original: The path to the original chunk. (.npz directory)
        copy: The path to the copy of the original chunk. (.npz directory)
    """

    if not copy.exists():
        return False

    # Load the original metadata
    original_metadata = original / 'state.json'
    with original_metadata.open('r') as f:
        original_metadata = json.load(f)
        if 'checksums' not in original_metadata:
            raise ValueError(f'No checksum found in {original_metadata}')
        original_checksums = original_metadata['checksums']

        if 'num_bytes' not in original_metadata:
            raise ValueError(f'No num_bytes found in {original_metadata}')
        original_sizes = original_metadata['num_bytes']
        original_arrow_files=original_metadata['_data_files']

    # Load the copy metadata
    copy_metadata = copy / 'state.json'
    with copy_metadata.open('r') as f:
        copy_metadata = json.load(f)
        if 'checksums' not in copy_metadata:
            raise ValueError(f'No checksum found in {copy_metadata}')
        copy_checksums = copy_metadata['checksums']

        if 'num_bytes' not in copy_metadata:
            raise ValueError(f'No num_bytes found in {copy_metadata}')
        copy_sizes = copy_metadata['num_bytes']
        copy_arrow_files=copy_metadata['_data_files']

    # Check if the checksums and sizes are the same to make sure there's no mixups
    if original_checksums != copy_checksums:
        return False
    if original_sizes != copy_sizes:
        return False
    if original_arrow_files != copy_arrow_files:
        return False
    
    # Check if the checksums and sizes of the files in the original and copy are the same
    for file in original_arrow_files:
        copy_info = datasets.utils.info_utils.get_size_checksum_dict(copy / file['filename'])
        copy_checksum = copy_info['checksum']
        copy_size = copy_info['num_bytes']

        if copy_checksum != original_checksums[file['filename']]:
            return False
        if copy_size != original_sizes[file['filename']]:
            return False
    return True

def check_round_copy_correctness(
    original: Path,
    copy: Path,
) -> None:
    original_files = list(original.glob('*.npz'))
    copy_files = list(copy.glob('*.npz'))

    # Check that original and copy both have the same files
    assert len(original_files) == len(copy_files)
    original_files = sorted(original_files)
    copy_files = sorted(copy_files)
    for original_file, copy_file in zip(original_files, copy_files):
        assert original_file.name == copy_file.name

    for original_file in original_files:
        copy_file = copy / original_file.name
        if not check_chunk_correctness(original_file, copy_file):
            print(f"Couldn't copy {original_file} to {copy_file} without errors, retryin once...")
            sysrsync.run(
                source=str(original),
                destination=str(copy),
                sync_source_contents=True,
                options=['-c', '-r'],
            )

            if not check_chunk_correctness(original_file, copy_file):
                raise ValueError(f"Couldn't copy {original_file} to {copy_file} without errors")
            print(f"Successfully copied {original_file} to {copy_file} after retrying!!")


def depth_image_to_openni(depth_image: np.ndarray) -> np.ndarray:
    """Converts a depth image to OpenNI format.

    Args:
        depth_image: A depth image in the format (H, W) with values in [0, inf], unit is meters.

    Returns:
        The depth image in OpenNI format, with values in [0, 2^16 - 1], unit is millimeters.
    """
    depth_image = depth_image * 1000  # Convert meters to millimeters
    depth_image = np.clip(depth_image, 0, 2 ** 16 - 1)
    depth_image = depth_image.round(decimals=0)
    return depth_image.astype(np.uint16)

def trajectory_batch_sample_collate_fn(
    batch: Sequence[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
) -> Dict[str, AnyTensor]:
    """
    Collate function for trajectory batch samples.
    """
    # sample[0] is the episode index, sample[1] is the observation
    episode_idxs = [sample[0] for sample in batch if sample is not None]
    batch = [sample[1] for sample in batch if sample is not None]

    batch_acts_and_dones = [
        {k: np.array(v) for k, v in sample.items() if k in ["acts", "dones"]}
        for sample in batch
    ]
    result = th_data.dataloader.default_collate(batch_acts_and_dones)
    assert isinstance(result, dict)

    if 'infos' in batch[0]:
        result["infos"] = [sample["infos"] for sample in batch]
        
    result["obs"] = th_data.dataloader.default_collate([sample["obs"] for sample in batch])
    result["episode_idx"] = th_data.dataloader.default_collate(episode_idxs)
    return result

def get_aspect_ratio_crop_size(in_height, in_width, out_aspect_ratio):
    """
    Get the crop size for a given aspect ratio.
    """
    ratioed_h = int(in_width / out_aspect_ratio)
    ratioed_w = int(in_height * out_aspect_ratio)

    if in_width>=in_height:
        if ratioed_h <= in_height:
            size = (ratioed_h, in_width)
        else:
            size = (in_height, ratioed_w)
    else:
        if ratioed_w <= in_width:
            size = (in_height, ratioed_w)
        else:
            size = (ratioed_h, in_width)
    return size