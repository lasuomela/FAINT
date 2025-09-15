import torch
import roma
from typing import Tuple

def shallow_copy_nested_dict(d):
    # Recursively shallow copy each dictionary in the nested structure
    if isinstance(d, dict):
        return {key: shallow_copy_nested_dict(value) for key, value in d.items()}
    return d  # If it's not a dict (e.g., a tensor), just return it

def to_ros_2d_coords(
    target_positions_local: torch.Tensor,
    target_orientations_local: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Change coordinate convention from Habitat to ROS coordinates.

    Args:
        target_positions: torch.Tensor, shape (B, N, 3)
        target_orientations: torch.Tensor, shape (B, N, 4)
            A quaternion in the form (w, x, y, z)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            target_positions_local: torch.Tensor, shape (B, N, 2)
                (x, y) coordinates with x as forward and y as left
            target_yaws: torch.Tensor, shape (B, N)
                Yaw angles in radians, right-handed rotation around the would-be z-axis
    """

    # Drop the y-coordinate (upward direction)
    # and swap the x and z coordinate places
    target_positions_local = target_positions_local[..., [2, 0]]

    # Habitat coordinate system has positive z-axis pointing backwards
    # So, negate the coordinates to have positive forward
    # Now we have (x, z) where x is forward and z is left
    target_positions_local *= -1

    # # Swap the x and z coordinate places: now we have (x, y) where x is forward and y is left
    # target_positions_local = target_positions_local[..., [1, 0]]

    # roma expects the quaternion in the form (x, y, z, w)
    target_orientations_local = roma.quat_wxyz_to_xyzw(target_orientations_local)
    _, target_yaws, _ = roma.unitquat_to_euler('xyz', target_orientations_local, as_tuple=True)
    target_yaws = target_yaws.unsqueeze(-1)

    return target_positions_local, target_yaws

def to_local_coords(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_orientation: torch.Tensor,
) -> torch.Tensor:
    """
    Change the global coordinates to the local coordinates of the robot.

    Args:
        positions: torch.Tensor, shape (B, N, 3)
        orientations: torch.Tensor, shape (B, N, 4)
            A quaternion in the form (w, x, y, z)
        curr_pos: torch.Tensor, shape (B, 3)
        curr_orientation: torch.Tensor, shape (B, 4)
            A quaternion in the form (w, x, y, z)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            target_positions_local: torch.Tensor, shape (B, N, 3)
            target_orientations_local: torch.Tensor, shape (B, N, 4)
                A quaternion in the form (w, x, y, z)
    """

    position_diffs = positions - curr_pos

    # Zero the y-coordinates
    position_diffs[:, :, 1] = 0

    # Remove the x and z direction rotation from the orientations
    orientations = roma.quat_wxyz_to_xyzw(orientations)
    orientations[:, :, [0, 2]] = 0
    orientations = roma.quat_normalize(orientations)

    curr_orientation = roma.quat_wxyz_to_xyzw(curr_orientation)
    curr_orientation[:, :, [0, 2]] = 0
    curr_orientation = roma.quat_normalize(curr_orientation)

    target_positions_local = roma.quat_action(
        roma.quat_conjugation(curr_orientation),
        position_diffs,
        is_normalized=True,
    )
    target_orientations_local = roma.quat_product(
        roma.quat_conjugation(curr_orientation),
        orientations,
    )
    target_orientations_local = roma.quat_xyzw_to_wxyz(
        target_orientations_local,
    )

    return target_positions_local, target_orientations_local