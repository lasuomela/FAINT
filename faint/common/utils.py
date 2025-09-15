import torch

def waypoint_to_velocity(
    waypoint: torch.Tensor,
    current_speed: torch.Tensor,
    dt: float,
    angular_error_threshold: float = torch.pi/4,
    max_linear_speed: float = 0.31,
    max_turn_speed: float = 1.9,
    smooth_acceleration: bool = True,
):
    """
    Given a waypoint, current position, current rotation, and current velocity,
    returns a new velocity and angular velocity to track the waypoint.

    Args:
        waypoint:
            The target waypoint [B, 2] to track in the local frame with x forward and y right.
        current_speed:
            The current speed [B,].
        current_angular_speed:
            The current angular speed [B,].
        dt:
            The time step.
        angular_error_threshold:
            The maximum angle error to consider the waypoint reached.
        max_linear_speed:
            The maximum linear speed.
        max_turn_speed:
            The maximum angular speed.
        smooth_acceleration:
            Whether to accelerate smoothly.
    """

    assert len(waypoint.shape) == 2

    # Compute the waypoint direction vector
    u_to_waypoint = waypoint / waypoint.norm(dim=(1), keepdim=True)

    # Create the forward vector in the local frame
    forward = torch.zeros_like(u_to_waypoint)
    forward[:, 0] = 1.0
    
    # Calculate the angle between the agent forward direction and the direction to the waypoint
    angle_error = torch.arccos(
        torch.clip(
            (
            forward * u_to_waypoint
            ).sum(dim=-1), # Batchwise dot product
            -1.0,
            1.0,
        )
    )

    if smooth_acceleration:
        condition = angle_error < angular_error_threshold
        new_speed = torch.where(
            condition,
            (current_speed + max_linear_speed) / 2.0,
            max_linear_speed,
        )
    else:
        condition = angle_error < angular_error_threshold
        new_speed = torch.where(
            condition,
            max_linear_speed,
            0.0,
        )

    # Calculate the rotation direction
    rot_dir = torch.where(
        (forward[:, 0] * u_to_waypoint[:, 1] - forward[:, 1] * u_to_waypoint[:, 0]) < 0,
        -1.0,
        1.0,
    )

    # Calculate the angular correction
    angular_correction = torch.where(
        angle_error > (max_turn_speed * 10.0 * dt),
        max_turn_speed * rot_dir,
        angle_error / 2.0 * rot_dir,
    )
    new_angular_speed = torch.clip(angular_correction, -max_turn_speed, max_turn_speed)
    return new_speed, new_angular_speed


def clamp_raw_velocity(
        v_raw: torch.Tensor,
        w_raw: torch.Tensor,
        max_lin_vel: float,
        max_ang_vel: float,
    ) -> torch.Tensor:
    """
    Convert the raw model output velocity to the linear and angular velocities
    restricted to the maximum values, while preserving the ratio of linear to angular velocity
    """

    # Clip negative linear velocities to zero
    v_raw = torch.max(v_raw, torch.zeros_like(v_raw))

    # Calculate the ratio of linear to angular velocity
    ratio = v_raw / abs(w_raw)

    # Check the proportion by which the linear and angular velocities exceed the maximum values
    v_excess = (v_raw - max_lin_vel) / max_lin_vel
    w_excess = (abs(w_raw) - max_ang_vel) / max_ang_vel

    # If the linear velocity exceeds the maximum, clamp it and adjust the angular velocity
    v_excess_greater_mask = ( v_excess > w_excess ) * ( v_excess > 0 )
    v_raw[ v_excess_greater_mask ] = max_lin_vel
    w_raw[ v_excess_greater_mask ] = max_lin_vel / ratio[ v_excess_greater_mask ] * torch.sign(w_raw[ v_excess_greater_mask ])

    # If the angular velocity exceeds the maximum, clamp it and adjust the linear velocity
    w_excess_greater_mask = ( w_excess > v_excess ) * ( w_excess > 0 )
    w_raw[ w_excess_greater_mask ] = max_ang_vel * torch.sign(w_raw[ w_excess_greater_mask ])
    v_raw[ w_excess_greater_mask ] = max_ang_vel * ratio[ w_excess_greater_mask ]

    # If the linear and angular velocities exceed the maximum values by the same proportion,
    # clamp both to the maximum values
    equal_excess_mask = ( v_excess == w_excess ) * ( v_excess > 0 )
    v_raw[ equal_excess_mask ] = max_lin_vel
    w_raw[ equal_excess_mask ] = max_ang_vel * torch.sign(w_raw[ equal_excess_mask ])

    return v_raw, w_raw