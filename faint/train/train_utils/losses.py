import torch

from faint.train.train_utils.utils import to_ros_2d_coords

class WaypointLoss(torch.nn.Module):

    """
    Loss function for the model output parametrization
    where output is the predicted waypoints in the robot coordinate system.
    """

    def __init__(self, loss_mode):
        super(WaypointLoss, self).__init__()
        self.loss_mode = loss_mode
    
    def __call__(
            self,
            output: torch.Tensor,
            target_positions: torch.Tensor,
            target_orientations: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            output: (B, prediction_horizon, 4)
                Predicted coordinates in robot coordinate system and orientation
                around the z-axis as sin(theta) and cos(theta)

            target_positions: (B, 1+prediction_horizon, 3)
                Global coordinates (Habitat coordinate system)
                First element is the current position

            target_orientations: (B, 1+prediction_horizon, 4)
                Global orientation in quaternion format (Habitat coordinate system)
                First element is the current orientation
        """

        # Go from Habitat to ROS coordinate convention
        target_positions_local, target_orientations_local = to_ros_2d_coords(
            target_positions,
            target_orientations,
        )

        # Concatenate the target positions and orientations
        target = torch.cat([
            target_positions_local,
            torch.cos(target_orientations_local),
            torch.sin(target_orientations_local)
            ], dim=-1)
        
        if self.loss_mode == 'last':
            target = target[:, -1]
            if len(output.shape) == 4:
                output = output[:,-1]

        # Check there are no NaN values in the output or target
        assert not torch.isnan(output).any(), f"Output contains NaN values: {output}"
        assert not torch.isnan(target).any(), f"Target contains NaN values: {target}"
        loss = torch.nn.functional.mse_loss(output, target)
        assert not torch.isnan(loss).any(), f"Loss contains NaN values: {loss}, output: {output}, target: {target}"

        return loss

# Legacy code below. Not used in the current implementation.
class RegressionLoss(torch.nn.Module):

    """
    Loss function for the model output parametrization
    where output is continuous linear and angular velocities.
    """

    def __init__(self, loss_mode):
        super(RegressionLoss, self).__init__()
        self.loss_mode = loss_mode
    
    def __call__(self, output: torch.Tensor, target_lin_vel: torch.Tensor, target_ang_vel: torch.Tensor):
        """
        Args:
            output: (B, seq_len, 2) or (B, 2)
            target_lin_vel: (B, seq_len, pred_horizon, 3)
            target_ang_vel: (B, seq_len, pred_horizon, 3)
        """
        if target_lin_vel.shape[-2] != 1:
            raise NotImplementedError("Multi step regression not implemented yet")
        
        target_lin_vel = target_lin_vel.view(target_lin_vel.shape[0], -1, target_lin_vel.shape[-1])
        target_ang_vel = target_ang_vel.view(target_ang_vel.shape[0], -1, target_ang_vel.shape[-1])

        # Forward linear velocity. Go from Habitat to ROS convention
        target_lin_vel = -target_lin_vel[..., 2]

        # Yaw is around the y-axis
        target_ang_vel = target_ang_vel[..., 1]
        target = torch.stack((target_lin_vel, target_ang_vel), dim=-1)

        if self.loss_mode == 'last':
            target = target[:,-1]
            if len(output.shape) == 3:
                output = output[:,-1]

        # Check there are no NaN values in the output or target
        assert not torch.isnan(output).any(), f"Output contains NaN values: {output}"
        assert not torch.isnan(target).any(), f"Target contains NaN values: {target}"

        loss = torch.nn.functional.mse_loss(output, target)
        assert not torch.isnan(loss).any(), f"Loss contains NaN values: {loss}, output: {output}, target: {target}"
        return loss