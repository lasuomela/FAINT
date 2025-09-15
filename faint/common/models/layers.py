import torch
import torch.nn.functional as F
import einops

class WaypointShaper(torch.nn.Module):
    """
    Converts model output into waypoint locations and orientations.
    """

    def __init__(
            self,
            num_action_params,
            len_trajectory_pred,
        ):
        super(WaypointShaper, self).__init__()
        self.num_action_params = num_action_params
        self.len_trajectory_pred = len_trajectory_pred

    def forward(
            self,
            action_pred,
        ):
        """
        Convert the action prediction into waypoints and normalize the angle prediction

        Args:
            action_pred (torch.Tensor): [B, action_pred_horizon * num_action_params]
                The action prediction from the model.

        Returns:
            torch.Tensor: [B, action_pred_horizon, num_action_params]
                The action prediction reshaped into a trajectory of waypoints.
        """

        action_pred = einops.rearrange(
                action_pred,
                "b (t p) -> b t p",
                t=self.len_trajectory_pred,
                p=self.num_action_params,
        )

        action_pred[..., :2] = torch.cumsum(
            action_pred[..., :2], dim=1
        )  # convert position deltas into waypoints

        action_pred[..., 2:] = F.normalize(
            action_pred[..., 2:].clone(), dim=-1
        )  # normalize the angle prediction
        return action_pred


class PredictionHead(torch.nn.Module):
    """
    FC layers for the prediction head.
    """
    def __init__(
            self,
            input_dim,
            output_layer_dims=[256, 128, 64, 32],
            dropout=0.2,
        ):
        super(PredictionHead, self).__init__()

        fc = []
        for i, (input_dim, output_dim) in enumerate(
            zip(
                [input_dim] + output_layer_dims[:-1],
                output_layer_dims
            )
            ):

            fc.append( torch.nn.Dropout(dropout))
            fc.append( torch.nn.Linear(input_dim, output_dim))
            if i < len(output_layer_dims) - 2:
                fc.append(torch.nn.ReLU())

        self.fc = torch.nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)