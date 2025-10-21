"""
A class to load the Faint policy.
"""

from ..base_model import BaseModel
from faint_deployment.utils import get_image_transform, get_depth_image_transform
from faint_deployment.policies.goal_reaching.utils import pd_control, load_model

import torch

from rclpy.logging import get_logger
logger = get_logger("faint_policy")

class FaintPolicy(BaseModel):

    default_conf = {
    }
    required_inputs = ['obs', 'goal', 'max_lin_vel', 'max_ang_vel']

    def _init(self, conf, device):

        self.device = device
        self.net = load_model(conf, device, logger)

        if not 'seq_type' in self.conf:
            self.conf['seq_type'] = 'channel_stack'

        # Initialize an observation queue for the model
        self._obs_queue = []

        if conf['obs_type'] == 'depth':
            self.transform = get_depth_image_transform(conf['image_size'])
        elif conf['obs_type'] == 'rgb':
            self.transform = get_image_transform(
                conf['image_size'],
                conf['normalization_type'],
            )
        else:
            raise ValueError(f"Invalid observation type: {conf['obs_type']}")

    def _transform_image(self, image):
        if self.conf['obs_type'] == 'depth':
            image = self.transform(image=image)['image'].unsqueeze(0).to(self.device)
        elif self.conf['obs_type'] == 'rgb':
            image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def _forward(self, data):
        """
        Preprocess images, forward pass through the model,
        and turn the waypoints into velocities.
        """
        obs = self._transform_image(data['obs'])
        goal = self._transform_image(data['goal'])

        # Add the current observation to the observation queue
        if len(self._obs_queue) == 0:
            self._obs_queue = [obs] * self.conf['sequence_length']
        else:
            self._obs_queue.pop(0)
            self._obs_queue.append(obs)

        # Concatenate the observation sequence along the channel dimension
        # or stack the observations along the sequence dimension
        if self.conf['seq_type'] == 'channel_stack':
            # List[1, C, H, W] -> [1, C*seq_len, H, W]
            obs = torch.cat(self._obs_queue, dim=1)
        elif self.conf['seq_type'] == 'seq_dim':
            # List[1, C, H, W] -> [1, seq_len, C, H, W]
            obs = torch.stack(self._obs_queue, dim=1)

        # Forward pass through the model
        out = self.net(obs, goal)

        if self.conf['action_space'] == 'waypoints':
            waypoints = out[0].cpu().detach().numpy()
            if self.conf['normalize']:
                 # If trained with the GNM / ViNT datasets, the waypoint predictions have to be unnormalized
                 waypoints[:, :2] *= (data['max_lin_vel'] / data['main_loop_frequency'])

            chosen_waypoint = waypoints[self.conf['target_waypoint_idx']]
            v, w = pd_control(
                chosen_waypoint,
                1/data['main_loop_frequency'],
                data['max_lin_vel'],
                data['max_ang_vel'],
            )
        else:
            raise ValueError(f"Unknown action space: {self.conf['action_space']}")

        return {
            'v': v.item(),
            'w': w.item(),
            'raw': out,
            'waypoints': waypoints
        }