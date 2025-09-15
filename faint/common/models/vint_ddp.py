"""
A DDP and TorchScript friendly version of the original ViNT model from
https://github.com/robodhruv/visualnav-transformer
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# These imports require for the ViNT repo to be installed
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder


class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet_b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        channels: Optional[int] = 3,
        mode: Optional[str] = "action",
        pretrained_encoder: Optional[bool] = True,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner.

        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.channels = channels
        self.mode = mode

        # late fusion is not implemented yet with the timm encoders
        self.late_fusion = late_fusion
        if self.late_fusion:
            raise NotImplementedError("Late fusion is not implemented yet with the timm encoders")

        # Use timm to load the EfficientNet encoders
        # since it's ddp and TorchScript friendly
        self.obs_encoder = timm.create_model(
            obs_encoder,
            pretrained=pretrained_encoder,
            in_chans=channels,
            num_classes=0,
        )
        self.num_obs_features = self.obs_encoder.num_features

        self.goal_encoder = timm.create_model(
            obs_encoder,
            pretrained=pretrained_encoder,
            in_chans=2*channels,
            num_classes=0,
        )
        self.num_goal_features = self.goal_encoder.num_features
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        if self.mode == "action":
            self.action_predictor = nn.Sequential(
                nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            )
        elif self.mode == "distance":
            self.dist_predictor = nn.Sequential(
                nn.Linear(32, 1),
            )
        else:
            raise ValueError(f"Invalid mode '{self.mode}', must be either 'action' or 'distance'")

        # Remove the the decoder self-attention layer
        # as it's not used and therefore messes up DDP
        self.decoder.sa_layer = None

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        obsgoal_img = torch.cat(
            [
                obs_img[:, self.channels*self.context_size:, :, :],
                goal_img,
            ], dim=1)
        goal_encoding = self.goal_encoder(obsgoal_img)

        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size
        
        # split the observation into context based on the context size
        # image size is [batch_size, self.channels*self.context_size, H, W]
        obs_img = torch.split(obs_img, self.channels, dim=1)

        # image size is [batch_size*self.context_size, self.channels, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder(obs_img)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)
        # currently, the size is [batch_size, 32]

        if self.mode == "action":
            action_pred = self.action_predictor(final_repr)

            # augment outputs to match labels size-wise
            action_pred = action_pred.reshape(
                (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
            )
            action_pred[:, :, :2] = torch.cumsum(
                action_pred[:, :, :2], dim=1
            )  # convert position deltas into waypoints
            if self.learn_angle:
                action_pred[:, :, 2:] = F.normalize(
                    action_pred[:, :, 2:].clone(), dim=-1
                )  # normalize the angle prediction
            return action_pred
        
        elif self.mode == "distance":
            dist_pred = self.dist_predictor(final_repr)
            return dist_pred