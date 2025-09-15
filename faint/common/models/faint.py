"""
A PyTorch implementation of the FAINT model.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import einops

from faint.common.models.layers import (
    WaypointShaper,
    PredictionHead,
)
from faint.common.models.faint_model.layers import (
    TimmEncoder,
    TheiaEncoder,
    CompressionLayer,
    ObsGoalFuser,
    SeqEncoder,
)

class FAINT(nn.Module):
    """
    Fast, Appearance-Invariant Navigation Transformer (FAINT).

    The model predicts robot navigation trajectory given a sequence of observations and a goal image.
    Image encoding with a pre-trained, frozen transformer enables sim2real transfer and invariance to changing
    deployment conditions such as illumination change.
    
    The model is composed of the following components:
    - Encoder: Encodes the observation and goal images.
    - ObsGoalFuser: Conditions the goal encoding on the latest observation encoding.
    - Compression Layer: Compresses the encodings into 1D vectors.
    - Sequence Encoder: Transformer encoder for the sequence of observations and goal.
    - Prediction Head: MLP that predicts the future waypoints from the sequence encoding.
    - Waypoint Shaper: Reshapes the output of the prediction head into waypoints.
    """

    def __init__(
        self,
        sequence_length: int = 6,
        len_traj_pred: Optional[int] = 5,
        input_size: Optional[List[int]] = [224, 224], # H, W
        channels: Optional[int] = 3,
        output_type: Optional[str] = "waypoints",
        obs_encoder: Optional[str] = "theaiinstitute/theia-tiny-patch16-224-cddsv",
        pretrained_encoder: Optional[bool] = True,
        freeze_encoder: Optional[bool] = True,
        compression_channels: Optional[int] = 2,
        compression_type: Optional[str] = "flatten",
        obsgoal_fusion_type: Optional[str] = "CrossBlock",
        obsgoal_fusion_num_attn_heads: Optional[int] = 4,
        obsgoal_fusion_num_attn_layers: Optional[int] = 4,
        obsgoal_fusion_ff_dim_factor: Optional[int] = 2,
        seq_encoder_num_attn_heads: Optional[int] = 4,
        seq_encoder_num_attn_layers: Optional[int] = 4,
        seq_encoder_ff_dim_factor: Optional[int] = 2,
        seq_encoding_type: Optional[str] = "cls",
        seq_use_cls_token: Optional[bool] = True,
        seq_pos_enc_type: Optional[str] = "learned",
        prediction_head_layer_dims: Optional[List[int]] = [256, 128, 64, 32],
        prediction_head_dropout: Optional[float] = 0.2,
    ) -> None:
        """
        Initialize the FAINT model.

        Args:
            sequence_length (int): Number of most recent observations in the input sequence.
            len_traj_pred (Optional[int]): Number of waypoints to predict.
            input_size (Optional[List[int]]): Size of the input images (H, W).
            channels (Optional[int]): Number of channels in the input images (RGB: 3 | Depth: 1).
            output_type (Optional[str]): Type of output, either 'waypoints'.
            obs_encoder (Optional[str]): Name of the image encoder model.
            pretrained_encoder (Optional[bool]): Whether to use a pretrained encoder.
            freeze_encoder (Optional[bool]): Whether to freeze the encoder weights duting training.
            compression_channels (Optional[int]): Number of channels to allocate per patch when using 'flatten' compression.
            compression_type (Optional[str]): Type of compression to use ('flatten' | 'mean').
            obsgoal_fusion_type (Optional[str]): Type of fusion block for observation and goal ('CrossBlock' | 'CatBlock' | 'ConvBlock' | 'EarlyConv' | 'none').
            obsgoal_fusion_num_attn_heads (Optional[int]): Number of attention heads in the fusion block.
            obsgoal_fusion_num_attn_layers (Optional[int]): Number of attention layers in the fusion block.
            obsgoal_fusion_ff_dim_factor (Optional[int]): Feed-forward dimension multiply factor in the fusion block.
            seq_encoder_num_attn_heads (Optional[int]): Number of attention heads in the sequence encoder.
            seq_encoder_num_attn_layers (Optional[int]): Number of attention layers in the sequence encoder.
            seq_encoder_ff_dim_factor (Optional[int]): Feed-forward dimension multiply factor in the sequence encoder.
            seq_encoding_type (Optional[str]): Which tokens to output from the sequence encoder. ('cls' | 'sequence').
            seq_use_cls_token (Optional[bool]): Whether to use a CLS token in the sequence encoder.
            seq_pos_enc_type (Optional[str]): Type of positional encoding in the sequence encoder ('learned' | 'sinusoidal').
            prediction_head_layer_dims (Optional[List[int]]): Dimensions of the layers in the prediction head.
            prediction_head_dropout (Optional[float]): Dropout rate in the prediction head.
        """
        super(FAINT, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.output_type = output_type
        self.sequence_length = sequence_length
        self.len_trajectory_pred = len_traj_pred
        self.obsgoal_fusion_type = obsgoal_fusion_type

        if "theia" in obs_encoder:
            self.obs_encoder = TheiaEncoder(
                obs_encoder,
                pretrained=pretrained_encoder,
                freeze=freeze_encoder,
                input_size=input_size,
            )
            self.goal_encoder = self.obs_encoder
        else:
            self.obs_encoder = TimmEncoder(
                obs_encoder,
                pretrained=pretrained_encoder,
                freeze=freeze_encoder,
                input_size=input_size,
            )
            if obsgoal_fusion_type != 'EarlyConv':
                self.goal_encoder = self.obs_encoder
            else:
                self.goal_encoder = TimmEncoder(
                    obs_encoder,
                    pretrained=pretrained_encoder,
                    freeze=freeze_encoder,
                    input_size=input_size,
                    num_channels=channels * 2,
                )

        self.obs_compressor = CompressionLayer(
            input_dim=self.obs_encoder.encoding_dim,
            input_size=self.obs_encoder.encoding_size,
            compression_channels=compression_channels,
            compression_type=compression_type,
        )

        self.obs_goal_fuser = ObsGoalFuser(
            input_dim=self.obs_encoder.encoding_dim,
            input_size=self.obs_encoder.encoding_size,
            compression_channels=compression_channels,
            compression_type=compression_type,
            fusion_block_type=obsgoal_fusion_type,
            num_layers=obsgoal_fusion_num_attn_layers,
            nhead=obsgoal_fusion_num_attn_heads,
            ff_dim_factor=obsgoal_fusion_ff_dim_factor,
        )

        self.seq_encoder = SeqEncoder(
            self.sequence_length,
            self.obs_compressor.output_dim,
            num_layers=seq_encoder_num_attn_layers,
            nhead=seq_encoder_num_attn_heads,
            ff_dim_factor=seq_encoder_ff_dim_factor,
            output_type=seq_encoding_type,
            use_cls_token=seq_use_cls_token,
            pos_enc_type=seq_pos_enc_type,
        )

        self.prediction_head = PredictionHead(
            input_dim=self.seq_encoder.encoding_dim,
            output_layer_dims=prediction_head_layer_dims,
            dropout=prediction_head_dropout,
        )

        if output_type == "waypoints":
            # Predict relative x, y, sin(θ), cos(θ)
            self.num_action_params = 4
            output_dim = self.len_trajectory_pred * self.num_action_params

            self.output_shaper = WaypointShaper(
                num_action_params=self.num_action_params,
                len_trajectory_pred=self.len_trajectory_pred,
            )
        else:
            raise ValueError(f"Invalid output_type '{self.output_type}', must be either 'waypoints'")

        # Projects the final representation from prediction head to the output shaper dimension
        self.output_projection = nn.Linear(prediction_head_layer_dims[-1], output_dim)


    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the FAINT model.

        Args:
            obs_img: [B, S, C, H, W] tensor of observations
            goal_img: [B, C, H, W] tensor of goal image

        Returns:
            [B, len_traj_pred, num_action_params] tensor of predicted waypoints
        """
        
        if self.obsgoal_fusion_type == 'EarlyConv':
            latest_obs_img = obs_img[:, -1]

        batch_size = obs_img.shape[0]
        obs_img = einops.rearrange(
            obs_img,
            "b s c h w -> (b s) c h w",
        )

        # Encode the observations
        # [B * S, C, H, W] -> [B * S, P, E]
        obs_encoding = self.obs_encoder(obs_img)

        # Pick the newest observation of each batch
        # [B, S, P, E] -> [B, P, E]
        latest_obs_encoding = einops.rearrange(
            obs_encoding,
            "(b s) p c -> b s p c",
            b=batch_size,
        )[:, -1]

        # Compress the observation encoding
        # [B * S, P, C] -> [B * S, obs_encoding_size]
        obs_encoding = self.obs_compressor(obs_encoding)

        obs_encoding = einops.rearrange(
            obs_encoding,
            "(b s) c -> b s c",
            b=batch_size,
        )

        # Encode the goal image
        if self.obsgoal_fusion_type == 'EarlyConv':
            goal_img = torch.concat([latest_obs_img, goal_img], dim=1)

        # [B, C, H, W] -> [B, P, E]
        goal_encoding = self.goal_encoder(goal_img)

        # [B, P, E], [B, P, E] -> [B, E]
        goal_encoding = self.obs_goal_fuser(latest_obs_encoding, goal_encoding)

        # [B, S, E], [B, E] -> [B, E]
        seq_encoding = self.seq_encoder(obs_encoding, goal_encoding)

        # [B, E] -> [B, 32]
        final_repr = self.prediction_head(seq_encoding)

        # [B, 32] -> [B, output_dim]
        output = self.output_projection(final_repr)

        # [B, output_dim] -> [B, len_traj_pred, num_action_params] | [B, 1]
        return self.output_shaper(output)

if __name__ == "__main__":
    for enc in ["theaiinstitute/theia-tiny-patch16-224-cddsv", "efficientnet_b0"]:
        for block in ["CrossBlock", "CatBlock", "ConvBlock", "none"]:
            for seq_type in ["cls", "sequence"]:
                model = FAINT(
                    obs_encoder=enc,
                    obsgoal_fusion_type=block,
                    seq_encoding_type=seq_type,
                )
                model.eval()

                obs_img = torch.randint(255, (1, 6, 3, 224, 224), dtype=torch.uint8).to(torch.float32) / 255
                goal_img = torch.randint(255, (1, 3, 224, 224), dtype=torch.uint8).to(torch.float32) / 255
                action_pred = model(obs_img, goal_img)
