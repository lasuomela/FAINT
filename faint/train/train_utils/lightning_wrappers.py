"""
Wrap Pytorch models in a Lightning module to use the Lightning Trainer API.
"""
from typing import Dict

import lightning as L
import timm
import torch
import einops

from torch import optim, Tensor
from hydra.utils import instantiate
from copy import deepcopy

from faint.common.models.vint_ddp import ViNT
from faint.common.models.faint import FAINT

from faint.train.train_utils.losses import RegressionLoss, WaypointLoss
from faint.train.config.toponav_registry import toponav_registry

class BaseWrapper(L.LightningModule):
    '''
    A base class to share some common functionality between models.
    '''
    def __init__(self, config, num_train_gpus: int = 1, channels: int = 3):
        super().__init__()
        self.save_hyperparameters()

        self._config = config
        self.num_train_gpus = num_train_gpus
        self.channels = channels
        self.obs_sensor_uuid = 'depth' if channels == 1 else 'rgb'
        self.output_type = config.output_type

        if config.output_type == 'continuous':
            # For continuous models, 'last' is the only supported loss mode
            self.loss = RegressionLoss(loss_mode=config.loss_mode)
            self.predictor_output_size = config.action_pred_horizon * 2

        elif config.output_type == 'waypoints':
            # Loss mode 'full' currently only implemented for RNN models
            self.loss = WaypointLoss(loss_mode=config.loss_mode)
            self.predictor_output_size = config.action_pred_horizon * 4
        else:
            raise ValueError(f'Unknown output type {config.output_type}')
    
    def configure_optimizers(self) -> optim.Optimizer:
        # Scale the base learning rate by number of GPU's
        optimizer = optim.AdamW(self.parameters(), lr=self._config.base_lr)

        schedulers = []
        schedule_milestones = []
        
        # Use the base learning rate for the first no_scheduler_epochs
        if self._config.no_scheduler_epochs > 0:
            schedulers += [
                torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=self._config.no_scheduler_epochs,
                ),
            ]
            schedule_milestones += [self._config.no_scheduler_epochs ]

        # Use the learning rate scheduler for the remaining epochs
        lr_scheduler_conf = deepcopy(self._config.lr_scheduler)
        schedulers += [ instantiate(lr_scheduler_conf, optimizer=optimizer) ]

        lr_schedule_composer = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers = schedulers,
            milestones = schedule_milestones,
        )
        return [optimizer], [{'scheduler': lr_schedule_composer}]
    
    def _compute_batch_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute the loss for a batch of data.

        Args:
            batch: Dict[str, Tensor]
                A dictionary containing the batch data
        Returns:
            loss: Tensor
                The computed loss
        """

        obs = batch["obs"][self.obs_sensor_uuid]
        goal = batch["obs"]["subgoal_image"]
        
        if self._config.loss_mode == 'last':
            # Pick the last goal image in the sequence
            goal = goal[:, -1]

        output = self._forward(obs, goal)
        
        if self._config.output_type == 'waypoints':
            target_positions = batch["obs"]['pos_cmds']
            target_rotations = batch["obs"]['rot_cmds']
            loss = self.loss(output, target_positions, target_rotations)
        else:
            linear_vel = batch["obs"]['linear_vel_cmds']
            angular_vel = batch["obs"]['angular_vel_cmds']
            loss = self.loss(output, linear_vel, angular_vel)
        
        return loss

class NonRecurrentModel(BaseWrapper):
    """
    Lightning wrapper base for non-recurrent models, such as those
    that utilize Transformers, MLPs or temporal convolutions for sequence modeling.
    """

    def __init__(self, config, num_train_gpus: int = 1, channels: int = 1):
        super(NonRecurrentModel, self).__init__(config, num_train_gpus, channels)

        if config.loss_mode == 'full':
            raise ValueError(f'Loss mode {config.loss_mode} not implented yet')

        # Define deployment mode inputs for conversion to TorchScript
        self.example_input_array = (
            torch.randn(1, config.sequence_length, channels, config.input_height, config.input_width),
            torch.randn(1, channels, config.input_height, config.input_width),
        )

    def forward(
            self,
            obs: Tensor,
            goal: Tensor,
        ) -> Tensor:
        """
        Args:
            obs: (B, S, C, H, W)
                Sequence of observation images
            goal: (B, C, H, W)
                Goal image corresponding to the last observation image.

        Returns:
            velocities: (B, 2)
                The predicted linear and angular velocities.
            or
            waypoints: (B, action_pred_horizon, 4)
                The predicted waypoints.
        """

        assert len(goal.shape) != 5, f"""
            Prediction with goal sequence not implemented yet.
            Expected shape (B, C, H, W), got {goal.shape}.
        """

        if self._config.output_type == 'continuous':
            out = self._forward(obs, goal)
            return out
        elif self._config.output_type == 'waypoints':
            out = self._forward(obs, goal)
            return out
        else:
            raise ValueError(f'Unknown output type {self._config.output_type}')


    def _forward(self, obs: Tensor, goal: Tensor) -> Tensor:
        """
        Args:
            obs: (B, S, C, H, W)
                Sequence of observation images
            goal: (B, C, H, W)
                Latest goal image

        Returns:
            velocities: (B, 2)
                The predicted linear and angular velocities
            or
            waypoints: (B, action_pred_horizon, 4)
                The predicted waypoints
        """
        preds = self.net.forward(obs, goal)
        return preds

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Lightning training step function. Computes the loss for a batch of data.
        """
        loss = self._compute_batch_loss(batch)
        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Lightning validation step function. Computes the loss for a batch of data.
        """
        loss = self._compute_batch_loss(batch)
        self.log(
            f"val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss
    
@toponav_registry.register_policy
class ViNTModel(NonRecurrentModel):
    """
    Lightning wrapper for the ViNT model.
    """

    def __init__(self, config, num_train_gpus: int = 1, channels: int = 1):
        super(ViNTModel, self).__init__(config, num_train_gpus, channels)

        if config.freeze_encoder:
            raise ValueError('Freezing the encoder not supported for ViNT!')

        # Create a ViNT model with parameters from 
        # https://github.com/robodhruv/visualnav-transformer/blob/main/train/config/vint.yaml
        self.net = ViNT(
            context_size=config.sequence_length-1,
            len_traj_pred=config.action_pred_horizon,
            learn_angle=True,
            obs_encoder=config.encoder, # Should be efficientnet_b0 for ViNT
            obs_encoding_size=config.encoding_dim, # Should be 512 for ViNT
            late_fusion=False,
            mha_num_attention_heads=4,
            mha_num_attention_layers=4,
            mha_ff_dim_factor=4,
            channels=channels,
            pretrained_encoder=config.pretrained_encoder,
        )

    def _forward(self, obs, goal):
        # The original ViNT model expects the observation sequence
        #  to be concatenated along the channel dimension
        obs = einops.rearrange(
            obs,
            'b s c h w -> b (s c) h w'
        )
        return super()._forward(obs, goal)

@toponav_registry.register_policy
class FAINTModel(NonRecurrentModel):
    """
    Lightning wrapper for the FAINT model.
    """

    def __init__(self, config, num_train_gpus: int = 1, channels: int = 1):
        super(FAINTModel, self).__init__(config, num_train_gpus, channels)

        self.net = FAINT(
            sequence_length=config.sequence_length,
            len_traj_pred=config.action_pred_horizon,
            input_size=[config.input_height, config.input_width],
            channels=channels,
            output_type=config.output_type,
            obs_encoder=config.encoder,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            compression_channels=config.compression_channels,
            compression_type=config.compression_type,
            obsgoal_fusion_type=config.obsgoal_fusion_type,
            obsgoal_fusion_num_attn_heads=config.obsgoal_fusion_num_attn_heads,
            obsgoal_fusion_num_attn_layers=config.obsgoal_fusion_num_attn_layers,
            obsgoal_fusion_ff_dim_factor=config.obsgoal_fusion_ff_dim_factor,
            seq_encoder_num_attn_heads=config.seq_encoder_num_attn_heads,
            seq_encoder_num_attn_layers=config.seq_encoder_num_attn_layers,
            seq_encoder_ff_dim_factor=config.seq_encoder_ff_dim_factor,
            seq_encoding_type=config.seq_encoding_type,
            seq_use_cls_token=config.seq_use_cls_token,
            seq_pos_enc_type=config.seq_pos_enc_type,
            prediction_head_layer_dims=config.prediction_head_layer_dims,
            prediction_head_dropout=config.prediction_head_dropout,
        )