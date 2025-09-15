"""
Class for training a policy using PyTorch Lightning.
"""
from lightning import LightningModule
from omegaconf import DictConfig

import torch
import lightning as L
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

class LightningTrainer():

    def __init__(self, policy: LightningModule, baselines_config: DictConfig):
        self.demonstrations = None
        self.policy = policy
        self._config = baselines_config
        self._configure_trainer()

    def set_demonstrations(self, demonstrations: torch.utils.data.DataLoader):
        self.demonstrations = demonstrations

    def train_data_size(self):
        return self.demonstrations.train_data_size()

    def _configure_trainer(self):
        """
        Set up the Lightning trainer callbacks and logger.
        """

        self.logger = WandbLogger(
            project=self._config.wb.project_name,
            name=self._config.wb.run_name,
            entity=self._config.wb.entity,
            group=self._config.wb.group,
            save_dir='checkpoints',
        )

        # Specify callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=self._config.il.trainer.save_top_k,
            monitor="val_loss_epoch",
            mode="min",
            save_last=True,
            filename=f"{self.policy.__class__.__name__}"+"-epoch_{epoch:02d}-val_loss_{val_loss:.3f}",
            auto_insert_metric_name = False,
        )

        # Set max depth to 2 to show the modules in lightning wrapper .net
        model_summary = ModelSummary(max_depth=2)

        self.callbacks = [lr_monitor, checkpoint_callback, model_summary]


    def train(self, round_num: int):
        """
        Train self.policy on the self.demonstrations dataset.
        """
        # Make sure the policy is in training mode
        self.policy.train()
        
        num_epochs = (1 + round_num) * self._config.il.trainer.num_epochs_per_round
        trainer = L.Trainer(
            max_epochs=num_epochs,
            num_nodes=self._config.il.trainer.num_nodes,
            devices=self._config.il.trainer.num_devices,
            strategy="ddp",
            accelerator=self._config.il.trainer.accelerator,
            check_val_every_n_epoch=self._config.il.trainer.check_val_every_n_epoch,
            val_check_interval=self._config.il.trainer.val_check_interval,
            logger=self.logger,
            callbacks=self.callbacks,
            use_distributed_sampler=False, # Do this manually
            gradient_clip_val=self._config.il.trainer.gradient_clip_val,
        )

        trainer.fit(
            model=self.policy,
            datamodule=self.demonstrations,
            ckpt_path='last'
        )

    def export_checkpoints(self):
        '''
        Export all checkpoints to torchscript format for deployment
        '''
        project_name = self.logger.experiment.project_name()
        if not project_name:
            project_name = self._config.wb.project_name

        checkpoint_dir = Path(self.logger.save_dir) / project_name / self.logger.experiment.id / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))

        for checkpoint in checkpoints:
            policy = type(self.policy).load_from_checkpoint(checkpoint)
            policy.eval()
            print(f"Exporting TorchScript model to {checkpoint.with_suffix('.pt')}")
            policy.to_torchscript(file_path = checkpoint.with_suffix(".pt"), method="trace")





