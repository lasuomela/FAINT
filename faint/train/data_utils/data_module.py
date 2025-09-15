"""
A LightningDataModule for imitation learning demonstrations collected from Habitat.
"""
from typing import List, Dict

import torch
import lightning as L
import pathlib
import albumentations as A
import torchvision.transforms.v2 as T

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from habitat import logger

from faint.train.data_utils.trajectory_dataset import TrajectoryDataset
from faint.train.data_utils.utils import trajectory_batch_sample_collate_fn
from faint.common.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class HabitatImitationDataModule(L.LightningDataModule):

    rgb_normalize_mean = IMAGENET_STANDARD_MEAN
    rgb_normalize_std = IMAGENET_STANDARD_STD

    def __init__(
            self,
            train_demo_paths: List[pathlib.Path],
            val_demo_paths: List[pathlib.Path],
            trainer_config: Dict,
            student_policy_config: Dict,
            image_sensor_config: Dict,
            load_full_goal_sequence: bool,
        ):
        super().__init__()

        # Calculate batch size per device
        batch_size = int(trainer_config.batch_size / (trainer_config.num_devices * trainer_config.num_nodes))

        self.train_demo_paths = train_demo_paths
        self.val_demo_paths = val_demo_paths
        self.batch_size = batch_size
        self.test_batch_size = batch_size
        self.sequence_length = trainer_config.sequence_length
        self.sequence_stride = trainer_config.sequence_stride
        self.action_pred_horizon = trainer_config.action_pred_horizon
        self.val_sequence_length = trainer_config.sequence_length
        self.num_workers = trainer_config.num_workers
        self.load_full_goal_sequence = load_full_goal_sequence
        self.is_distributed = trainer_config.num_devices > 1
        self.gpu_augmentation = trainer_config.gpu_augmentation

        self.model_input_height = student_policy_config.input_height
        self.model_input_width = student_policy_config.input_width
        self.raw_image_height = image_sensor_config.height
        self.raw_image_width = image_sensor_config.width

        self.setup_depth_transforms()
        self.setup_rgb_transforms()

    def setup_depth_transforms(self):
        """
        Transformations for depth images.
        """

        # Depth transforms
        self.train_transform_depth = A.Compose([
            A.ToFloat(max_value=2**16-1),
            A.CoarseDropout(max_holes=50, max_height=12, max_width=12, min_holes=10, min_height=3, min_width=3, fill_value=0, p=1.0),
            A.CoarseDropout(max_holes=200, max_height=3, max_width=3, min_holes=100, min_height=1, min_width=1, fill_value=0, p=1.0),
            ToTensorV2()
        ])
        self.val_transform_depth = A.Compose([
            A.ToFloat(max_value=2**16-1),
            ToTensorV2()
        ])
        self.test_transform_depth = self.val_transform_depth
        
    def setup_rgb_transforms(self):
        """
        Transformations for RGB images.

        Application order:
        1. Sequence transform
        2. Individual image transform
        3. Transforms to run on GPU (if gpu_augmentation is enabled)
        """

        # Applied to the entire sequence with the same random seed
        transform_list = [] # No sequence transform for now
        if transform_list:
            num_sequence_transform_targets = 2 * self.sequence_length if self.load_full_goal_sequence else self.sequence_length + 1
            self.train_transform_rgb_sequence = A.Compose(
                transform_list,
                additional_targets={
                    f"image_{i}": "image" for i in range(1, num_sequence_transform_targets)
                }
            )
        else:
            self.train_transform_rgb_sequence = None

        # Applied to each image in the sequence individually (after the sequence transform)
        transform_list = [
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.05,
                p=1.0
            ),
            A.Posterize(
                num_bits=(4,7),
                p=0.3,
            ),
            A.Resize(
                self.model_input_height,
                self.model_input_width,
            ) if not self.gpu_augmentation else A.NoOp(),
            A.Normalize(
                mean=self.rgb_normalize_mean,
                std=self.rgb_normalize_std,
                max_pixel_value=255.0,
            ) if not self.gpu_augmentation else A.NoOp(),
            ToTensorV2(),
        ]
        self.train_transform_rgb_individual = A.Compose(transform_list)

        # Applied batch-wise on the GPU
        # Obs and goals will have different random seed,
        # but all observations within a batch will have the same seed.
        # Same for goals.
        if self.gpu_augmentation:
            transform_list = [
                T.Resize(
                    size=(self.model_input_height, self.model_input_width),
                ),
            ]

            # These transforms are applied prior to uint8 -> float32 conversion & normalization
            self.train_transform_rgb_gpu = T.Compose(transform_list)

            # Applied after uint8 -> float32 conversion
            self.train_transform_rgb_gpu_norm = T.Compose([
                T.Normalize(mean=self.rgb_normalize_mean, std=self.rgb_normalize_std),
            ])

        # Validation and test transforms
        transform_list = [
            A.Resize(
                self.model_input_height,
                self.model_input_width,
            ),
            A.Normalize(
                mean=self.rgb_normalize_mean,
                std=self.rgb_normalize_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
        self.val_transform_rgb = A.Compose(transform_list)
        self.test_transform_rgb = self.val_transform_rgb


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Define train and val datasets
            self.train_dataset = TrajectoryDataset(
                self.train_demo_paths,
                depth_transform=self.train_transform_depth,
                rgb_transform=self.train_transform_rgb_individual,
                rgb_transform_sequence=self.train_transform_rgb_sequence,
                seq_len=self.sequence_length,
                seq_stride=self.sequence_stride,
                pred_horizon_len=self.action_pred_horizon,
                mode="train",
                load_full_goal_sequence=self.load_full_goal_sequence,
            )
            
            num_demos = [len(self.train_dataset)]
            logger.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds",
            )
            if len(self.train_dataset) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(self.train_dataset)}",
                )
            
            self.val_dataset = TrajectoryDataset(
                self.val_demo_paths,
                depth_transform=self.val_transform_depth,
                rgb_transform=self.val_transform_rgb,
                seq_len=self.val_sequence_length,
                seq_stride=self.sequence_stride,
                pred_horizon_len=self.action_pred_horizon,
                mode="val",
                load_full_goal_sequence=self.load_full_goal_sequence,
            )

            if len(self.val_dataset) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(self.val_dataset)}",
                )
            
        if stage == "validate":
            self.val_dataset = TrajectoryDataset(
                self.val_demo_paths,
                depth_transform=self.val_transform_depth,
                rgb_transform=self.val_transform_rgb,
                seq_len=self.val_sequence_length,
                seq_stride=self.sequence_stride,
                pred_horizon_len=self.action_pred_horizon,
                mode="val",
                load_full_goal_sequence=self.load_full_goal_sequence,
            )            
        if stage == "test":
            self.test_dataset = TrajectoryDataset(
                self.val_demo_paths,
                depth_transform=self.test_transform_depth,
                rgb_transform=self.test_transform_rgb,
                seq_len=1,
                seq_stride=1,
                pred_horizon_len=self.action_pred_horizon,
                mode="test",
                load_full_goal_sequence=self.load_full_goal_sequence,
            )
        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented yet")
        
    def train_dataloader(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": trajectory_batch_sample_collate_fn,
            "drop_last": True,
            "pin_memory": True,
        }
        if self.is_distributed:
            kwargs['sampler'] = torch.utils.data.DistributedSampler(
                self.train_dataset,
                shuffle=True,
            )
        else:
            kwargs['shuffle'] = True

        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": trajectory_batch_sample_collate_fn,
            "drop_last": True,
            "pin_memory": True,
        }
        if self.is_distributed:
            kwargs['sampler'] = torch.utils.data.DistributedSampler(
                self.val_dataset,
                shuffle=False,
            )
        else:
            kwargs['shuffle'] = False
        

        return DataLoader(self.val_dataset, **kwargs)
        
    def train_data_size(self):
        training_samples = len(self.train_dataset)
        raw_data_size = self.train_dataset.raw_data_size()
        return training_samples, raw_data_size

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Apply GPU-side augmentation to the RGB observations if enabled.
        """

        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        if self.gpu_augmentation and isinstance(batch, dict):
            if self.trainer.training:
                for key in ['rgb', 'subgoal_image']:
                    # Apply gpu-side transforms
                    batch['obs'][key] = self.train_transform_rgb_gpu(batch['obs'][key])
                    batch['obs'][key] = batch['obs'][key].to(torch.float32) / 255.0
                    batch['obs'][key] = self.train_transform_rgb_gpu_norm(batch['obs'][key])
        return batch
