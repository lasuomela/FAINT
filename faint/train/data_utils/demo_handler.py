"""
Class to manage file paths and loading of demonstrations for DAgger training.
"""
from typing import Optional, List, Tuple
from omegaconf import DictConfig

import os
import sysrsync
import torch

from pathlib import Path
from habitat import logger

from faint.train.data_utils.utils import check_round_copy_correctness
from faint.train.data_utils.data_module import HabitatImitationDataModule

class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""

class DemoHandler:

    def __init__(
            self,
            config: DictConfig,
            local_rank: int,
            world_rank: int,
            world_size: int = 1,
        ):

        self.config = config
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.world_size = world_size

        # Storage directory for the demonstrations
        self.scratch_dir = self.config.habitat_baselines.il.data_collection.scratch_dir
        if self.world_rank == 0:
            self.scratch_dir.mkdir(exist_ok=True, parents=True)

        # Optional fast temporary disk to place the demonstrations during training.
        # If specified, assume that each node has own fast storage disk.
        self.fast_tmp_dir = self._parse_fast_tmp_storage_path(
            self.config.habitat_baselines.il.data_collection.fast_tmp_dir
        )

        # Ugly but couldn't properly evaluate in the structured config
        self.load_full_goal_sequence = False
        if getattr(config.habitat_baselines.il.student_policy, 'loss_mode', '') == 'full':
            self.load_full_goal_sequence = True
        if getattr(config.habitat_baselines.il.student_policy, 'goal_fusion_stage', '') == 'pre_rnn':
            self.load_full_goal_sequence = True

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> Path:
        """
        Get the path to the directory containing demonstrations for the given round.
        """
        if round_num is None:            
            round_num = self.round_num if hasattr(self, "round_num") else None
        return self.scratch_dir / f"round-{round_num:03d}" if round_num is not None else Path()
    
    def demo_dir_path(self, round_num: int = None, split: str = "train") -> Path:
        """
        Get the path to the directory containing demonstrations for the given round and split.
        """
        if "train" in split:
            return self._demo_dir_path_for_round(round_num)
        elif "val" in split:
            return self.scratch_dir / "val"
        else:
            raise ValueError(f"Invalid split '{split}'")
        
    def has_demos(self, dir_path: Path) -> bool:
        """
        Check if the given directory contains demonstrations.
        """
        return dir_path.is_dir() and len(list(dir_path.glob("*.npz"))) > 0
        
    def _get_demo_paths(self, round_dir: Path) -> List[Path]:
        """
        Get the paths to the demonstrations in the given directory.
        """
        return sorted(round_dir.glob("*.npz"), key=lambda p: p.name)
    
    def _parse_fast_tmp_storage_path(self, fast_storage_path: str) -> Path:
        """
        Parse the path to the fast temporary storage directory.
        """
        if not fast_storage_path:
            return None
        
        # Parse any environment variables in the path
        fast_storage_path = os.path.expandvars(fast_storage_path)
        fast_storage_path = Path(fast_storage_path)

        # Check if the path exists and is a directory
        if not fast_storage_path.is_dir():
            raise ValueError(f"Fast storage path '{fast_storage_path}' is not a directory")
        return fast_storage_path
        
    def _sync_demos_to_fast_tmp_storage(self, round_demo_dir: Path, tmp_dir_name: Optional[str] = None) -> List[Path]:
        """
        Trasfer the demonstrations to a fast temporary storage directory.
        """
        if self.world_size > 1:
            # Do a sync here to make sure the distributed process group is properly set up.
            # Otherwise NCCL might time out at the post-copy barrier.
            torch.distributed.barrier()

        # Sync the demos to local fast storage if configured
        self.fast_tmp_dir.mkdir(exist_ok=True, parents=True)
        synced_demo_path = self.fast_tmp_dir / (round_demo_dir.name if tmp_dir_name is None else tmp_dir_name)

        # Only move data on worker 0 of each node
        if self.local_rank == 0 and not synced_demo_path.is_dir():
            logger.info(f"Rank worker {self.world_rank} syncing demos to fast storage: {synced_demo_path}")
            sysrsync.run(
                source=str(round_demo_dir),
                destination=str(synced_demo_path),
                sync_source_contents=True,
                options=['-r'],
            )
            os.sync()
            check_round_copy_correctness(round_demo_dir, synced_demo_path)

        # Wait for the rank 0 worker to finish syncing
        if self.world_size > 1:
            torch.distributed.barrier()

        # Return the paths to the synced demos
        return self._get_demo_paths(synced_demo_path)
    
    def try_load_demos(
            self,
            round_num: int,
            last_loaded_round: int,
            train=True,
            val=True,
        ) -> Tuple[HabitatImitationDataModule, int]:
        """
        Create a Lightning DataModule with demonstrations for the given round.
        """
        if train:
            # Train demonstrations
            train_demo_paths = []

            rounds = range(round_num + 1)
            for round in rounds:
                round_demo_dir = self._demo_dir_path_for_round(round)
                round_demo_paths = self._get_demo_paths(round_demo_dir) if round_demo_dir.is_dir() else []
                if len(round_demo_paths) == 0:
                    raise NeedsDemosException(
                        f"No demos found in dir '{round_demo_dir}'. Maybe you need to collect some demos?",
                    )
                
                # Sync the demos to local fast storage if configured
                if self.fast_tmp_dir is not None:
                    round_demo_paths = self._sync_demos_to_fast_tmp_storage(round_demo_dir)
                train_demo_paths.extend(round_demo_paths)
        else:
            train_demo_paths = []

        if val:
            # Val demonstrations
            val_demo_dir = self.demo_dir_path(split="val")
            val_demo_paths = self._get_demo_paths(val_demo_dir) if val_demo_dir.is_dir() else []
            if len(val_demo_paths) == 0:
                raise NeedsDemosException(
                    f"No demos found in dir '{val_demo_dir}'. Maybe you need to collect some demos?",
                )
            if self.fast_tmp_dir is not None:
                val_demo_paths = self._sync_demos_to_fast_tmp_storage(val_demo_dir)
        else:
            val_demo_paths = []
            
        if last_loaded_round < round_num:

            demonstrations = HabitatImitationDataModule(
                train_demo_paths,
                val_demo_paths,
                trainer_config=self.config.habitat_baselines.il.trainer,
                student_policy_config=self.config.habitat_baselines.il.student_policy,
                image_sensor_config=self.config.toponav.image_sensor,
                load_full_goal_sequence=self.load_full_goal_sequence,
            )
            return demonstrations, round_num
