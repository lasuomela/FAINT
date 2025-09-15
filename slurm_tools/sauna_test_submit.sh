#!/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=disabled

OMP_NUM_THREADS=24 torchrun --nnodes=1 --nproc-per-node=1 \
    -m faint.train.run \
    --config-name=experiments/main_experiment/ViNT.yaml \
    habitat_baselines.il.trainer.num_devices=1 \
    habitat_baselines.il.trainer.batch_size=32 \
    habitat_baselines.il.data_collection.scratch_dir=dagger_test/release \
    habitat_baselines.il.data_collection.chunk_size=200MB \
    habitat_baselines.il.data_collection.num_episodes_per_round=30 \
    habitat_baselines.il.num_rounds=2 \
    habitat_baselines.test_episode_count=20 \
    habitat_baselines.il.trainer.num_epochs_per_round=1 \
    habitat.environment.iterator_options.max_scene_repeat_episodes=100