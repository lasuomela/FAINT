#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=disabled

CWD=$(pwd)
cd ..
OMP_NUM_THREADS=48 torchrun --nnodes=1 --nproc-per-node=2 \
    -m faint.train.run \
    --config-name=experiments/test/oracle.yaml \
    habitat.environment.max_episode_steps=500 \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval.eval_type=online \
    habitat_baselines.num_environments=16 \
    habitat_baselines.il.trainer.num_devices=2 \
    habitat_baselines.test_episode_count=2 \
    habitat_baselines.eval.video_option=['disk'] \
    habitat.task.lab_sensors.imagesubgoal.planner_safety_margin=0.2 \
    habitat_baselines.il.trainer.action_pred_horizon=1 \

cd $CWD