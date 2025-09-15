#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export GIT_PYTHON_REFRESH=quiet

OMP_NUM_THREADS=24 torchrun --nnodes=1 --nproc-per-node=2 \
    -m depthgoals.train.run \
    --config-name=val_experiments/vint.yaml \
    val_experiments/agents@agents=rgb_agent_base \
    habitat_baselines.evaluate=False \
    habitat_baselines.num_environments=16 \
    habitat_baselines.il.data_collection.recollect_train_demos=False \
    habitat_baselines.il.trainer.sequence_length=6 \
    habitat_baselines.il.trainer.num_devices=2 \
    habitat_baselines.il.trainer.num_nodes=1 \
    habitat_baselines.il.trainer.val_check_interval=1.0 \
    habitat_baselines.il.trainer.num_workers=6 \
    habitat_baselines.il.trainer.batch_size=64 \
    habitat_baselines.il.trainer.num_epochs_per_round=1 \
    habitat_baselines.il.data_collection.num_episodes_per_round=10000 \
    habitat_baselines.il.num_rounds=10 \
    habitat_baselines.test_episode_count=-1 \
    habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
    habitat_baselines.il.data_collection.scratch_dir=/home/sgn/Data1/lauri/dagger_rgb \
    habitat_baselines.il.trainer.action_pred_horizon=5 \
    habitat_baselines.il.student_policy.base_lr=5e-5 \
    depthgoals.augmentation.subgoal_sampling_strategy=random \
    habitat.task.lab_sensors.imagesubgoal.planner_safety_margin=0.2 \
    habitat.task.actions.se2_velocity_action.add_noise=False \
    habitat_baselines.il.student_policy.encoder=theaiinstitute/theia-tiny-patch16-224-cddsv \
    habitat_baselines.il.student_policy.output_type=waypoints \
    habitat_baselines.il.student_policy.input_width=224 \
    habitat_baselines.il.student_policy.input_height=224 \
    depthgoals.image_sensor.height=126 \
    depthgoals.image_sensor.width=224 \
    habitat_baselines.il.student_policy.encoding_dim=392 \
    habitat_baselines.il.student_policy.obs_goal_fusion_type=no_obs \




# 
    # habitat_baselines.il.trainer.gradient_clip_val=2.0 \



# OMP_NUM_THREADS=48 torchrun --nnodes=1 --nproc-per-node=2 \
#     -m depthgoals.train.run \
#     --config-name=val_experiments/cnn.yaml \
#     val_experiments/agents@agents=rgb_fisheye_agent_base \
#     habitat_baselines.evaluate=False \
#     habitat_baselines.num_environments=16 \
#     habitat_baselines.il.data_collection.recollect_train_demos=False \
#     habitat_baselines.il.trainer.sequence_length=6 \
#     habitat_baselines.il.trainer.num_devices=2 \
#     habitat_baselines.il.trainer.num_nodes=1 \
#     habitat_baselines.il.trainer.val_check_interval=1.0 \
#     habitat_baselines.il.trainer.num_workers=8 \
#     habitat_baselines.il.trainer.batch_size=128 \
#     habitat_baselines.il.trainer.num_epochs_per_round=1 \
#     habitat_baselines.il.data_collection.num_episodes_per_round=10000 \
#     habitat_baselines.il.num_rounds=10 \
#     habitat_baselines.test_episode_count=-1 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
#     habitat_baselines.il.data_collection.scratch_dir=/home/sgn/Data1/lauri/dagger_rgb/ \
#     habitat_baselines.il.student_policy.output_type=waypoints \
#     habitat_baselines.il.trainer.action_pred_horizon=5 \
#     habitat_baselines.il.student_policy.base_lr=1e-3 \
#     depthgoals.augmentation.subgoal_sampling_strategy=random \
#     habitat.task.lab_sensors.imagesubgoal.planner_safety_margin=0.2 \
#     depthgoals.augmentation.timestep_noise_multiplier=0.1 \

#     habitat_baselines.il.student_policy.checkpoint=/home/lauri/Documents/DepthGoals/checkpoints/DepthGoals/y3tzpdtn/checkpoints/last.ckpt \

#     # habitat_baselines.il.student_policy.checkpoint=/home/lauri/Documents/DepthGoals/checkpoints/DepthGoals/y3tzpdtn/checkpoints/CNNModel-epoch_04-val_loss_0.003.ckpt \


# OMP_NUM_THREADS=24 WANDB_MODE=disabled torchrun --nnodes=1 --nproc-per-node=1 \
#     -m depthgoals.train.run \
#     --config-name=val_experiments/rnn.yaml \
#     habitat_baselines.evaluate=False \
#     habitat_baselines.num_environments=10 \
#     habitat_baselines.il.data_collection.recollect_train_demos=False \
#     habitat_baselines.il.trainer.sequence_length=1 \
#     habitat_baselines.il.trainer.num_devices=1 \
#     habitat_baselines.il.trainer.num_nodes=1 \
#     habitat_baselines.il.trainer.val_check_interval=1.0 \
#     habitat_baselines.il.trainer.num_workers=4 \
#     habitat_baselines.il.trainer.batch_size=80 \
#     habitat_baselines.il.trainer.num_epochs_per_round=1 \
#     habitat_baselines.il.data_collection.num_episodes_per_round=50 \
#     habitat_baselines.il.num_rounds=1 \
#     habitat_baselines.test_episode_count=20 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=False \
#     habitat_baselines.il.data_collection.scratch_dir=dagger_test \
#     habitat_baselines.il.student_policy.goal_fusion_stage='pre_rnn' \
#     habitat_baselines.il.student_policy.num_lstm_layers=1 \
#     habitat_baselines.il.student_policy.type=RNNModel \
#     habitat_baselines.il.student_policy.rnn_type=GRU \
