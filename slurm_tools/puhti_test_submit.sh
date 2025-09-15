#!/bin/bash
#SBATCH --job-name=faint_test
#SBATCH --output=test_log.out
#SBATCH --error=test_log.err
#SBATCH --account=project_2010179
#SBATCH --time=00:15:00
#SBATCH --mem=180G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1,nvme:1000
#SBATCH --partition=gputest

source ~/.bashrc
module load cuda
export PATH="/projappl/project_2010179/py310/bin:$PATH" 
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE="disabled"

srun python -u \
    -m faint.train.run \
    --config-name=experiments/main_experiment/FAINT_10x.yaml \
    habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
    habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    habitat_baselines.il.data_collection.scratch_dir=dagger_test/26571072 \
    habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \
    habitat_baselines.il.data_collection.num_episodes_per_round=100 \
    habitat_baselines.il.num_rounds=1 \
    habitat_baselines.test_episode_count=20 \
    habitat_baselines.il.trainer.batch_size=64 \
    habitat_baselines.verbose=True


#$SLURM_JOBID \

# srun python -u \
#     -m depthgoals.train.run \
#     --config-name=val_experiments/vint.yaml \
#     val_experiments/agents@agents=rgb_agent_base \
#     habitat_baselines.evaluate=False \
#     habitat_baselines.num_environments=16 \
#     habitat_baselines.il.data_collection.recollect_train_demos=False \
#     habitat_baselines.il.trainer.sequence_length=6 \
#     habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
#     habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
#     habitat_baselines.il.trainer.val_check_interval=1.0 \
#     habitat_baselines.il.trainer.num_workers=6 \
#     habitat_baselines.il.trainer.batch_size=64 \
#     habitat_baselines.il.trainer.num_epochs_per_round=1 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
#     habitat_baselines.il.data_collection.num_episodes_per_round=200 \
#     habitat_baselines.il.num_rounds=1 \
#     habitat_baselines.test_episode_count=-1 \
#     habitat_baselines.il.student_policy.output_type=waypoints \
#     habitat_baselines.il.student_policy.base_lr=1e-3 \
#     habitat_baselines.il.data_collection.scratch_dir=dagger_test/pytest \
#     habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \
#     habitat_baselines.il.data_collection.save_demonstrations=True \
#     habitat.task.lab_sensors.imagesubgoal.planner_safety_margin=0.2 \
#     habitat.task.actions.se2_velocity_action.add_noise=False \
#     habitat_baselines.il.trainer.action_pred_horizon=5 \
#     habitat_baselines.il.student_policy.encoder=theaiinstitute/theia-tiny-patch16-224-cddsv \
#     habitat_baselines.il.student_policy.encoding_dim=784 \
#     habitat_baselines.il.student_policy.input_width=224 \
#     habitat_baselines.il.student_policy.input_height=224 \
#     depthgoals.image_sensor.height=126 \
#     depthgoals.image_sensor.width=224 \
#     habitat_baselines.il.data_collection.shard_size=200MB \




    # dagger_test/dagger_test_rgb \


# srun python -u \
#     -m depthgoals.train.run \
#     --config-name=val_experiments/cnn.yaml \
#     habitat_baselines.evaluate=False \
#     habitat_baselines.num_environments=8 \
#     habitat_baselines.il.data_collection.recollect_train_demos=False \
#     habitat_baselines.il.trainer.sequence_length=5 \
#     habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
#     habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
#     habitat_baselines.il.trainer.val_check_interval=1.0 \
#     habitat_baselines.il.trainer.num_workers=4 \
#     habitat_baselines.il.trainer.batch_size=256 \
#     habitat_baselines.il.trainer.num_epochs_per_round=1 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
#     habitat_baselines.il.data_collection.num_episodes_per_round=50 \
#     habitat_baselines.il.num_rounds=2 \
#     habitat_baselines.test_episode_count=20 \
#     habitat_baselines.il.student_policy.output_type=continuous \
#     habitat_baselines.il.student_policy.base_lr=1e-5 \
#     habitat_baselines.il.data_collection.scratch_dir=dagger_test \
#     habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \


# srun python -u \
#     -m depthgoals.train.run \
#     --config-name=val_experiments/rnn.yaml \
#     val_experiments/agents@agents=depth_agent_base \
#     habitat_baselines.evaluate=False \
#     habitat_baselines.num_environments=16 \
#     habitat_baselines.il.data_collection.recollect_train_demos=False \
#     habitat_baselines.il.trainer.sequence_length=15 \
#     habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
#     habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
#     habitat_baselines.il.trainer.val_check_interval=1.0 \
#     habitat_baselines.il.trainer.num_workers=4 \
#     habitat_baselines.il.trainer.batch_size=80 \
#     habitat_baselines.il.trainer.num_epochs_per_round=1 \
#     habitat_baselines.il.data_collection.num_episodes_per_round=50 \
#     habitat_baselines.il.num_rounds=10 \
#     habitat_baselines.test_episode_count=20 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=False \
#     habitat_baselines.il.student_policy.type=RNNModel \
#     habitat_baselines.il.student_policy.goal_fusion_stage='post_rnn' \
#     habitat_baselines.il.student_policy.rnn_type=GRU \
#     habitat_baselines.il.student_policy.num_rnn_layers=2 \
#     habitat_baselines.il.student_policy.hidden_dim=512 \
#     habitat_baselines.il.student_policy.output_type=continuous \
#     habitat_baselines.il.student_policy.base_lr=1e-5 \
#     habitat_baselines.il.student_policy.loss_mode='last' \
#     habitat_baselines.il.data_collection.scratch_dir=dagger_examples/23478416 \
#     depthgoals.augmentation.timestep_noise_multiplier=0.1 \
#     habitat_baselines.il.trainer.skip_train_until=4 \
#     # habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \

