#!/bin/bash
#SBATCH --job-name=faint
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err
#SBATCH --account=project_2010179
#SBATCH --time=1-12:00:00
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4,nvme:3500
#SBATCH --partition=gpumedium

source ~/.bashrc
module load cuda
export PATH="/projappl/project_2010179/DepthGoals/bin:$PATH"
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# srun python -u \
#     -m depthgoals.train.run \
#     --config-name=val_experiments/vint.yaml \
#     val_experiments/agents@agents=depth_agent_base \
#     habitat_baselines.evaluate=False \
#     habitat_baselines.num_environments=16 \
#     habitat_baselines.il.data_collection.recollect_train_demos=False \
#     habitat_baselines.il.trainer.sequence_length=6 \
#     habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
#     habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
#     habitat_baselines.il.trainer.val_check_interval=1.0 \
#     habitat_baselines.il.trainer.num_workers=4 \
#     habitat_baselines.il.trainer.batch_size=196 \
#     habitat_baselines.il.trainer.num_epochs_per_round=1 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
#     habitat_baselines.il.data_collection.num_episodes_per_round=10000 \
#     habitat_baselines.il.num_rounds=10 \
#     habitat_baselines.test_episode_count=-1 \
#     habitat_baselines.il.student_policy.output_type=waypoints \
#     habitat_baselines.il.student_policy.base_lr=1e-4 \
#     habitat_baselines.il.data_collection.scratch_dir=dagger_examples/$SLURM_JOBID \
#     habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \
#     depthgoals.augmentation.timestep_noise_multiplier=0.1 \


srun python -u \
    -m depthgoals.train.run \
    --config-name=val_experiments/rnn.yaml \
    val_experiments/agents@agents=depth_agent_base \
    habitat_baselines.evaluate=False \
    habitat_baselines.num_environments=16 \
    habitat_baselines.il.data_collection.recollect_train_demos=False \
    habitat_baselines.il.trainer.sequence_length=15 \
    habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
    habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    habitat_baselines.il.trainer.val_check_interval=1.0 \
    habitat_baselines.il.trainer.num_workers=4 \
    habitat_baselines.il.trainer.batch_size=128 \
    habitat_baselines.il.trainer.num_epochs_per_round=1 \
    habitat_baselines.il.trainer.gradient_clip_val=2.0 \
    habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
    habitat_baselines.il.data_collection.num_episodes_per_round=10000 \
    habitat_baselines.il.num_rounds=10 \
    habitat_baselines.il.student_policy.goal_fusion_stage='post_rnn' \
    habitat_baselines.il.student_policy.num_rnn_layers=1 \
    habitat_baselines.il.student_policy.output_type=continuous \
    habitat_baselines.il.student_policy.base_lr=1e-5 \
    habitat_baselines.il.student_policy.loss_mode='last' \
    habitat_baselines.il.student_policy.type=RNNModel \
    habitat_baselines.il.student_policy.rnn_type=LSTM \
    habitat_baselines.il.data_collection.scratch_dir=dagger_examples/$SLURM_JOBID \
    depthgoals.augmentation.timestep_noise_multiplier=0.1 \
    habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \
    habitat.task.actions.se2_velocity_action.add_noise=True \
    habitat.task.lab_sensors.imagesubgoal.planner_safety_margin=0.2 \





    # 1e-5 has been the lr for continuous


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
#     habitat_baselines.il.data_collection.num_episodes_per_round=10000 \
#     habitat_baselines.il.num_rounds=10 \
#     habitat_baselines.il.student_policy.output_type=continuous \
#     habitat_baselines.il.student_policy.base_lr=1e-5 \
#     habitat_baselines.il.trainer.closed_loop_eval_every_round=True \
#     habitat_baselines.il.data_collection.scratch_dir=dagger_examples/$SLURM_JOBID \
#     habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH \

# lr 1e-3 for discrete and 1e-5 for continuous - CNN don't seem to care though?
