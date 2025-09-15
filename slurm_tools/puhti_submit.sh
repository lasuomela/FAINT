#!/bin/bash
#SBATCH --job-name=faint
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err
#SBATCH --account=project_2010179
#SBATCH --time=3-00:00:00
#SBATCH --mem=373G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:4,nvme:3600
#SBATCH --partition=gpu

source ~/.bashrc
module load cuda
export PATH="/projappl/project_2010179/py310/bin:$PATH" 
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=4

srun python -u \
    -m faint.train.run \
    --config-name=experiments/main_experiment/FAINT_10x.yaml \
    habitat_baselines.il.trainer.num_devices=$SLURM_GPUS_ON_NODE \
    habitat_baselines.il.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    habitat_baselines.il.data_collection.scratch_dir=dagger_examples/$SLURM_JOBID \
    habitat_baselines.il.data_collection.fast_tmp_dir=\$LOCAL_SCRATCH 

