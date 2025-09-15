#!/bin/bash
#SBATCH --job-name=faint_mkenv
#SBATCH --output=test_log.out
#SBATCH --error=test_log.err
#SBATCH --account=project_2010179
#SBATCH --time=00:15:00
#SBATCH --mem=90G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --partition=gputest

source ~/.bashrc
module load tykky
conda-containerize new --mamba --prefix /projappl/project_2010179/py310/ --post post_install.sh faint/train/environment_py310.yml
