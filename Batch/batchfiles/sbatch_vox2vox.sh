#!/bin/bash
#SBATCH --time=50:0:0
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/vox2vox_%A.out
#SBATCH --error=logs/vox2vox_%A.err

PYTHON_SCRIPT="/hpc/pmc_rios/1.projects/CK1_VirtualMultiplexing/2.experiments/VirtualMultiplexing/train.py"

CONDA_SOURCE="/hpc/local/CentOS7/pmc_rios/anaconda3/etc/profile.d/conda.sh"
source "$CONDA_SOURCE"
conda activate /home/pmc_research/ckenter/pmc_rios/2.personal/ckenter/conda_envs/stapl3d_coen/

echo "python ${PYTHON_SCRIPT} --c configs/train_conf.yml"
python ${PYTHON_SCRIPT} --c configs/train_conf.yml
