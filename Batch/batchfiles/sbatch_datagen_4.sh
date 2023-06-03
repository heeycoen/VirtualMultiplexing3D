#!/bin/bash
#SBATCH --time=100:0:0
#SBATCH --mem=100G
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/datagen_%A.out
#SBATCH --error=logs/datagen_%A.err

PYTHON_SCRIPT="/hpc/pmc_rios/1.projects/CK1_VirtualMultiplexing/2.experiments/VirtualMultiplexing/Preprocessing/Preprocessor.py"

CONDA_SOURCE="/hpc/local/CentOS7/pmc_rios/anaconda3/etc/profile.d/conda.sh"
source "$CONDA_SOURCE"
conda activate /home/pmc_research/ckenter/pmc_rios/2.personal/ckenter/conda_envs/stapl3d_coen/

echo "python ${PYTHON_SCRIPT} --c configs/VoxPreP_Conf_4.yml"
python ${PYTHON_SCRIPT} --c configs/VoxPreP_Conf_4.yml
