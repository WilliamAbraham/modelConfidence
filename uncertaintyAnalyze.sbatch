#!/bin/bash
#SBATCH --account=torch_pr_183_general
#SBATCH --job-name=modelConfidence
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jc14407@nyu.edu
#SBATCH --output=slurm_%j.out


#SBATCH --mail-type=END
#SBATCH --mail-user=jc14407@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/2024.02
source $HOME/.bashrc
conda activate /scratch/jc14407/.conda/env/torch2
export MPLCONFIGDIR=/scratch/jc14407/.config/matplotlib

python uncertainty_analysis.py



 