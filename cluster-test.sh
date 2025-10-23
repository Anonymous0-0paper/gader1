#!/bin/bash -l
#SBATCH --partition=IFIgpu
#SBATCH --job-name=MoEFL_Train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@institution.edu
#SBATCH --account=your_group
#SBATCH --uid=your_username
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output slurm.%N.%j.out
#SBATCH --error slurm.%N.%j.err

# Load required modules (adjust based on your cluster)
module load python/3.9 cuda/11.7

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Install requirements if not already installed
pip install -r requirements.txt

# Run the MoE-FL training script with default parameters
python train_moefl.py

echo "Training completed!"