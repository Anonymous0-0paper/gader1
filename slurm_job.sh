#!/bin/bash -l
#SBATCH --partition=IFIgpu4090
#SBATCH --job-name=MoEFL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.younesi@uibk.ac.at
#SBATCH --account=DPS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm_logs/slurm.%N.%j.out
#SBATCH --error=slurm_logs/slurm.%N.%j.err

echo "=== SLURM job started on $(hostname) at $(date) ==="
# Create logs directory
mkdir -p slurm_logs

# Load required modules (adjust based on your cluster)

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Activate virtual environment if needed
cd ~/gader1
source .venv/bin/activate

python --version
nvidia-smi

# Install requirements if not already installed
pip install -r requirements.txt

# Run the MoE-FL training script
python train_moefl.py --dataset CIFAR10 --num_rounds 200 --device cuda

echo "Job completed successfully!"
