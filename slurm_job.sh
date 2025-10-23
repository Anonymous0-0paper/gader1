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
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/slurm.%N.%j.out
#SBATCH --error=slurm_logs/slurm.%N.%j.err

# Create logs directory
mkdir -p slurm_logs

# Load required modules (adjust based on your cluster)
module load python/3.9 cuda/11.7

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Install requirements if not already installed
pip install -r requirements.txt

# Run the MoE-FL training script
python train_moefl.py --dataset CIFAR10 --num_rounds 200 --num_experts 8 --top_k 2 --clients_per_round 5

echo "Job completed successfully!"