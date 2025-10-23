#!/bin/bash -l
#SBATCH --partition=IFIgpu
#SBATCH --job-name=MoEFL_Experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@institution.edu
#SBATCH --account=your_group
#SBATCH --uid=your_username
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
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

# Define experiment parameters
DATASETS=("CIFAR10" "CIFAR100")
EXPERTS=(8 12)
TOPK=(2 3)

# Run experiments
for DATASET in "${DATASETS[@]}"; do
    for NUM_EXPERTS in "${EXPERTS[@]}"; do
        for TOP_K in "${TOPK[@]}"; do
            # Skip invalid combinations
            if [[ "$DATASET" == "CIFAR10" && "$NUM_EXPERTS" == "12" ]]; then
                continue
            fi
            
            EXP_NAME="${DATASET}_E${NUM_EXPERTS}_K${TOP_K}"
            echo "Running experiment: $EXP_NAME"
            
            python train_moefl.py \
                --dataset $DATASET \
                --num_experts $NUM_EXPERTS \
                --top_k $TOP_K \
                --clients_per_round 5 \
                --output_dir "./outputs/${EXP_NAME}" \
                --device cuda
                
            echo "Completed experiment: $EXP_NAME"
        done
    done
done

echo "All experiments completed successfully!"