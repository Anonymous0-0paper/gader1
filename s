python train_moefl.py --dataset CIFAR10
Train with custom parameters
python train_moefl.py --dataset CIFAR10 --num_experts 8 --top_k 2 --num_rounds 200 --clients_per_round 10
Run single experiment (faster)
python train_moefl.py --dataset CIFAR10 --single_run --seed 42
Compare with baselines
python compare_baselines.py --dataset CIFAR10 --output_dir ./outputs/comparison