#!/bin/bash
#SBATCH --job-name=placeforge_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e

CONFIG=$1

if [ -z "$CONFIG" ]; then
    echo "Usage: sbatch submit_train_job.sh path/to/config.yaml"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

mkdir -p logs

# ── Environment setup ────────────────────────────────────────────────────
# Activate your conda/venv environment (update this path for your cluster)
# source /path/to/your/conda/bin/activate placeforge
# source /path/to/your/venv/bin/activate

# Wandb offline mode (compute nodes have no internet)
export WANDB_MODE=offline

# Pre-populated model cache (download on login node, used on compute nodes)
# export TORCH_HOME=/path/to/shared/torch_cache

# Override dataset paths for HPC (uncomment and update if not using .env)

export PLACEFORGE_RAW_DIR=/home/oliver/datasets/
export PLACEFORGE_FEATURE_STORE_DIR=/home/oliver/datasets/placeforge/feature_store
export PLACEFORGE_PROCESSED_DIR=/home/oliver/datasets/placeforge/processed

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG"
echo "GPUs: $SLURM_GPUS_ON_NODE"

python -m cli train "$CONFIG"
