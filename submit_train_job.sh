#!/bin/bash
#SBATCH --job-name=placeforge_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG"
echo "GPUs: $SLURM_GPUS_ON_NODE"

python -m cli train "$CONFIG"
