#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_directory>" >&2
    exit 1
fi

CONFIG_DIR="$1"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: '$CONFIG_DIR' is not a directory" >&2
    exit 1
fi

configs=("$CONFIG_DIR"/*.yaml)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .yaml config files found in '$CONFIG_DIR'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="$SCRIPT_DIR/../submit_train_job.sh"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "Error: submit_train_job.sh not found at '$SUBMIT_SCRIPT'" >&2
    exit 1
fi

mkdir -p "$SCRIPT_DIR/../logs"

for config in "${configs[@]}"; do
    echo "Submitting: $config"
    sbatch "$SUBMIT_SCRIPT" "$config"
done

echo "All ${#configs[@]} jobs submitted."
