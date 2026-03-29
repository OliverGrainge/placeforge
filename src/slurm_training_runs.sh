#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file_or_directory>" >&2
    exit 1
fi

INPUT="$1"

if [ -f "$INPUT" ]; then
    configs=("$INPUT")
elif [ -d "$INPUT" ]; then
    configs=("$INPUT"/*.yaml)
    if [ ${#configs[@]} -eq 0 ]; then
        echo "No .yaml config files found in '$INPUT'" >&2
        exit 1
    fi
else
    echo "Error: '$INPUT' is not a file or directory" >&2
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
