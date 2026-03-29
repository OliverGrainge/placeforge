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

failures=0
for config in "${configs[@]}"; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    set +e
    python -m cli train "$config"
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "FAILED: $config (exit $status) — continuing to next config" >&2
        failures=$((failures + 1))
    fi
done

echo "All configs processed."
if [ "$failures" -gt 0 ]; then
    echo "$failures run(s) failed." >&2
    exit 1
fi
