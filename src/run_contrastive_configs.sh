#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs/sf_xl_classification"

configs=(
    "$CONFIG_DIR/curavpr_boq.yaml"
    "$CONFIG_DIR/cosplace_gem.yaml"
    "$CONFIG_DIR/eigenplaces_gem.yaml"
    "$CONFIG_DIR/curavpr_gem.yaml"
    "$CONFIG_DIR/cosplace_mixvpr.yaml"
    "$CONFIG_DIR/eigenplaces_mixvpr.yaml"
    "$CONFIG_DIR/curavpr_mixvpr.yaml"
    "$CONFIG_DIR/cosplace_salad.yaml"
    "$CONFIG_DIR/eigenplaces_salad.yaml"
    "$CONFIG_DIR/curavpr_salad.yaml"
)

for config in "${configs[@]}"; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    if ! python -m cli train "$config"; then
        echo "FAILED: $config — continuing to next config" >&2
    fi
done

echo "All configs processed."
