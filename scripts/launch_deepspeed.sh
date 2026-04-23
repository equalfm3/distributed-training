#!/usr/bin/env bash
# Launch DeepSpeed training with ZeRO optimization
#
# Usage:
#   bash scripts/launch_deepspeed.sh                           # Single GPU / CPU fallback
#   bash scripts/launch_deepspeed.sh --num_gpus 4              # 4 GPUs
#   bash scripts/launch_deepspeed.sh --num_gpus 8 --num_nodes 2 --hostfile hostfile.txt
#
# Environment variables:
#   NUM_GPUS        - GPUs per node (default: 1)
#   DS_CONFIG       - DeepSpeed config path (default: configs/deepspeed_zero3.json)
#   MAX_STEPS       - Training steps (default: 100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
NUM_GPUS="${NUM_GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
DS_CONFIG="${DS_CONFIG:-configs/deepspeed_zero3.json}"
MAX_STEPS="${MAX_STEPS:-100}"
HOSTFILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        --config)
            DS_CONFIG="$2"
            shift 2
            ;;
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=== DeepSpeed Training Launch ==="
echo "Nodes: ${NUM_NODES}, GPUs/node: ${NUM_GPUS}"
echo "Config: ${DS_CONFIG}"
echo "Steps: ${MAX_STEPS}"
echo "================================="

cd "$PROJECT_DIR"

DS_ARGS=(
    --num_gpus "$NUM_GPUS"
    --num_nodes "$NUM_NODES"
)

if [[ -n "$HOSTFILE" ]]; then
    DS_ARGS+=(--hostfile "$HOSTFILE")
fi

deepspeed "${DS_ARGS[@]}" \
    src/deepspeed/trainer.py \
    --deepspeed "$DS_CONFIG" \
    --steps "$MAX_STEPS"
