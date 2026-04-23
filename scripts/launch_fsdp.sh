#!/usr/bin/env bash
# Launch FSDP training with torchrun
#
# Usage:
#   bash scripts/launch_fsdp.sh                    # Single GPU / CPU
#   bash scripts/launch_fsdp.sh --nproc 4          # 4 GPUs on one node
#   bash scripts/launch_fsdp.sh --nproc 8 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1
#
# Environment variables:
#   NPROC_PER_NODE  - GPUs per node (default: 1)
#   MASTER_ADDR     - Master node address (default: 127.0.0.1)
#   MASTER_PORT     - Master port (default: 29500)
#   MAX_STEPS       - Training steps (default: 100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
NPROC="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-100}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
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

echo "=== FSDP Training Launch ==="
echo "Nodes: ${NNODES}, GPUs/node: ${NPROC}, Node rank: ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Steps: ${MAX_STEPS}"
echo "==========================="

cd "$PROJECT_DIR"

torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m src.fsdp.trainer \
    --steps "$MAX_STEPS"
