# Distributed Training at Scale

Multi-GPU and multi-node training implementations: FSDP (Fully Sharded Data Parallel), DeepSpeed ZeRO stages, pipeline parallelism, tensor parallelism, and gradient checkpointing. Benchmarks scaling efficiency.

## What This Covers

- PyTorch FSDP: sharding strategies, mixed precision, activation checkpointing
- DeepSpeed ZeRO: Stage 1/2/3 configuration and training
- Pipeline parallelism: micro-batch scheduling, GPipe-style
- Tensor parallelism: column/row parallel linear layers
- Gradient checkpointing for memory optimization
- Scaling efficiency benchmarks: throughput vs GPU count
- Communication profiling: NCCL collective analysis

## Structure

```
├── src/
│   ├── fsdp/
│   │   ├── trainer.py         # FSDP training loop
│   │   ├── sharding.py        # Sharding strategy config
│   │   └── checkpointing.py   # FSDP checkpoint save/load
│   ├── deepspeed/
│   │   ├── trainer.py         # DeepSpeed training loop
│   │   └── configs/           # ZeRO stage configs
│   ├── pipeline/
│   │   ├── schedule.py        # Pipeline schedule (GPipe, 1F1B)
│   │   └── partition.py       # Model partitioning
│   ├── tensor_parallel/
│   │   ├── layers.py          # Column/row parallel linear
│   │   └── comm.py            # All-reduce, all-gather wrappers
│   ├── profiling/
│   │   ├── throughput.py      # Tokens/sec measurement
│   │   └── comm_profile.py    # NCCL communication profiling
│   └── utils/
│       └── distributed.py     # Distributed setup utilities
├── configs/
│   ├── fsdp_config.yaml
│   └── deepspeed_zero3.json
├── scripts/
│   ├── launch_fsdp.sh         # torchrun launch script
│   └── launch_deepspeed.sh    # deepspeed launch script
├── notebooks/
│   └── walkthrough.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
# Single-node multi-GPU with FSDP
torchrun --nproc_per_node=2 -m src.fsdp.trainer --config configs/fsdp_config.yaml
# DeepSpeed ZeRO-3
deepspeed src/deepspeed/trainer.py --deepspeed configs/deepspeed_zero3.json
```
