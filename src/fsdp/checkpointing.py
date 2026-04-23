"""FSDP checkpoint save and load utilities.

Supports full state dict (consolidated on rank 0) and sharded state dict
(each rank saves its own shard) checkpoint strategies. Handles optimizer
state alongside model parameters.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

from src.utils.distributed import barrier, get_rank, is_main_process


class CheckpointStrategy(str, Enum):
    """Checkpoint saving strategies."""

    FULL_STATE_DICT = "full_state_dict"
    SHARDED_STATE_DICT = "sharded_state_dict"


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint operations.

    Attributes:
        save_dir: Directory to write checkpoints.
        strategy: Full or sharded state dict approach.
        save_optimizer: Whether to include optimizer state.
        max_checkpoints: Maximum checkpoints to keep (0 = unlimited).
    """

    save_dir: str = "./checkpoints"
    strategy: CheckpointStrategy = CheckpointStrategy.FULL_STATE_DICT
    save_optimizer: bool = True
    max_checkpoints: int = 3


def save_full_state_dict(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    save_path: str,
    step: int,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a full (consolidated) state dict checkpoint.

    Gathers all shards to rank 0 and saves a single checkpoint file.
    Only rank 0 performs the actual write.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer to save state from. None to skip.
        save_path: Directory to save the checkpoint.
        step: Current training step for naming.
        extra_state: Additional metadata to include.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        model_state = model.state_dict()

        checkpoint: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model_state,
        }

        if optimizer is not None:
            optim_state = FSDP.optim_state_dict(model, optimizer)
            checkpoint["optimizer_state_dict"] = optim_state

        if extra_state:
            checkpoint["extra_state"] = extra_state

        if is_main_process():
            filepath = save_dir / f"checkpoint_step_{step}.pt"
            torch.save(checkpoint, filepath)
            print(f"Saved full state dict checkpoint: {filepath}")

    barrier()


def load_full_state_dict(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
) -> Dict[str, Any]:
    """Load a full state dict checkpoint into an FSDP model.

    Broadcasts the checkpoint from rank 0 to all ranks, then loads
    model and optimizer state.

    Args:
        model: FSDP-wrapped model to load state into.
        optimizer: Optimizer to restore state. None to skip.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Dictionary with step number and any extra state.
    """
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    checkpoint: Dict[str, Any] = {}
    if is_main_process():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        if is_main_process():
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optim_state = (
                checkpoint["optimizer_state_dict"] if is_main_process() else {}
            )
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, optim_state
            )
            optimizer.load_state_dict(optim_state)

    barrier()

    return {
        "step": checkpoint.get("step", 0) if is_main_process() else 0,
        "extra_state": checkpoint.get("extra_state", {}) if is_main_process() else {},
    }


def save_sharded_state_dict(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    save_path: str,
    step: int,
) -> None:
    """Save a sharded state dict checkpoint.

    Each rank saves its own shard independently. Faster than full state
    dict for large models but requires the same world size to resume.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer to save. None to skip.
        save_path: Base directory for sharded checkpoint.
        step: Current training step.
    """
    save_dir = Path(save_path) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    rank = get_rank()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state = model.state_dict()
        shard_path = save_dir / f"shard_rank_{rank}.pt"

        shard: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model_state,
        }

        if optimizer is not None:
            optim_state = FSDP.optim_state_dict(model, optimizer)
            shard["optimizer_state_dict"] = optim_state

        torch.save(shard, shard_path)

    if is_main_process():
        print(f"Saved sharded checkpoint at step {step}: {save_dir}")

    barrier()


def load_sharded_state_dict(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_dir: str,
    step: int,
) -> Dict[str, Any]:
    """Load a sharded state dict checkpoint.

    Each rank loads its own shard file. Requires the same world size
    as when the checkpoint was saved.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer to restore. None to skip.
        checkpoint_dir: Base directory containing step subdirectories.
        step: Training step to load.

    Returns:
        Dictionary with step number.
    """
    shard_dir = Path(checkpoint_dir) / f"step_{step}"
    rank = get_rank()
    shard_path = shard_dir / f"shard_rank_{rank}.pt"

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        model.load_state_dict(shard["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in shard:
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, shard["optimizer_state_dict"]
            )
            optimizer.load_state_dict(optim_state)

    barrier()
    return {"step": shard.get("step", 0)}


def cleanup_old_checkpoints(save_dir: str, max_keep: int) -> None:
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        save_dir: Directory containing checkpoint files or subdirectories.
        max_keep: Maximum number of checkpoints to retain. 0 keeps all.
    """
    if max_keep <= 0 or not is_main_process():
        return

    save_path = Path(save_dir)
    if not save_path.exists():
        return

    checkpoints = sorted(save_path.glob("checkpoint_step_*.pt"))
    step_dirs = sorted(save_path.glob("step_*"))
    all_items = checkpoints + step_dirs

    if len(all_items) <= max_keep:
        return

    to_remove = all_items[: len(all_items) - max_keep]
    for item in to_remove:
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)
        print(f"Removed old checkpoint: {item}")


if __name__ == "__main__":
    print("FSDP checkpointing utilities loaded.")
    print(f"Strategies: {[s.value for s in CheckpointStrategy]}")
