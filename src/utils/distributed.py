"""Distributed setup utilities for multi-GPU training.

Provides helpers for initializing process groups, querying rank/world size,
and assigning devices. Gracefully handles single-process mode (world_size=1)
and falls back to gloo backend when NCCL is unavailable.
"""

import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistributedInfo:
    """Container for distributed training metadata.

    Attributes:
        rank: Global rank of this process.
        local_rank: Rank within the local node.
        world_size: Total number of processes.
        device: Torch device assigned to this process.
        backend: Communication backend in use.
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str


def get_free_port() -> int:
    """Find a free port on localhost.

    Returns:
        An available TCP port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _select_backend(backend: Optional[str] = None) -> str:
    """Choose the best available communication backend.

    Args:
        backend: Explicit backend choice. If None, auto-selects based on
            hardware availability.

    Returns:
        Backend string suitable for dist.init_process_group.
    """
    if backend is not None:
        return backend
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> DistributedInfo:
    """Initialize the distributed process group.

    Sets up torch.distributed with automatic backend selection and device
    assignment. Safe to call in single-process mode — will set up a group
    of size 1 using environment variables or explicit parameters.

    Args:
        backend: Communication backend ('nccl', 'gloo', or None for auto).
        init_method: URL for process group initialization. Defaults to
            env:// or a localhost TCP endpoint.
        world_size: Total number of processes. Read from env if None.
        rank: Global rank of this process. Read from env if None.

    Returns:
        DistributedInfo with rank, device, and backend metadata.
    """
    if dist.is_initialized():
        return _build_info(_select_backend(backend))

    chosen_backend = _select_backend(backend)

    env_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    env_world = int(os.environ.get("WORLD_SIZE", "1"))

    actual_rank = rank if rank is not None else env_rank
    actual_world = world_size if world_size is not None else env_world

    if init_method is None:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(get_free_port())
        init_method = "env://"

    dist.init_process_group(
        backend=chosen_backend,
        init_method=init_method,
        world_size=actual_world,
        rank=actual_rank,
    )

    return _build_info(chosen_backend)


def _build_info(backend: str) -> DistributedInfo:
    """Build DistributedInfo from the current process group state.

    Args:
        backend: The backend string used for initialization.

    Returns:
        Populated DistributedInfo dataclass.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count() if torch.cuda.is_available() else 0))

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return DistributedInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def get_rank() -> int:
    """Get the global rank of the current process.

    Returns:
        0 if distributed is not initialized, otherwise the process rank.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes in the group.

    Returns:
        1 if distributed is not initialized, otherwise the world size.
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_device() -> torch.device:
    """Get the device assigned to the current process.

    Returns:
        CUDA device for the local rank, or CPU if CUDA is unavailable.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def is_main_process() -> bool:
    """Check whether this is the main (rank 0) process.

    Returns:
        True if rank is 0 or distributed is not initialized.
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes.

    No-op if distributed is not initialized.
    """
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    """Destroy the distributed process group.

    Safe to call even if the group was never initialized.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    info = init_process_group()
    print(f"Rank {info.rank}/{info.world_size} on {info.device} via {info.backend}")
    cleanup()
