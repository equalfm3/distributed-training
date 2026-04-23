"""Communication primitives for tensor parallelism.

Wraps torch.distributed collective operations (all-reduce, all-gather,
reduce-scatter, scatter) with tensor-parallel-aware defaults. Gracefully
handles single-process mode by returning inputs unchanged.
"""

from typing import List, Optional

import torch
import torch.distributed as dist


_TENSOR_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None


def init_tensor_parallel_group(
    ranks: Optional[List[int]] = None,
    backend: Optional[str] = None,
) -> dist.ProcessGroup:
    """Initialize a process group for tensor parallelism.

    Creates a sub-group of the default process group for tensor-parallel
    communication. If distributed is not initialized, returns None and
    all operations fall back to single-process mode.

    Args:
        ranks: List of global ranks in the tensor parallel group.
            Defaults to all ranks.
        backend: Communication backend. Defaults to the global backend.

    Returns:
        The created process group, or None if not distributed.
    """
    global _TENSOR_PARALLEL_GROUP

    if not dist.is_initialized():
        return None

    if ranks is None:
        ranks = list(range(dist.get_world_size()))

    _TENSOR_PARALLEL_GROUP = dist.new_group(ranks=ranks, backend=backend)
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the current tensor parallel process group.

    Returns:
        The tensor parallel group, or None if not initialized.
    """
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_world_size() -> int:
    """Get the number of ranks in the tensor parallel group.

    Returns:
        World size of the tensor parallel group, or 1 if not distributed.
    """
    if _TENSOR_PARALLEL_GROUP is not None:
        return dist.get_world_size(group=_TENSOR_PARALLEL_GROUP)
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tensor_parallel_rank() -> int:
    """Get the rank within the tensor parallel group.

    Returns:
        Rank in the tensor parallel group, or 0 if not distributed.
    """
    if _TENSOR_PARALLEL_GROUP is not None:
        return dist.get_rank(group=_TENSOR_PARALLEL_GROUP)
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def all_reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """All-reduce a tensor across the tensor parallel group.

    Each rank ends up with the sum (or other reduction) of all ranks'
    tensors. No-op in single-process mode.

    Args:
        tensor: Input tensor to reduce.
        op: Reduction operation (SUM, AVG, etc.).

    Returns:
        Reduced tensor (in-place on the input).
    """
    if get_tensor_parallel_world_size() <= 1:
        return tensor

    dist.all_reduce(tensor, op=op, group=_TENSOR_PARALLEL_GROUP)
    return tensor


def all_gather_tensor(
    tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """All-gather a tensor along a specified dimension.

    Concatenates each rank's tensor along the given dimension so every
    rank has the full tensor. No-op in single-process mode.

    Args:
        tensor: Local tensor shard.
        dim: Dimension along which to concatenate.

    Returns:
        Gathered tensor with the full data.
    """
    world_size = get_tensor_parallel_world_size()
    if world_size <= 1:
        return tensor

    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous(), group=_TENSOR_PARALLEL_GROUP)
    return torch.cat(tensor_list, dim=dim)


def reduce_scatter_tensor(
    tensor: torch.Tensor,
    dim: int = 0,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """Reduce-scatter a tensor across the tensor parallel group.

    Reduces the tensor across ranks and scatters the result so each
    rank gets a different shard. No-op in single-process mode.

    Args:
        tensor: Input tensor to reduce and scatter.
        dim: Dimension along which to scatter.
        op: Reduction operation.

    Returns:
        This rank's shard of the reduced tensor.
    """
    world_size = get_tensor_parallel_world_size()
    if world_size <= 1:
        return tensor

    chunk_size = tensor.shape[dim] // world_size
    output = torch.empty(
        *tensor.shape[:dim],
        chunk_size,
        *tensor.shape[dim + 1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )

    input_chunks = list(tensor.chunk(world_size, dim=dim))
    dist.reduce_scatter(output, input_chunks, op=op, group=_TENSOR_PARALLEL_GROUP)
    return output


def scatter_tensor(
    tensor: torch.Tensor,
    dim: int = 0,
    src: int = 0,
) -> torch.Tensor:
    """Scatter a tensor from source rank to all ranks.

    Splits the tensor along the given dimension and sends one chunk
    to each rank. In single-process mode, returns the full tensor.

    Args:
        tensor: Full tensor on the source rank, any tensor on others.
        dim: Dimension along which to split.
        src: Source rank that holds the full tensor.

    Returns:
        This rank's chunk of the scattered tensor.
    """
    world_size = get_tensor_parallel_world_size()
    if world_size <= 1:
        return tensor

    rank = get_tensor_parallel_rank()
    chunk_size = tensor.shape[dim] // world_size

    output = torch.empty(
        *tensor.shape[:dim],
        chunk_size,
        *tensor.shape[dim + 1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )

    if rank == src:
        chunks = list(tensor.chunk(world_size, dim=dim))
        scatter_list = [c.contiguous() for c in chunks]
    else:
        scatter_list = None

    dist.scatter(output, scatter_list, src=src, group=_TENSOR_PARALLEL_GROUP)
    return output


def broadcast_tensor(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    """Broadcast a tensor from source rank to all ranks.

    Args:
        tensor: Tensor to broadcast (meaningful only on src rank).
        src: Source rank.

    Returns:
        Broadcast tensor (same on all ranks).
    """
    if get_tensor_parallel_world_size() <= 1:
        return tensor

    dist.broadcast(tensor, src=src, group=_TENSOR_PARALLEL_GROUP)
    return tensor


def send_tensor(tensor: torch.Tensor, dst: int) -> None:
    """Send a tensor to a specific rank.

    Args:
        tensor: Tensor to send.
        dst: Destination rank.
    """
    if not dist.is_initialized():
        return
    dist.send(tensor.contiguous(), dst=dst, group=_TENSOR_PARALLEL_GROUP)


def recv_tensor(tensor: torch.Tensor, src: int) -> torch.Tensor:
    """Receive a tensor from a specific rank.

    Args:
        tensor: Pre-allocated tensor to receive into.
        src: Source rank.

    Returns:
        The received tensor.
    """
    if not dist.is_initialized():
        return tensor
    dist.recv(tensor, src=src, group=_TENSOR_PARALLEL_GROUP)
    return tensor


if __name__ == "__main__":
    x = torch.randn(4, 8)
    print(f"TP world size: {get_tensor_parallel_world_size()}")
    print(f"TP rank: {get_tensor_parallel_rank()}")

    y = all_reduce_tensor(x.clone())
    print(f"All-reduce (single process): input == output: {torch.equal(x, y)}")

    z = all_gather_tensor(x.clone(), dim=0)
    print(f"All-gather (single process): shape {x.shape} -> {z.shape}")
