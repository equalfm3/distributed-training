"""Communication profiling for distributed training.

Measures latency and bandwidth of collective operations (all-reduce,
all-gather, reduce-scatter, broadcast) across different message sizes.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from src.utils.distributed import get_rank, get_world_size, is_main_process


@dataclass
class CommBenchmarkResult:
    """Result from benchmarking a single collective operation.

    Attributes:
        operation: Name of the collective (e.g., 'all_reduce').
        message_size_bytes: Size of the message in bytes.
        latency_ms: Average latency in milliseconds.
        bandwidth_gbps: Achieved bandwidth in GB/s.
        num_iterations: Number of iterations averaged over.
    """

    operation: str
    message_size_bytes: int
    latency_ms: float
    bandwidth_gbps: float
    num_iterations: int


@dataclass
class CommProfile:
    """Complete communication profile across operations and sizes."""

    results: List[CommBenchmarkResult] = field(default_factory=list)
    world_size: int = 1
    backend: str = "gloo"


def _sync_device(device: torch.device) -> None:
    """Synchronize the device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _timed_loop(
    fn: Callable[[], None],
    device: torch.device,
    num_warmup: int,
    num_iterations: int,
) -> float:
    """Run warmup then timed iterations, returning average latency in seconds."""
    for _ in range(num_warmup):
        fn()
        _sync_device(device)

    _sync_device(device)
    start = time.perf_counter()
    for _ in range(num_iterations):
        fn()
        _sync_device(device)
    return (time.perf_counter() - start) / num_iterations


def benchmark_all_reduce(
    size_bytes: int, num_warmup: int = 5, num_iterations: int = 20,
    device: Optional[torch.device] = None,
) -> CommBenchmarkResult:
    """Benchmark all-reduce latency and bandwidth."""
    device = device or _default_device()
    tensor = torch.randn(max(1, size_bytes // 4), device=device)

    def op() -> None:
        if dist.is_initialized():
            dist.all_reduce(tensor)

    avg_latency = _timed_loop(op, device, num_warmup, num_iterations)
    bw = _compute_allreduce_bandwidth(size_bytes, avg_latency, get_world_size())

    return CommBenchmarkResult("all_reduce", size_bytes, avg_latency * 1000, bw, num_iterations)


def benchmark_all_gather(
    size_bytes: int, num_warmup: int = 5, num_iterations: int = 20,
    device: Optional[torch.device] = None,
) -> CommBenchmarkResult:
    """Benchmark all-gather latency and bandwidth."""
    device = device or _default_device()
    world_size = get_world_size()
    tensor = torch.randn(max(1, size_bytes // 4), device=device)
    output_list = [torch.empty_like(tensor) for _ in range(world_size)]

    def op() -> None:
        if dist.is_initialized():
            dist.all_gather(output_list, tensor)

    avg_latency = _timed_loop(op, device, num_warmup, num_iterations)
    bw = (size_bytes * world_size) / avg_latency / 1e9 if avg_latency > 0 else 0

    return CommBenchmarkResult("all_gather", size_bytes, avg_latency * 1000, bw, num_iterations)


def benchmark_reduce_scatter(
    size_bytes: int, num_warmup: int = 5, num_iterations: int = 20,
    device: Optional[torch.device] = None,
) -> CommBenchmarkResult:
    """Benchmark reduce-scatter latency and bandwidth."""
    device = device or _default_device()
    world_size = get_world_size()
    per_rank = max(1, size_bytes // (4 * world_size))
    input_tensor = torch.randn(per_rank * world_size, device=device)
    output_tensor = torch.empty(per_rank, device=device)
    chunks = list(input_tensor.chunk(world_size))

    def op() -> None:
        if dist.is_initialized():
            dist.reduce_scatter(output_tensor, chunks)

    avg_latency = _timed_loop(op, device, num_warmup, num_iterations)
    bw = size_bytes / avg_latency / 1e9 if avg_latency > 0 else 0

    return CommBenchmarkResult("reduce_scatter", size_bytes, avg_latency * 1000, bw, num_iterations)


def benchmark_broadcast(
    size_bytes: int, src: int = 0, num_warmup: int = 5, num_iterations: int = 20,
    device: Optional[torch.device] = None,
) -> CommBenchmarkResult:
    """Benchmark broadcast latency and bandwidth."""
    device = device or _default_device()
    tensor = torch.randn(max(1, size_bytes // 4), device=device)

    def op() -> None:
        if dist.is_initialized():
            dist.broadcast(tensor, src=src)

    avg_latency = _timed_loop(op, device, num_warmup, num_iterations)
    bw = size_bytes / avg_latency / 1e9 if avg_latency > 0 else 0

    return CommBenchmarkResult("broadcast", size_bytes, avg_latency * 1000, bw, num_iterations)


def _compute_allreduce_bandwidth(
    size_bytes: int, latency_sec: float, world_size: int,
) -> float:
    """Compute algorithmic bandwidth for ring all-reduce.

    Effective data moved is 2 * (N-1)/N * message_size.
    """
    if latency_sec <= 0 or world_size <= 1:
        return 0.0
    algo_bytes = 2 * (world_size - 1) / world_size * size_bytes
    return algo_bytes / latency_sec / 1e9


def run_full_profile(
    sizes_bytes: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    num_iterations: int = 20,
) -> CommProfile:
    """Run a complete communication profile across operations and sizes.

    Args:
        sizes_bytes: Message sizes to benchmark. Defaults to powers of 2 from 1KB to 1GB.
        device: Device for benchmarking.
        num_iterations: Iterations per benchmark.

    Returns:
        CommProfile with all results.
    """
    if sizes_bytes is None:
        sizes_bytes = [2**i for i in range(10, 31, 2)]

    backend = dist.get_backend() if dist.is_initialized() else "unknown"
    profile = CommProfile(world_size=get_world_size(), backend=backend)

    benchmarks = [benchmark_all_reduce, benchmark_all_gather, benchmark_reduce_scatter, benchmark_broadcast]
    for size in sizes_bytes:
        for bench_fn in benchmarks:
            profile.results.append(bench_fn(size, num_iterations=num_iterations, device=device))

    return profile


def format_profile(profile: CommProfile) -> str:
    """Format a communication profile as a readable table."""
    lines = [
        f"Communication Profile (world_size={profile.world_size}, backend={profile.backend})",
        "-" * 80,
        f"{'Operation':<18} {'Size':>12} {'Latency':>12} {'Bandwidth':>12}",
        "-" * 80,
    ]
    for r in profile.results:
        size_str = _format_bytes(r.message_size_bytes)
        lines.append(f"{r.operation:<18} {size_str:>12} {r.latency_ms:>10.3f}ms {r.bandwidth_gbps:>10.2f} GB/s")
    return "\n".join(lines)


def _format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


if __name__ == "__main__":
    print("Communication profiling utilities loaded.")
    print("Run with torchrun for multi-GPU profiling.")
    print("Available operations: all_reduce, all_gather, reduce_scatter, broadcast")
