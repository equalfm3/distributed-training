"""Throughput measurement for distributed training.

Measures tokens per second, samples per second, TFLOPS utilization,
and scaling efficiency across multiple GPU configurations. Provides
both real-time tracking and summary statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class StepMetrics:
    """Metrics for a single training step.

    Attributes:
        step: Training step number.
        elapsed_sec: Wall-clock time for this step.
        tokens: Number of tokens processed.
        samples: Number of samples processed.
        loss: Training loss value.
    """

    step: int
    elapsed_sec: float
    tokens: int
    samples: int
    loss: Optional[float] = None


@dataclass
class ThroughputStats:
    """Aggregated throughput statistics.

    Attributes:
        total_tokens: Total tokens processed.
        total_samples: Total samples processed.
        total_time_sec: Total wall-clock time.
        tokens_per_sec: Average tokens per second.
        samples_per_sec: Average samples per second.
        avg_step_time_sec: Average time per step.
        peak_tokens_per_sec: Maximum tokens/sec observed.
        num_steps: Number of steps recorded.
    """

    total_tokens: int = 0
    total_samples: int = 0
    total_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    avg_step_time_sec: float = 0.0
    peak_tokens_per_sec: float = 0.0
    num_steps: int = 0


class ThroughputTracker:
    """Tracks training throughput in real time.

    Records per-step metrics and computes running averages for
    tokens/sec, samples/sec, and step latency.

    Args:
        world_size: Number of distributed processes (for aggregate stats).
        log_interval: Steps between automatic log prints.
    """

    def __init__(self, world_size: int = 1, log_interval: int = 10) -> None:
        self.world_size = world_size
        self.log_interval = log_interval
        self._steps: List[StepMetrics] = []
        self._start_time: Optional[float] = None
        self._step_start: Optional[float] = None

    def start_step(self) -> None:
        """Mark the beginning of a training step."""
        if self._start_time is None:
            self._start_time = time.perf_counter()
        self._step_start = time.perf_counter()

    def end_step(
        self,
        tokens: int,
        samples: int,
        loss: Optional[float] = None,
    ) -> StepMetrics:
        """Mark the end of a training step and record metrics.

        Args:
            tokens: Number of tokens processed in this step.
            samples: Number of samples processed in this step.
            loss: Optional loss value for this step.

        Returns:
            StepMetrics for the completed step.
        """
        end_time = time.perf_counter()
        elapsed = end_time - (self._step_start or end_time)

        metrics = StepMetrics(
            step=len(self._steps),
            elapsed_sec=elapsed,
            tokens=tokens * self.world_size,
            samples=samples * self.world_size,
            loss=loss,
        )
        self._steps.append(metrics)
        return metrics

    def get_stats(self) -> ThroughputStats:
        """Compute aggregate throughput statistics.

        Returns:
            ThroughputStats with averages and totals.
        """
        if not self._steps:
            return ThroughputStats()

        total_tokens = sum(s.tokens for s in self._steps)
        total_samples = sum(s.samples for s in self._steps)
        total_time = sum(s.elapsed_sec for s in self._steps)

        per_step_tps = [
            s.tokens / s.elapsed_sec for s in self._steps if s.elapsed_sec > 0
        ]

        return ThroughputStats(
            total_tokens=total_tokens,
            total_samples=total_samples,
            total_time_sec=total_time,
            tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
            samples_per_sec=total_samples / total_time if total_time > 0 else 0,
            avg_step_time_sec=total_time / len(self._steps),
            peak_tokens_per_sec=max(per_step_tps) if per_step_tps else 0,
            num_steps=len(self._steps),
        )

    def get_recent_throughput(self, window: int = 10) -> float:
        """Get tokens/sec averaged over the last N steps.

        Args:
            window: Number of recent steps to average.

        Returns:
            Recent tokens per second.
        """
        recent = self._steps[-window:]
        if not recent:
            return 0.0
        total_tokens = sum(s.tokens for s in recent)
        total_time = sum(s.elapsed_sec for s in recent)
        return total_tokens / total_time if total_time > 0 else 0.0

    def format_step(self, metrics: StepMetrics) -> str:
        """Format a step's metrics as a human-readable string.

        Args:
            metrics: Step metrics to format.

        Returns:
            Formatted string with throughput info.
        """
        tps = metrics.tokens / metrics.elapsed_sec if metrics.elapsed_sec > 0 else 0
        sps = metrics.samples / metrics.elapsed_sec if metrics.elapsed_sec > 0 else 0
        parts = [
            f"Step {metrics.step}",
            f"Tok/s: {tps:,.0f}",
            f"Samp/s: {sps:,.0f}",
            f"Step time: {metrics.elapsed_sec * 1000:.1f}ms",
        ]
        if metrics.loss is not None:
            parts.append(f"Loss: {metrics.loss:.4f}")
        return " | ".join(parts)


def compute_tflops(
    model_params: int,
    tokens_per_sec: float,
    seq_len: int,
    batch_size: int,
) -> float:
    """Estimate TFLOPS utilization for transformer training.

    Uses the approximation: 6 * N * T FLOPS per token for forward
    and backward passes, where N is parameter count and T is tokens.

    Args:
        model_params: Total number of model parameters.
        tokens_per_sec: Measured tokens per second throughput.
        seq_len: Sequence length.
        batch_size: Batch size.

    Returns:
        Estimated TFLOPS.
    """
    flops_per_token = 6 * model_params
    total_flops = flops_per_token * tokens_per_sec
    return total_flops / 1e12


def compute_scaling_efficiency(
    throughputs: Dict[int, float],
    baseline_gpus: int = 1,
) -> Dict[int, float]:
    """Compute scaling efficiency relative to a baseline GPU count.

    Scaling efficiency = (throughput_N / throughput_base) / (N / base).
    Perfect linear scaling gives 1.0.

    Args:
        throughputs: Mapping from GPU count to tokens/sec throughput.
        baseline_gpus: GPU count to use as the reference point.

    Returns:
        Mapping from GPU count to scaling efficiency (0 to 1).
    """
    if baseline_gpus not in throughputs:
        raise ValueError(f"Baseline GPU count {baseline_gpus} not in throughputs")

    base_throughput = throughputs[baseline_gpus]
    efficiencies: Dict[int, float] = {}

    for num_gpus, throughput in throughputs.items():
        speedup = throughput / base_throughput
        ideal_speedup = num_gpus / baseline_gpus
        efficiencies[num_gpus] = speedup / ideal_speedup

    return efficiencies


def compute_mfu(
    model_params: int,
    tokens_per_sec: float,
    gpu_peak_tflops: float,
    num_gpus: int = 1,
) -> float:
    """Compute Model FLOPS Utilization (MFU).

    MFU measures what fraction of the GPU's peak FLOPS is actually
    used for model computation (excluding communication overhead).

    Args:
        model_params: Total model parameters.
        tokens_per_sec: Measured throughput.
        gpu_peak_tflops: Peak TFLOPS per GPU (e.g., 312 for A100).
        num_gpus: Number of GPUs.

    Returns:
        MFU as a fraction between 0 and 1.
    """
    achieved_tflops = compute_tflops(model_params, tokens_per_sec, 0, 0)
    total_peak = gpu_peak_tflops * num_gpus
    return achieved_tflops / total_peak if total_peak > 0 else 0.0


if __name__ == "__main__":
    tracker = ThroughputTracker(world_size=1)

    for i in range(20):
        tracker.start_step()
        time.sleep(0.01)
        metrics = tracker.end_step(tokens=4096, samples=8, loss=2.5 - i * 0.05)
        if (i + 1) % 5 == 0:
            print(tracker.format_step(metrics))

    stats = tracker.get_stats()
    print(f"\nTotal: {stats.total_tokens:,} tokens in {stats.total_time_sec:.2f}s")
    print(f"Average: {stats.tokens_per_sec:,.0f} tok/s, {stats.samples_per_sec:,.0f} samp/s")
    print(f"Peak: {stats.peak_tokens_per_sec:,.0f} tok/s")

    throughputs = {1: 10000, 2: 18000, 4: 34000, 8: 60000}
    efficiencies = compute_scaling_efficiency(throughputs)
    for gpus, eff in efficiencies.items():
        print(f"{gpus} GPUs: {eff:.1%} scaling efficiency")
