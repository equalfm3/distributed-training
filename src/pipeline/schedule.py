"""Pipeline scheduling: GPipe (fill-drain) and 1F1B micro-batch scheduling.

Implements two pipeline parallelism schedules for executing micro-batches
across pipeline stages. GPipe fills then drains. 1F1B interleaves forward
and backward passes to reduce memory footprint.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.pipeline.partition import PipelineStage


class ScheduleType(str, Enum):
    """Pipeline schedule variants."""

    GPIPE = "gpipe"
    ONE_F_ONE_B = "1f1b"


@dataclass
class MicroBatchResult:
    """Result from processing a single micro-batch through the pipeline.

    Attributes:
        micro_batch_id: Index of this micro-batch.
        output: Final output tensor from the last stage.
        loss: Computed loss value if a loss function was provided.
    """

    micro_batch_id: int
    output: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


@dataclass
class ScheduleStep:
    """A single step in the pipeline schedule.

    Attributes:
        stage_id: Which pipeline stage executes this step.
        micro_batch_id: Which micro-batch is being processed.
        is_forward: True for forward pass, False for backward.
        clock_tick: Logical time step in the schedule.
    """

    stage_id: int
    micro_batch_id: int
    is_forward: bool
    clock_tick: int


def generate_gpipe_schedule(
    num_stages: int,
    num_micro_batches: int,
) -> List[ScheduleStep]:
    """Generate a GPipe (fill-drain) pipeline schedule.

    In GPipe, all forward passes complete before any backward passes
    begin. This maximizes pipeline utilization during each phase but
    requires storing all intermediate activations.

    Args:
        num_stages: Number of pipeline stages.
        num_micro_batches: Number of micro-batches to schedule.

    Returns:
        Ordered list of schedule steps.
    """
    steps: List[ScheduleStep] = []
    tick = 0

    for mb in range(num_micro_batches):
        for stage in range(num_stages):
            steps.append(ScheduleStep(
                stage_id=stage,
                micro_batch_id=mb,
                is_forward=True,
                clock_tick=tick + stage,
            ))
        tick += 1

    tick += num_stages - 1

    for mb in range(num_micro_batches):
        for stage in reversed(range(num_stages)):
            steps.append(ScheduleStep(
                stage_id=stage,
                micro_batch_id=mb,
                is_forward=False,
                clock_tick=tick + (num_stages - 1 - stage),
            ))
        tick += 1

    return steps


def generate_1f1b_schedule(
    num_stages: int,
    num_micro_batches: int,
) -> List[ScheduleStep]:
    """Generate a 1F1B (one-forward-one-backward) pipeline schedule.

    After the warmup phase fills the pipeline, each stage alternates
    between one forward and one backward pass. This reduces peak
    memory compared to GPipe by limiting in-flight micro-batches.

    Args:
        num_stages: Number of pipeline stages.
        num_micro_batches: Number of micro-batches to schedule.

    Returns:
        Ordered list of schedule steps.
    """
    steps: List[ScheduleStep] = []
    tick = 0

    warmup_batches = min(num_stages - 1, num_micro_batches)
    for mb in range(warmup_batches):
        for stage in range(num_stages):
            steps.append(ScheduleStep(
                stage_id=stage,
                micro_batch_id=mb,
                is_forward=True,
                clock_tick=tick + stage,
            ))
        tick += 1

    steady_batches = num_micro_batches - warmup_batches
    bwd_mb = 0
    fwd_mb = warmup_batches

    for _ in range(steady_batches):
        if fwd_mb < num_micro_batches:
            for stage in range(num_stages):
                steps.append(ScheduleStep(
                    stage_id=stage,
                    micro_batch_id=fwd_mb,
                    is_forward=True,
                    clock_tick=tick,
                ))
            tick += 1
            fwd_mb += 1

        if bwd_mb < num_micro_batches:
            for stage in reversed(range(num_stages)):
                steps.append(ScheduleStep(
                    stage_id=stage,
                    micro_batch_id=bwd_mb,
                    is_forward=False,
                    clock_tick=tick,
                ))
            tick += 1
            bwd_mb += 1

    while bwd_mb < num_micro_batches:
        for stage in reversed(range(num_stages)):
            steps.append(ScheduleStep(
                stage_id=stage,
                micro_batch_id=bwd_mb,
                is_forward=False,
                clock_tick=tick,
            ))
        tick += 1
        bwd_mb += 1

    return steps


def split_batch_into_micro_batches(
    batch: torch.Tensor,
    num_micro_batches: int,
) -> List[torch.Tensor]:
    """Split a batch tensor into micro-batches along the batch dimension.

    Args:
        batch: Input tensor of shape (batch_size, ...).
        num_micro_batches: Number of micro-batches to create.

    Returns:
        List of micro-batch tensors.

    Raises:
        ValueError: If batch size is not divisible by num_micro_batches.
    """
    batch_size = batch.shape[0]
    if batch_size % num_micro_batches != 0:
        raise ValueError(
            f"Batch size {batch_size} not divisible by {num_micro_batches}"
        )
    return list(batch.chunk(num_micro_batches, dim=0))


class PipelineEngine:
    """Executes micro-batches through pipeline stages according to a schedule.

    Args:
        stages: List of PipelineStage modules in order.
        schedule_type: Which scheduling algorithm to use.
        num_micro_batches: Number of micro-batches per batch.
        loss_fn: Optional loss function applied at the last stage.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        schedule_type: ScheduleType = ScheduleType.GPIPE,
        num_micro_batches: int = 4,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        self.stages = stages
        self.schedule_type = schedule_type
        self.num_micro_batches = num_micro_batches
        self.loss_fn = loss_fn
        self.num_stages = len(stages)

    def forward(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None,
    ) -> List[MicroBatchResult]:
        """Execute the pipeline on a batch of data.

        Splits the input into micro-batches and processes them through
        all stages according to the selected schedule.

        Args:
            input_batch: Full input batch tensor.
            target_batch: Optional target tensor for loss computation.

        Returns:
            List of MicroBatchResult for each micro-batch.
        """
        micro_inputs = split_batch_into_micro_batches(
            input_batch, self.num_micro_batches
        )
        micro_targets = None
        if target_batch is not None:
            micro_targets = split_batch_into_micro_batches(
                target_batch, self.num_micro_batches
            )

        activations: Dict[Tuple[int, int], torch.Tensor] = {}
        results: List[MicroBatchResult] = []

        for mb_id, micro_input in enumerate(micro_inputs):
            x = micro_input
            for stage_id, stage in enumerate(self.stages):
                x = stage(x)
                activations[(stage_id, mb_id)] = x

            result = MicroBatchResult(micro_batch_id=mb_id, output=x)

            if self.loss_fn is not None and micro_targets is not None:
                result.loss = self.loss_fn(x, micro_targets[mb_id])

            results.append(result)

        return results

    def get_schedule(self) -> List[ScheduleStep]:
        """Get the schedule steps for the current configuration.

        Returns:
            List of ScheduleStep objects.
        """
        if self.schedule_type == ScheduleType.GPIPE:
            return generate_gpipe_schedule(self.num_stages, self.num_micro_batches)
        return generate_1f1b_schedule(self.num_stages, self.num_micro_batches)

    def get_bubble_ratio(self) -> float:
        """Compute the pipeline bubble ratio. Lower is better.

        Returns:
            Bubble ratio between 0 and 1.
        """
        m = self.num_micro_batches
        p = self.num_stages
        if self.schedule_type == ScheduleType.GPIPE:
            return (p - 1) / (m + p - 1)
        return (p - 1) / (2 * m + p - 1)


if __name__ == "__main__":
    gpipe = generate_gpipe_schedule(num_stages=4, num_micro_batches=8)
    fwd = sum(1 for s in gpipe if s.is_forward)
    print(f"GPipe: {fwd} forward + {len(gpipe) - fwd} backward steps")
    print(f"1F1B: {len(generate_1f1b_schedule(4, 8))} total steps")
    engine = PipelineEngine([], ScheduleType.GPIPE, num_micro_batches=8)
    engine.num_stages = 4
    print(f"GPipe bubble: {engine.get_bubble_ratio():.2%}")
    engine.schedule_type = ScheduleType.ONE_F_ONE_B
    print(f"1F1B bubble: {engine.get_bubble_ratio():.2%}")
