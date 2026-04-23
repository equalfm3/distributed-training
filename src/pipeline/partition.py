"""Model partitioning across pipeline stages.

Splits a transformer model into sequential stages for pipeline parallelism.
Each stage contains a contiguous subset of layers and can be placed on a
different device. Supports balanced and manual partitioning strategies.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StageSpec:
    """Specification for a single pipeline stage.

    Attributes:
        stage_id: Index of this stage in the pipeline.
        layer_indices: Which transformer layers belong to this stage.
        device: Device this stage runs on.
        has_embedding: Whether this stage contains the embedding layer.
        has_head: Whether this stage contains the output head.
    """

    stage_id: int
    layer_indices: List[int]
    device: torch.device
    has_embedding: bool = False
    has_head: bool = False


@dataclass
class PipelinePartition:
    """Complete pipeline partition specification.

    Attributes:
        stages: List of stage specifications.
        num_stages: Total number of pipeline stages.
        num_layers: Total number of transformer layers.
    """

    stages: List[StageSpec] = field(default_factory=list)
    num_stages: int = 0
    num_layers: int = 0


def compute_balanced_partition(
    num_layers: int,
    num_stages: int,
) -> List[List[int]]:
    """Divide layers evenly across stages.

    Distributes layers as evenly as possible, with earlier stages
    receiving extra layers when the division is uneven.

    Args:
        num_layers: Total number of transformer layers.
        num_stages: Number of pipeline stages.

    Returns:
        List of layer index lists, one per stage.

    Raises:
        ValueError: If num_stages exceeds num_layers.
    """
    if num_stages > num_layers:
        raise ValueError(
            f"Cannot partition {num_layers} layers into {num_stages} stages"
        )

    base_size = num_layers // num_stages
    remainder = num_layers % num_stages

    partitions: List[List[int]] = []
    start = 0
    for i in range(num_stages):
        size = base_size + (1 if i < remainder else 0)
        partitions.append(list(range(start, start + size)))
        start += size

    return partitions


def compute_cost_balanced_partition(
    layer_costs: List[float],
    num_stages: int,
) -> List[List[int]]:
    """Partition layers by estimated compute cost.

    Uses dynamic programming to minimize the maximum stage cost,
    balancing the pipeline for heterogeneous layer sizes.

    Args:
        layer_costs: Estimated compute cost per layer.
        num_stages: Number of pipeline stages.

    Returns:
        List of layer index lists, one per stage.
    """
    n = len(layer_costs)
    if num_stages >= n:
        return [[i] for i in range(n)]

    prefix_sum = [0.0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + layer_costs[i]

    def range_cost(start: int, end: int) -> float:
        return prefix_sum[end] - prefix_sum[start]

    dp = [[float("inf")] * (num_stages + 1) for _ in range(n + 1)]
    split = [[0] * (num_stages + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for j in range(1, num_stages + 1):
        for i in range(j, n + 1):
            for k in range(j - 1, i):
                cost = max(dp[k][j - 1], range_cost(k, i))
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    partitions: List[List[int]] = []
    end = n
    for j in range(num_stages, 0, -1):
        start = split[end][j]
        partitions.append(list(range(start, end)))
        end = start
    partitions.reverse()

    return partitions


def create_partition(
    num_layers: int,
    num_stages: int,
    devices: Optional[List[torch.device]] = None,
    layer_costs: Optional[List[float]] = None,
) -> PipelinePartition:
    """Create a pipeline partition specification.

    Args:
        num_layers: Total transformer layers in the model.
        num_stages: Number of pipeline stages.
        devices: Device for each stage. Defaults to CPU for all.
        layer_costs: Per-layer compute costs for cost-balanced partitioning.
            If None, uses balanced (equal count) partitioning.

    Returns:
        PipelinePartition with stage specifications.
    """
    if devices is None:
        devices = [torch.device("cpu")] * num_stages

    if layer_costs is not None:
        layer_groups = compute_cost_balanced_partition(layer_costs, num_stages)
    else:
        layer_groups = compute_balanced_partition(num_layers, num_stages)

    stages: List[StageSpec] = []
    for i, (layers, device) in enumerate(zip(layer_groups, devices)):
        stages.append(
            StageSpec(
                stage_id=i,
                layer_indices=layers,
                device=device,
                has_embedding=(i == 0),
                has_head=(i == num_stages - 1),
            )
        )

    return PipelinePartition(
        stages=stages,
        num_stages=num_stages,
        num_layers=num_layers,
    )


class PipelineStage(nn.Module):
    """A single stage in a pipeline-parallel model.

    Contains a subset of transformer layers and optionally the
    embedding or output head.

    Args:
        layers: Sequential module of transformer layers for this stage.
        spec: Stage specification with metadata.
        embedding: Token embedding module (first stage only).
        pos_embedding: Positional embedding module (first stage only).
        ln_f: Final layer norm (last stage only).
        head: Output projection head (last stage only).
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        spec: StageSpec,
        embedding: Optional[nn.Embedding] = None,
        pos_embedding: Optional[nn.Embedding] = None,
        ln_f: Optional[nn.LayerNorm] = None,
        head: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.spec = spec
        self.embedding = embedding
        self.pos_embedding = pos_embedding
        self.ln_f = ln_f
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through this pipeline stage.

        Args:
            x: Input tensor. Token IDs for the first stage, hidden
                states for intermediate stages.

        Returns:
            Hidden states or logits (last stage).
        """
        if self.spec.has_embedding and self.embedding is not None:
            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        if self.spec.has_head and self.ln_f is not None:
            x = self.ln_f(x)
            x = self.head(x)

        return x


def partition_transformer(
    model: nn.Module,
    partition: PipelinePartition,
) -> List[PipelineStage]:
    """Split a transformer model into pipeline stages.

    Extracts layers from the model according to the partition spec
    and creates PipelineStage modules.

    Args:
        model: Transformer model with .decoder.layers, .embedding,
            .pos_embedding, .ln_f, and .head attributes.
        partition: Partition specification.

    Returns:
        List of PipelineStage modules, one per stage.
    """
    all_layers = list(model.decoder.layers)
    stages: List[PipelineStage] = []

    for spec in partition.stages:
        stage_layers = nn.ModuleList([all_layers[i] for i in spec.layer_indices])

        stage = PipelineStage(
            layers=stage_layers,
            spec=spec,
            embedding=model.embedding if spec.has_embedding else None,
            pos_embedding=model.pos_embedding if spec.has_embedding else None,
            ln_f=model.ln_f if spec.has_head else None,
            head=model.head if spec.has_head else None,
        )
        stage = stage.to(spec.device)
        stages.append(stage)

    return stages


if __name__ == "__main__":
    partition = create_partition(num_layers=12, num_stages=4)
    for stage in partition.stages:
        print(
            f"Stage {stage.stage_id}: layers {stage.layer_indices}, "
            f"embed={stage.has_embedding}, head={stage.has_head}"
        )
