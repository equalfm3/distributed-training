"""Column-parallel and row-parallel linear layers for tensor parallelism.

Implements Megatron-style tensor parallelism where linear layers are split
across devices. Column-parallel splits the output dimension; row-parallel
splits the input dimension. Communication (all-reduce or reduce-scatter)
synchronizes partial results.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tensor_parallel.comm import (
    all_gather_tensor,
    all_reduce_tensor,
    get_tensor_parallel_group,
    get_tensor_parallel_world_size,
    reduce_scatter_tensor,
    scatter_tensor,
)


class _AllReduceFunc(torch.autograd.Function):
    """Autograd function that all-reduces in the forward pass.

    Used in row-parallel layers to sum partial outputs across ranks.
    The backward pass is an identity (gradients flow through unchanged).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """All-reduce the input tensor across the tensor parallel group.

        Args:
            ctx: Autograd context.
            x: Input tensor to reduce.

        Returns:
            All-reduced tensor.
        """
        return all_reduce_tensor(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Pass gradients through unchanged.

        Args:
            ctx: Autograd context.
            grad: Gradient tensor.

        Returns:
            Unchanged gradient.
        """
        return grad


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.

    Used in column-parallel layers so that gradients from downstream
    row-parallel layers are properly aggregated.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Pass input through unchanged.

        Args:
            ctx: Autograd context.
            x: Input tensor.

        Returns:
            Unchanged input.
        """
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """All-reduce gradients across the tensor parallel group.

        Args:
            ctx: Autograd context.
            grad: Gradient tensor.

        Returns:
            All-reduced gradient.
        """
        return all_reduce_tensor(grad)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    Splits the weight matrix along the output dimension across ranks.
    Each rank computes a slice of the output. Optionally gathers the
    full output or leaves it partitioned for a downstream row-parallel layer.

    Args:
        in_features: Input feature dimension.
        out_features: Total output feature dimension (before splitting).
        bias: Whether to include a bias term.
        gather_output: If True, all-gather output across ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        tp_size = get_tensor_parallel_world_size()
        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"tensor parallel size ({tp_size})"
            )

        self.out_features_per_rank = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with scaled normal distribution."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute column-parallel linear transformation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor. Shape (..., out_features) if gather_output,
            otherwise (..., out_features_per_rank).
        """
        x = _IdentityForwardAllReduceBackward.apply(x)
        output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            output = all_gather_tensor(output, dim=-1)

        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    Splits the weight matrix along the input dimension across ranks.
    Each rank computes a partial output, then results are all-reduced
    to produce the final output.

    Args:
        in_features: Total input feature dimension (before splitting).
        out_features: Output feature dimension.
        bias: Whether to include a bias term.
        input_is_parallel: If True, input is already split across ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        tp_size = get_tensor_parallel_world_size()
        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"tensor parallel size ({tp_size})"
            )

        self.in_features_per_rank = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with scaled normal distribution."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute row-parallel linear transformation.

        Args:
            x: Input tensor. Shape (..., in_features) if not parallel,
                (..., in_features_per_rank) if already split.

        Returns:
            Output tensor of shape (..., out_features).
        """
        if not self.input_is_parallel:
            x = scatter_tensor(x, dim=-1)

        output = F.linear(x, self.weight)
        output = _AllReduceFunc.apply(output)

        if self.bias is not None:
            output = output + self.bias

        return output


class TensorParallelMLP(nn.Module):
    """Two-layer MLP with tensor parallelism.

    Uses column-parallel for the first linear (expanding) and
    row-parallel for the second linear (contracting). This avoids
    communication between the two layers.

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Expanded intermediate dimension.
        activation: Activation function to use.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size, intermediate_size, gather_output=False
        )
        self.fc2 = RowParallelLinear(
            intermediate_size, hidden_size, input_is_parallel=True
        )
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the tensor-parallel MLP.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    col = ColumnParallelLinear(512, 2048, gather_output=True)
    x = torch.randn(2, 10, 512)
    out = col(x)
    print(f"ColumnParallel: {x.shape} -> {out.shape}")

    row = RowParallelLinear(2048, 512, input_is_parallel=False)
    out2 = row(torch.randn(2, 10, 2048))
    print(f"RowParallel: (2, 10, 2048) -> {out2.shape}")
