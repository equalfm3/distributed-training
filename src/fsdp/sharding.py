"""FSDP sharding strategy configuration.

Provides factory functions for FSDP sharding strategies, mixed precision
policies, and wrapping policies. Supports full shard, shard grad op,
and no shard modes with configurable precision.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from functools import partial

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


class ShardMode(str, Enum):
    """Available FSDP sharding modes."""

    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"


class PrecisionMode(str, Enum):
    """Mixed precision configurations."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


_SHARD_STRATEGY_MAP: Dict[ShardMode, ShardingStrategy] = {
    ShardMode.FULL_SHARD: ShardingStrategy.FULL_SHARD,
    ShardMode.SHARD_GRAD_OP: ShardingStrategy.SHARD_GRAD_OP,
    ShardMode.NO_SHARD: ShardingStrategy.NO_SHARD,
    ShardMode.HYBRID_SHARD: ShardingStrategy.HYBRID_SHARD,
}


@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapping.

    Attributes:
        shard_mode: How parameters are sharded across ranks.
        precision_mode: Mixed precision policy to apply.
        min_num_params: Minimum parameter count for auto-wrapping.
        transformer_layer_cls: Set of layer classes for transformer wrapping.
        use_activation_checkpointing: Whether to enable gradient checkpointing.
        cpu_offload: Whether to offload parameters to CPU.
        forward_prefetch: Whether to prefetch next FSDP unit during forward.
        limit_all_gathers: Whether to rate-limit all-gather operations.
    """

    shard_mode: ShardMode = ShardMode.FULL_SHARD
    precision_mode: PrecisionMode = PrecisionMode.BF16
    min_num_params: int = 100_000
    transformer_layer_cls: Set[Type[nn.Module]] = field(default_factory=set)
    use_activation_checkpointing: bool = True
    cpu_offload: bool = False
    forward_prefetch: bool = True
    limit_all_gathers: bool = True


def get_sharding_strategy(mode: ShardMode) -> ShardingStrategy:
    """Map a ShardMode enum to a PyTorch ShardingStrategy.

    Args:
        mode: The desired sharding mode.

    Returns:
        Corresponding torch ShardingStrategy.

    Raises:
        ValueError: If the mode is not recognized.
    """
    if mode not in _SHARD_STRATEGY_MAP:
        raise ValueError(f"Unknown shard mode: {mode}. Choose from {list(ShardMode)}")
    return _SHARD_STRATEGY_MAP[mode]


def get_mixed_precision_policy(mode: PrecisionMode) -> Optional[MixedPrecision]:
    """Create a MixedPrecision policy for FSDP.

    Args:
        mode: The precision mode to configure.

    Returns:
        MixedPrecision policy, or None for full FP32.
    """
    if mode == PrecisionMode.FP32:
        return None

    dtype_map = {
        PrecisionMode.FP16: torch.float16,
        PrecisionMode.BF16: torch.bfloat16,
    }
    dtype = dtype_map[mode]

    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def get_auto_wrap_policy(config: FSDPConfig) -> Any:
    """Create an FSDP auto-wrap policy based on configuration.

    If transformer layer classes are specified, uses transformer-aware
    wrapping. Otherwise falls back to size-based wrapping.

    Args:
        config: FSDP configuration with wrapping parameters.

    Returns:
        A callable wrapping policy for FSDP.
    """
    if config.transformer_layer_cls:
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=config.transformer_layer_cls,
        )
    return partial(
        size_based_auto_wrap_policy,
        min_num_params=config.min_num_params,
    )


def build_fsdp_kwargs(config: FSDPConfig) -> Dict[str, Any]:
    """Build keyword arguments for FSDP constructor.

    Assembles sharding strategy, mixed precision, wrapping policy, and
    other options into a dict suitable for passing to FSDP().

    Args:
        config: Complete FSDP configuration.

    Returns:
        Dictionary of keyword arguments for FSDP wrapping.
    """
    kwargs: Dict[str, Any] = {
        "sharding_strategy": get_sharding_strategy(config.shard_mode),
        "auto_wrap_policy": get_auto_wrap_policy(config),
        "forward_prefetch": config.forward_prefetch,
        "limit_all_gathers": config.limit_all_gathers,
    }

    mp_policy = get_mixed_precision_policy(config.precision_mode)
    if mp_policy is not None:
        kwargs["mixed_precision"] = mp_policy

    if config.cpu_offload:
        from torch.distributed.fsdp import CPUOffload
        kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    return kwargs


def wrap_model_with_fsdp(
    model: nn.Module,
    config: Optional[FSDPConfig] = None,
) -> FSDP:
    """Wrap a model with Fully Sharded Data Parallel.

    Args:
        model: The model to wrap.
        config: FSDP configuration. Uses defaults if None.

    Returns:
        The FSDP-wrapped model.
    """
    if config is None:
        config = FSDPConfig()

    fsdp_kwargs = build_fsdp_kwargs(config)
    wrapped = FSDP(model, **fsdp_kwargs)

    if config.use_activation_checkpointing:
        apply_activation_checkpointing(wrapped, config)

    return wrapped


def apply_activation_checkpointing(
    model: FSDP,
    config: FSDPConfig,
) -> None:
    """Enable activation checkpointing on FSDP-wrapped model.

    Applies gradient checkpointing to transformer layers if specified,
    otherwise to all FSDP units.

    Args:
        model: FSDP-wrapped model.
        config: Configuration with layer class info.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing as _apply_ac,
        checkpoint_wrapper,
    )

    check_fn = None
    if config.transformer_layer_cls:
        check_fn = lambda module: isinstance(module, tuple(config.transformer_layer_cls))

    _apply_ac(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn,
    )


if __name__ == "__main__":
    config = FSDPConfig(
        shard_mode=ShardMode.FULL_SHARD,
        precision_mode=PrecisionMode.BF16,
    )
    kwargs = build_fsdp_kwargs(config)
    print(f"FSDP kwargs: {list(kwargs.keys())}")
    print(f"Sharding: {kwargs['sharding_strategy']}")
