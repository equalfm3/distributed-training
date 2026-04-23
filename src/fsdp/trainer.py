"""FSDP training loop with a simple transformer model.

Implements a complete training loop using Fully Sharded Data Parallel
with mixed precision, gradient checkpointing, and gradient clipping.
Works with torchrun for multi-GPU and single-process CPU mode.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.utils.distributed import (
    DistributedInfo,
    cleanup,
    get_rank,
    get_world_size,
    init_process_group,
    is_main_process,
)
from src.fsdp.sharding import FSDPConfig, PrecisionMode, ShardMode, wrap_model_with_fsdp


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        vocab_size: Size of the token vocabulary.
        d_model: Hidden dimension of the transformer.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        seq_len: Maximum sequence length.
        batch_size: Per-device batch size.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        max_steps: Total training steps.
        grad_clip: Maximum gradient norm.
        log_interval: Steps between log messages.
    """

    vocab_size: int = 1024
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    seq_len: int = 64
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100
    grad_clip: float = 1.0
    log_interval: int = 10


class SimpleTransformer(nn.Module):
    """Minimal decoder-only transformer for training demonstrations.

    Args:
        vocab_size: Token vocabulary size.
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer decoder layers.
        seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits.

        Args:
            input_ids: Token indices of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )
        x = self.decoder(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)
        x = self.ln_f(x)
        return self.head(x)


class SyntheticDataset(Dataset):
    """Generates random token sequences for training demos.

    Args:
        vocab_size: Range of token values.
        seq_len: Length of each sequence.
        num_samples: Total dataset size.
    """

    def __init__(self, vocab_size: int = 1024, seq_len: int = 64, num_samples: int = 1000) -> None:
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


def create_dataloader(config: TrainConfig, dist_info: DistributedInfo) -> DataLoader:
    """Build a DataLoader with optional distributed sampling.

    Args:
        config: Training configuration.
        dist_info: Distributed environment info.

    Returns:
        DataLoader yielding (input, target) batches.
    """
    dataset = SyntheticDataset(config.vocab_size, config.seq_len, num_samples=config.max_steps * config.batch_size * 2)

    sampler = None
    if dist_info.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=dist_info.world_size, rank=dist_info.rank
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )


def train(config: Optional[TrainConfig] = None) -> None:
    """Run the FSDP training loop.

    Initializes distributed, wraps the model with FSDP, and trains
    on synthetic data with mixed precision and gradient clipping.

    Args:
        config: Training configuration. Uses defaults if None.
    """
    if config is None:
        config = TrainConfig()

    dist_info = init_process_group()

    model = SimpleTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
    ).to(dist_info.device)

    fsdp_config = FSDPConfig(
        shard_mode=ShardMode.FULL_SHARD,
        precision_mode=PrecisionMode.BF16 if dist_info.device.type == "cuda" else PrecisionMode.FP32,
        min_num_params=1000,
        use_activation_checkpointing=False,
    )
    model = wrap_model_with_fsdp(model, fsdp_config)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_steps)
    loss_fn = nn.CrossEntropyLoss()

    dataloader = create_dataloader(config, dist_info)

    if is_main_process():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        print(f"Training for {config.max_steps} steps on {dist_info.world_size} device(s)")

    model.train()
    step = 0
    start_time = time.time()

    for inputs, targets in dataloader:
        if step >= config.max_steps:
            break

        inputs = inputs.to(dist_info.device)
        targets = targets.to(dist_info.device)

        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, config.vocab_size), targets.reshape(-1))

        loss.backward()
        model.clip_grad_norm_(config.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if is_main_process() and (step + 1) % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * config.batch_size * (config.seq_len - 1) / elapsed
            lr_current = scheduler.get_last_lr()[0]
            print(
                f"Step {step + 1}/{config.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr_current:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )

        step += 1

    if is_main_process():
        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s ({step} steps)")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP Trainer")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    config = TrainConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    train(config)
