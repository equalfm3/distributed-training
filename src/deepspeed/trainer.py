"""DeepSpeed training loop with ZeRO configuration.

Implements a training loop using DeepSpeed ZeRO optimization stages.
Supports ZeRO-1 (optimizer partitioning), ZeRO-2 (gradient partitioning),
and ZeRO-3 (parameter partitioning). Falls back to standard PyTorch
training when DeepSpeed is unavailable.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class DeepSpeedTrainConfig:
    """Training configuration for DeepSpeed."""

    vocab_size: int = 1024
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    seq_len: int = 64
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lr: float = 3e-4
    max_steps: int = 100
    log_interval: int = 10
    ds_config_path: Optional[str] = None


class SimpleTransformerDS(nn.Module):
    """Decoder-only transformer for DeepSpeed training.

    Args:
        vocab_size: Token vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of decoder layers.
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.seq_len = seq_len

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with loss computation.

        Args:
            input_ids: Input token indices (batch, seq_len).
            labels: Target token indices (batch, seq_len).

        Returns:
            Tuple of (loss, logits).
        """
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )
        x = self.decoder(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return loss, logits


class SyntheticDatasetDS(Dataset):
    """Random token dataset for DeepSpeed training demos."""

    def __init__(self, vocab_size: int = 1024, seq_len: int = 64, num_samples: int = 1000) -> None:
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_default_ds_config(config: DeepSpeedTrainConfig) -> Dict[str, Any]:
    """Create a default DeepSpeed ZeRO-2 configuration dict."""
    return {
        "train_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.lr,
                "warmup_num_steps": min(100, config.max_steps // 10),
                "total_num_steps": config.max_steps,
            },
        },
        "fp16": {"enabled": torch.cuda.is_available()},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }


def load_ds_config(path: Optional[str], config: DeepSpeedTrainConfig) -> Dict[str, Any]:
    """Load DeepSpeed config from file or generate defaults.

    Args:
        path: Path to JSON config file. None for defaults.
        config: Training config for default generation.

    Returns:
        DeepSpeed configuration dictionary.
    """
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return build_default_ds_config(config)


def train_with_deepspeed(config: Optional[DeepSpeedTrainConfig] = None) -> None:
    """Run training with DeepSpeed engine.

    Falls back to standard PyTorch training if DeepSpeed is not installed.

    Args:
        config: Training configuration. Uses defaults if None.
    """
    if config is None:
        config = DeepSpeedTrainConfig()

    try:
        import deepspeed
        _train_deepspeed(config)
    except ImportError:
        print("DeepSpeed not available, falling back to standard training")
        _train_standard(config)


def _train_deepspeed(config: DeepSpeedTrainConfig) -> None:
    """Internal DeepSpeed training implementation.

    Args:
        config: Training configuration.
    """
    import deepspeed

    model = SimpleTransformerDS(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
    )

    ds_config = load_ds_config(config.ds_config_path, config)
    dataset = SyntheticDatasetDS(config.vocab_size, config.seq_len, num_samples=config.max_steps * config.batch_size * 2)

    engine, optimizer, dataloader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )

    engine.train()
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        if step >= config.max_steps:
            break

        input_ids = batch["input_ids"].to(engine.device)
        labels = batch["labels"].to(engine.device)

        loss, _ = engine(input_ids, labels)
        engine.backward(loss)
        engine.step()

        if engine.local_rank == 0 and (step + 1) % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * config.batch_size * (config.seq_len - 1) / elapsed
            print(f"Step {step + 1}/{config.max_steps} | Loss: {loss.item():.4f} | Tok/s: {tokens_per_sec:.0f}")

    if engine.local_rank == 0:
        print(f"DeepSpeed training complete in {time.time() - start_time:.1f}s")


def _train_standard(config: DeepSpeedTrainConfig) -> None:
    """Fallback standard PyTorch training loop.

    Args:
        config: Training configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleTransformerDS(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    dataset = SyntheticDatasetDS(config.vocab_size, config.seq_len, num_samples=config.max_steps * config.batch_size * 2)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model.train()
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        if step >= config.max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        loss, _ = model(input_ids, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * config.batch_size * (config.seq_len - 1) / elapsed
            print(f"Step {step + 1}/{config.max_steps} | Loss: {loss.item():.4f} | Tok/s: {tokens_per_sec:.0f}")

    print(f"Standard training complete in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeed Trainer")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config path")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank (set by deepspeed)")
    args = parser.parse_args()

    train_config = DeepSpeedTrainConfig(max_steps=args.steps, ds_config_path=args.deepspeed)
    train_with_deepspeed(train_config)
