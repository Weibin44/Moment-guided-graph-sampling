"""Shared node-classification split used by all EXP3 baselines."""

import torch


def get_split(num_samples: int, seed: int) -> dict[str, torch.Tensor]:
    """Return an identical 10% train / 10% validation / 80% test split."""
    train_size = int(num_samples * 0.1)
    test_size = int(num_samples * 0.8)
    indices = torch.randperm(
        num_samples, generator=torch.Generator().manual_seed(seed)
    )
    return {
        "train": indices[:train_size],
        "test": indices[train_size:train_size + test_size],
        "valid": indices[train_size + test_size:],
    }
