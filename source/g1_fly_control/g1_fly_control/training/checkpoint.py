"""Self-describing checkpoints for functional policy reloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT = 1


def save_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    normalizer_state: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer_state or {},
        "metadata": metadata,
        "torch_rng": torch.get_rng_state(),
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}")
    policy.load_state_dict(payload["policy"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

