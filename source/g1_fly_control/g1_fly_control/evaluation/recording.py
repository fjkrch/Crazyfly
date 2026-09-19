"""Compact selected-episode recording with configuration provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class EpisodeRecorder:
    """Collect tensors at control rate; callers select episodes/neuron subsets."""

    metadata: dict[str, Any]
    neuron_ids: list[str]
    selected_neurons: list[int] | None = None
    frames: list[dict[str, np.ndarray]] = field(default_factory=list)

    def append(self, timestamp_s: float, **signals: torch.Tensor | np.ndarray | float | int) -> None:
        frame: dict[str, np.ndarray] = {"timestamp_s": np.asarray(timestamp_s, dtype=np.float64)}
        for key, value in signals.items():
            array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if key in {"neural_voltage", "neural_spikes", "neural_filtered_rate"} and self.selected_neurons is not None:
                array = array[..., self.selected_neurons]
            frame[key] = array
        self.frames.append(frame)

    def save(self, destination: str | Path) -> Path:
        if not self.frames:
            raise ValueError("Cannot save an empty recording.")
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for frame in self.frames for key in frame})
        arrays = {key: np.stack([frame[key] for frame in self.frames]) for key in keys}
        arrays["metadata_json"] = np.asarray(json.dumps({
            **self.metadata,
            "neuron_ids": self.neuron_ids,
            "selected_neurons": self.selected_neurons,
            "control_samples": len(self.frames),
        }, sort_keys=True))
        np.savez_compressed(output, **arrays)
        return output

