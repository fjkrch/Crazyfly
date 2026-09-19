"""Load a circuit without silently substituting synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import CircuitData, CircuitManifest, ConnectomeValidationError, validate_circuit_records


def _records(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        raise ConnectomeValidationError(f"Manifest references missing {label} file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConnectomeValidationError(f"Invalid JSON in {label} file {path}: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ConnectomeValidationError(f"{label} file must be a JSON list of objects: {path}")
    return data


def load_connectome(path: str | Path, *, allow_synthetic: bool = False) -> CircuitData:
    """Load a validated circuit.

    Research callers keep the default. Tests must opt in to a manifest tagged
    ``synthetic`` so a missing biological dataset cannot turn into a fabricated run.
    """
    manifest = CircuitManifest.from_file(path)
    if manifest.status != "real" and not allow_synthetic:
        raise ConnectomeValidationError(
            "Synthetic circuit manifest rejected for research use. Provide a real MaleCNS-derived manifest."
        )
    for label, data_path in (("neurons", manifest.neurons_path), ("edges", manifest.edges_path)):
        manifest.assert_checksum(label, data_path)
    return validate_circuit_records(
        manifest,
        _records(manifest.neurons_path, "neurons"),
        _records(manifest.edges_path, "edges"),
    )

