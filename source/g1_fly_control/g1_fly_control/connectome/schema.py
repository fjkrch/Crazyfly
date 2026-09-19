"""Schema and validation for derived, simulation-ready connectomes.

The schema intentionally separates a source/provenance manifest from the derived edge weights.
Weights are model values, not claims about physical synaptic conductance.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch


class ConnectomeValidationError(ValueError):
    """Raised when a circuit cannot safely be used as a frozen model."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CircuitManifest:
    """Provenance and paths for one frozen circuit transform."""

    path: Path
    schema_version: int
    status: str
    source_release: str
    extraction_rule: str
    neuron_subset: str
    duplicate_edge_policy: str
    weight_mapping: str
    neuron_model: dict[str, Any]
    neurons_path: Path
    edges_path: Path
    expected_checksums: dict[str, str]
    input_neuron_ids: tuple[str, ...]
    output_neuron_ids: tuple[str, ...]
    raw: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CircuitManifest":
        manifest_path = Path(path).expanduser().resolve()
        if not manifest_path.is_file():
            raise ConnectomeValidationError(
                f"Connectome manifest is required and was not found: {manifest_path}. "
                "Supply a real-data manifest; synthetic fixtures are test-only."
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConnectomeValidationError(f"Invalid JSON in manifest {manifest_path}: {exc}") from exc
        required = {
            "schema_version", "status", "source_release", "extraction_rule", "neuron_subset",
            "duplicate_edge_policy", "weight_mapping", "neuron_model", "neurons_path", "edges_path",
            "input_neuron_ids", "output_neuron_ids",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ConnectomeValidationError(f"Manifest missing required fields: {', '.join(missing)}")
        if raw["schema_version"] != 1:
            raise ConnectomeValidationError(f"Unsupported manifest schema_version: {raw['schema_version']!r}")
        if raw["status"] not in {"real", "synthetic"}:
            raise ConnectomeValidationError("Manifest status must be 'real' or 'synthetic'.")
        if not isinstance(raw["neuron_model"], dict):
            raise ConnectomeValidationError("neuron_model must be an object with declared LIF constants.")
        checksums = raw.get("checksums", {})
        if not isinstance(checksums, dict):
            raise ConnectomeValidationError("checksums must be an object when supplied.")
        if raw["status"] == "real" and not {"neurons", "edges"}.issubset(checksums):
            raise ConnectomeValidationError("Real manifests require SHA-256 checksums for neurons and edges.")
        for label in ("input_neuron_ids", "output_neuron_ids"):
            if not isinstance(raw[label], list) or not raw[label] or not all(isinstance(item, str) and item for item in raw[label]):
                raise ConnectomeValidationError(f"{label} must be a non-empty list of stable neuron IDs.")
        return cls(
            path=manifest_path,
            schema_version=raw["schema_version"],
            status=raw["status"],
            source_release=str(raw["source_release"]),
            extraction_rule=str(raw["extraction_rule"]),
            neuron_subset=str(raw["neuron_subset"]),
            duplicate_edge_policy=str(raw["duplicate_edge_policy"]),
            weight_mapping=str(raw["weight_mapping"]),
            neuron_model=raw["neuron_model"],
            neurons_path=(manifest_path.parent / raw["neurons_path"]).resolve(),
            edges_path=(manifest_path.parent / raw["edges_path"]).resolve(),
            expected_checksums={str(k): str(v) for k, v in checksums.items()},
            input_neuron_ids=tuple(raw["input_neuron_ids"]),
            output_neuron_ids=tuple(raw["output_neuron_ids"]),
            raw=raw,
        )

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json(self.raw)).hexdigest()

    def assert_checksum(self, label: str, path: Path) -> None:
        if not path.is_file():
            raise ConnectomeValidationError(f"Manifest references missing {label} file: {path}")
        expected = self.expected_checksums.get(label)
        if expected is not None and file_sha256(path) != expected:
            raise ConnectomeValidationError(f"Checksum mismatch for {label}: {path}")


@dataclass(frozen=True)
class CircuitData:
    """A validated directed graph using the explicit ``pre -> post`` convention."""

    manifest: CircuitManifest
    neuron_ids: tuple[str, ...]
    edge_index: torch.Tensor  # [2, E], row 0 is pre, row 1 is post
    weights: torch.Tensor  # [E]
    annotations: dict[str, dict[str, Any]]

    @property
    def num_neurons(self) -> int:
        return len(self.neuron_ids)

    @property
    def checksum(self) -> str:
        payload = {
            "manifest": self.manifest.fingerprint,
            "neuron_ids": self.neuron_ids,
            "edge_index": self.edge_index.cpu().tolist(),
            "weights": [float(x) for x in self.weights.cpu().tolist()],
        }
        return sha256(canonical_json(payload)).hexdigest()

    def dense_weight_matrix(self) -> torch.Tensor:
        """Return W[post, pre], matching the core's recurrent-current convention."""
        matrix = torch.zeros((self.num_neurons, self.num_neurons), dtype=self.weights.dtype)
        matrix.index_put_((self.edge_index[1], self.edge_index[0]), self.weights, accumulate=True)
        return matrix


def validate_circuit_records(manifest: CircuitManifest, neurons: list[dict[str, Any]], edges: list[dict[str, Any]]) -> CircuitData:
    if not neurons:
        raise ConnectomeValidationError("Circuit contains no neurons.")
    ids = [str(row.get("id", "")) for row in neurons]
    if any(not item for item in ids):
        raise ConnectomeValidationError("Every neuron must have a non-empty stable id.")
    if len(set(ids)) != len(ids):
        raise ConnectomeValidationError("Neuron IDs are not unique.")
    id_to_index = {value: index for index, value in enumerate(ids)}
    for label, subset in (("input_neuron_ids", manifest.input_neuron_ids), ("output_neuron_ids", manifest.output_neuron_ids)):
        if len(set(subset)) != len(subset) or any(item not in id_to_index for item in subset):
            raise ConnectomeValidationError(f"{label} contains duplicate or unknown IDs.")
    annotations = {str(row["id"]): dict(row.get("annotations", {})) for row in neurons}

    pairs: set[tuple[str, str]] = set()
    pre: list[int] = []
    post: list[int] = []
    weights: list[float] = []
    for edge_index, row in enumerate(edges):
        try:
            src, dst = str(row["pre"]), str(row["post"])
            weight = float(row["weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConnectomeValidationError(f"Invalid edge at index {edge_index}: {row!r}") from exc
        if src not in id_to_index or dst not in id_to_index:
            raise ConnectomeValidationError(f"Edge endpoint is not in neuron set: {src!r} -> {dst!r}")
        if not torch.isfinite(torch.tensor(weight)):
            raise ConnectomeValidationError(f"Edge weight is not finite: {src!r} -> {dst!r}")
        pair = (src, dst)
        if pair in pairs and manifest.duplicate_edge_policy == "reject":
            raise ConnectomeValidationError(f"Duplicate edge under reject policy: {src!r} -> {dst!r}")
        pairs.add(pair)
        pre.append(id_to_index[src])
        post.append(id_to_index[dst])
        weights.append(weight)
    if manifest.duplicate_edge_policy not in {"reject", "sum"}:
        raise ConnectomeValidationError("duplicate_edge_policy must be 'reject' or 'sum'.")
    if manifest.duplicate_edge_policy == "sum":
        aggregate: dict[tuple[int, int], float] = {}
        for source, target, value in zip(pre, post, weights, strict=True):
            aggregate[(source, target)] = aggregate.get((source, target), 0.0) + value
        pre, post, weights = [], [], []
        for (source, target), value in sorted(aggregate.items()):
            pre.append(source)
            post.append(target)
            weights.append(value)
    return CircuitData(
        manifest=manifest,
        neuron_ids=tuple(ids),
        edge_index=torch.tensor([pre, post], dtype=torch.long),
        weights=torch.tensor(weights, dtype=torch.float32),
        annotations=annotations,
    )
