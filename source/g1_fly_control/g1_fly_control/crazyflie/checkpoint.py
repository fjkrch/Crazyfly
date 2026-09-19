"""Atomic, fingerprint-checked checkpoints for the Crazyflie experiment.

Counter semantics are explicit: ``completed_updates`` is the number of PPO
updates already committed, so it is also the zero-based index of the next
update.  Loading a checkpoint never increments either counter.  This avoids
the common resume bug that repeats a saved rollout or skips the next one.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import stat
import tempfile
from threading import RLock
from typing import Any

import numpy as np
import torch
from torch import nn

from .controllers import controller_core_checksum, verify_frozen_core


CHECKPOINT_FORMAT = "flyg1.crazyflie.training-checkpoint"
CHECKPOINT_VERSION = 2


class CheckpointValidationError(ValueError):
    """A checkpoint is malformed or internally inconsistent."""


class CheckpointCompatibilityError(CheckpointValidationError):
    """A valid checkpoint does not match the requested run."""


@dataclass(frozen=True)
class ResumeCounters:
    """Monotone counters at a completed rollout/update boundary."""

    completed_updates: int
    total_interactions: int
    completed_episodes: int = 0
    resume_count: int = 0
    resume_reset: bool = False

    def __post_init__(self) -> None:
        values = {
            "completed_updates": self.completed_updates,
            "total_interactions": self.total_interactions,
            "completed_episodes": self.completed_episodes,
            "resume_count": self.resume_count,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.resume_reset, bool):
            raise TypeError("resume_reset must be Boolean.")

    @classmethod
    def from_value(cls, value: "ResumeCounters | Mapping[str, Any]") -> "ResumeCounters":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("counters must be ResumeCounters or a mapping.")
        required = {"completed_updates", "total_interactions"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"Checkpoint counters are missing: {', '.join(missing)}")
        allowed = required | {"completed_episodes", "resume_count", "resume_reset"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown checkpoint counters: {', '.join(unknown)}")
        return cls(
            completed_updates=value["completed_updates"],
            total_interactions=value["total_interactions"],
            completed_episodes=value.get("completed_episodes", 0),
            resume_count=value.get("resume_count", 0),
            resume_reset=value.get("resume_reset", False),
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "completed_updates": self.completed_updates,
            "total_interactions": self.total_interactions,
            "completed_episodes": self.completed_episodes,
            "resume_count": self.resume_count,
            "resume_reset": self.resume_reset,
        }

    @property
    def next_update_index(self) -> int:
        return self.completed_updates

    def resumed(self, *, reset_environments: bool) -> "ResumeCounters":
        """Record a resume without changing already-counted work."""

        return replace(self, resume_count=self.resume_count + 1, resume_reset=bool(reset_environments))

    def advanced(self, *, rollout_interactions: int, completed_episodes: int = 0) -> "ResumeCounters":
        """Commit exactly one new PPO update after its rollout completes."""

        if (
            not isinstance(rollout_interactions, int)
            or isinstance(rollout_interactions, bool)
            or rollout_interactions < 1
        ):
            raise ValueError("rollout_interactions must be a positive integer.")
        if (
            not isinstance(completed_episodes, int)
            or isinstance(completed_episodes, bool)
            or completed_episodes < 0
        ):
            raise ValueError("completed_episodes must be a non-negative integer.")
        return replace(
            self,
            completed_updates=self.completed_updates + 1,
            total_interactions=self.total_interactions + rollout_interactions,
            completed_episodes=self.completed_episodes + completed_episodes,
        )


def validate_interaction_alignment(
    counters: ResumeCounters | Mapping[str, Any], interactions_per_update: int
) -> None:
    """Validate exact counting when every completed update has one fixed rollout."""

    state = ResumeCounters.from_value(counters)
    if (
        not isinstance(interactions_per_update, int)
        or isinstance(interactions_per_update, bool)
        or interactions_per_update < 1
    ):
        raise ValueError("interactions_per_update must be a positive integer.")
    expected = state.completed_updates * interactions_per_update
    if state.total_interactions != expected:
        raise CheckpointValidationError(
            f"Counter mismatch: {state.completed_updates} updates x {interactions_per_update} interactions "
            f"= {expected}, checkpoint records {state.total_interactions}."
        )


def _history_position(record: Mapping[str, Any], index: int) -> tuple[int, int]:
    missing = [key for key in ("completed_updates", "total_interactions") if key not in record]
    if missing:
        raise CheckpointValidationError(
            f"History record {index} is missing explicit counter fields: {', '.join(missing)}."
        )
    update = record["completed_updates"]
    interactions = record["total_interactions"]
    if (
        not isinstance(update, int)
        or isinstance(update, bool)
        or update < 0
        or not isinstance(interactions, int)
        or isinstance(interactions, bool)
        or interactions < 0
    ):
        raise CheckpointValidationError(f"History record {index} has invalid counters.")
    return update, interactions


def validate_history(
    history: Sequence[Mapping[str, Any]],
    counters: ResumeCounters | Mapping[str, Any] | None = None,
) -> None:
    """Reject repeated/out-of-order samples and samples beyond saved counters."""

    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence of mappings.")
    previous: tuple[int, int] | None = None
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise CheckpointValidationError(f"History record {index} is not a mapping.")
        position = _history_position(record, index)
        if previous is not None and (position[0] <= previous[0] or position[1] <= previous[1]):
            raise CheckpointValidationError(
                "Training history must have strictly increasing completed_updates and total_interactions; "
                f"record {index} would duplicate or reorder counted work."
            )
        previous = position
    if counters is not None and previous is not None:
        state = ResumeCounters.from_value(counters)
        if previous[0] > state.completed_updates or previous[1] > state.total_interactions:
            raise CheckpointValidationError("Training history extends beyond the checkpoint counters.")


def validate_history_alignment(
    history: Sequence[Mapping[str, Any]],
    counters: ResumeCounters | Mapping[str, Any],
    interactions_per_update: int,
) -> None:
    """Require exactly one contiguous metric row for every committed update."""

    state = ResumeCounters.from_value(counters)
    validate_interaction_alignment(state, interactions_per_update)
    validate_history(history, state)
    if len(history) != state.completed_updates:
        raise CheckpointValidationError(
            f"History has {len(history)} rows for {state.completed_updates} completed updates."
        )
    for index, record in enumerate(history):
        expected_update = index + 1
        expected_interactions = expected_update * interactions_per_update
        if _history_position(record, index) != (expected_update, expected_interactions):
            raise CheckpointValidationError(
                "Training history skips or invents counted work: "
                f"row {index} must be update {expected_update} at "
                f"{expected_interactions} interactions."
            )


def append_history_record(
    history: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return a validated copy with one strictly-new sample appended."""

    result = [dict(item) for item in history]
    validate_history(result)
    candidate = dict(record)
    _history_position(candidate, len(result))
    validate_history([*result, candidate])
    result.append(candidate)
    return result


def append_validated_history_record_inplace(
    history: list[dict[str, Any]],
    record: Mapping[str, Any],
) -> None:
    """Append one row to an already-validated live history in constant time.

    Training validates a loaded checkpoint history in full before entering the
    update loop, and fresh histories start empty.  Revalidating and copying all
    preceding rows on every one of 50,000 main-matrix updates would make that
    loop quadratic.  This narrow helper checks the new row and its boundary
    against the last validated row; checkpoint creation still validates the
    complete history and its immutable external segments.
    """

    if not isinstance(history, list):
        raise TypeError("history must be a mutable list.")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping.")
    candidate = dict(record)
    position = _history_position(candidate, len(history))
    if history:
        previous = history[-1]
        if not isinstance(previous, Mapping):
            raise CheckpointValidationError("The last validated history row is not a mapping.")
        previous_position = _history_position(previous, len(history) - 1)
        if position[0] <= previous_position[0] or position[1] <= previous_position[1]:
            raise CheckpointValidationError(
                "Training history must have strictly increasing completed_updates and "
                "total_interactions."
            )
    history.append(candidate)


def capture_rng_states(
    *,
    environment: Any = None,
    task_schedule: Any = None,
    include_cuda: bool | None = None,
) -> dict[str, Any]:
    """Capture global and caller-owned random generators without launching CUDA."""

    if include_cuda is None:
        include_cuda = torch.cuda.is_initialized()
    if include_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG capture was requested but CUDA is unavailable.")
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if include_cuda else None,
        "environment": deepcopy(environment),
        "task_schedule": deepcopy(task_schedule),
    }


def restore_rng_states(states: Mapping[str, Any], *, strict_cuda: bool = True) -> dict[str, Any]:
    """Restore global RNGs and return caller-owned environment/task states."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda", "environment", "task_schedule"}
    missing = sorted(required - states.keys())
    if missing:
        raise CheckpointValidationError(f"RNG state is missing: {', '.join(missing)}")
    cuda_states = states["torch_cuda"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            if strict_cuda:
                raise CheckpointCompatibilityError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
        elif len(cuda_states) != torch.cuda.device_count():
            if strict_cuda:
                raise CheckpointCompatibilityError(
                    f"Checkpoint has RNG states for {len(cuda_states)} CUDA devices; runtime has "
                    f"{torch.cuda.device_count()}."
                )
        else:
            # A whole-checkpoint ``map_location='cuda'`` also maps these byte
            # tensors.  Generator APIs require CPU byte-state tensors.
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"].detach().cpu())
    return {
        "environment": deepcopy(states["environment"]),
        "task_schedule": deepcopy(states["task_schedule"]),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-serializable checkpoint metadata.")


def _json_checksum(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _snapshot(value: Any) -> Any:
    """Detach tensors from live graphs and move them to CPU before serialization."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, OrderedDict):
        result = OrderedDict((key, _snapshot(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)  # type: ignore[attr-defined]
        return result
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: _snapshot(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return deepcopy(value)


def _normalize_command(command: str | Sequence[str]) -> str | list[str]:
    if isinstance(command, str):
        if not command.strip():
            raise ValueError("command must not be empty.")
        return command
    if isinstance(command, (bytes, bytearray)) or not isinstance(command, Sequence):
        raise TypeError("command must be a string or a sequence of arguments.")
    result = [str(argument) for argument in command]
    if not result:
        raise ValueError("command must not be empty.")
    return result


def _normalize_normalizers(
    normalizers: Mapping[str, Any] | None,
    observation_normalizer_state: Any,
    reward_normalizer_state: Any,
) -> dict[str, Any]:
    result = dict(normalizers or {})
    if observation_normalizer_state is not None:
        if "observation" in result:
            raise ValueError("Observation normalizer state was supplied twice.")
        result["observation"] = observation_normalizer_state
    if reward_normalizer_state is not None:
        if "reward" in result:
            raise ValueError("Reward normalizer state was supplied twice.")
        result["reward"] = reward_normalizer_state
    # Explicit entries make absence visible instead of conflating it with a
    # serialization mistake.
    result.setdefault("observation", None)
    result.setdefault("reward", None)
    return result


def _validate_fingerprints(fingerprints: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ValueError("fingerprints must be a non-empty mapping.")
    normalized = _json_safe(fingerprints)
    if not isinstance(normalized, dict):
        raise TypeError("fingerprints must normalize to an object.")
    for key, value in normalized.items():
        if not key or value == "":
            raise ValueError("Fingerprint keys and non-null values must not be empty.")
    return normalized


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        # The experiment queue runs one writer.  Recheck immediately before
        # the atomic rename so the default path cannot replace an earlier
        # numbered checkpoint during ordinary operation.
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def link_checkpoint_snapshot(source: str | Path, destination: str | Path) -> Path:
    """Durably archive an already-committed checkpoint without serializing twice.

    The source and destination must share a directory because external history
    references are relative to that checkpoint directory.  A hard link gives
    the numbered archive the exact bytes and inode that were already fsynced as
    ``latest.pt``.  Replacing ``latest.pt`` at the next boundary leaves the
    prior numbered inode intact.

    The authoritative source is deliberately not rolled back if linking or the
    directory fsync fails: it remains a valid restart boundary and a resume can
    retry the missing archive without repeating an update.
    """

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("Checkpoint source and archive destination must be distinct.")
    if source_path.parent != destination_path.parent:
        raise ValueError("Checkpoint archive must share the source checkpoint directory.")
    try:
        source_mode = os.lstat(source_path).st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Checkpoint source does not exist: {source_path}") from exc
    if not stat.S_ISREG(source_mode):
        raise ValueError(f"Checkpoint source must be a regular file: {source_path}")
    if os.path.lexists(destination_path):
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination_path}")

    os.link(source_path, destination_path, follow_symlinks=False)
    directory_fd = os.open(source_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination_path


def write_history_segment(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    reference_directory: str | Path,
) -> dict[str, Any]:
    """Atomically stream one immutable, non-overlapping JSONL history segment.

    Only one canonical encoded row is resident at a time.  This matters for
    long-running training: constructing a second list of metric dictionaries
    plus one segment-sized ``bytes`` object at every checkpoint fragmented the
    process allocator even though both objects became unreachable immediately.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("History records must be a sequence of mappings.")
    if len(records) == 0:
        raise ValueError("A history segment must contain at least one record.")
    validate_history(records)
    destination = Path(path).expanduser().resolve()
    reference_root = Path(reference_directory).expanduser().resolve()
    allowed_root = reference_root.parent.resolve()
    if not destination.is_relative_to(allowed_root):
        raise ValueError("History segments must remain inside the checkpoint run directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite history segment: {destination}")
    file_digest = sha256()
    byte_count = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                encoded = json.dumps(
                    _json_safe(dict(record)),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8") + b"\n"
                handle.write(encoded)
                file_digest.update(encoded)
                byte_count += len(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes the already-fsynced inode while
        # providing create-if-absent semantics.  Unlike an exists-check followed
        # by ``os.replace``, a concurrent writer can never be overwritten.
        try:
            os.link(temporary_path, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite history segment: {destination}") from exc
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    first_update, first_interactions = _history_position(records[0], 0)
    last_update, last_interactions = _history_position(records[-1], len(records) - 1)
    return {
        "path": os.path.relpath(destination, reference_root),
        "sha256": file_digest.hexdigest(),
        "byte_count": byte_count,
        "row_count": len(records),
        "first_completed_updates": first_update,
        "first_total_interactions": first_interactions,
        "last_completed_updates": last_update,
        "last_total_interactions": last_interactions,
    }


def build_history_reference(
    history: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the compact checkpoint cursor for immutable history segments."""

    rows = [dict(record) for record in history]
    validate_history(rows)
    normalized_segments = [_json_safe(dict(segment)) for segment in segments]
    if sum(int(segment.get("row_count", -1)) for segment in normalized_segments) != len(rows):
        raise CheckpointValidationError("History segment row counts do not cover the complete history.")
    last_update, last_interactions = _history_position(rows[-1], len(rows) - 1) if rows else (0, 0)
    reference = {
        "schema_version": 1,
        "storage": "immutable_jsonl_segments",
        "row_count": len(rows),
        "history_sha256": _json_checksum(rows),
        "last_completed_updates": last_update,
        "last_total_interactions": last_interactions,
        "segments": normalized_segments,
    }
    _validate_history_reference(reference, history=rows)
    return reference


def _validate_history_accumulator_arguments(
    segments: Sequence[Mapping[str, Any]], interactions_per_update: int
) -> None:
    if (
        not isinstance(interactions_per_update, int)
        or isinstance(interactions_per_update, bool)
        or interactions_per_update < 1
    ):
        raise ValueError("interactions_per_update must be a positive integer.")
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
        raise TypeError("segments must be a sequence of mappings.")


def _provisional_history_reference(
    segments: Sequence[Mapping[str, Any]], interactions_per_update: int
) -> dict[str, Any]:
    """Normalize and validate the closed version-1 segment-reference schema."""

    _validate_history_accumulator_arguments(segments, interactions_per_update)
    normalized_segments = [_json_safe(dict(segment)) for segment in segments]
    row_count = sum(int(segment.get("row_count", -1)) for segment in normalized_segments)
    previous_last = (0, 0)
    if normalized_segments:
        final_segment = normalized_segments[-1]
        previous_last = (
            int(final_segment.get("last_completed_updates", -1)),
            int(final_segment.get("last_total_interactions", -1)),
        )
    # Validate the closed segment schema, ordering, counts, and final cursor
    # before touching any referenced file.  The placeholder is replaced by
    # the streamed aggregate digest below.
    return _validate_history_reference({
        "schema_version": 1,
        "storage": "immutable_jsonl_segments",
        "row_count": row_count,
        "history_sha256": "0" * 64,
        "last_completed_updates": previous_last[0],
        "last_total_interactions": previous_last[1],
        "segments": normalized_segments,
    })


def _history_stat_signature(status: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(status.st_mode):
        raise CheckpointValidationError("History segment is not a regular file.")
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _history_file_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Return a cheap mutation signature for an authenticated segment."""

    return _history_stat_signature(path.stat())


class HistoryReferenceAccumulator:
    """Incrementally authenticate history segments and reproduce the v1 digest.

    A fresh process, including a resumed trainer, reconstructs this state once
    from immutable segments.  Later checkpoint boundaries authenticate only
    newly appended segments.  The running SHA-256 state represents the exact
    canonical JSON-list prefix, so :meth:`reference` remains byte-for-byte
    compatible with :func:`build_history_reference`.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        interactions_per_update: int,
    ) -> None:
        _validate_history_accumulator_arguments([], interactions_per_update)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        self._reference_directory = checkpoint.parent
        self._allowed_root = self._reference_directory.parent.resolve()
        self._interactions_per_update = interactions_per_update
        self._segments: list[dict[str, Any]] = []
        self._signatures: list[tuple[int, int, int, int, int]] = []
        self._history_digest = sha256()
        self._history_digest.update(b"[")
        self._row_count = 0

    @classmethod
    def reconstruct(
        cls,
        segments: Sequence[Mapping[str, Any]],
        *,
        checkpoint_path: str | Path,
        interactions_per_update: int,
    ) -> "HistoryReferenceAccumulator":
        """Authenticate an existing history once and return resumable state."""

        accumulator = cls(
            checkpoint_path=checkpoint_path,
            interactions_per_update=interactions_per_update,
        )
        accumulator.synchronize(segments)
        return accumulator

    @property
    def validated_segment_count(self) -> int:
        return len(self._segments)

    @property
    def row_count(self) -> int:
        return self._row_count

    def _source_path(self, segment: Mapping[str, Any]) -> Path:
        source = (self._reference_directory / segment["path"]).resolve()
        if not source.is_relative_to(self._allowed_root):
            raise CheckpointValidationError("History segment escapes the checkpoint run directory.")
        return source

    def _matches_authenticated_prefix(self, segments: Sequence[Mapping[str, Any]]) -> bool:
        if len(self._segments) > len(segments):
            return False
        if list(segments[:len(self._segments)]) != self._segments:
            return False
        try:
            return all(
                _history_file_signature(self._source_path(segment)) == signature
                for segment, signature in zip(self._segments, self._signatures)
            )
        except (OSError, CheckpointValidationError):
            return False

    def _append_authenticated_segment(
        self,
        segment: Mapping[str, Any],
        *,
        segment_index: int,
    ) -> None:
        """Validate one segment transactionally before advancing live state."""

        source = self._source_path(segment)
        candidate_digest = self._history_digest.copy()
        candidate_row_count = self._row_count
        file_digest = sha256()
        byte_count = 0
        segment_row_count = 0
        first_position: tuple[int, int] | None = None
        last_position: tuple[int, int] | None = None
        try:
            stream = source.open("rb")
        except OSError as exc:
            raise CheckpointValidationError(f"Cannot read history segment {source}: {exc}") from exc
        with stream:
            opened_signature = _history_stat_signature(os.fstat(stream.fileno()))
            for raw_line in stream:
                byte_count += len(raw_line)
                file_digest.update(raw_line)
                if not raw_line.endswith(b"\n") or raw_line == b"\n":
                    raise CheckpointValidationError(
                        f"History segment {segment_index} is not canonical newline-delimited JSON."
                    )
                encoded = raw_line[:-1]
                try:
                    row = json.loads(encoded.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CheckpointValidationError(
                        f"History segment {segment_index} is not valid JSONL."
                    ) from exc
                if not isinstance(row, dict):
                    raise CheckpointValidationError(
                        f"History segment {segment_index} contains a non-mapping row."
                    )
                try:
                    canonical = json.dumps(
                        _json_safe(row), sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False,
                    ).encode("utf-8")
                except (TypeError, ValueError) as exc:
                    raise CheckpointValidationError(
                        f"History segment {segment_index} contains a non-canonical row."
                    ) from exc
                if encoded != canonical:
                    raise CheckpointValidationError(
                        f"History segment {segment_index} row encoding is not canonical."
                    )
                position = _history_position(row, candidate_row_count)
                expected_position = (
                    candidate_row_count + 1,
                    (candidate_row_count + 1) * self._interactions_per_update,
                )
                if position != expected_position:
                    raise CheckpointValidationError(
                        "Training history skips or invents counted work: "
                        f"row {candidate_row_count} must be update {expected_position[0]} at "
                        f"{expected_position[1]} interactions."
                    )
                if candidate_row_count:
                    candidate_digest.update(b",")
                candidate_digest.update(encoded)
                first_position = position if first_position is None else first_position
                last_position = position
                segment_row_count += 1
                candidate_row_count += 1
            completed_signature = _history_stat_signature(os.fstat(stream.fileno()))

        if completed_signature != opened_signature:
            raise CheckpointValidationError(
                f"History segment {segment_index} changed while it was being validated."
            )

        if byte_count != segment["byte_count"] or file_digest.hexdigest() != segment["sha256"]:
            raise CheckpointValidationError(
                f"History segment {segment_index} size or checksum mismatch."
            )
        if segment_row_count != segment["row_count"]:
            raise CheckpointValidationError(
                f"History segment {segment_index} row count/type mismatch."
            )
        if first_position != (
            segment["first_completed_updates"], segment["first_total_interactions"]
        ) or last_position != (
            segment["last_completed_updates"], segment["last_total_interactions"]
        ):
            raise CheckpointValidationError(
                f"History segment {segment_index} counter bounds mismatch."
            )
        try:
            current_source = self._source_path(segment)
            if current_source != source:
                raise CheckpointValidationError(
                    f"History segment {segment_index} changed while it was being validated."
                )
            signature = _history_file_signature(current_source)
        except OSError as exc:
            raise CheckpointValidationError(f"Cannot stat history segment {source}: {exc}") from exc
        if signature != completed_signature or signature[2] != byte_count:
            raise CheckpointValidationError(
                f"History segment {segment_index} changed while it was being validated."
            )

        self._history_digest = candidate_digest
        self._row_count = candidate_row_count
        self._segments.append(dict(segment))
        self._signatures.append(signature)

    def synchronize(self, segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Extend this state to ``segments`` and return the exact v1 reference."""

        provisional = _provisional_history_reference(
            segments, self._interactions_per_update
        )
        normalized_segments = provisional["segments"]
        if not self._matches_authenticated_prefix(normalized_segments):
            raise CheckpointValidationError(
                "History accumulator prefix changed; reconstruct it from immutable segments."
            )
        prior_digest = self._history_digest
        prior_row_count = self._row_count
        prior_segment_count = len(self._segments)
        try:
            for segment_index in range(prior_segment_count, len(normalized_segments)):
                self._append_authenticated_segment(
                    normalized_segments[segment_index], segment_index=segment_index
                )
            if self._row_count != provisional["row_count"]:
                raise CheckpointValidationError(
                    "History reference segment counts do not match row_count."
                )
            reference = self.reference()
            if reference["segments"] != normalized_segments:
                raise CheckpointValidationError("History accumulator segment sequence differs.")
            return reference
        except BaseException:
            # A multi-segment suffix is one logical update.  Never expose a
            # partially advanced prefix when a later segment fails.
            self._history_digest = prior_digest
            self._row_count = prior_row_count
            del self._segments[prior_segment_count:]
            del self._signatures[prior_segment_count:]
            raise

    def reference(self) -> dict[str, Any]:
        """Return the closed version-1 reference without mutating digest state."""

        digest = self._history_digest.copy()
        digest.update(b"]")
        previous_last = (0, 0)
        if self._segments:
            final_segment = self._segments[-1]
            previous_last = (
                final_segment["last_completed_updates"],
                final_segment["last_total_interactions"],
            )
        return _validate_history_reference({
            "schema_version": 1,
            "storage": "immutable_jsonl_segments",
            "row_count": self._row_count,
            "history_sha256": digest.hexdigest(),
            "last_completed_updates": previous_last[0],
            "last_total_interactions": previous_last[1],
            "segments": [dict(segment) for segment in self._segments],
        })


_HISTORY_ACCUMULATOR_CACHE_LIMIT = 8
_HISTORY_ACCUMULATOR_CACHE: OrderedDict[
    tuple[str, int], HistoryReferenceAccumulator
] = OrderedDict()
_HISTORY_ACCUMULATOR_CACHE_LOCK = RLock()
_HISTORY_ACCUMULATOR_CACHE_PID = os.getpid()


def build_history_reference_from_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: str | Path,
    interactions_per_update: int,
) -> dict[str, Any]:
    """Incrementally validate segments and build their exact version-1 cursor.

    The first call for a checkpoint directory, including the first call after a
    process resume, reconstructs the canonical digest one row at a time.  The
    bounded process-local cache then authenticates only appended segments.
    Prefix file signatures are checked on every call; a changed file discards
    cached state and forces full checksum/canonical revalidation.
    """

    provisional = _provisional_history_reference(segments, interactions_per_update)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    cache_key = (str(checkpoint.parent), interactions_per_update)
    with _HISTORY_ACCUMULATOR_CACHE_LOCK:
        global _HISTORY_ACCUMULATOR_CACHE_PID
        current_pid = os.getpid()
        if current_pid != _HISTORY_ACCUMULATOR_CACHE_PID:
            _HISTORY_ACCUMULATOR_CACHE.clear()
            _HISTORY_ACCUMULATOR_CACHE_PID = current_pid
        accumulator = _HISTORY_ACCUMULATOR_CACHE.get(cache_key)
        if accumulator is None or not accumulator._matches_authenticated_prefix(
            provisional["segments"]
        ):
            _HISTORY_ACCUMULATOR_CACHE.pop(cache_key, None)
            accumulator = HistoryReferenceAccumulator(
                checkpoint_path=checkpoint,
                interactions_per_update=interactions_per_update,
            )
        try:
            reference = accumulator.synchronize(provisional["segments"])
        except BaseException:
            _HISTORY_ACCUMULATOR_CACHE.pop(cache_key, None)
            raise
        _HISTORY_ACCUMULATOR_CACHE[cache_key] = accumulator
        _HISTORY_ACCUMULATOR_CACHE.move_to_end(cache_key)
        while len(_HISTORY_ACCUMULATOR_CACHE) > _HISTORY_ACCUMULATOR_CACHE_LIMIT:
            _HISTORY_ACCUMULATOR_CACHE.popitem(last=False)
        return reference


def _validate_history_reference(
    reference: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "storage", "row_count", "history_sha256",
        "last_completed_updates", "last_total_interactions", "segments",
    }
    if not isinstance(reference, Mapping) or set(reference) != required:
        raise CheckpointValidationError("History reference differs from the closed version-1 schema.")
    if reference.get("schema_version") != 1 or reference.get("storage") != "immutable_jsonl_segments":
        raise CheckpointValidationError("Unsupported history-reference format.")
    row_count = reference.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise CheckpointValidationError("History reference row_count is invalid.")
    digest = reference.get("history_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise CheckpointValidationError("History reference checksum is invalid.")
    segments = reference.get("segments")
    if not isinstance(segments, list):
        raise CheckpointValidationError("History reference segments must be a list.")
    segment_fields = {
        "path", "sha256", "byte_count", "row_count",
        "first_completed_updates", "first_total_interactions",
        "last_completed_updates", "last_total_interactions",
    }
    total_rows = 0
    previous_last: tuple[int, int] | None = None
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or set(segment) != segment_fields:
            raise CheckpointValidationError(f"History segment reference {index} has invalid fields.")
        if not isinstance(segment["path"], str) or not segment["path"]:
            raise CheckpointValidationError(f"History segment reference {index} has an invalid path.")
        if not isinstance(segment["sha256"], str) or len(segment["sha256"]) != 64:
            raise CheckpointValidationError(f"History segment reference {index} has an invalid checksum.")
        for field in segment_fields - {"path", "sha256"}:
            value = segment[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CheckpointValidationError(
                    f"History segment reference {index} has invalid {field}."
                )
        if segment["row_count"] < 1 or segment["byte_count"] < 1:
            raise CheckpointValidationError(f"History segment reference {index} is empty.")
        first = (segment["first_completed_updates"], segment["first_total_interactions"])
        last = (segment["last_completed_updates"], segment["last_total_interactions"])
        if last[0] < first[0] or last[1] < first[1]:
            raise CheckpointValidationError(f"History segment reference {index} is reversed.")
        if previous_last is not None and (first[0] <= previous_last[0] or first[1] <= previous_last[1]):
            raise CheckpointValidationError("History segment references overlap or are out of order.")
        previous_last = last
        total_rows += segment["row_count"]
    if total_rows != row_count:
        raise CheckpointValidationError("History reference segment counts do not match row_count.")
    expected_last = previous_last or (0, 0)
    if expected_last != (
        reference["last_completed_updates"], reference["last_total_interactions"]
    ):
        raise CheckpointValidationError("History reference last counters do not match its segments.")
    if history is not None:
        rows = [dict(record) for record in history]
        validate_history(rows)
        if len(rows) != row_count or _json_checksum(rows) != digest:
            raise CheckpointValidationError("History reference does not match supplied history rows.")
        actual_last = _history_position(rows[-1], len(rows) - 1) if rows else (0, 0)
        if actual_last != expected_last:
            raise CheckpointValidationError("History reference counters do not match supplied history rows.")
    return _json_safe(dict(reference))


def load_history_reference(
    reference: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    """Resolve and verify every immutable segment named by a checkpoint cursor."""

    normalized = _validate_history_reference(reference)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    reference_directory = checkpoint.parent
    allowed_root = reference_directory.parent.resolve()
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(normalized["segments"]):
        source = (reference_directory / segment["path"]).resolve()
        if not source.is_relative_to(allowed_root):
            raise CheckpointValidationError("History segment escapes the checkpoint run directory.")
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise CheckpointValidationError(f"Cannot read history segment {source}: {exc}") from exc
        if len(data) != segment["byte_count"] or sha256(data).hexdigest() != segment["sha256"]:
            raise CheckpointValidationError(f"History segment {index} size or checksum mismatch.")
        try:
            segment_rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(f"History segment {index} is not valid JSONL.") from exc
        if len(segment_rows) != segment["row_count"] or not all(
            isinstance(row, dict) for row in segment_rows
        ):
            raise CheckpointValidationError(f"History segment {index} row count/type mismatch.")
        validate_history(segment_rows)
        if _history_position(segment_rows[0], 0) != (
            segment["first_completed_updates"], segment["first_total_interactions"]
        ) or _history_position(segment_rows[-1], len(segment_rows) - 1) != (
            segment["last_completed_updates"], segment["last_total_interactions"]
        ):
            raise CheckpointValidationError(f"History segment {index} counter bounds mismatch.")
        rows.extend(segment_rows)
    _validate_history_reference(normalized, history=rows)
    return rows


def save_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    normalizers: Mapping[str, Any] | None = None,
    observation_normalizer_state: Any = None,
    reward_normalizer_state: Any = None,
    counters: ResumeCounters | Mapping[str, Any],
    recurrent_state: Any,
    rng_states: Mapping[str, Any] | None = None,
    environment_rng_state: Any = None,
    task_schedule_rng_state: Any = None,
    resolved_config: Mapping[str, Any],
    command: str | Sequence[str],
    task_manifest_id: str,
    evaluation_manifest_id: str,
    fingerprints: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    history_reference: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    interactions_per_update: int,
    overwrite: bool = False,
) -> Path:
    """Atomically write a complete checkpoint at a rollout boundary.

    By default an existing path is never replaced.  Periodic callers should
    use unique update-numbered filenames; a separately managed ``latest`` link
    may be replaced by the orchestration layer if desired.
    """

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch module.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer.")
    state = ResumeCounters.from_value(counters)
    validate_interaction_alignment(state, interactions_per_update)
    if history_reference is None:
        validate_history_alignment(history, state, interactions_per_update)
        normalized_history_reference = None
    else:
        # A compact main-run checkpoint intentionally passes an empty live
        # history after all pending rows have been committed to authenticated
        # JSONL segments.  Full-history callers remain supported and receive
        # the stronger row-by-row cross-check used by the legacy API.
        normalized_history_reference = _validate_history_reference(
            history_reference,
            history=history if history else None,
        )
        if history:
            validate_history_alignment(history, state, interactions_per_update)
        elif (
            normalized_history_reference["row_count"] != state.completed_updates
            or normalized_history_reference["last_completed_updates"]
            != state.completed_updates
            or normalized_history_reference["last_total_interactions"]
            != state.total_interactions
        ):
            raise CheckpointValidationError(
                "External history reference does not align with checkpoint counters."
            )
    if not isinstance(resolved_config, Mapping) or not resolved_config:
        raise ValueError("resolved_config must be a non-empty mapping.")
    if not isinstance(task_manifest_id, str) or not task_manifest_id:
        raise ValueError("task_manifest_id must be a non-empty string.")
    if not isinstance(evaluation_manifest_id, str) or not evaluation_manifest_id:
        raise ValueError("evaluation_manifest_id must be a non-empty string.")
    normalized_config = _json_safe(resolved_config)
    normalized_fingerprints = _validate_fingerprints(fingerprints)
    normalized_metadata = _json_safe(metadata or {})
    random_states = (
        dict(rng_states)
        if rng_states is not None
        else capture_rng_states(environment=environment_rng_state, task_schedule=task_schedule_rng_state)
    )
    # Validate externally supplied RNG state now instead of discovering a
    # partial payload only at resume time.  This does not mutate global RNGs.
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "environment", "task_schedule"}
    missing_rng = sorted(required_rng - random_states.keys())
    if missing_rng:
        raise ValueError(f"rng_states is missing: {', '.join(missing_rng)}")
    core_checksum = controller_core_checksum(policy)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy_state": _snapshot(policy.state_dict()),
        "optimizer_state": _snapshot(optimizer.state_dict()),
        "scheduler_state": _snapshot(scheduler.state_dict()) if scheduler is not None else None,
        "normalizers": _snapshot(
            _normalize_normalizers(normalizers, observation_normalizer_state, reward_normalizer_state)
        ),
        "counters": state.as_dict(),
        "interactions_per_update": interactions_per_update,
        "recurrent_state": _snapshot(recurrent_state),
        "rng_states": _snapshot(random_states),
        "resolved_config": normalized_config,
        "resolved_config_checksum": _json_checksum(normalized_config),
        "command": _normalize_command(command),
        "task_manifest_id": task_manifest_id,
        "evaluation_manifest_id": evaluation_manifest_id,
        "fingerprints": normalized_fingerprints,
        # Main runs keep metric rows once in immutable JSONL segments.  Each
        # checkpoint stores their checksummed cursor instead of duplicating an
        # ever-growing list into all 500 periodic snapshots.
        "history": (
            [] if normalized_history_reference is not None
            else _snapshot([dict(record) for record in history])
        ),
        "history_reference": normalized_history_reference,
        "metadata": normalized_metadata,
        "core_checksum": core_checksum,
        "legacy_core_checksum": getattr(getattr(policy, "core", None), "frozen_checksum", None),
    }
    destination = Path(path).expanduser().resolve()
    _atomic_torch_save(payload, destination, overwrite=overwrite)
    return destination


def save_checkpoint_boundary(
    latest_path: str | Path,
    numbered_path: str | Path,
    **checkpoint_kwargs: Any,
) -> Path:
    """Commit one restart checkpoint, then durably link its immutable archive."""

    if "overwrite" in checkpoint_kwargs:
        raise TypeError("save_checkpoint_boundary manages overwrite semantics internally.")
    latest = save_checkpoint(latest_path, **checkpoint_kwargs, overwrite=True)
    numbered = Path(numbered_path).expanduser().resolve()
    if not os.path.lexists(numbered):
        link_checkpoint_snapshot(latest, numbered)
    return latest


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "format",
        "version",
        "policy_class",
        "policy_state",
        "optimizer_state",
        "scheduler_state",
        "normalizers",
        "counters",
        "interactions_per_update",
        "recurrent_state",
        "rng_states",
        "resolved_config",
        "resolved_config_checksum",
        "command",
        "task_manifest_id",
        "evaluation_manifest_id",
        "fingerprints",
        "history",
        "history_reference",
        "metadata",
        "core_checksum",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise CheckpointValidationError(f"Checkpoint payload is missing: {', '.join(missing)}")
    if payload["format"] != CHECKPOINT_FORMAT or payload["version"] != CHECKPOINT_VERSION:
        raise CheckpointValidationError(
            f"Unsupported checkpoint format/version: {payload.get('format')!r} v{payload.get('version')!r}."
        )
    counters = ResumeCounters.from_value(payload["counters"])
    interactions_per_update = payload["interactions_per_update"]
    validate_interaction_alignment(counters, interactions_per_update)
    history_reference = payload["history_reference"]
    if history_reference is None:
        validate_history_alignment(payload["history"], counters, interactions_per_update)
    else:
        if payload["history"] != []:
            raise CheckpointValidationError(
                "Externally referenced history requires an empty embedded history list."
            )
        normalized_reference = _validate_history_reference(history_reference)
        if normalized_reference["row_count"] != counters.completed_updates or (
            normalized_reference["last_completed_updates"] != counters.completed_updates
            or normalized_reference["last_total_interactions"] != counters.total_interactions
        ):
            raise CheckpointValidationError(
                "History reference does not end at the checkpoint counters."
            )
    if _json_checksum(payload["resolved_config"]) != payload["resolved_config_checksum"]:
        raise CheckpointValidationError("Resolved configuration checksum does not match its content.")
    _validate_fingerprints(payload["fingerprints"])
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "environment", "task_schedule"}
    if not isinstance(payload["rng_states"], Mapping):
        raise CheckpointValidationError("Checkpoint RNG state is not a mapping.")
    missing_rng = sorted(required_rng - payload["rng_states"].keys())
    if missing_rng:
        raise CheckpointValidationError(f"Checkpoint RNG state is missing: {', '.join(missing_rng)}")


def _compatibility_mismatches(
    payload: Mapping[str, Any],
    policy: nn.Module,
    *,
    expected_fingerprints: Mapping[str, Any] | None,
    expected_config: Mapping[str, Any] | None,
    expected_task_manifest_id: str | None,
    expected_evaluation_manifest_id: str | None,
) -> list[str]:
    mismatches: list[str] = []
    policy_class = f"{type(policy).__module__}.{type(policy).__qualname__}"
    if payload["policy_class"] != policy_class:
        mismatches.append(f"policy_class: checkpoint={payload['policy_class']!r}, expected={policy_class!r}")
    actual_core = controller_core_checksum(policy)
    if payload["core_checksum"] != actual_core:
        mismatches.append(f"core_checksum: checkpoint={payload['core_checksum']!r}, expected={actual_core!r}")
    # This fixed buffer can differ between controller-contract versions without
    # changing shape. A strict state-dict load alone would silently replace the
    # active contract. Compare it before policy mutation even when a read-only
    # caller did not supply external fingerprints.
    residual_scale_name = "crazyflie_residual_latent_scale"
    expected_residual_scale = getattr(policy, residual_scale_name, None)
    checkpoint_residual_scale = payload["policy_state"].get(residual_scale_name)
    if isinstance(expected_residual_scale, torch.Tensor):
        if (
            not isinstance(checkpoint_residual_scale, torch.Tensor)
            or checkpoint_residual_scale.shape != expected_residual_scale.shape
            or checkpoint_residual_scale.dtype != expected_residual_scale.dtype
            or not torch.equal(
                checkpoint_residual_scale.detach().cpu(),
                expected_residual_scale.detach().cpu(),
            )
        ):
            mismatches.append(
                "crazyflie_residual_latent_scale: checkpoint="
                f"{checkpoint_residual_scale!r}, expected="
                f"{expected_residual_scale.detach().cpu()!r}"
            )
    if expected_config is not None:
        expected_checksum = _json_checksum(expected_config)
        if payload["resolved_config_checksum"] != expected_checksum:
            mismatches.append(
                "resolved_config_checksum: "
                f"checkpoint={payload['resolved_config_checksum']!r}, expected={expected_checksum!r}"
            )
    if expected_task_manifest_id is not None and payload["task_manifest_id"] != expected_task_manifest_id:
        mismatches.append(
            "task_manifest_id: "
            f"checkpoint={payload['task_manifest_id']!r}, expected={expected_task_manifest_id!r}"
        )
    if (
        expected_evaluation_manifest_id is not None
        and payload["evaluation_manifest_id"] != expected_evaluation_manifest_id
    ):
        mismatches.append(
            "evaluation_manifest_id: "
            f"checkpoint={payload['evaluation_manifest_id']!r}, expected={expected_evaluation_manifest_id!r}"
        )
    if expected_fingerprints is not None:
        for key, expected in _validate_fingerprints(expected_fingerprints).items():
            actual = payload["fingerprints"].get(key, "<missing>")
            if actual != expected:
                mismatches.append(f"fingerprint[{key!r}]: checkpoint={actual!r}, expected={expected!r}")
    return mismatches


def read_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    resolve_external_history: bool = False,
) -> dict[str, Any]:
    """Read and internally validate a checkpoint before policy construction.

    Fresh evaluation/resume processes can use the resolved configuration in
    this result to construct the correct policy, then call :func:`load_checkpoint`
    with expected fingerprints to perform compatibility checks and state load.
    """

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointValidationError("Checkpoint root is not a mapping.")
    validate_checkpoint_payload(payload)
    result = dict(payload)
    if resolve_external_history and result.get("history_reference") is not None:
        result["history"] = load_history_reference(
            result["history_reference"], checkpoint_path=source
        )
        validate_history_alignment(
            result["history"], ResumeCounters.from_value(result["counters"]),
            result["interactions_per_update"],
        )
    return result


_WARM_START_ACTOR_PREFIXES = {
    "frozen_lif_original": ("encoder.", "decoder."),
    "frozen_lif_degree_rewired": ("encoder.", "decoder."),
    "gru_matched": ("gru.", "actor."),
    "mlp_normal": ("actor.",),
}
_WARM_START_CONTROLLER_KINDS = {
    "frozen_lif_original": "frozen_lif",
    "frozen_lif_degree_rewired": "frozen_lif_rewired",
    "gru_matched": "gru",
    "mlp_normal": "mlp",
}


def _state_subset_sha256(state: Mapping[str, torch.Tensor], keys: Sequence[str]) -> str:
    """Hash an ordered tensor-state subset without device-dependent serialization."""

    digest = sha256()
    for name in sorted(keys):
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise CheckpointValidationError(f"Policy state {name!r} is not a tensor.")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _warm_start_policy_key_partition(
    policy: nn.Module,
    *,
    controller: str,
) -> tuple[list[str], list[str], list[str]]:
    """Partition state keys into actor-load, critic-reset, and fixed-verify sets."""

    prefixes = _WARM_START_ACTOR_PREFIXES.get(controller)
    if prefixes is None:
        raise ValueError(f"Unsupported warm-start controller: {controller!r}")
    trainable = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    actor = sorted(
        name
        for name in trainable
        if name == "log_std" or any(name.startswith(prefix) for prefix in prefixes)
    )
    critic = sorted(name for name in trainable if name.startswith("critic."))
    unassigned = sorted(trainable - set(actor) - set(critic))
    if unassigned:
        raise CheckpointCompatibilityError(
            "Warm-start trainable-state partition has unassigned parameters: "
            + ", ".join(unassigned)
        )
    if not actor or not critic:
        raise CheckpointCompatibilityError(
            "Warm-start requires nonempty actor and critic parameter partitions."
        )
    state_keys = set(policy.state_dict())
    if not set(actor).issubset(state_keys) or not set(critic).issubset(state_keys):
        raise CheckpointCompatibilityError("Warm-start parameter names are absent from state_dict.")
    fixed = sorted(state_keys - set(actor) - set(critic))
    return actor, critic, fixed


def warm_start_actor_from_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    expected_controller: str,
    expected_connectome_checksum: str,
    expected_frozen_core_checksum: str | None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only compatible actor parameters into a newly initialized policy.

    This is deliberately not resume. Task/reward/reproduction fingerprints may
    differ, but controller identity, policy class, connectome, frozen LIF core,
    and every other fixed policy-state tensor must match exactly. The critic
    and optimizer/scheduler/RNG/recurrent/history state remain freshly reset.
    """

    if expected_controller not in _WARM_START_ACTOR_PREFIXES:
        raise ValueError(f"Unsupported warm-start controller: {expected_controller!r}")
    if not isinstance(expected_connectome_checksum, str) or not expected_connectome_checksum:
        raise ValueError("expected_connectome_checksum must be a nonempty string")
    source = Path(path).expanduser().resolve()
    payload = read_checkpoint(source, map_location=map_location, resolve_external_history=True)
    expected_policy_class = f"{type(policy).__module__}.{type(policy).__qualname__}"
    mismatches: list[str] = []
    if payload["policy_class"] != expected_policy_class:
        mismatches.append(
            f"policy_class: checkpoint={payload['policy_class']!r}, expected={expected_policy_class!r}"
        )
    source_config = payload.get("resolved_config")
    source_controller = source_config.get("controller") if isinstance(source_config, Mapping) else None
    if source_controller != expected_controller:
        mismatches.append(
            f"controller: checkpoint={source_controller!r}, expected={expected_controller!r}"
        )
    source_fingerprints = payload.get("fingerprints")
    source_connectome = (
        source_fingerprints.get("connectome") if isinstance(source_fingerprints, Mapping) else None
    )
    if source_connectome != expected_connectome_checksum:
        mismatches.append(
            "connectome checksum: "
            f"checkpoint={source_connectome!r}, expected={expected_connectome_checksum!r}"
        )
    actual_target_core = controller_core_checksum(policy)
    if actual_target_core != expected_frozen_core_checksum:
        mismatches.append(
            "target frozen-core checksum: "
            f"policy={actual_target_core!r}, expected={expected_frozen_core_checksum!r}"
        )
    if payload.get("core_checksum") != expected_frozen_core_checksum:
        mismatches.append(
            "source frozen-core checksum: "
            f"checkpoint={payload.get('core_checksum')!r}, expected={expected_frozen_core_checksum!r}"
        )
    source_frozen_fingerprint = (
        source_fingerprints.get("frozen_core")
        if isinstance(source_fingerprints, Mapping) else "<missing>"
    )
    if source_frozen_fingerprint != expected_frozen_core_checksum:
        mismatches.append(
            "frozen-core fingerprint: "
            f"checkpoint={source_frozen_fingerprint!r}, expected={expected_frozen_core_checksum!r}"
        )
    metadata = payload.get("metadata")
    controller_report = metadata.get("controller_report") if isinstance(metadata, Mapping) else None
    if not isinstance(controller_report, Mapping):
        mismatches.append("source checkpoint lacks metadata.controller_report")
    else:
        expected_kind = _WARM_START_CONTROLLER_KINDS[expected_controller]
        if controller_report.get("controller_kind") != expected_kind:
            mismatches.append(
                "controller kind: "
                f"checkpoint={controller_report.get('controller_kind')!r}, expected={expected_kind!r}"
            )
        if controller_report.get("connectome_checksum") != expected_connectome_checksum:
            mismatches.append("metadata controller-report connectome checksum differs")
        if controller_report.get("core_checksum") != expected_frozen_core_checksum:
            mismatches.append("metadata controller-report frozen-core checksum differs")

    target_state = policy.state_dict()
    source_state = payload.get("policy_state")
    if not isinstance(source_state, Mapping):
        raise CheckpointValidationError("Warm-start source policy_state is not a mapping.")
    if set(source_state) != set(target_state):
        missing = sorted(set(target_state) - set(source_state))
        unexpected = sorted(set(source_state) - set(target_state))
        mismatches.append(f"policy state keys differ (missing={missing}, unexpected={unexpected})")
    actor_keys, critic_keys, fixed_keys = _warm_start_policy_key_partition(
        policy, controller=expected_controller
    )
    for name in sorted(set(target_state) & set(source_state)):
        source_value = source_state[name]
        target_value = target_state[name]
        if not isinstance(source_value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
            mismatches.append(f"policy state {name!r} is not tensor-valued")
            continue
        if source_value.shape != target_value.shape or source_value.dtype != target_value.dtype:
            mismatches.append(
                f"policy state {name!r} shape/dtype differs: "
                f"checkpoint={tuple(source_value.shape)}/{source_value.dtype}, "
                f"target={tuple(target_value.shape)}/{target_value.dtype}"
            )
        elif name in fixed_keys and not torch.equal(
            source_value.detach().cpu(), target_value.detach().cpu()
        ):
            mismatches.append(f"fixed policy state {name!r} differs")
    if mismatches:
        raise CheckpointCompatibilityError(
            "Incompatible actor warm-start checkpoint:\n- " + "\n- ".join(mismatches)
        )

    source_actor_sha256 = _state_subset_sha256(source_state, actor_keys)
    source_critic_sha256 = _state_subset_sha256(source_state, critic_keys)
    target_critic_before_sha256 = _state_subset_sha256(target_state, critic_keys)
    target_fixed_before_sha256 = _state_subset_sha256(target_state, fixed_keys)
    actor_before = {name: target_state[name].detach().clone() for name in actor_keys}
    try:
        with torch.no_grad():
            live_state = policy.state_dict()
            for name in actor_keys:
                live_state[name].copy_(
                    source_state[name].detach().to(
                        device=live_state[name].device, dtype=live_state[name].dtype
                    )
                )
    except BaseException:
        with torch.no_grad():
            live_state = policy.state_dict()
            for name, value in actor_before.items():
                live_state[name].copy_(value.to(device=live_state[name].device))
        raise

    loaded_state = policy.state_dict()
    loaded_actor_sha256 = _state_subset_sha256(loaded_state, actor_keys)
    target_critic_after_sha256 = _state_subset_sha256(loaded_state, critic_keys)
    target_fixed_after_sha256 = _state_subset_sha256(loaded_state, fixed_keys)
    if loaded_actor_sha256 != source_actor_sha256:
        raise RuntimeError("Warm-start actor tensor checksum differs after loading.")
    if target_critic_after_sha256 != target_critic_before_sha256:
        raise RuntimeError("Warm-start unexpectedly changed target critic state.")
    if target_fixed_after_sha256 != target_fixed_before_sha256:
        raise RuntimeError("Warm-start unexpectedly changed fixed policy state.")
    verify_frozen_core(policy, expected_frozen_core_checksum)

    return {
        "schema_version": 1,
        "mode": "actor_trainable_state_only_new_run",
        "source_checkpoint": {
            "absolute_path": str(source),
            "sha256": checkpoint_sha256(source),
            "policy_class": payload["policy_class"],
            "controller": source_controller,
            "task": source_config.get("task") if isinstance(source_config, Mapping) else None,
            "completed_updates": int(payload["counters"]["completed_updates"]),
            "total_interactions": int(payload["counters"]["total_interactions"]),
            "resolved_config_checksum": payload["resolved_config_checksum"],
            "task_manifest_id": payload["task_manifest_id"],
            "evaluation_manifest_id": payload["evaluation_manifest_id"],
            "fingerprints": dict(source_fingerprints),
        },
        "compatibility": {
            "policy_class_verified": True,
            "controller_verified": True,
            "connectome_checksum_verified": True,
            "frozen_core_checksum_verified": True,
            "fixed_policy_state_verified": True,
            "cross_task_and_reward_fingerprint_change_allowed": True,
        },
        "loaded_policy_state_keys": actor_keys,
        "loaded_policy_state_sha256": loaded_actor_sha256,
        "verified_fixed_policy_state_keys": fixed_keys,
        "verified_fixed_policy_state_sha256": target_fixed_after_sha256,
        "reset_policy_state_keys": critic_keys,
        "source_critic_state_sha256": source_critic_sha256,
        "target_reset_critic_state_sha256": target_critic_after_sha256,
        "reset_checkpoint_state_keys": [
            "optimizer_state",
            "scheduler_state",
            "normalizers.observation",
            "normalizers.reward",
            "counters",
            "recurrent_state",
            "rng_states",
            "history",
            "history_reference",
            "metadata.memory_samples",
        ],
    }


def load_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    expected_fingerprints: Mapping[str, Any] | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_task_manifest_id: str | None = None,
    expected_evaluation_manifest_id: str | None = None,
    for_resume: bool = False,
    restore_rng: bool = False,
    reset_environments_on_resume: bool = True,
    allow_incompatible: bool = False,
    materialize_external_history: bool = True,
) -> dict[str, Any]:
    """Validate, then load a checkpoint; incompatible inputs fail by default."""

    if not isinstance(materialize_external_history, bool):
        raise TypeError("materialize_external_history must be Boolean.")
    # A checkpoint's compact cursor is part of the artifact even when the
    # caller only evaluates the policy.  Resolve every row only for consumers
    # that explicitly need it; training/evaluation can stream-authenticate the
    # same files without retaining O(updates) Python objects.
    payload = read_checkpoint(
        path,
        map_location=map_location,
        resolve_external_history=materialize_external_history,
    )
    if not materialize_external_history and payload.get("history_reference") is not None:
        streamed_reference = build_history_reference_from_segments(
            payload["history_reference"]["segments"],
            checkpoint_path=path,
            interactions_per_update=payload["interactions_per_update"],
        )
        if streamed_reference != payload["history_reference"]:
            raise CheckpointValidationError(
                "Streamed external history validation differs from checkpoint cursor."
            )
    if for_resume and optimizer is None:
        raise ValueError("An optimizer is required when loading for resume.")
    if for_resume and expected_fingerprints is None:
        raise ValueError("expected_fingerprints is required when loading for resume.")
    if for_resume and payload["scheduler_state"] is not None and scheduler is None:
        raise ValueError("This checkpoint contains scheduler state; a scheduler is required for resume.")
    mismatches = _compatibility_mismatches(
        payload,
        policy,
        expected_fingerprints=expected_fingerprints,
        expected_config=expected_config,
        expected_task_manifest_id=expected_task_manifest_id,
        expected_evaluation_manifest_id=expected_evaluation_manifest_id,
    )
    if mismatches and not allow_incompatible:
        raise CheckpointCompatibilityError("Incompatible checkpoint:\n- " + "\n- ".join(mismatches))

    policy.load_state_dict(payload["policy_state"], strict=True)
    verify_frozen_core(policy, payload["core_checksum"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload["scheduler_state"] is None:
            if for_resume:
                raise CheckpointCompatibilityError(
                    "A scheduler was supplied but the checkpoint has no scheduler state."
                )
        else:
            scheduler.load_state_dict(payload["scheduler_state"])
    caller_rng_states = None
    if restore_rng:
        caller_rng_states = restore_rng_states(payload["rng_states"])
    result = dict(payload)
    saved_counters = ResumeCounters.from_value(payload["counters"])
    result["resume_counters"] = (
        saved_counters.resumed(reset_environments=reset_environments_on_resume).as_dict()
        if for_resume
        else saved_counters.as_dict()
    )
    result["next_update_index"] = saved_counters.next_update_index
    result["tainted"] = bool(mismatches)
    result["taint_reasons"] = mismatches
    result["restored_caller_rng_states"] = caller_rng_states
    return result


def checkpoint_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Explicit aliases make call sites self-documenting while retaining concise
# names for tests and small utilities.
save_drone_checkpoint = save_checkpoint
load_drone_checkpoint = load_checkpoint


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointValidationError",
    "HistoryReferenceAccumulator",
    "ResumeCounters",
    "append_history_record",
    "append_validated_history_record_inplace",
    "build_history_reference",
    "build_history_reference_from_segments",
    "capture_rng_states",
    "checkpoint_sha256",
    "link_checkpoint_snapshot",
    "load_checkpoint",
    "load_drone_checkpoint",
    "load_history_reference",
    "read_checkpoint",
    "restore_rng_states",
    "save_checkpoint",
    "save_checkpoint_boundary",
    "save_drone_checkpoint",
    "validate_checkpoint_payload",
    "validate_history",
    "validate_history_alignment",
    "validate_interaction_alignment",
    "warm_start_actor_from_checkpoint",
    "write_history_segment",
]
