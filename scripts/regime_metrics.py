#!/usr/bin/env python3
"""CPU-only descriptive posture/contact proxies from held-out G1 recordings.

These measurements describe sampled root pose and *net* forces on named bodies.
They do not identify the contact partner, ground support, a gait, or a behavior
chosen by a policy. Samples are active, pre-action states; post-termination
auto-reset states are excluded by ``valid_step``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


SAMPLE_PHASE = "pre_action_before_first_episode_reset"


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


def _metadata(array: np.ndarray) -> dict[str, Any]:
    if array.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string.")
    raw = array.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("metadata_json must be a scalar JSON string.")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must contain a JSON object.")
    for field in ("condition", "scenario"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(f"metadata_json requires nonempty {field}.")
    for field in ("training_seed", "evaluation_seed"):
        if isinstance(metadata.get(field), bool) or not isinstance(metadata.get(field), int):
            raise ValueError(f"metadata_json requires integer {field}.")
    _finite_positive(metadata.get("control_dt_s"), "control_dt_s")
    if metadata.get("sample_phase") != SAMPLE_PHASE:
        raise ValueError(f"sample_phase must be {SAMPLE_PHASE!r}.")
    return metadata


def _body_names(array: np.ndarray, count: int) -> list[str]:
    if count < 1:
        raise ValueError("The force sensor must contain at least one named body.")
    if array.shape != (count,):
        raise ValueError("body_names must have one entry per force-sensor body.")
    names = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str) or not item:
            raise ValueError("body_names entries must be nonempty strings.")
        names.append(item)
    if len(set(names)) != len(names):
        raise ValueError("body_names contains duplicates.")
    return names


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_summary(
    root: np.ndarray,
    forces: np.ndarray,
    names: list[str],
    episode_id: int,
    dt_s: float,
    force_threshold_n: float,
    upright_cos_threshold: float,
) -> dict[str, Any]:
    if not np.isfinite(root).all() or not np.isfinite(forces).all():
        raise ValueError(f"Episode {episode_id} has non-finite values in valid samples.")
    # Isaac root_state_w stores position xyz followed by quaternion wxyz.
    quaternion = root[:, 3:7]
    qnorm = np.linalg.norm(quaternion, axis=1)
    if not np.isfinite(qnorm).all() or np.any(qnorm <= 1e-8):
        raise ValueError(f"Episode {episode_id} has an invalid root quaternion.")
    q = quaternion / qnorm[:, None]
    # World-z component of the root body's local +z axis (R_zz).
    alignment = np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)
    height = root[:, 2]
    magnitudes = np.linalg.norm(forces, axis=-1)
    contacts = {
        name: {
            "mean_magnitude_n": float(magnitudes[:, index].mean()),
            "max_magnitude_n": float(magnitudes[:, index].max()),
            "fraction_at_or_above_threshold": float((magnitudes[:, index] >= force_threshold_n).mean()),
        }
        for index, name in enumerate(names)
    }
    return {
        "episode_id": episode_id,
        "valid_control_steps": int(len(root)),
        "sampled_duration_s": float(len(root) * dt_s),
        "root_height_m": {"mean": float(height.mean()), "min": float(height.min()), "max": float(height.max())},
        "root_up_axis_alignment_cosine": {
            "mean": float(alignment.mean()),
            "min": float(alignment.min()),
            "max": float(alignment.max()),
            "fraction_at_or_above_positive_threshold": float((alignment >= upright_cos_threshold).mean()),
            "fraction_at_or_below_negative_threshold": float((alignment <= -upright_cos_threshold).mean()),
        },
        "net_body_force": contacts,
    }


def _seed_summary(
    episodes: list[dict[str, Any]],
    *,
    condition: str,
    scenario: str,
    training_seed: int,
    names: list[str],
) -> dict[str, Any]:
    heights = [episode["root_height_m"] for episode in episodes]
    alignments = [episode["root_up_axis_alignment_cosine"] for episode in episodes]
    return {
        "condition": condition,
        "scenario": scenario,
        "training_seed": training_seed,
        "episode_count": len(episodes),
        "valid_control_steps": sum(episode["valid_control_steps"] for episode in episodes),
        "mean_episode_sampled_duration_s": mean(episode["sampled_duration_s"] for episode in episodes),
        "root_height_m": {
            "mean_of_episode_means": mean(item["mean"] for item in heights),
            "min_observed": min(item["min"] for item in heights),
            "max_observed": max(item["max"] for item in heights),
        },
        "root_up_axis_alignment_cosine": {
            "mean_of_episode_means": mean(item["mean"] for item in alignments),
            "mean_episode_positive_threshold_fraction": mean(
                item["fraction_at_or_above_positive_threshold"] for item in alignments
            ),
            "mean_episode_negative_threshold_fraction": mean(
                item["fraction_at_or_below_negative_threshold"] for item in alignments
            ),
        },
        "net_body_force": {
            name: {
                "mean_of_episode_mean_magnitudes_n": mean(
                    episode["net_body_force"][name]["mean_magnitude_n"] for episode in episodes
                ),
                "max_observed_magnitude_n": max(
                    episode["net_body_force"][name]["max_magnitude_n"] for episode in episodes
                ),
                "mean_episode_threshold_fraction": mean(
                    episode["net_body_force"][name]["fraction_at_or_above_threshold"] for episode in episodes
                ),
            }
            for name in names
        },
    }


def summarize_recording(
    path: str | Path,
    *,
    force_threshold_n: float,
    upright_cos_threshold: float = 0.5,
) -> dict[str, Any]:
    """Summarize one held-out NPZ, with each episode weighted equally at seed level.

    Required arrays: root_state [time, episode, 13] (xyz, wxyz, velocities),
    contact_net_forces_w [time, episode, body, 3], valid_step [time,
    episode], episode_id [episode], body_names [body], metadata_json scalar.
    Invalid samples must be a suffix for each episode and are never read.
    """
    threshold = _finite_positive(force_threshold_n, "force_threshold_n")
    if isinstance(upright_cos_threshold, bool) or not isinstance(upright_cos_threshold, (int, float)):
        raise ValueError("upright_cos_threshold must be in (0, 1].")
    upright = float(upright_cos_threshold)
    if not math.isfinite(upright) or not 0.0 < upright <= 1.0:
        raise ValueError("upright_cos_threshold must be in (0, 1].")
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as recording:
        required = {
            "root_state", "contact_net_forces_w", "valid_step", "episode_id", "body_names", "metadata_json"
        }
        missing = required.difference(recording.files)
        if missing:
            raise ValueError(f"Recording is missing required arrays: {sorted(missing)}")
        root = recording["root_state"]
        forces = recording["contact_net_forces_w"]
        valid = recording["valid_step"]
        ids = recording["episode_id"]
        metadata = _metadata(recording["metadata_json"])
        if root.ndim != 3 or root.shape[2] != 13:
            raise ValueError("root_state must have shape [time, episode, 13].")
        steps, episode_count, _ = root.shape
        if steps < 1 or episode_count < 1:
            raise ValueError("Recording must contain at least one sample and episode.")
        if forces.ndim != 4 or forces.shape[:2] != (steps, episode_count) or forces.shape[-1] != 3:
            raise ValueError("contact_net_forces_w must have shape [time, episode, body, 3].")
        names = _body_names(recording["body_names"], forces.shape[2])
        if valid.shape != (steps, episode_count) or valid.dtype != np.bool_:
            raise ValueError("valid_step must be boolean with shape [time, episode].")
        if ids.shape != (episode_count,) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("episode_id must be integer with shape [episode].")
        episode_ids = [int(item) for item in ids.tolist()]
        if len(set(episode_ids)) != episode_count:
            raise ValueError("episode_id contains duplicates.")
        episodes = []
        for index, episode_id in enumerate(episode_ids):
            n_valid = int(valid[:, index].sum())
            if n_valid == 0 or not valid[:n_valid, index].all() or valid[n_valid:, index].any():
                raise ValueError(f"valid_step for episode {episode_id} must be a nonempty true prefix.")
            episodes.append(_episode_summary(
                root[:n_valid, index], forces[:n_valid, index], names, episode_id,
                float(metadata["control_dt_s"]), threshold, upright,
            ))
    return {
        "schema_version": 1,
        "recording_path": str(source),
        "recording_sha256": _file_sha256(source),
        "metadata": metadata,
        "definitions": {
            "sample_phase": SAMPLE_PHASE,
            "root_quaternion_order": "wxyz",
            "root_height": "world-z coordinate of root, not body clearance or a posture label",
            "root_up_axis_alignment": "dot(root local +z in world frame, world +z); signed cosine",
            "contact_measure": "magnitude of sensor net force on named body; contact partner and ground support unknown",
            "force_threshold_n": threshold,
            "force_fraction_rule": "fraction of valid pre-action samples with net-force magnitude >= threshold",
            "upright_cos_threshold": upright,
            "seed_aggregation": "arithmetic mean of episode metrics; episodes have equal weight",
        },
        "body_names": names,
        "episodes": episodes,
        "per_training_seed": _seed_summary(
            episodes, condition=metadata["condition"], scenario=metadata["scenario"],
            training_seed=metadata["training_seed"], names=names,
        ),
    }


def summarize_recordings(
    paths: Iterable[str | Path],
    *,
    force_threshold_n: float,
    upright_cos_threshold: float = 0.5,
) -> dict[str, Any]:
    """Merge recordings only within matching condition/scenario/training seeds."""
    records = [summarize_recording(
        path, force_threshold_n=force_threshold_n, upright_cos_threshold=upright_cos_threshold
    ) for path in paths]
    if not records:
        raise ValueError("At least one recording is required.")
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    names_by_group: dict[tuple[str, str, int], list[str]] = {}
    seen: set[tuple[str, str, int, int, int]] = set()
    for record in records:
        meta = record["metadata"]
        key = (meta["condition"], meta["scenario"], meta["training_seed"])
        names = record["body_names"]
        if key in names_by_group and set(names_by_group[key]) != set(names):
            raise ValueError(f"Body-name sets differ within training-seed group {key}.")
        names_by_group.setdefault(key, names)
        for episode in record["episodes"]:
            identity = (*key, meta["evaluation_seed"], episode["episode_id"])
            if identity in seen:
                raise ValueError(f"Duplicate recorded evaluation episode: {identity}.")
            seen.add(identity)
        groups.setdefault(key, []).extend(record["episodes"])
    per_seed = [
        _seed_summary(groups[key], condition=key[0], scenario=key[1], training_seed=key[2],
                      names=names_by_group[key])
        for key in sorted(groups)
    ]
    return {
        "schema_version": 1,
        "status": "descriptive_posthoc_summary",
        "definitions": records[0]["definitions"],
        "recordings": [{
            "path": record["recording_path"], "sha256": record["recording_sha256"],
            "condition": record["metadata"]["condition"],
            "scenario": record["metadata"]["scenario"],
            "training_seed": record["metadata"]["training_seed"],
            "evaluation_seed": record["metadata"]["evaluation_seed"],
            "episode_count": len(record["episodes"]),
        } for record in records],
        "per_training_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--force-threshold-n", type=float, required=True)
    parser.add_argument("--upright-cos-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_recordings(
        args.recordings, force_threshold_n=args.force_threshold_n,
        upright_cos_threshold=args.upright_cos_threshold,
    )
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(body)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
