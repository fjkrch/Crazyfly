#!/usr/bin/env python3
"""Create an auditable static training summary for a Crazyflie LIF checkpoint.

The checkpoint is the authority for the history boundary.  In particular, the
script resolves only the immutable JSONL segments referenced by that
checkpoint, so plotting an older boundary while a run is still training cannot
silently include later updates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source" / "g1_fly_control"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_fly_control.crazyflie.checkpoint import read_checkpoint  # noqa: E402


BLUE = "#2563eb"
GOLD = "#d97706"
ORANGE = "#ea580c"
OLIVE = "#65a30d"
PINK = "#db2777"
SLATE = "#475569"
LIGHT_SLATE = "#94a3b8"
GRID = "#cbd5e1"
TRAINABLE_FILL = "#dbeafe"
FROZEN_FILL = "#fef3c7"
FIXED_FILL = "#f1f5f9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trailing_mean(values: Sequence[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("rolling window must be positive")
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("rolling mean received a non-finite value")
        running += number
        if index >= window:
            running -= float(values[index - window])
        result.append(running / min(index + 1, window))
    return result


def trailing_sum(values: Sequence[int | float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("rolling window must be positive")
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("rolling sum received a non-finite value")
        running += number
        if index >= window:
            running -= float(values[index - window])
        result.append(running)
    return result


def rolling_episode_outcomes(
    rows: Sequence[Mapping[str, Any]], window: int
) -> list[dict[str, int | float | None]]:
    """Return rates with their exact, shared completed-episode denominator.

    Target success is not an exclusive terminal cause: an episode can reach a
    target and later time out or fail.  Therefore these three rates must not be
    stacked or interpreted as a partition of 100 percent.
    """

    field_names = (
        "completed_episode_count",
        "time_limit_truncation_count",
        "successful_episode_count",
        "failure_termination_count",
    )
    sums = {
        name: trailing_sum([int(row[name]) for row in rows], window)
        for name in field_names
    }
    result: list[dict[str, int | float | None]] = []
    for index, row in enumerate(rows):
        denominator = int(sums["completed_episode_count"][index])
        survival = int(sums["time_limit_truncation_count"][index])
        success = int(sums["successful_episode_count"][index])
        failure = int(sums["failure_termination_count"][index])
        if survival + failure != denominator:
            raise ValueError(
                "rolling time-limit and failure terminal counts do not equal "
                "the completed-episode denominator"
            )
        result.append(
            {
                "completed_updates": int(row["completed_updates"]),
                "completed_episode_denominator": denominator,
                "time_limit_survival_numerator": survival,
                "target_success_episode_numerator": success,
                "failure_termination_numerator": failure,
                "time_limit_survival_rate": survival / denominator if denominator else None,
                "target_success_episode_rate": success / denominator if denominator else None,
                "failure_termination_rate": failure / denominator if denominator else None,
            }
        )
    return result


def positive_axis_scale(values: Iterable[float], *, ratio_threshold: float = 100.0) -> str:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite or any(value <= 0.0 for value in finite):
        return "linear"
    return "log" if max(finite) / min(finite) >= ratio_threshold else "linear"


def _finite_value(row: Mapping[str, Any], field: str, *, nullable: bool = False) -> float | None:
    if field not in row:
        raise ValueError(f"history row is missing required field {field!r}")
    value = row[field]
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"history field {field!r} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"history field {field!r} is non-finite")
    return number


def _validate_history(
    rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]
) -> None:
    if not rows:
        raise ValueError("checkpoint history is empty")
    counters = checkpoint["counters"]
    expected_updates = int(counters["completed_updates"])
    interactions_per_update = int(checkpoint["interactions_per_update"])
    if len(rows) != expected_updates:
        raise ValueError(
            f"history has {len(rows)} rows but checkpoint records {expected_updates} updates"
        )
    required_numeric = (
        "loss",
        "value_loss",
        "policy_loss",
        "mean_rollout_reward",
        "episodic_return_sum",
        "completed_episode_count",
        "time_limit_truncation_count",
        "successful_episode_count",
        "failure_termination_count",
        "action_mean_abs",
        "action_change_mean_l2",
        "action_saturation_fraction",
        "approx_kl",
        "attempted_kl",
        "grad_norm",
        "encoder_gradient_norm",
        "decoder_gradient_norm",
    )
    for index, row in enumerate(rows, start=1):
        if int(row.get("completed_updates", -1)) != index:
            raise ValueError("history updates are not contiguous and ordered")
        expected_interactions = index * interactions_per_update
        if int(row.get("total_interactions", -1)) != expected_interactions:
            raise ValueError("history interactions are not aligned with checkpoint cadence")
        for field in required_numeric:
            _finite_value(row, field)
        _finite_value(row, "episodic_return_mean", nullable=True)
        completed = int(row["completed_episode_count"])
        survival = int(row["time_limit_truncation_count"])
        failure = int(row["failure_termination_count"])
        success = int(row["successful_episode_count"])
        if min(completed, survival, failure, success) < 0:
            raise ValueError("episode counts must be non-negative")
        if survival + failure != completed or success > completed:
            raise ValueError("episode numerators disagree with the completed-episode denominator")
    if int(rows[-1]["total_interactions"]) != int(counters["total_interactions"]):
        raise ValueError("last history row does not match checkpoint interactions")


def _history_sources(checkpoint: Mapping[str, Any], checkpoint_path: Path) -> list[dict[str, Any]]:
    reference = checkpoint.get("history_reference")
    if not isinstance(reference, Mapping):
        return []
    result = []
    for segment in reference["segments"]:
        path = (checkpoint_path.parent / str(segment["path"])).resolve()
        result.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
                "row_count": int(segment["row_count"]),
                "first_completed_updates": int(segment["first_completed_updates"]),
                "last_completed_updates": int(segment["last_completed_updates"]),
            }
        )
    return result


def _resolve_connectome_manifest(
    checkpoint: Mapping[str, Any], explicit_path: Path | None
) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    metadata = checkpoint.get("metadata", {})
    report = metadata.get("controller_report", {}) if isinstance(metadata, Mapping) else {}
    candidate = report.get("connectome_manifest") if isinstance(report, Mapping) else None
    if isinstance(candidate, str) and candidate:
        return Path(candidate).expanduser().resolve()
    return (ROOT / "data" / "connectome" / "manifest.json").resolve()


def _load_connectome(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "real":
        raise ValueError("connectome manifest must describe the provenance-tracked real circuit")
    neurons_path = (path.parent / manifest["neurons_path"]).resolve()
    edges_path = (path.parent / manifest["edges_path"]).resolve()
    if sha256_file(neurons_path) != manifest["checksums"]["neurons"]:
        raise ValueError("connectome neuron-file checksum mismatch")
    if sha256_file(edges_path) != manifest["checksums"]["edges"]:
        raise ValueError("connectome edge-file checksum mismatch")
    neurons = json.loads(neurons_path.read_text(encoding="utf-8"))
    edges = json.loads(edges_path.read_text(encoding="utf-8"))
    roles: dict[str, int] = {}
    for neuron in neurons:
        role = str(neuron.get("annotations", {}).get("model_role", "unknown"))
        roles[role] = roles.get(role, 0) + 1
    positive = sum(float(edge["weight"]) > 0.0 for edge in edges)
    negative = sum(float(edge["weight"]) < 0.0 for edge in edges)
    zero = len(edges) - positive - negative
    declared = manifest["derived_counts"]
    observed = {
        "neurons": len(neurons),
        "edges": len(edges),
        "excitatory_edges": positive,
        "inhibitory_edges": negative,
    }
    for key, value in observed.items():
        if int(declared[key]) != value:
            raise ValueError(f"connectome derived count mismatch for {key}")
    architecture = {
        "source_release": manifest["source_release"],
        "neuron_subset": manifest["neuron_subset"],
        "role_counts": roles,
        "neuron_count": len(neurons),
        "edge_count": len(edges),
        "excitatory_edge_count": positive,
        "inhibitory_edge_count": negative,
        "zero_weight_edge_count": zero,
        "input_population_size": len(manifest["input_neuron_ids"]),
        "output_population_size": len(manifest["output_neuron_ids"]),
        "neuron_model": manifest["neuron_model"],
        "weight_mapping": manifest["weight_mapping"],
    }
    sources = {
        "manifest": {"path": str(path), "sha256": sha256_file(path)},
        "neurons": {"path": str(neurons_path), "sha256": sha256_file(neurons_path)},
        "edges": {"path": str(edges_path), "sha256": sha256_file(edges_path)},
    }
    return architecture, sources


def _parameter_architecture(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {})
    report = metadata.get("controller_report", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(report, Mapping) or report.get("controller_kind") != "frozen_lif":
        raise ValueError("checkpoint is not a frozen-LIF Crazyflie controller")
    fields = (
        "encoder_trainable_parameters",
        "decoder_trainable_parameters",
        "distribution_trainable_parameters",
        "critic_trainable_parameters",
        "actor_trainable_parameters",
        "total_trainable_parameters",
        "frozen_parameters",
        "model_total_parameters",
        "frozen_synaptic_weights",
        "fixed_core_scalar_buffers",
        "graph_index_values",
        "population_index_values",
        "derived_dense_recurrent_buffer_values",
        "total_registered_buffer_values",
        "continuous_dynamic_state_per_environment",
        "discrete_dynamic_state_per_environment",
        "total_dynamic_state_per_environment",
    )
    result = {field: int(report[field]) for field in fields}
    if result["actor_trainable_parameters"] != (
        result["encoder_trainable_parameters"]
        + result["decoder_trainable_parameters"]
        + result["distribution_trainable_parameters"]
    ):
        raise ValueError("actor parameter breakdown does not reconcile")
    if result["total_trainable_parameters"] != (
        result["actor_trainable_parameters"] + result["critic_trainable_parameters"]
    ):
        raise ValueError("trainable parameter breakdown does not reconcile")
    if result["model_total_parameters"] != (
        result["total_trainable_parameters"] + result["frozen_parameters"]
    ):
        raise ValueError("model parameter total does not reconcile")
    result["core_checksum"] = str(report["core_checksum"])
    result["stabilization_and_residual_contract"] = report[
        "stabilization_and_residual_contract"
    ]
    return result


def _lif_samples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        sampled = row.get("lif_activity_sampled")
        activity = row.get("lif_activity")
        if sampled is True:
            if not isinstance(activity, Mapping):
                raise ValueError("sampled LIF row lacks an activity payload")
            sample = {"completed_updates": int(row["completed_updates"]), **dict(activity)}
            for key, value in sample.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"non-finite LIF activity value {key!r}")
            expected_samples = int(activity["rollout_control_steps"]) * int(
                activity["environment_count"]
            )
            if int(activity["neuron_sample_count"]) != expected_samples:
                raise ValueError("LIF activity sample denominator is inconsistent")
            samples.append(sample)
        elif activity is not None:
            raise ValueError("unscheduled history row contains a LIF activity payload")
    if not samples:
        raise ValueError("checkpoint history contains no scheduled LIF activity rows")
    return samples


def _weighted_trailing_episode_return(
    rows: Sequence[Mapping[str, Any]], window: int
) -> list[float | None]:
    numerators = trailing_sum([float(row["episodic_return_sum"]) for row in rows], window)
    denominators = trailing_sum([int(row["completed_episode_count"]) for row in rows], window)
    return [
        numerator / denominator if denominator else None
        for numerator, denominator in zip(numerators, denominators, strict=True)
    ]


def _stage_transitions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    previous: str | None = None
    for row in rows:
        stage = str(row.get("active_training_curriculum_stage_name", "unknown"))
        if stage != previous:
            transitions.append(
                {
                    "completed_updates": int(row["completed_updates"]),
                    "total_interactions": int(row["total_interactions"]),
                    "stage": stage,
                }
            )
            previous = stage
    return transitions


def _style_time_axis(axis: Any, *, transitions: Sequence[Mapping[str, Any]]) -> None:
    axis.grid(True, color=GRID, linewidth=0.6, alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    for transition in transitions[1:]:
        axis.axvline(
            int(transition["completed_updates"]), color=LIGHT_SLATE,
            linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.8,
        )


def _combined_legend(axis: Any, secondary: Any | None = None, *, location: str = "best") -> None:
    handles, labels = axis.get_legend_handles_labels()
    if secondary is not None:
        second_handles, second_labels = secondary.get_legend_handles_labels()
        handles += second_handles
        labels += second_labels
    if handles:
        axis.legend(handles, labels, loc=location, frameon=False, fontsize=8)


def _draw_architecture(
    axis: Any, connectome: Mapping[str, Any], parameters: Mapping[str, Any]
) -> None:
    axis.set_axis_off()
    axis.set_title("Controller architecture and frozen boundary", loc="left", fontweight="bold")
    boxes = (
        (0.02, 0.56, 0.16, 0.25, FIXED_FILL, "Observation\n12-D\nfixed scaling"),
        (
            0.22, 0.56, 0.17, 0.25, TRAINABLE_FILL,
            f"Encoder\ntrainable\n{parameters['encoder_trainable_parameters']:,} params",
        ),
        (
            0.43, 0.50, 0.25, 0.37, FROZEN_FILL,
            "MaleCNS LIF core — frozen\n"
            f"{connectome['neuron_count']:,} neurons · {connectome['edge_count']:,} edges\n"
            f"{connectome['input_population_size']} input · {connectome['output_population_size']} output\n"
            f"{connectome['excitatory_edge_count']:,} + / {connectome['inhibitory_edge_count']:,} − weights",
        ),
        (
            0.72, 0.56, 0.17, 0.25, TRAINABLE_FILL,
            f"Decoder + dist.\ntrainable\n{parameters['decoder_trainable_parameters']:,} + "
            f"{parameters['distribution_trainable_parameters']} params",
        ),
        (0.92, 0.56, 0.07, 0.25, FIXED_FILL, "4-D\naction"),
    )
    for x, y, width, height, color, label in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.0, edgecolor=SLATE, facecolor=color, transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=8.5, color="#0f172a", transform=axis.transAxes,
        )
    for left, right in ((0.18, 0.22), (0.39, 0.43), (0.68, 0.72), (0.89, 0.92)):
        axis.add_patch(
            FancyArrowPatch(
                (left, 0.685), (right, 0.685), arrowstyle="-|>", mutation_scale=12,
                linewidth=1.2, color=SLATE, transform=axis.transAxes,
            )
        )
    critic = FancyBboxPatch(
        (0.22, 0.12), 0.30, 0.20, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0, edgecolor=SLATE, facecolor=TRAINABLE_FILL,
        transform=axis.transAxes,
    )
    axis.add_patch(critic)
    axis.text(
        0.37, 0.22,
        f"Critic branch — trainable\n{parameters['critic_trainable_parameters']:,} params",
        ha="center", va="center", fontsize=9, transform=axis.transAxes,
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.10, 0.56), (0.22, 0.30), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.1, color=SLATE, transform=axis.transAxes,
        )
    )
    roles = connectome["role_counts"]
    axis.text(
        0.57, 0.22,
        "Core roles\n"
        f"sensory {roles.get('sensory_input', 0)} · descending {roles.get('descending_input', 0)}\n"
        f"interneuron {roles.get('vnc_interneuron', 0)} · motor {roles.get('motor_output', 0)}",
        ha="left", va="center", fontsize=8.5, color="#334155", transform=axis.transAxes,
    )
    axis.text(
        0.02, 0.02,
        "Blue = PPO-trainable parameters · Gold = frozen biological graph weights · Grey = fixed transform/state",
        fontsize=8, color="#475569", transform=axis.transAxes,
    )


def _draw_parameter_counts(axis: Any, parameters: Mapping[str, Any]) -> None:
    labels = ["Encoder", "Decoder", "Distribution", "Critic", "Frozen synapses"]
    values = [
        parameters["encoder_trainable_parameters"],
        parameters["decoder_trainable_parameters"],
        parameters["distribution_trainable_parameters"],
        parameters["critic_trainable_parameters"],
        parameters["frozen_synaptic_weights"],
    ]
    colors = [BLUE, BLUE, BLUE, BLUE, GOLD]
    positions = list(range(len(labels)))
    axis.barh(positions, values, color=colors, alpha=0.85)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.set_xlabel("Parameter count (zero-based absolute scale)")
    axis.set_title("Trainable versus frozen parameter inventory", loc="left", fontweight="bold")
    axis.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    offset = max(values) * 0.015
    for position, value in zip(positions, values, strict=True):
        axis.text(value + offset, position, f"{value:,}", va="center", fontsize=8)
    axis.text(
        0.99, 0.03,
        f"Trainable total {parameters['total_trainable_parameters']:,}  ·  "
        f"Frozen {parameters['frozen_parameters']:,}  ·  "
        f"Model total {parameters['model_total_parameters']:,}",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#334155",
    )


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source-data JSON cannot contain non-finite values")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    raise TypeError(f"unsupported source-data value {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(_safe_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_figure(fig: Any, path: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, format="png", dpi=170, bbox_inches="tight", facecolor="white")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_visualization(
    *,
    checkpoint_path: Path,
    output_path: Path,
    source_data_path: Path,
    connectome_manifest_path: Path | None,
    rolling_window: int,
    title: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    source_data_path = source_data_path.expanduser().resolve()
    if rolling_window < 1:
        raise ValueError("--window must be positive")
    if output_path == source_data_path:
        raise ValueError("PNG and source-data paths must differ")
    for path in (output_path, source_data_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite explicitly")
        path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = read_checkpoint(
        checkpoint_path, map_location="cpu", resolve_external_history=True
    )
    rows = checkpoint["history"]
    _validate_history(rows, checkpoint)
    history_sources = _history_sources(checkpoint, checkpoint_path)
    manifest_path = _resolve_connectome_manifest(checkpoint, connectome_manifest_path)
    connectome, connectome_sources = _load_connectome(manifest_path)
    parameters = _parameter_architecture(checkpoint)
    if checkpoint["core_checksum"] != parameters["core_checksum"]:
        raise ValueError("checkpoint and controller report disagree on frozen-core checksum")
    if checkpoint["fingerprints"].get("frozen_core") != checkpoint["core_checksum"]:
        raise ValueError("checkpoint fingerprint disagrees with frozen-core checksum")
    samples = _lif_samples(rows)
    outcomes = rolling_episode_outcomes(rows, rolling_window)
    transitions = _stage_transitions(rows)

    updates = [int(row["completed_updates"]) for row in rows]
    total_loss = [float(row["loss"]) for row in rows]
    value_loss = [float(row["value_loss"]) for row in rows]
    policy_loss = [float(row["policy_loss"]) for row in rows]
    total_loss_rolling = trailing_mean(total_loss, rolling_window)
    value_loss_rolling = trailing_mean(value_loss, rolling_window)
    policy_loss_rolling = trailing_mean(policy_loss, rolling_window)
    reward = [float(row["mean_rollout_reward"]) for row in rows]
    reward_rolling = trailing_mean(reward, rolling_window)
    episode_return_rolling = _weighted_trailing_episode_return(rows, rolling_window)
    action_abs = [float(row["action_mean_abs"]) for row in rows]
    action_change = [float(row["action_change_mean_l2"]) for row in rows]
    action_saturation = [float(row["action_saturation_fraction"]) for row in rows]
    approx_kl = [float(row["approx_kl"]) for row in rows]
    attempted_kl = [float(row["attempted_kl"]) for row in rows]
    grad_norm = [float(row["grad_norm"]) for row in rows]
    encoder_grad = [float(row["encoder_gradient_norm"]) for row in rows]
    decoder_grad = [float(row["decoder_gradient_norm"]) for row in rows]
    loss_scale = positive_axis_scale([*total_loss, *value_loss])
    kl_scale = positive_axis_scale([*approx_kl, *attempted_kl])
    gradient_scale = positive_axis_scale([*grad_norm, *encoder_grad, *decoder_grad])
    target_kl = float(checkpoint["resolved_config"]["target_kl"])

    chart_rows = []
    for index, row in enumerate(rows):
        chart_rows.append(
            {
                "completed_updates": updates[index],
                "total_interactions": int(row["total_interactions"]),
                "curriculum_stage": str(row["active_training_curriculum_stage_name"]),
                "loss": total_loss[index],
                "loss_trailing_mean": total_loss_rolling[index],
                "value_loss": value_loss[index],
                "value_loss_trailing_mean": value_loss_rolling[index],
                "policy_loss": policy_loss[index],
                "mean_rollout_reward": reward[index],
                "mean_rollout_reward_trailing_mean": reward_rolling[index],
                "episodic_return_trailing_weighted_mean": episode_return_rolling[index],
                "action_mean_abs": action_abs[index],
                "action_change_mean_l2": action_change[index],
                "action_saturation_fraction": action_saturation[index],
                "approx_kl": approx_kl[index],
                "attempted_kl": attempted_kl[index],
                "rejected_step": float(row["rejected_step"]),
                "grad_norm": grad_norm[index],
                "encoder_gradient_norm": encoder_grad[index],
                "decoder_gradient_norm": decoder_grad[index],
                **{
                    key: value
                    for key, value in outcomes[index].items()
                    if key != "completed_updates"
                },
                "last_step_reward_components": dict(row["last_step_reward_components"]),
            }
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(5, 2, figsize=(18, 23), constrained_layout=False)

    axis = axes[0, 0]
    axis.plot(updates, total_loss, color=BLUE, linewidth=0.55, alpha=0.18)
    axis.plot(
        updates, total_loss_rolling, color=BLUE, linewidth=1.8,
        label=f"total loss · trailing {rolling_window}",
    )
    axis.plot(updates, value_loss, color=GOLD, linewidth=0.55, alpha=0.16)
    axis.plot(
        updates, value_loss_rolling, color=GOLD,
        linewidth=1.8, linestyle="--", label=f"value loss · trailing {rolling_window}",
    )
    axis.set_yscale(loss_scale)
    axis.set_ylabel(f"Total / value loss ({loss_scale})")
    policy_axis = axis.twinx()
    policy_axis.plot(
        updates, policy_loss_rolling, color=PINK,
        linewidth=1.3, linestyle=":", label=f"policy loss · trailing {rolling_window}",
    )
    policy_axis.axhline(0.0, color=SLATE, linewidth=0.6, alpha=0.6)
    policy_axis.set_ylabel("Policy loss (signed)", color=PINK)
    axis.set_title("PPO losses", loc="left", fontweight="bold")
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, policy_axis, location="upper right")

    axis = axes[0, 1]
    axis.plot(updates, reward, color=BLUE, linewidth=0.55, alpha=0.18)
    axis.plot(
        updates, reward_rolling, color=BLUE, linewidth=1.8,
        label=f"mean interval reward · trailing {rolling_window}",
    )
    axis.axhline(0.0, color=SLATE, linewidth=0.6, alpha=0.65)
    axis.set_ylabel("Reward per environment-step")
    return_axis = axis.twinx()
    valid_return_x = [
        update for update, value in zip(updates, episode_return_rolling, strict=True)
        if value is not None
    ]
    valid_returns = [value for value in episode_return_rolling if value is not None]
    return_axis.plot(
        valid_return_x, valid_returns, color=GOLD, linewidth=1.6, linestyle="--",
        label=f"episode return · weighted trailing {rolling_window}",
    )
    return_axis.set_ylabel("Completed-episode return", color=GOLD)
    axis.set_title("Reward evidence", loc="left", fontweight="bold")
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, return_axis, location="lower left")

    axis = axes[1, 0]
    denominators = [int(row["completed_episode_denominator"]) for row in outcomes]
    outcome_axis = axis.twinx()
    outcome_axis.fill_between(
        updates, 0, denominators, color=LIGHT_SLATE, alpha=0.18,
        label=f"completed episodes · trailing {rolling_window} denominator",
    )
    outcome_axis.set_ylabel("Completed episodes in window", color=SLATE)
    outcome_specs = (
        ("time_limit_survival_rate", BLUE, "-", "time-limit survival / completed"),
        ("target_success_episode_rate", OLIVE, "--", "target-success episodes / completed"),
        ("failure_termination_rate", ORANGE, ":", "failure terminations / completed"),
    )
    for field, color, linestyle, label in outcome_specs:
        x_values = []
        y_values = []
        for update, record in zip(updates, outcomes, strict=True):
            if record[field] is not None:
                x_values.append(update)
                y_values.append(100.0 * float(record[field]))
        axis.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    axis.set_ylim(-2, 102)
    axis.set_ylabel("Rate (%)")
    axis.set_title(
        "Rolling survival, success, and failure (shared denominator)",
        loc="left", fontweight="bold",
    )
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, outcome_axis, location="upper left")

    axis = axes[1, 1]
    for values, color, linestyle, label in (
        (action_abs, BLUE, "-", "mean |action|"),
        (action_change, GOLD, "--", "mean action change L2"),
    ):
        axis.plot(updates, values, color=color, linewidth=0.5, alpha=0.16)
        axis.plot(
            updates, trailing_mean(values, rolling_window), color=color,
            linestyle=linestyle, linewidth=1.8, label=f"{label} · trailing {rolling_window}",
        )
    saturation_axis = axis.twinx()
    saturation_axis.plot(
        updates, [100.0 * value for value in trailing_mean(action_saturation, rolling_window)],
        color=PINK, linestyle=":", linewidth=1.5,
        label=f"saturation fraction · trailing {rolling_window}",
    )
    saturation_axis.set_ylabel("Saturated action elements (%)", color=PINK)
    axis.set_ylabel("Action magnitude")
    axis.set_title("Action magnitude and smoothness", loc="left", fontweight="bold")
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, saturation_axis, location="upper right")

    axis = axes[2, 0]
    axis.plot(updates, attempted_kl, color=GOLD, linewidth=0.7, alpha=0.5, label="attempted KL")
    axis.plot(updates, approx_kl, color=BLUE, linewidth=1.1, label="accepted approx. KL")
    axis.axhline(target_kl, color=ORANGE, linewidth=1.2, linestyle="--", label=f"target KL {target_kl:g}")
    rejected_updates = [
        updates[index] for index, row in enumerate(rows) if float(row["rejected_step"]) > 0.0
    ]
    rejected_values = [
        attempted_kl[index] for index, row in enumerate(rows) if float(row["rejected_step"]) > 0.0
    ]
    if rejected_updates:
        axis.scatter(rejected_updates, rejected_values, color=ORANGE, marker="x", s=22, label="rejected update")
    axis.set_yscale(kl_scale)
    axis.set_ylabel(f"Approximate KL ({kl_scale})")
    axis.set_title("PPO KL guard", loc="left", fontweight="bold")
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, location="upper right")

    axis = axes[2, 1]
    for values, color, linestyle, label in (
        (grad_norm, SLATE, "-", "global raw grad norm"),
        (encoder_grad, BLUE, "--", "encoder grad norm"),
        (decoder_grad, GOLD, ":", "decoder grad norm"),
    ):
        axis.plot(updates, values, color=color, linestyle=linestyle, linewidth=1.2, label=label)
    axis.set_yscale(gradient_scale)
    axis.set_ylabel(f"Gradient norm ({gradient_scale})")
    axis.set_title("Gradient diagnostics", loc="left", fontweight="bold")
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, location="upper right")

    sample_updates = [int(sample["completed_updates"]) for sample in samples]
    axis = axes[3, 0]
    spike_rate = [float(sample["rollout_sampled_spike_rate_hz_per_neuron"]) for sample in samples]
    dead_fraction = [100.0 * float(sample["rollout_dead_neuron_fraction"]) for sample in samples]
    saturated_fraction = [
        100.0 * float(sample["rollout_saturated_neuron_fraction"]) for sample in samples
    ]
    axis.plot(sample_updates, spike_rate, color=BLUE, marker="o", linewidth=1.8, label="sampled spike rate")
    axis.set_ylabel("Spike rate (Hz / neuron)")
    fraction_axis = axis.twinx()
    fraction_axis.plot(
        sample_updates, dead_fraction, color=ORANGE, marker="s", linestyle="--",
        linewidth=1.5, label="dead-neuron fraction",
    )
    fraction_axis.plot(
        sample_updates, saturated_fraction, color=PINK, marker="^", linestyle=":",
        linewidth=1.5, label="saturated-neuron fraction",
    )
    fraction_axis.set_ylabel("Neuron fraction (%)")
    axis.set_title(
        f"Scheduled LIF activity samples only (n={len(samples)})",
        loc="left", fontweight="bold",
    )
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, fraction_axis, location="best")

    axis = axes[3, 1]
    membrane_mean = [float(sample["rollout_membrane_l2_mean"]) for sample in samples]
    membrane_rms = [float(sample["rollout_membrane_l2_rms"]) for sample in samples]
    synapse_mean = [float(sample["rollout_synapse_l2_mean"]) for sample in samples]
    synapse_rms = [float(sample["rollout_synapse_l2_rms"]) for sample in samples]
    axis.plot(sample_updates, membrane_mean, color=BLUE, marker="o", linewidth=1.7, label="membrane L2 mean")
    axis.plot(sample_updates, membrane_rms, color=BLUE, marker="s", linestyle="--", linewidth=1.3, label="membrane L2 RMS")
    synapse_axis = axis.twinx()
    synapse_axis.plot(sample_updates, synapse_mean, color=GOLD, marker="o", linewidth=1.7, label="synapse L2 mean")
    synapse_axis.plot(sample_updates, synapse_rms, color=GOLD, marker="s", linestyle="--", linewidth=1.3, label="synapse L2 RMS")
    axis.set_ylabel("Membrane L2")
    synapse_axis.set_ylabel("Synapse L2", color=GOLD)
    axis.set_title("Scheduled LIF state norms", loc="left", fontweight="bold")
    _style_time_axis(axis, transitions=transitions)
    _combined_legend(axis, synapse_axis, location="best")

    _draw_architecture(axes[4, 0], connectome, parameters)
    _draw_parameter_counts(axes[4, 1], parameters)

    for row_axes in axes[:4]:
        for axis in row_axes:
            axis.set_xlim(updates[0], updates[-1])
            axis.set_xlabel("Completed PPO update")

    checkpoint_update = int(checkpoint["counters"]["completed_updates"])
    run_title = title or "Crazyflie original frozen-LIF checkpoint training summary"
    fig.suptitle(run_title, fontsize=17, fontweight="bold", x=0.055, ha="left", y=0.992)
    fig.text(
        0.055, 0.974,
        f"Checkpoint update {checkpoint_update:,} · "
        f"{int(checkpoint['counters']['total_interactions']):,} interactions · "
        f"{len(rows):,} ordered history rows · rolling window {rolling_window} updates",
        fontsize=10, color="#334155", ha="left",
    )
    fig.text(
        0.055, 0.012,
        "Outcome rates use trailing completed episodes as the displayed denominator. "
        "Target success can overlap a later timeout/failure and is not stacked. "
        "LIF points are scheduled rollout summaries, not continuous traces; bootstrap calls are excluded. "
        "Thin time-series lines are raw rows and thick lines are trailing means where shown.",
        fontsize=8.5, color="#334155", ha="left",
    )
    fig.tight_layout(rect=(0.035, 0.035, 0.985, 0.955), h_pad=2.4, w_pad=2.0)
    _save_figure(fig, output_path)
    plt.close(fig)

    source_data = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "scope": {
            "checkpoint_completed_updates": checkpoint_update,
            "checkpoint_total_interactions": int(checkpoint["counters"]["total_interactions"]),
            "history_row_count": len(rows),
            "rolling_window_updates": rolling_window,
            "controller": checkpoint["resolved_config"]["controller"],
            "task": checkpoint["resolved_config"]["task"],
            "seed": int(checkpoint["resolved_config"]["seed"]),
        },
        "definitions": {
            "time_axis": "completed PPO updates in strict checkpoint-history order",
            "rolling_window": (
                "inclusive trailing update rows; shorter prefix windows use all available rows"
            ),
            "episode_rate_denominator": (
                "sum(completed_episode_count) over the same trailing update window"
            ),
            "time_limit_survival_numerator": (
                "sum(time_limit_truncation_count) over the trailing window"
            ),
            "target_success_episode_numerator": (
                "sum(successful_episode_count) over the trailing window; can overlap terminal status"
            ),
            "failure_termination_numerator": (
                "sum(failure_termination_count) over the trailing window"
            ),
            "episodic_return_trailing_weighted_mean": (
                "sum(episodic_return_sum) / sum(completed_episode_count) over the trailing window"
            ),
            "lif_activity_sampling": samples[0]["sampling_semantics"],
            "lif_activity_denominator": (
                "rollout_control_steps * environment_count; bootstrap forward call excluded"
            ),
        },
        "axis_scale_decisions": {
            "total_and_value_loss": loss_scale,
            "policy_loss": "linear_signed_secondary_axis",
            "kl": kl_scale,
            "gradient_norms": gradient_scale,
        },
        "curriculum_stage_transitions": transitions,
        "chart_rows": chart_rows,
        "scheduled_lif_activity_rows": samples,
        "architecture": {
            "connectome": connectome,
            "parameters": parameters,
        },
        "integrity": {
            "checkpoint_history_boundary_enforced": True,
            "history_contiguous_from_update_one": True,
            "history_interactions_aligned": True,
            "frozen_core_checksum": checkpoint["core_checksum"],
            "history_reference_sha256": (
                checkpoint["history_reference"]["history_sha256"]
                if checkpoint.get("history_reference") is not None else None
            ),
            "episode_terminal_partition_validated": True,
            "connectome_counts_and_file_hashes_validated": True,
        },
        "sources": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "history_jsonl_segments": history_sources,
            "connectome": connectome_sources,
        },
    }
    _atomic_json(source_data_path, source_data)
    return {
        "status": "PASS",
        "figure": str(output_path),
        "figure_sha256": sha256_file(output_path),
        "source_data": str(source_data_path),
        "source_data_sha256": sha256_file(source_data_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_completed_updates": checkpoint_update,
        "history_rows": len(rows),
        "history_segments": len(history_sources),
        "scheduled_lif_activity_rows": len(samples),
        "rolling_window_updates": rolling_window,
        "loss_axis_scale": loss_scale,
        "kl_axis_scale": kl_scale,
        "gradient_axis_scale": gradient_scale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--connectome_manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source_data", type=Path, default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--title", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None else checkpoint.parent.parent.resolve()
    )
    if not checkpoint.is_relative_to(run_dir):
        raise ValueError("checkpoint must be inside --run_dir")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "visualizations" / f"{checkpoint.stem}-training-summary.png"
    )
    source_data = (
        args.source_data.expanduser().resolve()
        if args.source_data is not None else output.with_suffix(".source.json")
    )
    report = build_visualization(
        checkpoint_path=checkpoint,
        output_path=output,
        source_data_path=source_data,
        connectome_manifest_path=args.connectome_manifest,
        rolling_window=args.window,
        title=args.title,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
