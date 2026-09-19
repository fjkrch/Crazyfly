#!/usr/bin/env python3
"""Build the authenticated Crazyflie command-control comparison report.

The reporter is intentionally outside the ``drone_*.py`` reproduction source
set.  It reads only paths declared by the reviewed command queue, verifies the
training/evaluation identities and immutable history/checkpoint hashes, and
never treats a missing, failed, or inconsistent job as a measured result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/crazyflie_command_seed0_500k.json"
DEFAULT_REPORT = ROOT / "docs/crazyflie_command_control_report.md"
V2_DEFAULT_CONFIG = (
    ROOT / "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json"
)
V2_DEFAULT_REPORT = ROOT / "docs/crazyflie_command_optic_wind_report.md"
TASK = "FlyCrazyflie-CommandFollow-v0"
CONTRACT_PROFILE = "command_v1"
QUEUE_KIND = "crazyflie_command_comparison_queue_v1"
ANALYSIS_KIND = "crazyflie_command_follow_heldout_v1"
CONTROLLERS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "wing_lif",
    "leg_wing_lif",
    "gru_matched",
    "mlp_normal",
)
LIF_CONTROLLERS = frozenset(CONTROLLERS[:4])
EPISODES = 16
STEPS = 600
INTERACTIONS = 500_000
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
# Evaluation summaries are accumulated in float32 tensors, whereas the
# per-episode values are serialized and recomputed here with Python float64.
# Across 16 episodes this can differ by a few float32 ULPs (the observed
# maximum is below 2e-6 for values near 20).  Keep a narrow absolute tolerance
# so genuine edits are still rejected while accepting the original evidence.
FLOAT32_EPISODE_MEAN_ATOL = 5.0e-6
PLOT_NAMES = {
    "score": "command_score_comparison.png",
    "reward": "command_training_reward.png",
    "loss": "command_training_loss.png",
    "activity": "command_activity_comparison.png",
}
V2_TASKS = (
    "FlyCrazyflie-CommandFollowWide-v0",
    "FlyCrazyflie-CommandFollowWideWind-v0",
)
V2_CONTROLLERS = (
    "original_lif",
    "rewired_lif",
    "wing_lif",
    "leg_wing_lif",
    "optic_lif",
    "gru_matched",
    "mlp_normal",
)
V2_POLICY_BY_CONTROLLER = {
    "original_lif": "frozen_lif_original",
    "rewired_lif": "frozen_lif_degree_rewired",
    "wing_lif": "wing_lif",
    "leg_wing_lif": "leg_wing_lif",
    "optic_lif": "optic_lif",
    "gru_matched": "gru_matched",
    "mlp_normal": "mlp_normal",
}
V2_LIF_CONTROLLERS = frozenset(V2_CONTROLLERS[:5])
V2_QUEUE_KIND = "crazyflie_command_big_matrix_queue_v2"
V2_ANALYSIS_KIND = "crazyflie_command_follow_heldout_v2"
V2_INTERACTIONS = 1_000_000
V2_JOB_COUNT = len(V2_TASKS) * len(V2_CONTROLLERS)
V2_PLOT_NAMES = {
    "score": "command_v2_score_by_condition.png",
    "reward": "command_v2_training_reward.png",
    "loss": "command_v2_training_loss.png",
    "activity": "command_v2_lif_activity.png",
    "wind_delta": "command_v2_wind_delta.png",
}
REWARD_COMPONENTS = frozenset(
    {
        "linear_tracking",
        "yaw_tracking",
        "tracking_progress",
        "wrong_direction_acceleration",
        "jerk",
        "target_retention",
        "attitude_stability",
        "angular_stability",
        "control_effort",
        "action_smoothness",
        "survival",
        "failure",
        "total",
    }
)


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_episode_mean_matches(reported: float, recomputed: float) -> bool:
    """Compare float32-reduced evaluator means with a float64 recomputation."""

    return math.isclose(
        reported,
        recomputed,
        rel_tol=0.0,
        abs_tol=FLOAT32_EPISODE_MEAN_ATOL,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return not isinstance(value, bool) and math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the active command output root: {path}")


def _short(value: Any, length: int = 12) -> str:
    if not isinstance(value, str) or not value:
        return "N/A"
    return value if len(value) <= length else value[:length] + "…"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    if number != 0.0 and (abs(number) < 10 ** (-digits) or abs(number) >= 10_000):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


@dataclass
class Cell:
    controller: str
    job_id: str
    complete: bool = False
    reason: str = "not evaluated"
    training_manifest: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    score: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)
    activity: dict[str, Any] | None = None
    core_before: str | None = None
    core_after: str | None = None
    input_hashes: dict[Path, str] = field(default_factory=dict)


@dataclass
class ReportData:
    config_path: Path
    queue_path: Path
    output_root: Path
    config_sha256: str
    queue_sha256: str
    cells: list[Cell]
    input_hashes: dict[Path, str]
    generated_at_utc: str

    @property
    def complete(self) -> bool:
        return len(self.cells) == len(CONTROLLERS) and all(cell.complete for cell in self.cells)


@dataclass
class V2Cell(Cell):
    """One authenticated controller/condition cell in the additive v2 matrix."""

    task: str = ""
    policy: str = ""
    condition: str = ""


@dataclass
class V2ReportData:
    config_path: Path
    queue_path: Path
    output_root: Path
    config_sha256: str
    queue_sha256: str
    cells: list[V2Cell]
    input_hashes: dict[Path, str]
    generated_at_utc: str

    @property
    def complete(self) -> bool:
        expected = [
            (controller, task)
            for controller in V2_CONTROLLERS
            for task in V2_TASKS
        ]
        actual = [(cell.controller, cell.task) for cell in self.cells]
        return actual == expected and all(cell.complete for cell in self.cells)


def _validate_config(config: Mapping[str, Any], config_path: Path) -> Path:
    expected_keys = {
        "schema_version",
        "label",
        "output_root",
        "isaac_python",
        "task",
        "contract_profile",
        "controllers",
        "seed",
        "total_interactions_per_job",
        "training",
        "command_envelope",
        "evaluation",
        "connectomes",
        "rewire",
        "queue",
        "comparison_contract",
    }
    if set(config) != expected_keys:
        raise ValueError("active command config top-level fields differ from schema v1")
    exact = {
        "schema_version": 1,
        "label": "crazyflie_command_seed0_500k",
        "task": TASK,
        "contract_profile": CONTRACT_PROFILE,
        "controllers": list(CONTROLLERS),
        "seed": 0,
        "total_interactions_per_job": INTERACTIONS,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"active command config {key} must be exactly {expected!r}")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    if (
        evaluation.get("protocol") != CONTRACT_PROFILE
        or evaluation.get("episodes_per_controller") != EPISODES
        or evaluation.get("steps_per_episode") != STEPS
        or evaluation.get("deterministic_actions") is not True
        or evaluation.get("activity_from_actual_controller") is not True
    ):
        raise ValueError("active command evaluation declaration differs from command_v1")
    training = _mapping(config.get("training"), "config.training")
    num_envs = _integer(training.get("num_envs"), "training.num_envs", minimum=1)
    horizon = _integer(training.get("horizon"), "training.horizon", minimum=1)
    if INTERACTIONS % (num_envs * horizon):
        raise ValueError("500k interaction budget is not divisible by num_envs*horizon")
    queue_contract = _mapping(config.get("queue"), "config.queue")
    if (
        queue_contract.get("lif_first") is not True
        or queue_contract.get("gpu_used_mib_exclusive") != GPU_LIMIT_MIB
        or queue_contract.get("system_ram_percent_exclusive") != RAM_LIMIT_PERCENT
    ):
        raise ValueError("active command queue resource/order contract differs from plan")
    output_value = config.get("output_root")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("config.output_root must be a non-empty path")
    output_root = _resolve(output_value)
    # This keeps the reporter away from every paused comparison namespace.
    if output_root.name != "crazyflie_command_seed0_500k":
        raise ValueError("reporter accepts only the active crazyflie_command_seed0_500k root")
    if config_path.resolve() != DEFAULT_CONFIG.resolve() and not config_path.is_file():
        raise ValueError("config path does not exist")
    return output_root


def _validate_queue(
    queue: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    queue_path: Path,
    output_root: Path,
) -> Sequence[Mapping[str, Any]]:
    if queue.get("schema_version") != 1 or queue.get("kind") != QUEUE_KIND:
        raise ValueError("queue is not the command comparison queue v1")
    if queue.get("config_file_sha256") != config_sha256:
        raise ValueError("queue config file SHA-256 does not match the selected config")
    if queue.get("config_identity_sha256") != _canonical_sha256(config):
        raise ValueError("queue embedded config identity does not match the selected config")
    if queue.get("config") != dict(config):
        raise ValueError("queue embedded config differs from the selected config")
    declared_config = queue.get("config_path")
    if not isinstance(declared_config, str) or _resolve(declared_config) != config_path.resolve():
        raise ValueError("queue config path differs from the selected config")
    declared_queue = queue.get("queue_file")
    if declared_queue is not None and (
        not isinstance(declared_queue, str) or _resolve(declared_queue) != queue_path.resolve()
    ):
        raise ValueError("queue file identity differs from the selected queue")
    if queue.get("controller_order") != list(CONTROLLERS):
        raise ValueError("queue does not preserve the six-controller LIF-first order")
    declared_root = queue.get("output_root")
    if not isinstance(declared_root, str) or _resolve(declared_root) != output_root:
        raise ValueError("queue output_root differs from the active command output root")
    for path_field, hash_field, expected_name in (
        ("queue_runner", "queue_runner_sha256", "crazyflie_command_queue.py"),
        ("evaluation_script", "evaluation_script_sha256", "crazyflie_command_evaluate.py"),
    ):
        raw_path = queue.get(path_field)
        expected_hash = queue.get(hash_field)
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"queue lacks {path_field}/{hash_field} identity")
        dependency = _resolve(raw_path)
        if dependency.name != expected_name or not dependency.is_file():
            raise ValueError(f"queue {path_field} is missing or has the wrong identity")
        if _sha256_file(dependency) != expected_hash:
            raise ValueError(f"queue {path_field} SHA-256 is stale")
    reports = _mapping(queue.get("controller_reports"), "queue.controller_reports")
    if set(reports) != set(CONTROLLERS):
        raise ValueError("queue controller-report set differs from the six-controller comparison")
    for controller in CONTROLLERS:
        report = _mapping(reports[controller], f"queue controller report {controller}")
        for count_field in (
            "actor_trainable_parameters",
            "total_trainable_parameters",
            "frozen_parameters",
        ):
            _integer(report.get(count_field), f"{controller} {count_field}")
        if type(report.get("parameter_matching_required")) is not bool:
            raise ValueError(f"{controller} lacks an exact parameter-matching requirement")
        if type(report.get("actor_parameter_match_passed")) is not bool:
            raise ValueError(f"{controller} lacks an exact parameter-match result")
        if controller in LIF_CONTROLLERS:
            checksum = report.get("core_checksum")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"{controller} lacks an authenticated frozen-core checksum")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(CONTROLLERS):
        raise ValueError("queue must contain exactly six command jobs")
    if queue.get("job_count") not in (None, len(CONTROLLERS)):
        raise ValueError("queue job_count differs from its job list")
    for index, (job, controller) in enumerate(zip(jobs, CONTROLLERS, strict=True), start=1):
        if not isinstance(job, Mapping):
            raise ValueError(f"queue job {index} is not an object")
        expected_id_prefix = f"{index:02d}__{controller}__command__seed-0"
        if job.get("id") != expected_id_prefix:
            raise ValueError(f"queue job {index} has the wrong stable ID")
        exact = {
            "controller": controller,
            "task": TASK,
            "seed": 0,
            "total_interactions": INTERACTIONS,
        }
        for field_name, expected in exact.items():
            if job.get(field_name) != expected:
                raise ValueError(f"queue job {index} {field_name} differs from {expected!r}")
        for field_name in ("run_dir", "training_manifest", "checkpoint", "evaluation_output"):
            raw = job.get(field_name)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"queue job {index} lacks {field_name}")
            resolved = _resolve(raw)
            _require_under(resolved, output_root, f"job {index} {field_name}")
        if _resolve(job["training_manifest"]) != _resolve(job["run_dir"]) / "training_manifest.json":
            raise ValueError(f"queue job {index} training manifest is outside its run directory")
        if _resolve(job["checkpoint"]) != _resolve(job["run_dir"]) / "checkpoints" / "latest.pt":
            raise ValueError(f"queue job {index} checkpoint is not its latest.pt boundary")
        for identity in (
            "expected_fingerprint",
            "evaluation_manifest_id",
            "command_training_contract_sha256",
        ):
            value = job.get(identity)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"queue job {index} has an invalid {identity}")
        if job.get("expected_updates") != INTERACTIONS // (
            int(config["training"]["num_envs"]) * int(config["training"]["horizon"])
        ):
            raise ValueError(f"queue job {index} expected-update count is wrong")
        if not isinstance(job.get("fingerprint_payload"), Mapping):
            raise ValueError(f"queue job {index} lacks its fingerprint payload")
        evaluation_manifest = job.get("evaluation_manifest")
        if (
            not isinstance(evaluation_manifest, Mapping)
            or evaluation_manifest.get("manifest_id") != job["evaluation_manifest_id"]
            or evaluation_manifest.get("protocol") != CONTRACT_PROFILE
            or evaluation_manifest.get("task") != TASK
        ):
            raise ValueError(f"queue job {index} evaluation manifest identity is invalid")
        if (
            job.get("evaluation_script") != queue.get("evaluation_script")
            or job.get("evaluation_script_sha256") != queue.get("evaluation_script_sha256")
        ):
            raise ValueError(f"queue job {index} evaluator identity differs from queue")
    all_completed = all(job.get("status") == "completed" for job in jobs)
    if all_completed and (
        queue.get("status") != "completed"
        or queue.get("dry_run") is not False
        or queue.get("counts") != {"completed": len(CONTROLLERS)}
    ):
        raise ValueError("queue cannot claim a complete matrix without completed global state")
    return jobs


def _validate_memory_gate(value: Any, label: str) -> None:
    gate = _mapping(value, label)
    if gate.get("passed") is not True:
        raise ValueError(f"{label} did not pass")
    gpu = _finite_number(gate.get("max_device_gpu_used_mib"), f"{label} GPU usage")
    ram = _finite_number(gate.get("max_system_ram_percent"), f"{label} RAM usage")
    if gpu >= GPU_LIMIT_MIB or ram >= RAM_LIMIT_PERCENT:
        raise ValueError(f"{label} exceeds the strict GPU/RAM cap")


def _load_history(
    reference: Mapping[str, Any],
    *,
    checkpoint: Path,
    expected_updates: int,
    interactions_per_update: int,
    total_interactions: int = INTERACTIONS,
) -> tuple[list[dict[str, Any]], dict[Path, str]]:
    if reference.get("schema_version") != 1 or reference.get("storage") != "immutable_jsonl_segments":
        raise ValueError("training history does not use immutable JSONL segments")
    if reference.get("row_count") != expected_updates:
        raise ValueError("training history row count differs from completed updates")
    if reference.get("last_completed_updates") != expected_updates:
        raise ValueError("training history final update differs from completed updates")
    if reference.get("last_total_interactions") != total_interactions:
        raise ValueError(
            f"training history does not end at {total_interactions:,} interactions"
        )
    expected_digest = reference.get("history_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("training history aggregate SHA-256 is invalid")
    segments = reference.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("training history has no immutable segments")
    allowed_root = checkpoint.parent.parent.resolve()
    rows: list[dict[str, Any]] = []
    snapshots: dict[Path, str] = {}
    for segment_index, segment_raw in enumerate(segments):
        segment = _mapping(segment_raw, f"history segment {segment_index}")
        relative = segment.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"history segment {segment_index} lacks a path")
        source = (checkpoint.parent / relative).resolve()
        _require_under(source, allowed_root, f"history segment {segment_index}")
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read history segment {source}: {exc}") from exc
        actual_sha = sha256(data).hexdigest()
        if segment.get("sha256") != actual_sha or segment.get("byte_count") != len(data):
            raise ValueError(f"history segment {segment_index} size/SHA-256 mismatch")
        try:
            decoded = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"history segment {segment_index} is invalid JSONL") from exc
        if len(decoded) != segment.get("row_count") or not all(isinstance(row, dict) for row in decoded):
            raise ValueError(f"history segment {segment_index} row count/type mismatch")
        rows.extend(decoded)
        snapshots[source] = actual_sha
    if len(rows) != expected_updates or _canonical_sha256(rows) != expected_digest:
        raise ValueError("training history aggregate row count/SHA-256 mismatch")
    for index, row in enumerate(rows, start=1):
        if not _finite_tree(row):
            raise ValueError(f"training history row {index} contains nonfinite evidence")
        if row.get("completed_updates") != index:
            raise ValueError("training history updates are not contiguous")
        if row.get("total_interactions") != index * interactions_per_update:
            raise ValueError("training history interactions are not contiguous")
        for field_name in ("mean_rollout_reward", "loss", "policy_loss", "value_loss"):
            _finite_number(row.get(field_name), f"history row {index} {field_name}")
    return rows, snapshots


def _validate_score(score: Mapping[str, Any]) -> None:
    total = _finite_number(score.get("score"), "evaluation total score")
    if not 0.0 <= total <= 100.0:
        raise ValueError("evaluation total score is outside [0, 100]")
    components = _mapping(score.get("component_scores"), "evaluation component scores")
    expected = {
        "linear_tracking",
        "yaw_tracking",
        "direction",
        "response",
        "braking",
        "hover",
        "safety",
        "effort",
        "smoothness",
    }
    if set(components) != expected:
        raise ValueError("evaluation score components differ from command_v1")
    for name, value in components.items():
        numeric = _finite_number(value, f"evaluation score component {name}")
        if not 0.0 <= numeric <= 100.0:
            raise ValueError(f"evaluation score component {name} is outside [0, 100]")
    weights = _mapping(score.get("component_weights"), "evaluation score weights")
    expected_weights = {
        "linear_tracking": 0.30,
        "yaw_tracking": 0.10,
        "direction": 0.10,
        "response": 0.10,
        "braking": 0.10,
        "hover": 0.10,
        "safety": 0.15,
        "effort": 0.025,
        "smoothness": 0.025,
    }
    if dict(weights) != expected_weights:
        raise ValueError("evaluation score weights differ from command_v1")
    recomputed_total = sum(float(components[name]) * weight for name, weight in expected_weights.items())
    if not math.isclose(total, recomputed_total, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("evaluation total score does not recompute from its components")
    raw = _mapping(score.get("raw"), "evaluation raw score measurements")
    if raw.get("steps") != STEPS or raw.get("episodes") != EPISODES:
        raise ValueError("evaluation raw dimensions differ from 16 x 600")
    for name, value in raw.items():
        if name not in {"steps", "episodes"}:
            _finite_number(value, f"evaluation raw measurement {name}")
    recomputed = {
        "linear_tracking": 100.0 * math.exp(-((float(raw["linear_tracking_rmse_m_s"]) / 0.35) ** 2)),
        "yaw_tracking": 100.0 * math.exp(-((float(raw["yaw_tracking_rmse_rad_s"]) / 0.50) ** 2)),
        "direction": 100.0 * max(0.0, 1.0 - float(raw["wrong_direction_fraction"])),
        "response": 100.0 * math.exp(-(float(raw["response_latency_mean_s"]) / 0.80)),
        "braking": 100.0 * math.exp(-(float(raw["brake_settling_mean_s"]) / 1.00)),
        "hover": 100.0 * math.exp(
            -((float(raw["hover_speed_rms_m_s"]) / 0.15) ** 2)
            - ((float(raw["hover_drift_mean_m"]) / 0.15) ** 2)
        ),
        "safety": 100.0
        * float(raw["survival_fraction"])
        * (1.0 if int(raw["invalid_state_count"]) == 0 else 0.0),
        "effort": 100.0 * math.exp(-((float(raw["action_effort_rms"]) / 0.25) ** 2)),
        "smoothness": 100.0 * math.exp(-((float(raw["action_delta_rms"]) / 0.08) ** 2)),
    }
    for name, expected_value in recomputed.items():
        if not math.isclose(float(components[name]), expected_value, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError(f"evaluation score component {name} does not recompute")


def _validate_quality(quality: Mapping[str, Any]) -> None:
    components = _mapping(
        quality.get("component_scores_0_100"), "control-quality component scores"
    )
    expected = {
        "acceleration_quality",
        "command_response",
        "flight_stability",
        "survival_not_die",
    }
    if set(components) != expected:
        raise ValueError("control-quality components differ from command_v1")
    for name, value in components.items():
        numeric = _finite_number(value, f"control-quality component {name}")
        if not 0.0 <= numeric <= 100.0:
            raise ValueError(f"control-quality component {name} is outside [0, 100]")
    raw = _mapping(quality.get("raw"), "control-quality raw measurements")
    for name, value in raw.items():
        _finite_number(value, f"control-quality raw measurement {name}")
    recomputed = {
        "acceleration_quality": 100.0
        * math.exp(-((float(raw["jerk_rms_m_s3"]) / 80.0) ** 2))
        * max(0.0, 1.0 - float(raw["wrong_direction_acceleration_fraction"])),
        "command_response": None,
        "flight_stability": 100.0
        * math.exp(
            -((float(raw["projected_gravity_xy_rms"]) / 0.25) ** 2)
            - ((float(raw["angular_velocity_rms_rad_s"]) / 1.5) ** 2)
        ),
        "survival_not_die": 100.0
        * float(raw["survival_fraction"])
        * (1.0 if int(raw["invalid_state_count"]) == 0 else 0.0),
    }
    for name, expected_value in recomputed.items():
        if expected_value is not None and not math.isclose(
            float(components[name]), expected_value, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(f"control-quality component {name} does not recompute")


def _validate_activity(activity: Mapping[str, Any], controller: str) -> None:
    if activity.get("controller") != controller:
        raise ValueError("activity controller differs from its queue job")
    if activity.get("source") != "exact_forward_pass_that_produced_each_evaluated_action":
        raise ValueError("activity was not measured from the evaluated forward pass")
    unit_count = _integer(activity.get("unit_count"), "activity.unit_count", minimum=1)
    overall = _mapping(activity.get("overall"), "activity.overall")
    for field_name in (
        "mean_absolute_activity_per_unit",
        "rms_activity_per_unit",
        "active_fraction_per_unit",
    ):
        _finite_number(overall.get(field_name), f"activity.overall.{field_name}")
    roles = _mapping(overall.get("roles"), "activity.overall.roles")
    if not roles:
        raise ValueError("activity has no role groups")
    role_total = 0
    for role, summary_raw in roles.items():
        if not isinstance(role, str) or not role:
            raise ValueError("activity contains an invalid role label")
        summary = _mapping(summary_raw, f"activity role {role}")
        role_total += _integer(summary.get("unit_count"), f"activity role {role} unit_count", minimum=1)
        _finite_number(
            summary.get("mean_absolute_activity_per_unit"),
            f"activity role {role} mean absolute activity",
        )
        _finite_number(
            summary.get("active_fraction_per_unit"),
            f"activity role {role} active fraction",
        )
    if role_total != unit_count:
        raise ValueError("activity role unit counts differ from total units")


def _validate_detailed_activity(
    activity: Mapping[str, Any], controller: str, *, require_unit_detail: bool
) -> None:
    """Validate the v2 role and per-neuron evidence used by the report."""

    _validate_activity(activity, controller)
    unit_count = int(activity["unit_count"])
    per_unit = activity.get("per_unit")
    top_units = activity.get("top_units")
    if not isinstance(per_unit, list) or not isinstance(top_units, list):
        if require_unit_detail:
            raise ValueError("LIF activity lacks per-unit/top-unit evidence")
        return
    if len(per_unit) != unit_count:
        raise ValueError("activity per-unit evidence does not cover every unit")
    seen_ids: set[str] = set()
    normalized: list[tuple[str, str, float, float, float]] = []
    for index, raw in enumerate(per_unit):
        unit = _mapping(raw, f"activity per_unit[{index}]")
        stable_id = unit.get("id")
        role = unit.get("role")
        if not isinstance(stable_id, str) or not stable_id or stable_id in seen_ids:
            raise ValueError("activity unit IDs must be non-empty and unique")
        if not isinstance(role, str) or not role:
            raise ValueError("activity unit role must be non-empty")
        if unit.get("index") != index:
            raise ValueError("activity unit indices are not contiguous")
        mean = _finite_number(unit.get("mean_absolute_activity"), "unit mean activity")
        rms = _finite_number(unit.get("rms_activity"), "unit RMS activity")
        active = _finite_number(unit.get("active_fraction"), "unit active fraction")
        if mean < 0.0 or rms < 0.0 or not 0.0 <= active <= 1.0:
            raise ValueError("activity per-unit values are outside physical bounds")
        seen_ids.add(stable_id)
        normalized.append((stable_id, role, mean, rms, active))
    expected_top = sorted(normalized, key=lambda row: (-row[2], row[0]))[: min(10, unit_count)]
    if len(top_units) != len(expected_top):
        raise ValueError("activity top-unit count is inconsistent")
    for rank, (raw, expected) in enumerate(zip(top_units, expected_top, strict=True), start=1):
        unit = _mapping(raw, f"activity top_units[{rank - 1}]")
        actual = (
            unit.get("id"),
            unit.get("role"),
            _finite_number(unit.get("mean_absolute_activity"), "top-unit mean activity"),
            _finite_number(unit.get("rms_activity"), "top-unit RMS activity"),
            _finite_number(unit.get("active_fraction"), "top-unit active fraction"),
        )
        if actual != expected:
            raise ValueError(f"activity top-unit rank {rank} does not recompute")


def _validate_v2_physical_wind(
    value: Any, *, condition: str
) -> Mapping[str, Any]:
    telemetry = _mapping(value, "evaluation physical_wind")
    if telemetry.get("frame") != "world":
        raise ValueError("physical wind telemetry is not world-frame")
    expected_condition = "still_air" if condition == "still" else "wind"
    if telemetry.get("condition") != expected_condition:
        raise ValueError("physical wind telemetry condition differs from queue task")
    measured = _mapping(telemetry.get("measured"), "physical wind measured values")
    for name, raw in measured.items():
        number = _finite_number(raw, f"physical wind measured {name}")
        if number < 0.0:
            raise ValueError(f"physical wind measured {name} is negative")
    samples = _mapping(telemetry.get("samples"), "physical wind samples")
    if samples.get("planned_intervals") != EPISODES * STEPS:
        raise ValueError("physical wind planned interval count is not 16 x 600")
    observed = _integer(
        samples.get("observed_intervals_through_first_done"),
        "physical wind observed intervals",
        minimum=1,
    )
    if observed > EPISODES * STEPS:
        raise ValueError("physical wind observed interval count exceeds the protocol")
    integrity = _mapping(telemetry.get("integrity"), "physical wind integrity")
    for name in (
        "all_values_finite",
        "category_codes_valid_all_simulated_intervals",
        "force_within_declared_bound",
        "torque_within_declared_bound",
        "expected_force_matches_on_observed_intervals",
        "expected_torque_matches_on_observed_intervals",
        "expected_category_matches_on_observed_intervals",
        "pulse_window_matches_on_observed_intervals",
        "passed",
    ):
        if integrity.get(name) is not True:
            raise ValueError(f"physical wind integrity check {name} did not pass")
    if condition == "still":
        if integrity.get("still_air_exact_zero_all_simulated_intervals") is not True:
            raise ValueError("still-air cell did not record an exactly zero physical wrench")
    elif integrity.get("wind_nonzero_wrench_observed") is not True:
        raise ValueError("wind cell did not observe a nonzero physical wrench")
    return telemetry


def _validate_v2_config(config: Mapping[str, Any], config_path: Path) -> Path:
    expected_keys = {
        "schema_version",
        "kind",
        "label",
        "output_root",
        "isaac_python",
        "tasks",
        "task_protocols",
        "contract_profile",
        "controllers",
        "trainer_policies",
        "seed",
        "total_interactions_per_job",
        "training",
        "command_envelope",
        "evaluation",
        "connectomes",
        "rewire",
        "queue",
        "comparison_contract",
        "prior_parallel_paging_evidence",
    }
    if set(config) != expected_keys:
        raise ValueError("active command config top-level fields differ from schema v2")
    exact = {
        "schema_version": 2,
        "kind": "crazyflie_command_optic_wind_matrix_v2",
        "label": "crazyflie_command_optic_wind_seed0_1m",
        "tasks": list(V2_TASKS),
        "task_protocols": {task: "command_v2" for task in V2_TASKS},
        "contract_profile": "command_v2",
        "controllers": list(V2_CONTROLLERS),
        "trainer_policies": V2_POLICY_BY_CONTROLLER,
        "seed": 0,
        "total_interactions_per_job": V2_INTERACTIONS,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"active v2 command config {key} must be exactly {expected!r}")
    training = _mapping(config.get("training"), "config.training")
    num_envs = _integer(training.get("num_envs"), "training.num_envs", minimum=1)
    horizon = _integer(training.get("horizon"), "training.horizon", minimum=1)
    if V2_INTERACTIONS % (num_envs * horizon):
        raise ValueError("1M interaction budget is not divisible by num_envs*horizon")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    if (
        evaluation.get("episodes_per_job") != EPISODES
        or evaluation.get("steps_per_episode") != STEPS
        or evaluation.get("deterministic_actions") is not True
        or evaluation.get("activity_from_actual_controller") is not True
        or evaluation.get("analysis_kind") != V2_ANALYSIS_KIND
        or evaluation.get("explicit_task_argument") is not True
    ):
        raise ValueError("active v2 evaluation declaration differs from reviewed 16 x 600 protocol")
    envelope = _mapping(config.get("command_envelope"), "config.command_envelope")
    if (
        envelope.get("maximum_horizontal_speed_mps") != 1.0
        or envelope.get("maximum_vertical_speed_mps") != 0.5
        or envelope.get("maximum_yaw_rate_radps") != 1.5
        or envelope.get("simultaneous_axes") is not True
    ):
        raise ValueError("active v2 command envelope differs from reviewed wide range")
    queue_contract = _mapping(config.get("queue"), "config.queue")
    if (
        queue_contract.get("lif_first") is not True
        or queue_contract.get("default_max_parallel") != 1
        or queue_contract.get("maximum_parallel") != 1
        or queue_contract.get("gpu_used_mib_exclusive") != GPU_LIMIT_MIB
        or queue_contract.get("system_ram_percent_exclusive") != RAM_LIMIT_PERCENT
    ):
        raise ValueError("active v2 queue resource/order contract differs from plan")
    output_value = config.get("output_root")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("config.output_root must be a non-empty path")
    output_root = _resolve(output_value)
    if output_root.name != "crazyflie_command_optic_wind_seed0_1m":
        raise ValueError("v2 reporter accepts only the additive optic/wind output root")
    if config_path.resolve() != V2_DEFAULT_CONFIG.resolve() and not config_path.is_file():
        raise ValueError("v2 config path does not exist")
    return output_root


def _validate_v2_queue(
    queue: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    queue_path: Path,
    output_root: Path,
) -> Sequence[Mapping[str, Any]]:
    if queue.get("schema_version") != 2 or queue.get("kind") != V2_QUEUE_KIND:
        raise ValueError("queue is not the additive command comparison queue v2")
    if queue.get("config_file_sha256") != config_sha256:
        raise ValueError("v2 queue config file SHA-256 does not match the selected config")
    if queue.get("config_identity_sha256") != _canonical_sha256(config):
        raise ValueError("v2 queue embedded config identity does not match the selected config")
    if queue.get("config") != dict(config):
        raise ValueError("v2 queue embedded config differs from the selected config")
    if _resolve(str(queue.get("config_path", ""))) != config_path.resolve():
        raise ValueError("v2 queue config path differs from the selected config")
    if queue.get("controller_order") != list(V2_CONTROLLERS):
        raise ValueError("v2 queue controller order differs from the LIF-first contract")
    if queue.get("task_order") != list(V2_TASKS):
        raise ValueError("v2 queue task order differs from the still/wind pair")
    if queue.get("job_count") != V2_JOB_COUNT:
        raise ValueError("v2 queue job_count is not 14")
    if queue.get("predicted_training_interactions") != V2_JOB_COUNT * V2_INTERACTIONS:
        raise ValueError("v2 queue interaction total is not 14,000,000")
    if queue.get("predicted_evaluation_episodes") != V2_JOB_COUNT * EPISODES:
        raise ValueError("v2 queue evaluation total is not 224")
    if queue.get("lif_first") is not True or queue.get("maximum_parallel") != 1:
        raise ValueError("v2 queue is not sequential and LIF-first")
    declared_root = queue.get("output_root")
    if not isinstance(declared_root, str) or _resolve(declared_root) != output_root:
        raise ValueError("v2 queue output_root differs from the active output root")
    declared_queue = queue.get("queue_file")
    if declared_queue is not None and (
        not isinstance(declared_queue, str) or _resolve(declared_queue) != queue_path.resolve()
    ):
        raise ValueError("v2 queue file identity differs from the selected queue")
    for path_field, hash_field, expected_name in (
        ("queue_runner", "queue_runner_sha256", "crazyflie_command_queue_v2.py"),
        ("trainer", "trainer_sha256", "drone_train.py"),
        ("evaluator", "evaluator_sha256", "crazyflie_command_evaluate.py"),
    ):
        raw_path = queue.get(path_field)
        expected_hash = queue.get(hash_field)
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"v2 queue lacks {path_field}/{hash_field} identity")
        dependency = _resolve(raw_path)
        if dependency.name != expected_name or not dependency.is_file():
            raise ValueError(f"v2 queue {path_field} is missing or has the wrong identity")
        if _sha256_file(dependency) != expected_hash:
            raise ValueError(f"v2 queue {path_field} SHA-256 is stale")
    reports = _mapping(queue.get("controller_reports"), "queue.controller_reports")
    if set(reports) != set(V2_CONTROLLERS):
        raise ValueError("v2 queue controller-report set differs from seven controllers")
    for controller in V2_CONTROLLERS:
        report = _mapping(reports[controller], f"v2 controller report {controller}")
        for count_field in (
            "actor_trainable_parameters",
            "total_trainable_parameters",
            "frozen_parameters",
        ):
            _integer(report.get(count_field), f"{controller} {count_field}")
        if type(report.get("parameter_matching_required")) is not bool:
            raise ValueError(f"{controller} lacks its parameter-matching declaration")
        if type(report.get("actor_parameter_match_passed")) is not bool:
            raise ValueError(f"{controller} lacks its parameter-match result")
        if controller in V2_LIF_CONTROLLERS:
            checksum = report.get("core_checksum")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"{controller} lacks an authenticated frozen-core checksum")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != V2_JOB_COUNT:
        raise ValueError("v2 queue must contain exactly 14 jobs")
    expected_pairs = [
        (controller, task)
        for controller in V2_CONTROLLERS
        for task in V2_TASKS
    ]
    for index, (job, pair) in enumerate(zip(jobs, expected_pairs, strict=True), start=1):
        if not isinstance(job, Mapping):
            raise ValueError(f"v2 queue job {index} is not an object")
        controller, task = pair
        slug = "still" if task == V2_TASKS[0] else "wind"
        expected_id = f"{index:02d}__{controller}__{slug}__seed-0"
        exact = {
            "id": expected_id,
            "controller": controller,
            "policy": V2_POLICY_BY_CONTROLLER[controller],
            "task": task,
            "seed": 0,
            "command_schedule_seed": 0,
            "contract_profile": "command_v2",
            "evaluation_protocol": "command_v2",
            "total_interactions": V2_INTERACTIONS,
        }
        for field_name, expected in exact.items():
            if job.get(field_name) != expected:
                raise ValueError(
                    f"v2 queue job {index} {field_name} differs from {expected!r}"
                )
        if job.get("paired_task_seed_key") != f"{controller}__seed-0":
            raise ValueError(f"v2 queue job {index} breaks paired still/wind seeding")
        if job.get("architecture_class") != (
            "lif" if controller in V2_LIF_CONTROLLERS else "baseline"
        ):
            raise ValueError(f"v2 queue job {index} architecture class is wrong")
        for field_name in ("run_dir", "training_manifest", "checkpoint", "evaluation_output"):
            raw = job.get(field_name)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"v2 queue job {index} lacks {field_name}")
            _require_under(_resolve(raw), output_root, f"v2 job {index} {field_name}")
        if _resolve(job["training_manifest"]) != _resolve(job["run_dir"]) / "training_manifest.json":
            raise ValueError(f"v2 queue job {index} training manifest is misplaced")
        if _resolve(job["checkpoint"]) != _resolve(job["run_dir"]) / "checkpoints" / "latest.pt":
            raise ValueError(f"v2 queue job {index} checkpoint is not latest.pt")
        for identity in (
            "expected_fingerprint",
            "evaluation_manifest_id",
            "command_training_contract_sha256",
        ):
            value = job.get(identity)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"v2 queue job {index} has an invalid {identity}")
        if job.get("expected_updates") != V2_INTERACTIONS // (
            int(config["training"]["num_envs"]) * int(config["training"]["horizon"])
        ):
            raise ValueError(f"v2 queue job {index} expected-update count is wrong")
        evaluation_manifest = _mapping(
            job.get("evaluation_manifest"), f"v2 job {index} evaluation_manifest"
        )
        if evaluation_manifest.get("manifest_id") != job["evaluation_manifest_id"]:
            raise ValueError(f"v2 queue job {index} evaluation manifest identity is invalid")
    all_completed = all(job.get("status") == "completed" for job in jobs)
    if all_completed and (
        queue.get("status") != "completed"
        or queue.get("dry_run") is not False
        or queue.get("counts") != {"completed": V2_JOB_COUNT}
    ):
        raise ValueError("v2 queue cannot claim a complete matrix without completed global state")
    return jobs


def _validate_cell(
    job: Mapping[str, Any],
    *,
    controller_report: Mapping[str, Any],
    output_root: Path,
) -> Cell:
    controller = str(job["controller"])
    cell = Cell(controller=controller, job_id=str(job["id"]), parameters=dict(controller_report))
    try:
        required_queue_states = {
            "status": "completed",
            "training_status": "completed",
            "evaluation_status": "completed",
        }
        for field_name, expected in required_queue_states.items():
            if job.get(field_name) != expected:
                raise ValueError(f"queue {field_name} is {job.get(field_name)!r}, not {expected!r}")

        manifest_path = _resolve(job["training_manifest"])
        evaluation_path = _resolve(job["evaluation_output"])
        checkpoint_path = _resolve(job["checkpoint"])
        for path, label in (
            (manifest_path, "training manifest"),
            (evaluation_path, "evaluation"),
            (checkpoint_path, "checkpoint"),
        ):
            _require_under(path, output_root, label)
            if not path.is_file():
                raise ValueError(f"{label} is missing")
        manifest = _read_json(manifest_path)
        evaluation = _read_json(evaluation_path)
        input_hashes = {
            manifest_path: _sha256_file(manifest_path),
            evaluation_path: _sha256_file(evaluation_path),
            checkpoint_path: _sha256_file(checkpoint_path),
        }
        expected_updates = INTERACTIONS // (
            _integer(manifest.get("num_envs"), "training num_envs", minimum=1)
            * _integer(manifest.get("horizon"), "training horizon", minimum=1)
        )
        exact_manifest = {
            "schema_version": 1,
            "status": "completed",
            "task": TASK,
            "contract_profile": CONTRACT_PROFILE,
            "controller": controller,
            "seed": 0,
            "requested_interactions": INTERACTIONS,
            "environment_interactions": INTERACTIONS,
            "completed_updates": expected_updates,
            "fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": input_hashes[checkpoint_path],
        }
        for field_name, expected in exact_manifest.items():
            actual = manifest.get(field_name)
            if field_name in {"checkpoint"} and isinstance(actual, str):
                actual = str(_resolve(actual))
            if actual != expected:
                raise ValueError(f"training manifest {field_name} differs from queue identity")
        resolved = _mapping(manifest.get("resolved_config"), "training resolved_config")
        resolved_exact = {
            "task": TASK,
            "contract_profile": CONTRACT_PROFILE,
            "controller": controller,
            "seed": 0,
            "total_interactions": INTERACTIONS,
            "evaluation_protocol": CONTRACT_PROFILE,
            "command_training_contract_sha256": job["command_training_contract_sha256"],
        }
        for field_name, expected in resolved_exact.items():
            if resolved.get(field_name) != expected:
                raise ValueError(f"training resolved_config {field_name} differs from queue")
        if manifest.get("fingerprint_payload") != job["fingerprint_payload"]:
            raise ValueError("training fingerprint payload differs from queue")
        manifest_report = _mapping(manifest.get("controller_report"), "training controller_report")
        if _canonical_sha256(manifest_report) != _canonical_sha256(controller_report):
            raise ValueError("training controller report differs from queue declaration")
        _validate_memory_gate(manifest.get("memory_gate"), "training memory gate")
        command_schedule = _mapping(manifest.get("command_schedule"), "training command_schedule")
        schedule_state = _mapping(command_schedule.get("state"), "training command_schedule.state")
        if (
            command_schedule.get("command_training_contract_sha256")
            != job["command_training_contract_sha256"]
            or schedule_state.get("training_interactions") != INTERACTIONS
        ):
            raise ValueError("training command schedule/contract does not end at 500,000")
        interactions_per_update = int(manifest["num_envs"]) * int(manifest["horizon"])
        history, history_hashes = _load_history(
            _mapping(manifest.get("history_reference"), "training history_reference"),
            checkpoint=checkpoint_path,
            expected_updates=expected_updates,
            interactions_per_update=interactions_per_update,
        )
        input_hashes.update(history_hashes)

        core_before = manifest.get("core_checksum_before")
        core_after = manifest.get("core_checksum_after")
        if controller in LIF_CONTROLLERS:
            expected_core = controller_report.get("core_checksum")
            if (
                not isinstance(core_before, str)
                or len(core_before) != 64
                or core_before != core_after
                or core_before != expected_core
            ):
                raise ValueError("frozen LIF core checksum changed or differs from queue")

        exact_evaluation = {
            "schema_version": 1,
            "analysis_kind": ANALYSIS_KIND,
            "status": "PASS",
            "task": TASK,
            "controller": controller,
            "episodes_requested": EPISODES,
            "episodes_evaluated": EPISODES,
            "steps_per_episode": STEPS,
            "deterministic_actions": True,
            "policy_action_source": "actual_trained_controller_no_assist",
        }
        for field_name, expected in exact_evaluation.items():
            if evaluation.get(field_name) != expected:
                raise ValueError(f"evaluation {field_name} differs from command_v1")
        episodes = evaluation.get("episodes")
        if (
            not isinstance(episodes, list)
            or len(episodes) != EPISODES
            or not all(isinstance(row, dict) and row for row in episodes)
        ):
            raise ValueError("evaluation does not contain exactly 16 episodes")
        if not _finite_tree(evaluation):
            raise ValueError("evaluation contains nonfinite or unsupported evidence")
        checkpoint = _mapping(evaluation.get("checkpoint"), "evaluation checkpoint")
        checkpoint_exact = {
            "path": str(checkpoint_path),
            "sha256": input_hashes[checkpoint_path],
            "training_seed": 0,
            "total_interactions": INTERACTIONS,
            "reproduction_fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
        }
        for field_name, expected in checkpoint_exact.items():
            actual = checkpoint.get(field_name)
            if field_name == "path" and isinstance(actual, str):
                actual = str(_resolve(actual))
            if actual != expected:
                raise ValueError(f"evaluation checkpoint {field_name} differs from training")
        if checkpoint.get("evaluation_manifest") != job["evaluation_manifest"]:
            raise ValueError("evaluation manifest payload differs from queue")
        if (
            evaluation.get("protocol")
            != job["evaluation_manifest"].get("evaluation_protocol")
            or evaluation.get("protocol_sha256")
            != job["evaluation_manifest"].get("evaluation_protocol_sha256")
        ):
            raise ValueError("evaluation protocol identity differs from queue")
        _validate_memory_gate(evaluation.get("memory_gate"), "evaluation memory gate")
        integrity = _mapping(evaluation.get("integrity"), "evaluation integrity")
        for name in (
            "task_manifest_matched",
            "evaluation_manifest_matched",
            "source_set_matched",
            "checkpoint_completed_budget",
            "all_actions_finite_and_bounded",
            "activity_from_actual_forward",
        ):
            if integrity.get(name) is not True:
                raise ValueError(f"evaluation integrity check {name} did not pass")
        summary = _mapping(evaluation.get("summary"), "evaluation summary")
        score = _mapping(summary.get("score"), "evaluation summary score")
        quality = _mapping(summary.get("control_quality"), "evaluation control quality")
        _validate_score(score)
        _validate_quality(quality)
        if not math.isclose(
            float(quality["component_scores_0_100"]["command_response"]),
            float(score["component_scores"]["response"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("control-quality response score differs from primary response score")
        reward_components_raw = _mapping(
            summary.get("reward_component_mean_per_episode"),
            "evaluation reward component means",
        )
        if not reward_components_raw:
            raise ValueError("evaluation has no reward component means")
        if set(reward_components_raw) != REWARD_COMPONENTS:
            raise ValueError("evaluation reward-component set differs from command_v1")
        reward_components = {
            str(name): _finite_number(value, f"reward component {name}")
            for name, value in reward_components_raw.items()
        }
        episode_component_sums = {name: 0.0 for name in REWARD_COMPONENTS}
        for episode_index, episode in enumerate(episodes):
            episode_rewards = _mapping(
                episode.get("reward_components"),
                f"evaluation episode {episode_index} reward components",
            )
            if set(episode_rewards) != REWARD_COMPONENTS:
                raise ValueError(
                    f"evaluation episode {episode_index} reward-component set differs from command_v1"
                )
            for name, value in episode_rewards.items():
                episode_component_sums[name] += _finite_number(
                    value, f"evaluation episode {episode_index} reward component {name}"
                )
        for name, total_value in episode_component_sums.items():
            if not _float32_episode_mean_matches(
                reward_components[name], total_value / EPISODES
            ):
                raise ValueError(f"evaluation reward-component mean {name} does not recompute")
        activity = _mapping(evaluation.get("activity"), "evaluation activity")
        _validate_activity(activity, controller)
        eval_controller_report = _mapping(
            evaluation.get("controller_report"), "evaluation controller_report"
        )
        if _canonical_sha256(eval_controller_report) != _canonical_sha256(controller_report):
            raise ValueError("evaluation controller report differs from queue declaration")

        cell.complete = True
        cell.reason = "verified complete"
        cell.training_manifest = manifest
        cell.evaluation = evaluation
        cell.history = history
        cell.score = dict(score)
        cell.quality = dict(quality)
        cell.reward_components = reward_components
        cell.activity = dict(activity)
        cell.core_before = core_before if isinstance(core_before, str) else None
        cell.core_after = core_after if isinstance(core_after, str) else None
        cell.input_hashes = input_hashes
    except (KeyError, TypeError, ValueError, OSError) as exc:
        cell.complete = False
        cell.reason = str(exc)
    return cell


def collect_report_data(config_path: Path, queue_path: Path | None = None) -> ReportData:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"active command config is missing: {config_path}")
    config = _read_json(config_path)
    output_root = _validate_config(config, config_path)
    queue_path = (
        queue_path.expanduser().resolve()
        if queue_path is not None
        else (output_root / "queue.json").resolve()
    )
    _require_under(queue_path, output_root, "queue")
    if not queue_path.is_file():
        raise ValueError(f"active command queue is missing: {queue_path}")
    config_sha = _sha256_file(config_path)
    queue_sha = _sha256_file(queue_path)
    queue = _read_json(queue_path)
    jobs = _validate_queue(
        queue,
        config,
        config_path=config_path,
        config_sha256=config_sha,
        queue_path=queue_path,
        output_root=output_root,
    )
    reports = _mapping(queue["controller_reports"], "queue.controller_reports")
    cells = [
        _validate_cell(job, controller_report=_mapping(reports[job["controller"]], "controller report"), output_root=output_root)
        for job in jobs
    ]
    input_hashes: dict[Path, str] = {config_path: config_sha, queue_path: queue_sha}
    for field_name in ("queue_runner", "evaluation_script"):
        dependency = _resolve(queue[field_name])
        input_hashes[dependency] = _sha256_file(dependency)
    for cell in cells:
        input_hashes.update(cell.input_hashes)
    return ReportData(
        config_path=config_path,
        queue_path=queue_path,
        output_root=output_root,
        config_sha256=config_sha,
        queue_sha256=queue_sha,
        cells=cells,
        input_hashes=input_hashes,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _validate_v2_cell(
    job: Mapping[str, Any],
    *,
    controller_report: Mapping[str, Any],
    output_root: Path,
) -> V2Cell:
    controller = str(job["controller"])
    policy = str(job["policy"])
    task = str(job["task"])
    condition = "still" if task == V2_TASKS[0] else "wind"
    cell = V2Cell(
        controller=controller,
        policy=policy,
        task=task,
        condition=condition,
        job_id=str(job["id"]),
        parameters=dict(controller_report),
    )
    try:
        for field_name in ("status", "training_status", "evaluation_status"):
            if job.get(field_name) != "completed":
                raise ValueError(f"queue {field_name} is {job.get(field_name)!r}, not 'completed'")
        manifest_path = _resolve(job["training_manifest"])
        evaluation_path = _resolve(job["evaluation_output"])
        checkpoint_path = _resolve(job["checkpoint"])
        for path, label in (
            (manifest_path, "training manifest"),
            (evaluation_path, "evaluation"),
            (checkpoint_path, "checkpoint"),
        ):
            _require_under(path, output_root, f"v2 {label}")
            if not path.is_file():
                raise ValueError(f"{label} is missing")
        manifest = _read_json(manifest_path)
        evaluation = _read_json(evaluation_path)
        input_hashes = {
            manifest_path: _sha256_file(manifest_path),
            evaluation_path: _sha256_file(evaluation_path),
            checkpoint_path: _sha256_file(checkpoint_path),
        }
        num_envs = _integer(manifest.get("num_envs"), "training num_envs", minimum=1)
        horizon = _integer(manifest.get("horizon"), "training horizon", minimum=1)
        expected_updates = V2_INTERACTIONS // (num_envs * horizon)
        exact_manifest = {
            "schema_version": 1,
            "status": "completed",
            "task": task,
            "contract_profile": "command_v2",
            "controller": policy,
            "seed": 0,
            "requested_interactions": V2_INTERACTIONS,
            "environment_interactions": V2_INTERACTIONS,
            "completed_updates": expected_updates,
            "fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": input_hashes[checkpoint_path],
        }
        for field_name, expected in exact_manifest.items():
            actual = manifest.get(field_name)
            if field_name == "checkpoint" and isinstance(actual, str):
                actual = str(_resolve(actual))
            if actual != expected:
                raise ValueError(f"training manifest {field_name} differs from v2 queue identity")
        resolved = _mapping(manifest.get("resolved_config"), "training resolved_config")
        resolved_exact = {
            "task": task,
            "contract_profile": "command_v2",
            "controller": policy,
            "seed": 0,
            "total_interactions": V2_INTERACTIONS,
            "evaluation_protocol": "command_v2",
            "command_training_contract_sha256": job["command_training_contract_sha256"],
        }
        for field_name, expected in resolved_exact.items():
            if resolved.get(field_name) != expected:
                raise ValueError(f"training resolved_config {field_name} differs from v2 queue")
        if manifest.get("fingerprint_payload") != job.get("fingerprint_payload"):
            raise ValueError("training fingerprint payload differs from v2 queue")
        manifest_report = _mapping(manifest.get("controller_report"), "training controller_report")
        if _canonical_sha256(manifest_report) != _canonical_sha256(controller_report):
            raise ValueError("training controller report differs from v2 queue declaration")
        _validate_memory_gate(manifest.get("memory_gate"), "training memory gate")
        command_schedule = _mapping(manifest.get("command_schedule"), "training command_schedule")
        schedule_state = _mapping(command_schedule.get("state"), "training command_schedule.state")
        if (
            command_schedule.get("command_training_contract_sha256")
            != job["command_training_contract_sha256"]
            or schedule_state.get("training_interactions") != V2_INTERACTIONS
        ):
            raise ValueError("training command schedule/contract does not end at 1,000,000")
        history, history_hashes = _load_history(
            _mapping(manifest.get("history_reference"), "training history_reference"),
            checkpoint=checkpoint_path,
            expected_updates=expected_updates,
            interactions_per_update=num_envs * horizon,
            total_interactions=V2_INTERACTIONS,
        )
        input_hashes.update(history_hashes)

        core_before = manifest.get("core_checksum_before")
        core_after = manifest.get("core_checksum_after")
        if controller in V2_LIF_CONTROLLERS:
            expected_core = controller_report.get("core_checksum")
            if (
                not isinstance(core_before, str)
                or len(core_before) != 64
                or core_before != core_after
                or core_before != expected_core
            ):
                raise ValueError("frozen v2 LIF core checksum changed or differs from queue")

        exact_evaluation = {
            "schema_version": 1,
            "analysis_kind": V2_ANALYSIS_KIND,
            "status": "PASS",
            "task": task,
            "controller": policy,
            "episodes_requested": EPISODES,
            "episodes_evaluated": EPISODES,
            "steps_per_episode": STEPS,
            "deterministic_actions": True,
            "policy_action_source": "actual_trained_controller_no_assist",
        }
        for field_name, expected in exact_evaluation.items():
            if evaluation.get(field_name) != expected:
                raise ValueError(f"evaluation {field_name} differs from command_v2")
        episodes = evaluation.get("episodes")
        if (
            not isinstance(episodes, list)
            or len(episodes) != EPISODES
            or not all(isinstance(row, dict) and row for row in episodes)
        ):
            raise ValueError("v2 evaluation does not contain exactly 16 episodes")
        if not _finite_tree(evaluation):
            raise ValueError("v2 evaluation contains nonfinite or unsupported evidence")
        checkpoint = _mapping(evaluation.get("checkpoint"), "evaluation checkpoint")
        checkpoint_exact = {
            "path": str(checkpoint_path),
            "sha256": input_hashes[checkpoint_path],
            "training_seed": 0,
            "total_interactions": V2_INTERACTIONS,
            "reproduction_fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
        }
        for field_name, expected in checkpoint_exact.items():
            actual = checkpoint.get(field_name)
            if field_name == "path" and isinstance(actual, str):
                actual = str(_resolve(actual))
            if actual != expected:
                raise ValueError(f"evaluation checkpoint {field_name} differs from v2 training")
        if checkpoint.get("evaluation_manifest") != job.get("evaluation_manifest"):
            raise ValueError("evaluation manifest payload differs from v2 queue")
        evaluation_manifest = _mapping(job.get("evaluation_manifest"), "job evaluation_manifest")
        if (
            evaluation.get("protocol") != evaluation_manifest.get("evaluation_protocol")
            or evaluation.get("protocol_sha256")
            != evaluation_manifest.get("evaluation_protocol_sha256")
        ):
            raise ValueError("evaluation protocol identity differs from v2 queue")
        _validate_memory_gate(evaluation.get("memory_gate"), "evaluation memory gate")
        integrity = _mapping(evaluation.get("integrity"), "evaluation integrity")
        for name in (
            "task_manifest_matched",
            "evaluation_manifest_matched",
            "source_set_matched",
            "checkpoint_completed_budget",
            "all_actions_finite_and_bounded",
            "activity_from_actual_forward",
        ):
            if integrity.get(name) is not True:
                raise ValueError(f"evaluation integrity check {name} did not pass")
        for name in (
            "physical_wind_telemetry_passed",
            "physical_wind_uses_terminal_actual_interval",
        ):
            if integrity.get(name) is not True:
                raise ValueError(f"evaluation integrity check {name} did not pass")
        summary = _mapping(evaluation.get("summary"), "evaluation summary")
        _validate_v2_physical_wind(
            summary.get("physical_wind"), condition=condition
        )
        score = _mapping(summary.get("score"), "evaluation summary score")
        quality = _mapping(summary.get("control_quality"), "evaluation control quality")
        _validate_score(score)
        _validate_quality(quality)
        if not math.isclose(
            float(quality["component_scores_0_100"]["command_response"]),
            float(score["component_scores"]["response"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("v2 control-quality response differs from primary response")
        reward_components_raw = _mapping(
            summary.get("reward_component_mean_per_episode"),
            "v2 evaluation reward component means",
        )
        if set(reward_components_raw) != REWARD_COMPONENTS:
            raise ValueError("v2 evaluation reward-component set differs from contract")
        reward_components = {
            str(name): _finite_number(value, f"v2 reward component {name}")
            for name, value in reward_components_raw.items()
        }
        episode_component_sums = {name: 0.0 for name in REWARD_COMPONENTS}
        for episode_index, episode in enumerate(episodes):
            episode_rewards = _mapping(
                episode.get("reward_components"),
                f"v2 episode {episode_index} reward components",
            )
            if set(episode_rewards) != REWARD_COMPONENTS:
                raise ValueError(f"v2 episode {episode_index} reward-component set differs")
            for name, value in episode_rewards.items():
                episode_component_sums[name] += _finite_number(
                    value, f"v2 episode {episode_index} reward component {name}"
                )
        for name, total_value in episode_component_sums.items():
            if not _float32_episode_mean_matches(
                reward_components[name], total_value / EPISODES
            ):
                raise ValueError(f"v2 reward-component mean {name} does not recompute")
        activity = _mapping(evaluation.get("activity"), "evaluation activity")
        _validate_detailed_activity(
            activity,
            policy,
            require_unit_detail=controller in V2_LIF_CONTROLLERS,
        )
        eval_controller_report = _mapping(
            evaluation.get("controller_report"), "evaluation controller_report"
        )
        if _canonical_sha256(eval_controller_report) != _canonical_sha256(controller_report):
            raise ValueError("evaluation controller report differs from v2 queue declaration")

        cell.complete = True
        cell.reason = "verified complete"
        cell.training_manifest = manifest
        cell.evaluation = evaluation
        cell.history = history
        cell.score = dict(score)
        cell.quality = dict(quality)
        cell.reward_components = reward_components
        cell.activity = dict(activity)
        cell.core_before = core_before if isinstance(core_before, str) else None
        cell.core_after = core_after if isinstance(core_after, str) else None
        cell.input_hashes = input_hashes
    except (KeyError, TypeError, ValueError, OSError) as exc:
        cell.complete = False
        cell.reason = str(exc)
    return cell


def collect_v2_report_data(
    config_path: Path, queue_path: Path | None = None
) -> V2ReportData:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"active v2 command config is missing: {config_path}")
    config = _read_json(config_path)
    output_root = _validate_v2_config(config, config_path)
    queue_path = (
        queue_path.expanduser().resolve()
        if queue_path is not None
        else (output_root / "queue.json").resolve()
    )
    _require_under(queue_path, output_root, "v2 queue")
    if not queue_path.is_file():
        raise ValueError(f"active v2 command queue is missing: {queue_path}")
    config_sha = _sha256_file(config_path)
    queue_sha = _sha256_file(queue_path)
    queue = _read_json(queue_path)
    jobs = _validate_v2_queue(
        queue,
        config,
        config_path=config_path,
        config_sha256=config_sha,
        queue_path=queue_path,
        output_root=output_root,
    )
    reports = _mapping(queue["controller_reports"], "v2 queue.controller_reports")
    cells = [
        _validate_v2_cell(
            job,
            controller_report=_mapping(reports[job["controller"]], "v2 controller report"),
            output_root=output_root,
        )
        for job in jobs
    ]
    input_hashes: dict[Path, str] = {config_path: config_sha, queue_path: queue_sha}
    for field_name in ("queue_runner", "trainer", "evaluator"):
        dependency = _resolve(queue[field_name])
        input_hashes[dependency] = _sha256_file(dependency)
    for cell in cells:
        input_hashes.update(cell.input_hashes)
    return V2ReportData(
        config_path=config_path,
        queue_path=queue_path,
        output_root=output_root,
        config_sha256=config_sha,
        queue_sha256=queue_sha,
        cells=cells,
        input_hashes=input_hashes,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _status(cell: Cell) -> str:
    return "VERIFIED COMPLETE" if cell.complete else f"N/A ({cell.reason})"


def _score_value(cell: Cell, key: str) -> float | None:
    if not cell.complete or cell.score is None:
        return None
    if key == "total":
        return float(cell.score["score"])
    return float(cell.score["component_scores"][key])


def _quality_value(cell: Cell, key: str) -> float | None:
    if not cell.complete or cell.quality is None:
        return None
    return float(cell.quality["component_scores_0_100"][key])


def _raw_value(cell: Cell, key: str) -> float | None:
    if not cell.complete or cell.score is None:
        return None
    return float(cell.score["raw"][key])


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_escape(value) for value in row) + " |" for row in rows),
    ]


def render_markdown(data: ReportData, plot_paths: Mapping[str, Path]) -> str:
    complete_count = sum(cell.complete for cell in data.cells)
    status = "COMPLETE — all six jobs independently verified" if data.complete else (
        f"INCOMPLETE — {complete_count}/6 verified; every other result is N/A"
    )
    lines = [
        "# Crazyflie learned keyboard-command control report",
        "",
        f"Report state: **{status}**",
        "",
        f"Generated (UTC): `{data.generated_at_utc}`  ",
        f"Task: `{TASK}`  ",
        f"Config SHA-256: `{data.config_sha256}`  ",
        f"Queue snapshot SHA-256: `{data.queue_sha256}`",
        "",
        (
            "A verified row requires a completed 500,000-interaction training manifest, "
            "an authenticated immutable history, a matching checkpoint hash, a passing "
            "16 × 600 command evaluation, passing memory gates, and measured activity "
            "from the same action-producing forward pass. Missing or inconsistent evidence "
            "is shown as N/A; task quality is not inferred from PPO loss."
        ),
        "",
        "## Controller score and capacity",
        "",
    ]
    score_rows = []
    for cell in data.cells:
        params = cell.parameters
        required = params.get("parameter_matching_required")
        matched = params.get("actor_parameter_match_passed")
        if required is True:
            match_text = "PASS" if matched is True else "FAIL"
        elif required is False:
            match_text = "not required"
        else:
            match_text = "N/A"
        score_rows.append(
            (
                cell.controller,
                _status(cell),
                params.get("actor_trainable_parameters", "N/A"),
                params.get("total_trainable_parameters", "N/A"),
                params.get("frozen_parameters", "N/A"),
                match_text,
                _fmt(_score_value(cell, "total")),
                _fmt(_quality_value(cell, "acceleration_quality")),
                _fmt(_quality_value(cell, "command_response")),
                _fmt(_quality_value(cell, "flight_stability")),
                _fmt(_quality_value(cell, "survival_not_die")),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "Controller",
                "Evidence state",
                "Actor trainable",
                "Total trainable",
                "Frozen weights",
                "Parameter match",
                "Total /100",
                "Acceleration /100",
                "Response /100",
                "Stability /100",
                "Survival /100",
            ),
            score_rows,
        )
    )
    lines.extend(("", f"![Score comparison]({_escape(plot_paths['score'])})", ""))

    lines.extend(("## Raw held-out control measurements", ""))
    raw_fields = (
        ("Linear tracking RMSE (m/s)", "linear_tracking_rmse_m_s"),
        ("Yaw tracking RMSE (rad/s)", "yaw_tracking_rmse_rad_s"),
        ("Wrong-direction fraction", "wrong_direction_fraction"),
        ("Mean command projection ratio", "mean_command_projection_ratio"),
        ("Response latency (s)", "response_latency_mean_s"),
        ("Overshoot ratio", "overshoot_ratio_mean"),
        ("Brake settling (s)", "brake_settling_mean_s"),
        ("Hover drift (m)", "hover_drift_mean_m"),
        ("Survival fraction", "survival_fraction"),
        ("Invalid states", "invalid_state_count"),
        ("Action effort RMS", "action_effort_rms"),
        ("Action delta RMS", "action_delta_rms"),
    )
    lines.extend(
        _markdown_table(
            ("Measurement", *CONTROLLERS),
            [
                (label, *(_fmt(_raw_value(cell, key)) for cell in data.cells))
                for label, key in raw_fields
            ],
        )
    )

    lines.extend(("", "## Evaluation reward components", ""))
    component_names = sorted(
        {name for cell in data.cells if cell.complete for name in cell.reward_components}
    )
    if component_names:
        lines.extend(
            _markdown_table(
                ("Mean per 16-episode evaluation", *CONTROLLERS),
                [
                    (
                        component,
                        *(
                            _fmt(cell.reward_components.get(component))
                            if cell.complete
                            else "N/A"
                            for cell in data.cells
                        ),
                    )
                    for component in component_names
                ],
            )
        )
    else:
        lines.append("N/A — no evaluation reward-component evidence is verified.")
    lines.extend(
        (
            "",
            f"![Training reward]({_escape(plot_paths['reward'])})",
            "",
            f"![Training loss]({_escape(plot_paths['loss'])})",
            "",
            "## Frozen-core integrity",
            "",
        )
    )
    frozen_rows = []
    for cell in data.cells:
        if cell.controller not in LIF_CONTROLLERS:
            frozen_rows.append((cell.controller, "N/A (non-LIF)", "N/A", "N/A", "N/A"))
        elif not cell.complete:
            frozen_rows.append((cell.controller, "N/A (incomplete)", "N/A", "N/A", "N/A"))
        else:
            per_core = cell.parameters.get("per_core_checksums", {})
            frozen_rows.append(
                (
                    cell.controller,
                    "PASS" if cell.core_before == cell.core_after else "FAIL",
                    cell.core_before,
                    cell.core_after,
                    "; ".join(f"{key}={value}" for key, value in sorted(per_core.items()))
                    if isinstance(per_core, Mapping)
                    else "N/A",
                )
            )
    lines.extend(
        _markdown_table(
            ("Controller", "Frozen check", "Before", "After", "Authenticated per-core identities"),
            frozen_rows,
        )
    )

    lines.extend(("", "## Measured controller activity", ""))
    activity_rows = []
    all_roles: set[str] = set()
    for cell in data.cells:
        if not cell.complete or cell.activity is None:
            activity_rows.append((cell.controller, "N/A", "N/A", "N/A", "N/A", "N/A"))
            continue
        overall = cell.activity["overall"]
        all_roles.update(overall["roles"])
        activity_rows.append(
            (
                cell.controller,
                cell.activity.get("kind", "N/A"),
                cell.activity.get("unit_count", "N/A"),
                _fmt(overall["mean_absolute_activity_per_unit"]),
                _fmt(overall["rms_activity_per_unit"]),
                _fmt(overall["active_fraction_per_unit"]),
            )
        )
    lines.extend(
        _markdown_table(
            ("Controller", "Activity type", "Units", "Mean |activity|", "RMS", "Active fraction"),
            activity_rows,
        )
    )
    lines.extend(
        (
            "",
            "LIF values are sampled spikes; GRU/MLP values are absolute hidden activations. "
            "They are descriptive within-controller measurements, not causal evidence and "
            "not directly equivalent in physical units.",
            "",
        )
    )
    if all_roles:
        role_rows = []
        for role in sorted(all_roles):
            values = []
            for cell in data.cells:
                if not cell.complete or cell.activity is None:
                    values.append("N/A")
                    continue
                role_value = cell.activity["overall"]["roles"].get(role)
                values.append(
                    "N/A"
                    if role_value is None
                    else (
                        f"{_fmt(role_value['mean_absolute_activity_per_unit'])} "
                        f"({_fmt(role_value['active_fraction_per_unit'])} active)"
                    )
                )
            role_rows.append((role, *values))
        lines.extend(_markdown_table(("Role", *CONTROLLERS), role_rows))
    else:
        lines.append("N/A — no role-resolved activity evidence is verified.")
    lines.extend(("", f"![Activity comparison]({_escape(plot_paths['activity'])})", ""))

    lines.extend(("## Per-job validation", ""))
    lines.extend(
        _markdown_table(
            ("Queue job", "Controller", "Validation result", "Checkpoint", "History rows"),
            [
                (
                    cell.job_id,
                    cell.controller,
                    _status(cell),
                    _short(
                        cell.training_manifest.get("checkpoint_sha256")
                        if cell.training_manifest
                        else None
                    ),
                    len(cell.history) if cell.complete else "N/A",
                )
                for cell in data.cells
            ],
        )
    )
    lines.extend(
        (
            "",
            "## Input boundary",
            "",
            "The reporter read only the selected command config, its queue, and paths declared "
            "by those six queue jobs. Paused waypoint/gust queues and Unitree G1 artifacts are "
            "outside this report's input boundary.",
            "",
        )
    )
    return "\n".join(lines) + "\n"


def _v2_cells_by_pair(data: V2ReportData) -> dict[tuple[str, str], V2Cell]:
    return {(cell.controller, cell.condition): cell for cell in data.cells}


def _v2_delta(still: V2Cell, wind: V2Cell, getter: Any) -> float | None:
    if not still.complete or not wind.complete:
        return None
    still_value = getter(still)
    wind_value = getter(wind)
    if still_value is None or wind_value is None:
        return None
    return float(wind_value) - float(still_value)


def _v2_wind_telemetry(cell: V2Cell) -> Mapping[str, Any] | None:
    """Find an evaluator's compact physical-wrench summary without guessing values."""

    if not cell.complete or cell.evaluation is None:
        return None
    summary = cell.evaluation.get("summary")
    candidates = (
        cell.evaluation.get("wind_telemetry"),
        summary.get("wind_telemetry") if isinstance(summary, Mapping) else None,
        summary.get("physical_wind") if isinstance(summary, Mapping) else None,
    )
    return next((value for value in candidates if isinstance(value, Mapping)), None)


def render_v2_markdown(
    data: V2ReportData, plot_paths: Mapping[str, Path]
) -> str:
    complete_count = sum(cell.complete for cell in data.cells)
    status = (
        "COMPLETE — all 14 jobs independently verified"
        if data.complete
        else f"INCOMPLETE — {complete_count}/14 verified; every other result is N/A"
    )
    pair_map = _v2_cells_by_pair(data)
    lines = [
        "# Crazyflie wide-command optic/wind comparison report",
        "",
        f"Report state: **{status}**",
        "",
        f"Generated (UTC): `{data.generated_at_utc}`  ",
        "Tasks: `FlyCrazyflie-CommandFollowWide-v0` and "
        "`FlyCrazyflie-CommandFollowWideWind-v0`  ",
        f"Config SHA-256: `{data.config_sha256}`  ",
        f"Queue snapshot SHA-256: `{data.queue_sha256}`",
        "",
        (
            "Every numeric row below comes from an authenticated completed 1,000,000-interaction "
            "checkpoint and its matching deterministic 16 × 600 held-out evaluation. Pending, "
            "failed, missing, hash-mismatched, nonfinite, or memory-gate-failing cells remain N/A. "
            "Still and wind jobs are independently trained but use the same seed, PPO budget, "
            "command schedule seed, and per-controller architecture."
        ),
        "",
        "## Controller/task scores",
        "",
    ]
    score_rows = []
    for cell in data.cells:
        score_rows.append(
            (
                cell.controller,
                cell.condition,
                _status(cell),
                cell.parameters.get("actor_trainable_parameters", "N/A"),
                _fmt(_score_value(cell, "total")),
                _fmt(_score_value(cell, "linear_tracking")),
                _fmt(_raw_value(cell, "linear_tracking_rmse_m_s")),
                _fmt(_raw_value(cell, "yaw_tracking_rmse_rad_s")),
                _fmt(_quality_value(cell, "acceleration_quality")),
                _fmt(_quality_value(cell, "command_response")),
                _fmt(_quality_value(cell, "flight_stability")),
                _fmt(_quality_value(cell, "survival_not_die")),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "Controller",
                "Condition",
                "Evidence state",
                "Actor params",
                "Total /100",
                "Linear tracking /100",
                "Linear RMSE m/s",
                "Yaw RMSE rad/s",
                "Acceleration /100",
                "Response /100",
                "Stability /100",
                "Survival /100",
            ),
            score_rows,
        )
    )
    lines.extend(("", f"![Score by condition]({_escape(plot_paths['score'])})", ""))

    lines.extend(("## Wind effect (wind minus still)", ""))
    delta_rows = []
    for controller in V2_CONTROLLERS:
        still = pair_map[(controller, "still")]
        wind = pair_map[(controller, "wind")]
        delta_rows.append(
            (
                controller,
                "paired" if still.complete and wind.complete else "N/A (incomplete pair)",
                _fmt(_v2_delta(still, wind, lambda cell: _score_value(cell, "total"))),
                _fmt(
                    _v2_delta(
                        still, wind, lambda cell: _score_value(cell, "linear_tracking")
                    )
                ),
                _fmt(
                    _v2_delta(
                        still,
                        wind,
                        lambda cell: _raw_value(cell, "linear_tracking_rmse_m_s"),
                    )
                ),
                _fmt(
                    _v2_delta(
                        still,
                        wind,
                        lambda cell: _quality_value(cell, "acceleration_quality"),
                    )
                ),
                _fmt(
                    _v2_delta(
                        still,
                        wind,
                        lambda cell: _quality_value(cell, "flight_stability"),
                    )
                ),
                _fmt(
                    _v2_delta(
                        still,
                        wind,
                        lambda cell: _quality_value(cell, "survival_not_die"),
                    )
                ),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "Controller",
                "Pair state",
                "Δ total",
                "Δ tracking score",
                "Δ linear RMSE",
                "Δ acceleration",
                "Δ stability",
                "Δ survival",
            ),
            delta_rows,
        )
    )
    lines.extend(
        (
            "",
            "For score columns, negative means wind reduced performance. For RMSE, positive means "
            "wind increased error. These are descriptive paired-condition differences, not "
            "uncertainty-adjusted causal estimates.",
            "",
            f"![Wind deltas]({_escape(plot_paths['wind_delta'])})",
            "",
            "## Capacity and frozen-core contract",
            "",
        )
    )
    capacity_rows = []
    for controller in V2_CONTROLLERS:
        representative = pair_map[(controller, "still")]
        params = representative.parameters
        required = params.get("parameter_matching_required")
        matched = params.get("actor_parameter_match_passed")
        capacity_rows.append(
            (
                controller,
                params.get("actor_trainable_parameters", "N/A"),
                params.get("total_trainable_parameters", "N/A"),
                params.get("frozen_parameters", "N/A"),
                (
                    "PASS"
                    if required is True and matched is True
                    else "FAIL"
                    if required is True
                    else "not required"
                    if required is False
                    else "N/A"
                ),
                (
                    "PASS in both conditions"
                    if controller in V2_LIF_CONTROLLERS
                    and pair_map[(controller, "still")].complete
                    and pair_map[(controller, "wind")].complete
                    and pair_map[(controller, "still")].core_before
                    == pair_map[(controller, "still")].core_after
                    and pair_map[(controller, "wind")].core_before
                    == pair_map[(controller, "wind")].core_after
                    else "N/A (non-LIF)"
                    if controller not in V2_LIF_CONTROLLERS
                    else "N/A (incomplete pair)"
                ),
                _short(params.get("core_checksum")),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "Controller",
                "Actor trainable",
                "Total trainable",
                "Frozen weights",
                "Parameter match",
                "Frozen-core check",
                "Core SHA-256",
            ),
            capacity_rows,
        )
    )

    lines.extend(
        (
            "",
            "## LIF role activity",
            "",
            "Activity is taken from the exact forward pass that produced each evaluated action. "
            "The table reports sampled spike activity; it does not imply causal importance.",
            "",
        )
    )
    role_rows = []
    for cell in data.cells:
        if cell.controller not in V2_LIF_CONTROLLERS:
            continue
        if not cell.complete or cell.activity is None:
            role_rows.append((cell.controller, cell.condition, "N/A", "N/A", "N/A", "N/A"))
            continue
        roles = cell.activity["overall"]["roles"]
        for role, summary in sorted(roles.items()):
            role_rows.append(
                (
                    cell.controller,
                    cell.condition,
                    role,
                    summary["unit_count"],
                    _fmt(summary["mean_absolute_activity_per_unit"]),
                    _fmt(summary["active_fraction_per_unit"]),
                )
            )
    lines.extend(
        _markdown_table(
            ("Controller", "Condition", "Role", "Units", "Mean |spike|", "Active fraction"),
            role_rows,
        )
    )
    lines.extend(("", f"![LIF activity]({_escape(plot_paths['activity'])})", ""))

    lines.extend(("## Most active LIF neurons", ""))
    top_rows = []
    for cell in data.cells:
        if cell.controller not in V2_LIF_CONTROLLERS:
            continue
        if not cell.complete or cell.activity is None:
            top_rows.append(
                (cell.controller, cell.condition, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
            )
            continue
        for rank, unit in enumerate(cell.activity["top_units"], start=1):
            top_rows.append(
                (
                    cell.controller,
                    cell.condition,
                    rank,
                    unit["id"],
                    unit["role"],
                    _fmt(unit["mean_absolute_activity"]),
                    _fmt(unit["rms_activity"]),
                    _fmt(unit["active_fraction"]),
                )
            )
    lines.extend(
        _markdown_table(
            (
                "Controller",
                "Condition",
                "Rank",
                "Neuron ID",
                "Role",
                "Mean |activity|",
                "RMS",
                "Active fraction",
            ),
            top_rows,
        )
    )

    lines.extend(("", "## Physical wind evidence", ""))
    telemetry_rows = []
    for cell in data.cells:
        telemetry = _v2_wind_telemetry(cell)
        integrity = (
            cell.evaluation.get("integrity", {})
            if cell.complete and cell.evaluation is not None
            else {}
        )
        telemetry_rows.append(
            (
                cell.controller,
                cell.condition,
                _status(cell),
                (
                    json.dumps(telemetry, sort_keys=True, separators=(",", ":"))
                    if telemetry is not None
                    else "N/A — compact physical-wrench summary not serialized"
                ),
                integrity.get("physical_wind_telemetry_passed", "N/A"),
                integrity.get("physical_wind_uses_terminal_actual_interval", "N/A"),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "Controller",
                "Condition",
                "Evidence state",
                "Recorded telemetry summary",
                "Physical telemetry integrity",
                "Actual-interval source",
            ),
            telemetry_rows,
        )
    )

    lines.extend(
        (
            "",
            f"![Training reward]({_escape(plot_paths['reward'])})",
            "",
            f"![Training loss]({_escape(plot_paths['loss'])})",
            "",
            "## Per-job validation",
            "",
        )
    )
    lines.extend(
        _markdown_table(
            ("Queue job", "Controller", "Condition", "Validation", "Checkpoint", "History rows"),
            [
                (
                    cell.job_id,
                    cell.controller,
                    cell.condition,
                    _status(cell),
                    _short(
                        cell.training_manifest.get("checkpoint_sha256")
                        if cell.training_manifest
                        else None
                    ),
                    len(cell.history) if cell.complete else "N/A",
                )
                for cell in data.cells
            ],
        )
    )
    incomplete = [cell for cell in data.cells if not cell.complete]
    lines.extend(("", "## Missing or rejected evidence", ""))
    if incomplete:
        lines.extend(
            _markdown_table(
                ("Queue job", "Controller", "Condition", "Reason"),
                [
                    (cell.job_id, cell.controller, cell.condition, cell.reason)
                    for cell in incomplete
                ],
            )
        )
    else:
        lines.append("None — all 14 cells passed artifact validation.")
    lines.extend(
        (
            "",
            "## Input boundary",
            "",
            "The reporter read only the selected v2 config, its queue, queue-authenticated "
            "runner/trainer/evaluator files, and artifact paths declared by those 14 jobs. "
            "It did not read or modify paused Unitree G1, legacy waypoint/gust, or v1 run artifacts.",
            "",
        )
    )
    return "\n".join(lines) + "\n"


def _plot_placeholder(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def render_plots(data: ReportData, destinations: Mapping[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [cell.controller.replace("frozen_lif_", "lif_") for cell in data.cells]
    colors = ["#214761", "#397a9a", "#47a69a", "#7b61a8", "#d07a36", "#b64d57"]

    fig, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    complete = [cell for cell in data.cells if cell.complete]
    if complete:
        metrics = (
            ("Total", lambda cell: _score_value(cell, "total")),
            ("Acceleration", lambda cell: _quality_value(cell, "acceleration_quality")),
            ("Response", lambda cell: _quality_value(cell, "command_response")),
            ("Stability", lambda cell: _quality_value(cell, "flight_stability")),
            ("Survival", lambda cell: _quality_value(cell, "survival_not_die")),
        )
        x = np.arange(len(data.cells), dtype=float)
        width = 0.16
        for metric_index, (label, getter) in enumerate(metrics):
            values = [getter(cell) if cell.complete else np.nan for cell in data.cells]
            axis.bar(x + (metric_index - 2) * width, values, width, label=label)
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Held-out score (0–100)")
        axis.legend(ncols=5, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    else:
        _plot_placeholder(axis, "N/A — no verified completed evaluations")
    axis.set_title("Crazyflie command-control score")
    fig.savefig(destinations["score"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for cell, color in zip(data.cells, colors, strict=True):
        if not cell.complete:
            continue
        axis.plot(
            [row["total_interactions"] for row in cell.history],
            [row["mean_rollout_reward"] for row in cell.history],
            label=cell.controller,
            color=color,
            linewidth=1.7,
        )
    if axis.lines:
        axis.legend(fontsize=8)
        axis.set_xlabel("Training interactions")
        axis.set_ylabel("Mean rollout reward")
        axis.grid(alpha=0.25)
    else:
        _plot_placeholder(axis, "N/A — no authenticated completed training histories")
    axis.set_title("Command-follow training reward")
    fig.savefig(destinations["reward"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    loss_fields = (("loss", "Total loss"), ("policy_loss", "Policy loss"), ("value_loss", "Value loss"))
    any_loss = False
    for axis, (field_name, label) in zip(axes, loss_fields, strict=True):
        for cell, color in zip(data.cells, colors, strict=True):
            if not cell.complete:
                continue
            any_loss = True
            axis.plot(
                [row["total_interactions"] for row in cell.history],
                [row[field_name] for row in cell.history],
                label=cell.controller,
                color=color,
                linewidth=1.4,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    if any_loss:
        axes[0].legend(ncols=3, fontsize=8)
        axes[-1].set_xlabel("Training interactions")
    else:
        for axis in axes:
            _plot_placeholder(axis, "N/A")
    fig.suptitle("Command-follow PPO losses")
    fig.savefig(destinations["loss"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True)
    mean_activity = []
    active_fraction = []
    for cell in data.cells:
        if cell.complete and cell.activity is not None:
            mean_activity.append(cell.activity["overall"]["mean_absolute_activity_per_unit"])
            active_fraction.append(cell.activity["overall"]["active_fraction_per_unit"])
        else:
            mean_activity.append(np.nan)
            active_fraction.append(np.nan)
    if complete:
        x = np.arange(len(data.cells))
        axes[0].bar(x, mean_activity, color=colors)
        axes[0].set_ylabel("Mean |activity| / unit")
        axes[1].bar(x, active_fraction, color=colors)
        axes[1].set_ylabel("Active fraction / unit")
        axes[1].set_xticks(x, labels, rotation=20, ha="right")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
    else:
        for axis in axes:
            _plot_placeholder(axis, "N/A — no verified measured activity")
    fig.suptitle("Actual action-producing controller activity\n(spikes for LIF; absolute hidden activation for GRU/MLP)")
    fig.savefig(destinations["activity"], dpi=170, facecolor="white")
    plt.close(fig)


def render_v2_plots(
    data: V2ReportData, destinations: Mapping[str, Path]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {
        "original_lif": "#214761",
        "rewired_lif": "#397a9a",
        "wing_lif": "#47a69a",
        "leg_wing_lif": "#7b61a8",
        "optic_lif": "#5b8e3e",
        "gru_matched": "#d07a36",
        "mlp_normal": "#b64d57",
    }
    pair_map = _v2_cells_by_pair(data)
    x = np.arange(len(V2_CONTROLLERS), dtype=float)

    fig, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    any_score = False
    for offset, condition in ((-0.19, "still"), (0.19, "wind")):
        values = []
        for controller in V2_CONTROLLERS:
            cell = pair_map[(controller, condition)]
            value = _score_value(cell, "total")
            values.append(np.nan if value is None else value)
            any_score |= value is not None
        axis.bar(
            x + offset,
            values,
            0.36,
            label=condition,
            color=[colors[controller] for controller in V2_CONTROLLERS],
            alpha=1.0 if condition == "still" else 0.55,
            edgecolor="black" if condition == "wind" else "none",
            linewidth=0.5,
        )
    if any_score:
        axis.set_xticks(x, V2_CONTROLLERS, rotation=20, ha="right")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Held-out total score (0–100)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    else:
        _plot_placeholder(axis, "N/A — no verified completed v2 evaluations")
    axis.set_title("Crazyflie wide-command score: still vs wind")
    fig.savefig(destinations["score"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    any_history = False
    for cell in data.cells:
        if not cell.complete:
            continue
        any_history = True
        axis.plot(
            [row["total_interactions"] for row in cell.history],
            [row["mean_rollout_reward"] for row in cell.history],
            label=f"{cell.controller}/{cell.condition}",
            color=colors[cell.controller],
            linestyle="-" if cell.condition == "still" else "--",
            linewidth=1.4,
        )
    if any_history:
        axis.set_xlabel("Training interactions")
        axis.set_ylabel("Mean rollout reward")
        axis.grid(alpha=0.25)
        axis.legend(ncols=2, fontsize=7)
    else:
        _plot_placeholder(axis, "N/A — no authenticated completed training histories")
    axis.set_title("Wide-command training reward (solid still; dashed wind)")
    fig.savefig(destinations["reward"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    loss_fields = (
        ("loss", "Total loss"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
    )
    any_loss = False
    for axis, (field_name, label) in zip(axes, loss_fields, strict=True):
        for cell in data.cells:
            if not cell.complete:
                continue
            any_loss = True
            axis.plot(
                [row["total_interactions"] for row in cell.history],
                [row[field_name] for row in cell.history],
                label=f"{cell.controller}/{cell.condition}",
                color=colors[cell.controller],
                linestyle="-" if cell.condition == "still" else "--",
                linewidth=1.2,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    if any_loss:
        axes[0].legend(ncols=3, fontsize=6.5)
        axes[-1].set_xlabel("Training interactions")
    else:
        for axis in axes:
            _plot_placeholder(axis, "N/A")
    fig.suptitle("Wide-command PPO losses (solid still; dashed wind)")
    fig.savefig(destinations["loss"], dpi=170, facecolor="white")
    plt.close(fig)

    lif_labels = list(V2_CONTROLLERS[:5])
    lif_x = np.arange(len(lif_labels), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True)
    any_activity = False
    for offset, condition in ((-0.19, "still"), (0.19, "wind")):
        means = []
        active = []
        for controller in lif_labels:
            cell = pair_map[(controller, condition)]
            if cell.complete and cell.activity is not None:
                any_activity = True
                means.append(cell.activity["overall"]["mean_absolute_activity_per_unit"])
                active.append(cell.activity["overall"]["active_fraction_per_unit"])
            else:
                means.append(np.nan)
                active.append(np.nan)
        axes[0].bar(
            lif_x + offset,
            means,
            0.36,
            label=condition,
            color=[colors[label] for label in lif_labels],
            alpha=1.0 if condition == "still" else 0.55,
        )
        axes[1].bar(
            lif_x + offset,
            active,
            0.36,
            label=condition,
            color=[colors[label] for label in lif_labels],
            alpha=1.0 if condition == "still" else 0.55,
        )
    if any_activity:
        axes[0].set_ylabel("Mean |spike| per unit")
        axes[1].set_ylabel("Active fraction per unit")
        axes[1].set_xticks(lif_x, lif_labels, rotation=20, ha="right")
        axes[0].legend()
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
    else:
        for axis in axes:
            _plot_placeholder(axis, "N/A — no verified LIF activity")
    fig.suptitle("Action-producing LIF activity")
    fig.savefig(destinations["activity"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    metrics = (
        ("Total", lambda cell: _score_value(cell, "total")),
        ("Tracking", lambda cell: _score_value(cell, "linear_tracking")),
        ("Acceleration", lambda cell: _quality_value(cell, "acceleration_quality")),
        ("Stability", lambda cell: _quality_value(cell, "flight_stability")),
        ("Survival", lambda cell: _quality_value(cell, "survival_not_die")),
    )
    any_delta = False
    width = 0.15
    for metric_index, (label, getter) in enumerate(metrics):
        values = []
        for controller in V2_CONTROLLERS:
            value = _v2_delta(
                pair_map[(controller, "still")],
                pair_map[(controller, "wind")],
                getter,
            )
            values.append(np.nan if value is None else value)
            any_delta |= value is not None
        axis.bar(x + (metric_index - 2) * width, values, width, label=label)
    if any_delta:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, V2_CONTROLLERS, rotation=20, ha="right")
        axis.set_ylabel("Wind minus still (score points)")
        axis.legend(ncols=5, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    else:
        _plot_placeholder(axis, "N/A — no complete still/wind pair")
    axis.set_title("Paired wind performance deltas")
    fig.savefig(destinations["wind_delta"], dpi=170, facecolor="white")
    plt.close(fig)


def _revalidate_input_snapshot(input_hashes: Mapping[Path, str]) -> None:
    for path, expected in input_hashes.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"input changed while report was being generated: {path}")


def _publish_no_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite report evidence: {destination}")
    os.link(source, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_report_bundle(data: ReportData, report_path: Path) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    plot_paths = {name: data.output_root / filename for name, filename in PLOT_NAMES.items()}
    destinations = [report_path, *plot_paths.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing report evidence: " + ", ".join(map(str, existing))
        )
    data.output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".crazyflie-command-report-", dir=data.output_root))
    report_temp: Path | None = None
    try:
        staged_plots = {name: staging / filename for name, filename in PLOT_NAMES.items()}
        render_plots(data, staged_plots)
        relative_plots = {
            name: Path(os.path.relpath(path, report_path.parent)) for name, path in plot_paths.items()
        }
        markdown = render_markdown(data, relative_plots)
        handle, raw_path = tempfile.mkstemp(
            prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
        )
        report_temp = Path(raw_path)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(markdown)
            stream.flush()
            os.fsync(stream.fileno())
        _revalidate_input_snapshot(data.input_hashes)
        # All content is fully rendered and every input has been rechecked
        # before any destination becomes visible. Each link is no-overwrite.
        for name in ("score", "reward", "loss", "activity"):
            _publish_no_overwrite(staged_plots[name], plot_paths[name])
        _publish_no_overwrite(report_temp, report_path)
        return {
            "status": "COMPLETE" if data.complete else "INCOMPLETE",
            "verified_jobs": sum(cell.complete for cell in data.cells),
            "job_count": len(data.cells),
            "report": str(report_path),
            "plots": {name: str(path) for name, path in plot_paths.items()},
        }
    finally:
        if report_temp is not None:
            report_temp.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def write_v2_report_bundle(
    data: V2ReportData, report_path: Path
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    plot_paths = {
        name: data.output_root / filename for name, filename in V2_PLOT_NAMES.items()
    }
    destinations = [report_path, *plot_paths.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing v2 report evidence: "
            + ", ".join(map(str, existing))
        )
    data.output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".crazyflie-command-v2-report-", dir=data.output_root))
    report_temp: Path | None = None
    try:
        staged_plots = {
            name: staging / filename for name, filename in V2_PLOT_NAMES.items()
        }
        render_v2_plots(data, staged_plots)
        relative_plots = {
            name: Path(os.path.relpath(path, report_path.parent))
            for name, path in plot_paths.items()
        }
        markdown = render_v2_markdown(data, relative_plots)
        handle, raw_path = tempfile.mkstemp(
            prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
        )
        report_temp = Path(raw_path)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(markdown)
            stream.flush()
            os.fsync(stream.fileno())
        _revalidate_input_snapshot(data.input_hashes)
        for name in ("score", "reward", "loss", "activity", "wind_delta"):
            _publish_no_overwrite(staged_plots[name], plot_paths[name])
        _publish_no_overwrite(report_temp, report_path)
        return {
            "status": "COMPLETE" if data.complete else "INCOMPLETE",
            "schema_version": 2,
            "verified_jobs": sum(cell.complete for cell in data.cells),
            "job_count": len(data.cells),
            "evaluation_episodes": sum(EPISODES for cell in data.cells if cell.complete),
            "report": str(report_path),
            "plots": {name: str(path) for name, path in plot_paths.items()},
        }
    finally:
        if report_temp is not None:
            report_temp.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _read_json(args.config.expanduser().resolve())
        if config.get("schema_version") == 2:
            data = collect_v2_report_data(args.config, args.queue)
            report_path = (
                V2_DEFAULT_REPORT if args.report == DEFAULT_REPORT else args.report
            )
            result = write_v2_report_bundle(data, report_path)
        else:
            data = collect_report_data(args.config, args.queue)
            result = write_report_bundle(data, args.report)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    # An incomplete artifact is useful and truthful, but is not an accepted
    # completed comparison; expose that distinction to automation.
    return 0 if data.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
