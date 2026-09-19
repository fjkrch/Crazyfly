#!/usr/bin/env python3
"""Validate and compare the 60 held-out root-pose/net-force recordings on CPU.

All values are descriptive proxies from valid pre-action samples. Named sensor
net force cannot identify a ground-contact partner or establish a gait.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:  # Support both `python scripts/...py` and package imports in CPU tests.
    from .regime_metrics import summarize_recording
    from .evaluation_protocol import schedule_manifest
except ImportError:
    from regime_metrics import summarize_recording
    from evaluation_protocol import schedule_manifest


CONDITIONS = (
    "frozen_lif_original", "frozen_lif_degree_rewired",
    "gru_trainable", "mlp_engineering_baseline",
)
SCENARIOS = (
    "FlyG1-GoalReach-FreePosture-v0",
    "FlyG1-GoalSwitch-FreePosture-v0",
    "FlyG1-PushRecovery-FreePosture-v0",
)
SEEDS = tuple(range(5))
EPISODES = 16
EVALUATION_SEED = 101
BASELINE = CONDITIONS[0]
EXPECTED = {(condition, seed, scenario)
            for condition in CONDITIONS for seed in SEEDS for scenario in SCENARIOS}
METRICS = (
    "root_height_m_mean_of_episode_means",
    "root_up_axis_alignment_cosine_mean_of_episode_means",
    "root_up_axis_positive_threshold_fraction",
    "root_up_axis_negative_threshold_fraction",
)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} needs a nonempty path")
    return Path(value).expanduser().resolve(strict=True)


def _check_hash(value: Any, path: Path, field: str, cache: dict[Path, str]) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field} requires a SHA-256 digest")
    if path not in cache:
        cache[path] = _sha(path)
    if cache[path] != value:
        raise ValueError(f"{field} SHA-256 differs from {path}")


def _identity(metadata: dict[str, Any]) -> tuple[str, int, str]:
    key = (metadata.get("condition"), metadata.get("training_seed"), metadata.get("scenario"))
    if key not in EXPECTED:
        raise ValueError(f"unexpected condition/training seed/scenario: {key}")
    return key


def _source_identity(
    record: dict[str, Any], *, expected_job: dict[str, Any] | None,
    main_job: dict[str, Any] | None, fingerprint: str | None,
    sha_cache: dict[Path, str],
) -> dict[str, Any]:
    metadata = record["metadata"]
    condition, seed, scenario = _identity(metadata)
    if metadata.get("schema_version") != "heldout_regimes_v1":
        raise ValueError("recording metadata schema_version is not heldout_regimes_v1")
    if metadata.get("episode_count") != EPISODES or len(record["episodes"]) != EPISODES:
        raise ValueError("recording must contain exactly 16 episodes")
    if [row["episode_id"] for row in record["episodes"]] != list(range(EPISODES)):
        raise ValueError("recording episode IDs must be 0 through 15 in order")
    if metadata.get("evaluation_seed") != EVALUATION_SEED:
        raise ValueError("recording evaluation seed is not 101")
    if metadata.get("task") != scenario:
        raise ValueError("recording task differs from scenario")
    if (not isinstance(metadata.get("execution_source_fingerprint"), str)
            or len(metadata["execution_source_fingerprint"]) != 64):
        raise ValueError("recording execution fingerprint is missing")
    code_hashes = metadata.get("code_sha256")
    if (not isinstance(code_hashes, dict) or not code_hashes
            or any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
                   for key, value in code_hashes.items())):
        raise ValueError("recording code source hashes are missing")
    if fingerprint is not None and metadata.get("execution_source_fingerprint") != fingerprint:
        raise ValueError("recording execution fingerprint differs from main matrix")
    if expected_job is not None:
        for field, expected in (
            ("condition", condition), ("training_seed", seed),
            ("scenario", scenario), ("evaluation_seed", EVALUATION_SEED),
        ):
            if expected_job.get(field) != expected:
                raise ValueError(f"batch job {field} differs from recording")
        if Path(str(expected_job.get("output_npz", ""))).resolve() != Path(record["recording_path"]):
            raise ValueError("batch output_npz path differs from recording")
        _check_hash(expected_job.get("output_npz_sha256"), Path(record["recording_path"]),
                    "batch output_npz_sha256", sha_cache)
        if metadata.get("code_sha256") != expected_job.get("_recorder_code_sha256"):
            raise ValueError("recording recorder source hashes differ from batch manifest")
        for field in ("connectome_manifest", "connectome_manifest_sha256"):
            if metadata.get(field) != expected_job.get(field):
                raise ValueError(f"recording {field} differs from batch job")

    checkpoint = _path(metadata.get("checkpoint"), "checkpoint")
    evaluation_path = _path(metadata.get("evaluation_json"), "evaluation_json")
    _check_hash(metadata.get("checkpoint_sha256"), checkpoint, "checkpoint_sha256", sha_cache)
    _check_hash(metadata.get("evaluation_json_sha256"), evaluation_path,
                "evaluation_json_sha256", sha_cache)
    if metadata.get("connectome_manifest") is None:
        if metadata.get("connectome_manifest_sha256") is not None:
            raise ValueError("recording connectome manifest SHA has no path")
    else:
        connectome = _path(metadata["connectome_manifest"], "connectome_manifest")
        _check_hash(metadata.get("connectome_manifest_sha256"), connectome,
                    "connectome_manifest_sha256", sha_cache)
    if expected_job is not None:
        for field, path in (("checkpoint", checkpoint), ("evaluation_json", evaluation_path)):
            if Path(str(expected_job.get(field, ""))).resolve() != path:
                raise ValueError(f"batch {field} path differs from recording")
            _check_hash(expected_job.get(field + "_sha256"), path,
                        "batch " + field + "_sha256", sha_cache)
    if main_job is not None:
        if (main_job.get("status") != "passed" or main_job.get("condition") != condition
                or main_job.get("seed") != seed or main_job.get("id") != f"{condition}__seed-{seed}"):
            raise ValueError("main-matrix job identity or passed status differs")
        if Path(str(main_job.get("checkpoint", ""))).resolve() != checkpoint:
            raise ValueError("main-matrix checkpoint path differs")
        eval_job = main_job.get("evaluations", {}).get(scenario, {})
        if eval_job.get("status") != "passed" or Path(str(eval_job.get("result_file", ""))).resolve() != evaluation_path:
            raise ValueError("main-matrix evaluation path/status differs")

    evaluation = _read_json(evaluation_path)
    scenario_data = evaluation.get("scenario")
    if (evaluation.get("status") != "executed" or evaluation.get("task") != scenario
            or evaluation.get("training_seed") != seed
            or evaluation.get("evaluation_seed") != EVALUATION_SEED
            or evaluation.get("n_episodes") != EPISODES
            or evaluation.get("ablation") is not None
            or Path(str(evaluation.get("checkpoint", ""))).resolve() != checkpoint
            or not isinstance(scenario_data, dict)
            or scenario_data.get("task") != scenario
            or scenario_data.get("evaluation_protocol") != "heldout_v1"
            or scenario_data.get("policy_action_mode") != "deterministic"):
        raise ValueError("evaluation JSON has different replay identity or is not executed heldout_v1")
    if not math.isclose(float(scenario_data.get("control_dt_s", -1)),
                        float(metadata["control_dt_s"]), abs_tol=1e-9):
        raise ValueError("evaluation/recording control_dt_s differs")
    schedule = schedule_manifest(EVALUATION_SEED, EPISODES, scenario)
    if (json.dumps(scenario_data.get("schedule"), sort_keys=True)
            != json.dumps(schedule, sort_keys=True)
            or schedule["sha256"] != metadata.get("scenario_schedule_sha256")):
        raise ValueError("evaluation/recording schedule differs from deterministic heldout_v1")
    rows = evaluation.get("episodes")
    if not isinstance(rows, list) or len(rows) != EPISODES:
        raise ValueError("evaluation JSON needs 16 episode rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("evaluation episode rows must be objects")
    for field in ("initial_state_sha256", "paired_plan_sha256"):
        hashes = metadata.get(field)
        if (not isinstance(hashes, list) or len(hashes) != EPISODES
                or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
                or hashes != [row.get(field) for row in rows]):
            raise ValueError(f"evaluation/recording {field} list differs")
    for episode_id, row in enumerate(rows):
        if (not isinstance(row, dict) or row.get("episode_id") != episode_id
                or row.get("seed") != seed):
            raise ValueError("evaluation episode IDs or training seeds differ")
        expected_plan_sha = sha256(json.dumps({
            "scenario_schedule_sha256": schedule["sha256"],
            "environment_plan": schedule["plan"][episode_id],
        }, sort_keys=True).encode()).hexdigest()
        if row.get("paired_plan_sha256") != expected_plan_sha:
            raise ValueError(f"evaluation episode {episode_id} has a non-protocol paired plan hash")
    checks = metadata.get("reference_checks")
    if not isinstance(checks, dict) or checks.get("required_match") is not True:
        raise ValueError("companion replay did not match reference outcomes and scheduled events")
    for field in ("success_match_by_episode", "target_time_match_by_episode",
                  "scheduled_event_match_by_episode"):
        values = checks.get(field)
        if not isinstance(values, list) or len(values) != EPISODES or any(value is not True for value in values):
            raise ValueError(f"companion replay {field} is incomplete or false")
    full_event_flags = checks.get("full_event_match_by_episode")
    if (not isinstance(full_event_flags, list) or len(full_event_flags) != EPISODES
            or any(type(value) is not bool for value in full_event_flags)):
        raise ValueError("companion replay full-event diagnostics are missing")
    if (checks.get("control_steps_match") is not True
            or checks.get("replay_control_steps") != evaluation.get("execution", {}).get("control_steps")
            or checks.get("replay_success_by_episode") != [row.get("success") for row in rows]):
        raise ValueError("companion replay control steps or success differ from evaluation")
    replay_times = checks.get("replay_target_time_s_by_episode")
    if not isinstance(replay_times, list) or len(replay_times) != EPISODES:
        raise ValueError("companion replay target times are missing")
    for actual, row in zip(replay_times, rows):
        expected_time = row.get("time_to_target_s")
        if actual is None and expected_time is None:
            continue
        if (type(actual) not in (int, float) or type(expected_time) not in (int, float)
                or not math.isfinite(actual) or not math.isfinite(expected_time)
                or not math.isclose(actual, expected_time, rel_tol=0.0, abs_tol=1e-4)):
            raise ValueError("companion replay target time differs from evaluation")
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha_cache[checkpoint],
        "evaluation_json": str(evaluation_path), "evaluation_json_sha256": sha_cache[evaluation_path],
        "scenario_schedule_sha256": schedule["sha256"],
        "initial_state_sha256": metadata["initial_state_sha256"],
        "paired_plan_sha256": metadata["paired_plan_sha256"],
        "execution_source_fingerprint": metadata.get("execution_source_fingerprint"),
        "companion_full_event_matches": sum(full_event_flags),
        "code_sha256": code_hashes,
        "connectome_manifest": metadata.get("connectome_manifest"),
        "connectome_manifest_sha256": metadata.get("connectome_manifest_sha256"),
    }


def _seed_metrics(record: dict[str, Any]) -> dict[str, float]:
    seed = record["per_training_seed"]
    root = seed["root_up_axis_alignment_cosine"]
    result = {
        "root_height_m_mean_of_episode_means": seed["root_height_m"]["mean_of_episode_means"],
        "root_up_axis_alignment_cosine_mean_of_episode_means": root["mean_of_episode_means"],
        "root_up_axis_positive_threshold_fraction": root["mean_episode_positive_threshold_fraction"],
        "root_up_axis_negative_threshold_fraction": root["mean_episode_negative_threshold_fraction"],
    }
    for name, item in seed["net_body_force"].items():
        result[f"net_body_force_threshold_fraction::{name}"] = item["mean_episode_threshold_fraction"]
    for name, value in result.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"nonfinite seed metric {name}")
    return result


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {"n_training_seeds": len(values), "mean": mean(values) if values else None,
            "sample_sd": stdev(values) if len(values) > 1 else None}


def _main_index(manifest: dict[str, Any], path: Path, cache: dict[Path, str]) -> dict[tuple[str, int], dict]:
    if manifest.get("schema_version") != "regime_matrix_v1":
        raise ValueError("batch manifest schema_version is not regime_matrix_v1")
    main_path = _path(manifest.get("main_matrix"), "main_matrix")
    _check_hash(manifest.get("main_matrix_sha256"), main_path, "main_matrix_sha256", cache)
    main = _read_json(main_path)
    if main.get("status") != "complete" or main.get("execution_source_fingerprint") != manifest.get("execution_source_fingerprint"):
        raise ValueError("main matrix is incomplete or its execution fingerprint differs")
    config = main.get("config", {})
    if (config.get("conditions") != list(CONDITIONS) or config.get("seeds") != list(SEEDS)
            or set(config.get("evaluation", {}).get("scenario_tasks", [])) != set(SCENARIOS)
            or config.get("evaluation", {}).get("episodes") != EPISODES):
        raise ValueError("main matrix does not specify the expected 4×5×3×16 evaluation")
    jobs = main.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(CONDITIONS) * len(SEEDS):
        raise ValueError("main matrix must contain exactly 20 training jobs")
    index = {(row.get("condition"), row.get("seed")): row for row in jobs}
    if len(index) != len(jobs):
        raise ValueError("main matrix has duplicate condition/seed jobs")
    return index


def summarize_regime_matrix(
    recordings: list[str | Path] | None = None, *, manifest_path: str | Path | None = None,
    force_threshold_n: float = 20.0, upright_cos_threshold: float = 0.5,
) -> dict[str, Any]:
    """Produce a complete report only when all 60 source-backed cells validate."""
    if (recordings is None) == (manifest_path is None):
        raise ValueError("Provide either recordings or a batch manifest")
    cache: dict[Path, str] = {}
    errors: list[str] = []
    manifest = None
    main_index: dict[tuple[str, int], dict] = {}
    manifest_jobs: dict[tuple[str, int, str], dict] = {}
    if manifest_path is not None:
        manifest_file = _path(str(manifest_path), "manifest_path")
        manifest = _read_json(manifest_file)
        main_index = _main_index(manifest, manifest_file, cache)
        if manifest.get("status") != "complete" or manifest.get("counts") != {"passed": len(EXPECTED)}:
            errors.append("Batch manifest status/counts do not claim exactly 60 passed recordings")
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != len(EXPECTED):
            raise ValueError("batch manifest must list exactly 60 recording jobs")
        for row in jobs:
            if not isinstance(row, dict):
                raise ValueError("batch manifest job must be an object")
            key = (row.get("condition"), row.get("training_seed"), row.get("scenario"))
            if key not in EXPECTED or key in manifest_jobs:
                raise ValueError(f"unexpected or duplicate batch job: {key}")
            if row.get("id") != f"{key[0]}__seed-{key[1]}__{key[2]}":
                raise ValueError(f"batch job ID differs: {key}")
            if row.get("status") not in {"ready", "running", "passed", "failed"}:
                raise ValueError(f"batch job status is invalid: {key}")
            manifest_jobs[key] = row
        if set(manifest_jobs) != EXPECTED:
            raise ValueError("batch manifest is missing required jobs")
        paths = [row["output_npz"] for row in jobs if row.get("status") == "passed"]
    else:
        manifest_file = None
        paths = list(recordings or [])
    accepted: dict[tuple[str, int, str], dict] = {}
    provenance: dict[tuple[str, int, str], dict] = {}
    body_names: list[str] | None = None
    pairing: dict[tuple[str, int], tuple[list[str], list[str], str]] = {}
    direct_fingerprint: str | None = None
    for raw_path in paths:
        try:
            path = _path(str(raw_path), "recording")
            record = summarize_recording(path, force_threshold_n=force_threshold_n,
                                         upright_cos_threshold=upright_cos_threshold)
            key = _identity(record["metadata"])
            if key in accepted:
                raise ValueError(f"duplicate recording for {key}")
            if body_names is not None and record["body_names"] != body_names:
                raise ValueError("named force-sensor body list/order differs across recordings")
            expected_job = manifest_jobs.get(key) if manifest is not None else None
            if manifest is not None and expected_job is None:
                raise ValueError(f"recording has no matching batch job: {key}")
            if expected_job is not None:
                expected_job = {**expected_job,
                                "_recorder_code_sha256": manifest.get("recorder_code_sha256")}
            source = _source_identity(
                record, expected_job=expected_job, main_job=main_index.get((key[0], key[1])),
                fingerprint=manifest.get("execution_source_fingerprint") if manifest else None,
                sha_cache=cache,
            )
            if (manifest is None and direct_fingerprint is not None
                    and source["execution_source_fingerprint"] != direct_fingerprint):
                raise ValueError("direct recordings have different execution source fingerprints")
            pair = (source["initial_state_sha256"], source["paired_plan_sha256"],
                    source["scenario_schedule_sha256"])
            pair_key = (key[1], SCENARIOS.index(key[2]))
            if pair_key in pairing and pairing[pair_key] != pair:
                raise ValueError(f"held-out paired plans/initial states differ for seed {key[1]} scenario {key[2]}")
            _seed_metrics(record)
            accepted[key] = record
            provenance[key] = source
            body_names = record["body_names"]
            pairing[pair_key] = pair
            if manifest is None:
                direct_fingerprint = source["execution_source_fingerprint"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{raw_path}: {exc}")
    missing = sorted(EXPECTED - set(accepted))
    metric_rows: dict[str, dict[str, dict]] = {}
    paired_rows: dict[str, dict[str, dict]] = {}
    for scenario in SCENARIOS:
        metric_rows[scenario] = {}
        paired_rows[scenario] = {}
        for condition in CONDITIONS:
            seeds = [seed for seed in SEEDS if (condition, seed, scenario) in accepted]
            values = {seed: _seed_metrics(accepted[(condition, seed, scenario)]) for seed in seeds}
            keys = [*METRICS, *(f"net_body_force_threshold_fraction::{name}" for name in body_names or [])]
            metric_rows[scenario][condition] = {
                "training_seeds": seeds,
                "episode_count": len(seeds) * EPISODES,
                "metrics": {metric: _stats([values[seed][metric] for seed in seeds]) for metric in keys},
            }
            if condition == BASELINE:
                continue
            paired = [seed for seed in seeds if (BASELINE, seed, scenario) in accepted]
            paired_rows[scenario][condition] = {
                "training_seeds": paired,
                "difference_definition": f"{condition} minus {BASELINE}, matched by training seed and held-out plan",
                "metrics": {metric: _stats([
                    values[seed][metric] - _seed_metrics(accepted[(BASELINE, seed, scenario)])[metric]
                    for seed in paired
                ]) for metric in keys},
            }
    return {
        "schema_version": "regime_matrix_comparison_v1",
        "analysis_code_sha256": {
            name: _sha(Path(__file__).with_name(name))
            for name in ("summarize_regime_matrix.py", "regime_metrics.py", "run_regime_matrix.py")
        },
        "status": (
            "complete" if (manifest is not None and manifest.get("status") == "complete"
                           and len(accepted) == len(EXPECTED) and not errors)
            else "descriptive_only" if (manifest is None and len(accepted) == len(EXPECTED)
                                     and not errors)
            else "incomplete"
        ),
        "interpretation": "Descriptive root-pose and named-body net-force proxies only; no contact-partner, ground-support, gait, walking, or crawling inference.",
        "validation_limitations": (
            "This CPU report verifies saved files, hashes, identities, and the deterministic heldout_v1 plan. "
            "The recorder checked live initial states, success outcomes, target times, scheduled event sequences, "
            "and aggregate control steps against each reference evaluation. State-dependent event payloads are "
            "reported separately; action hashes and bitwise trajectory equivalence are unavailable. "
            "This CPU report cannot remeasure simulator state or prove that traces came from the stated checkpoint. "
            "Direct NPZ inputs lack a validated completed main-matrix manifest and can only yield descriptive_only status."
        ),
        "definitions": {
            "sample_phase": "pre-action states before each first episode reset; no terminal post-action sample",
            "unit_of_aggregation": "one training seed; each seed is the equal-weight mean of 16 episode metrics",
            "cross_seed_statistic": "arithmetic mean and sample SD over independent training seeds",
            "force_threshold_n": float(force_threshold_n),
            "upright_cos_threshold": float(upright_cos_threshold),
            "force_measure": "magnitude of ContactSensor net_forces_w for each exact named body; contact partner unknown",
            "root_height_measure": "root world-z position, not body clearance or a posture label",
            "orientation_measure": "signed cosine between root local +z and world +z",
        },
        "batch_manifest": str(manifest_file) if manifest_file else None,
        "batch_manifest_sha256": _sha(manifest_file) if manifest_file else None,
        "batch_manifest_status": manifest.get("status") if manifest else None,
        "main_matrix": manifest.get("main_matrix") if manifest else None,
        "main_matrix_sha256": manifest.get("main_matrix_sha256") if manifest else None,
        "execution_source_fingerprint": (
            manifest.get("execution_source_fingerprint") if manifest else direct_fingerprint
        ),
        "expected_recordings": len(EXPECTED),
        "validated_recordings": len(accepted),
        "expected_episodes": len(EXPECTED) * EPISODES,
        "validated_episodes": len(accepted) * EPISODES,
        "companion_full_event_matches": sum(
            provenance[key]["companion_full_event_matches"] for key in accepted
        ),
        "body_names": body_names or [],
        "missing_recordings": [dict(condition=c, training_seed=s, scenario=t) for c, s, t in missing],
        "validation_errors": errors,
        "recordings": [dict(condition=key[0], training_seed=key[1], scenario=key[2],
                            path=accepted[key]["recording_path"],
                            sha256=accepted[key]["recording_sha256"],
                            **provenance[key]) for key in sorted(accepted)],
        "by_scenario": metric_rows,
        "paired_difference_vs_original": paired_rows,
    }


def _fmt(value: float | None, *, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Held-out root-pose and named net-force comparison", "",
             f"**Status:** {report['status']} — {report['validated_recordings']}/{report['expected_recordings']} recordings; "
             f"{report['validated_episodes']}/{report['expected_episodes']} episodes.", "",
             report["interpretation"], "",
             report["validation_limitations"], "",
             "Each value is a mean of 16 equal-weight episode summaries per training seed, then a mean ± sample SD across independent seeds. "
             "Paired differences match training seed and held-out plan. Fractions use valid pre-action samples only.", "",
             f"State-dependent event payloads matched the reference within tolerance for "
             f"{report['companion_full_event_matches']}/{report['validated_episodes']} accepted episodes; "
             "matching outcomes and schedules does not prove identical trajectories.", "",
             f"Net-force threshold: {report['definitions']['force_threshold_n']:g} N; root-axis cosine threshold: "
             f"{report['definitions']['upright_cos_threshold']:g}.", ""]
    if report["validation_errors"]:
        lines += [f"Validation errors: {len(report['validation_errors'])} (see JSON).", ""]
    for scenario in SCENARIOS:
        lines += [f"## {scenario}", "",
                  "| Condition | Seeds | Root height m | Root up cosine | Positive-axis fraction | Negative-axis fraction |",
                  "|---|---:|---:|---:|---:|---:|"]
        for condition in CONDITIONS:
            row = report["by_scenario"][scenario][condition]
            stats = row["metrics"]
            def cell(key: str) -> str:
                item = stats[key]
                return f"{_fmt(item['mean'])} ± {_fmt(item['sample_sd'])}"
            lines.append(f"| {condition} | {len(row['training_seeds'])} | "
                         f"{cell(METRICS[0])} | {cell(METRICS[1])} | {cell(METRICS[2])} | {cell(METRICS[3])} |")
        lines += ["", "**Named-body net-force fraction at or above threshold**", "",
                  "| Condition | Body | Seeds | Fraction mean ± sample SD | Paired Δ vs original mean ± sample SD |",
                  "|---|---|---:|---:|---:|"]
        for condition in CONDITIONS:
            row = report["by_scenario"][scenario][condition]
            for name in report["body_names"]:
                metric = f"net_body_force_threshold_fraction::{name}"
                stat = row["metrics"][metric]
                difference = report["paired_difference_vs_original"][scenario].get(condition)
                delta = difference["metrics"][metric] if difference else None
                delta_text = "—" if delta is None else f"{_fmt(delta['mean'])} ± {_fmt(delta['sample_sd'])}"
                lines.append(f"| {condition} | {name} | {len(row['training_seeds'])} | "
                             f"{_fmt(stat['mean'])} ± {_fmt(stat['sample_sd'])} | {delta_text} |")
        lines += ["", "**Paired root-pose differences vs original** (condition minus original)", "",
                  "| Condition | Paired seeds | Δ root height m | Δ positive-axis fraction | Δ negative-axis fraction |",
                  "|---|---:|---:|---:|---:|"]
        for condition in CONDITIONS[1:]:
            row = report["paired_difference_vs_original"][scenario][condition]
            stats = row["metrics"]
            lines.append(f"| {condition} | {len(row['training_seeds'])} | "
                         f"{_fmt(stats[METRICS[0]]['mean'])} ± {_fmt(stats[METRICS[0]]['sample_sd'])} | "
                         f"{_fmt(stats[METRICS[2]]['mean'])} ± {_fmt(stats[METRICS[2]]['sample_sd'])} | "
                         f"{_fmt(stats[METRICS[3]]['mean'])} ± {_fmt(stats[METRICS[3]]['sample_sd'])} |")
        lines.append("")
    lines += ["Root height and orientation cannot by themselves classify posture. Body net-force magnitude does not identify contact with the ground, locomotion, or gait.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--recordings", nargs="+", type=Path)
    parser.add_argument("--force-threshold-n", type=float, default=20.0)
    parser.add_argument("--upright-cos-threshold", type=float, default=0.5)
    parser.add_argument("--output-prefix", required=True, type=Path,
                        help="Write PREFIX.json and PREFIX.md.")
    args = parser.parse_args()
    report = summarize_regime_matrix(
        args.recordings, manifest_path=args.manifest,
        force_threshold_n=args.force_threshold_n,
        upright_cos_threshold=args.upright_cos_threshold,
    )
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                           encoding="utf-8")
    prefix.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "validated_recordings": report["validated_recordings"],
                      "output_json": str(prefix.with_suffix(".json")),
                      "output_md": str(prefix.with_suffix(".md"))}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
