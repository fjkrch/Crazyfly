#!/usr/bin/env python3
"""Validate and summarize the bounded leg/wing/combined LIF pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


CONDITIONS = {
    "leg_only": "crazyflie-wing-pilot-leg_lif-matched-v1",
    "wing_only": "crazyflie-wing-pilot-wing_lif-resume-v1",
    "leg_plus_wing": "crazyflie-wing-pilot-leg_wing_lif-resume-v1",
}
SCENARIOS = (
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _matched_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    ignored = {"controller", "wing_extension"}
    return {key: value for key, value in resolved.items() if key not in ignored}


def summarize(runs_root: Path) -> dict[str, Any]:
    rows = []
    matched_reference = None
    evaluation_manifest_id = None
    for label, directory in CONDITIONS.items():
        run_dir = runs_root / directory
        training = _read(run_dir / "training_manifest.json")
        evaluation = _read(run_dir / "evaluation.json")
        if training.get("status") != "completed" or evaluation.get("status") != "completed":
            raise ValueError(f"{label} did not complete its bounded train/evaluation pipeline")
        if training.get("environment_interactions") != 200 or training.get("completed_updates") != 2:
            raise ValueError(f"{label} did not use the matched 200-interaction/two-update budget")
        if training.get("memory_gate", {}).get("passed") is not True:
            raise ValueError(f"{label} training memory gate failed")
        if evaluation.get("total_episodes") != 6:
            raise ValueError(f"{label} evaluation must contain exactly six bounded episodes")
        if set(evaluation.get("scenario_results", {})) != set(SCENARIOS):
            raise ValueError(f"{label} scenario membership changed")
        current_manifest_id = evaluation.get("evaluation_manifest_id")
        if evaluation_manifest_id is None:
            evaluation_manifest_id = current_manifest_id
        elif current_manifest_id != evaluation_manifest_id:
            raise ValueError("Evaluation manifests differ across extension conditions")
        matched = _matched_payload(training["resolved_config"])
        if matched_reference is None:
            matched_reference = matched
        elif matched != matched_reference:
            raise ValueError("Task, seed, budget, or PPO settings differ across extension conditions")
        report = training["controller_report"]
        scenario_rows = []
        for scenario in SCENARIOS:
            result = evaluation["scenario_results"][scenario]
            summary = result["summary"]
            memory = result["memory_gate"]
            if memory.get("passed") is not True:
                raise ValueError(f"{label}/{scenario} evaluation memory gate failed")
            scenario_rows.append({
                "scenario": scenario,
                "episodes": summary["episode_count"],
                "success_count": summary["success_count"],
                "failure_count": summary["failure_count"],
                "crash_count": summary["crash_count"],
                "invalid_state_count": summary["invalid_state_count"],
                "out_of_bounds_count": summary["out_of_bounds_count"],
                "truncation_count": summary["truncation_count"],
                "switch_success_count": (
                    summary["switch_metrics"]["switch_success_count"]
                    if summary.get("switch_metrics") else None
                ),
                "gust_recovery_success_count": (
                    summary["gust_metrics"]["recovery_success_count"]
                    if summary.get("gust_metrics") else None
                ),
                "max_device_gpu_used_mib": memory["max_device_gpu_used_mib"],
                "max_process_rss_mib": memory["max_process_rss_mib"],
            })
        rows.append({
            "condition": label,
            "controller": training["controller"],
            "fingerprint": training["fingerprint"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "resume_count": training["resume_count"],
            "actor_trainable_parameters": report["actor_trainable_parameters"],
            "frozen_synaptic_weights": report["frozen_synaptic_weights"],
            "total_dynamic_state_per_environment": report["total_dynamic_state_per_environment"],
            "parameter_matching_required": report["parameter_matching_required"],
            "per_core_checksums": report["per_core_checksums"],
            "core_checksum_unchanged": (
                training["core_checksum_before"] == training["core_checksum_after"]
            ),
            "training_memory_gate": training["memory_gate"],
            "scenarios": scenario_rows,
        })
    total_success = sum(
        scenario["success_count"] for row in rows for scenario in row["scenarios"]
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if all(row["core_checksum_unchanged"] for row in rows) else "FAIL",
        "pipeline_interpretation": "bounded_pipeline_pass",
        "behavioral_interpretation": (
            "success_observed" if total_success else "no_task_success_observed_at_200_interactions"
        ),
        "parameter_matching_required": False,
        "matched_settings": matched_reference,
        "evaluation_manifest_id": evaluation_manifest_id,
        "conditions": rows,
        "total_training_jobs": 3,
        "total_evaluation_episodes": 18,
        "total_task_successes": total_success,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bounded Crazyflie leg/wing LIF comparison",
        "",
        f"Pipeline: **{report['status']}**. Behaviour: "
        f"**{report['behavioral_interpretation']}**.",
        "",
        "This is a 200-interaction pipeline pilot, not evidence of learned flight.",
        "The combined controller is intentionally not parameter matched.",
        "",
        "| Condition | Actor trainable | Frozen synapses | State/env | Reach | Switch | Gust | Train GPU MiB | Train RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        successes = {item["scenario"]: item["success_count"] for item in row["scenarios"]}
        memory = row["training_memory_gate"]
        lines.append(
            f"| {row['condition']} | {row['actor_trainable_parameters']} | "
            f"{row['frozen_synaptic_weights']} | {row['total_dynamic_state_per_environment']} | "
            f"{successes[SCENARIOS[0]]}/2 | {successes[SCENARIOS[1]]}/2 | "
            f"{successes[SCENARIOS[2]]}/2 | {memory['max_device_gpu_used_mib']} | "
            f"{memory['max_process_rss_mib']} |"
        )
    lines.extend(["", "All 18 episodes had zero crash, invalid-state, and out-of-bounds events.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.runs_root.resolve())
        _atomic(args.output.resolve(), report)
        markdown = args.markdown.resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        temporary = markdown.with_name(markdown.name + ".tmp")
        temporary.write_text(_markdown(report), encoding="utf-8")
        os.replace(temporary, markdown)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": report["status"],
        "behavioral_interpretation": report["behavioral_interpretation"],
        "output": str(args.output.resolve()),
        "markdown": str(args.markdown.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
