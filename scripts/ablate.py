#!/usr/bin/env python3
"""Run paired sham, target, and degree-matched acute LIF ablations."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from _bootstrap import ROOT
from g1_fly_control.connectome import load_connectome
from g1_fly_control.connectome.schema import file_sha256
from g1_fly_control.evaluation.ablations import (
    degree_match_report,
    matched_random_groups,
    resolve_target_indices,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _annotation(value: str) -> tuple[str, str]:
    field, separator, selected = value.partition("=")
    if not separator or not field or not selected:
        raise ValueError("--annotation must be FIELD=VALUE with both parts non-empty.")
    return field, selected


def _paired_effect(sham: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
    """Episode-level intervention minus sham differences under a common seed."""
    if sham.get("evaluation_seed") != intervention.get("evaluation_seed"):
        raise ValueError("Ablation and sham evaluations have different evaluation seeds.")
    sham_schedule = sham.get("scenario", {}).get("schedule", {}).get("sha256")
    intervention_schedule = intervention.get("scenario", {}).get("schedule", {}).get("sha256")
    if not sham_schedule or sham_schedule != intervention_schedule:
        raise ValueError("Ablation and sham evaluations have different or missing held-out schedules.")
    left = {int(row["episode_id"]): row for row in sham["episodes"]}
    right = {int(row["episode_id"]): row for row in intervention["episodes"]}
    if left.keys() != right.keys() or not left:
        raise ValueError("Ablation and sham evaluations do not contain identical episode IDs.")
    for episode_id in left:
        for field in ("paired_plan_sha256", "initial_state_sha256"):
            if not left[episode_id].get(field) or left[episode_id][field] != right[episode_id].get(field):
                raise ValueError(f"Episode {episode_id} has different or missing {field} across conditions.")
    metrics = (
        "success", "accumulated_goal_relative_progress_m", "mechanical_work_proxy",
        "excessive_impact", "joint_limit_frequency", "saturation_frequency",
    )
    rows = []
    for episode_id in sorted(left):
        rows.append({
            "episode_id": episode_id,
            **{f"delta_{metric}": float(right[episode_id][metric]) - float(left[episode_id][metric])
               for metric in metrics},
        })
    return {
        "definition": "intervention minus sham, matched by episode_id with the same evaluation seed",
        "pairing_verified": True,
        "schedule_sha256": sham_schedule,
        "episodes": rows,
        "mean_deltas": {
            metric: sum(row[f"delta_{metric}"] for row in rows) / len(rows) for metric in metrics
        },
    }


def _run_condition(job: dict[str, Any], *, args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    name = job["name"]
    id_file = artifact_dir / "source_ids" / f"{name}.json"
    result_file = artifact_dir / "evaluations" / f"{name}.json"
    log_file = artifact_dir / "logs" / f"{name}.log"
    _write_json(id_file, job["source_neuron_ids"])
    command = [
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--checkpoint", str(args.checkpoint),
        "--connectome_manifest", str(args.connectome_manifest),
        "--task", args.task,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--protocol", "heldout_v1",
        "--ablate_ids", str(id_file),
        "--output", str(result_file),
    ]
    if args.headless:
        command.append("--headless")
    job.update({"status": "running", "command": command, "evaluation_path": str(result_file),
                "log_path": str(log_file)})
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not result_file.is_file():
        job.update({"status": "failed", "exit_code": completed.returncode})
        raise RuntimeError(f"{name} evaluation failed (exit {completed.returncode}); see {log_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    actual_ids = result.get("ablation", {}).get("source_neuron_ids")
    if result.get("status") != "executed" or actual_ids != job["source_neuron_ids"]:
        job.update({"status": "failed", "exit_code": completed.returncode})
        raise RuntimeError(f"{name} evaluation did not verify its requested source neuron IDs: {result_file}")
    job.update({"status": "passed", "exit_code": completed.returncode,
                "summary": result.get("per_seed", []), "scenario": result.get("scenario")})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fixed original frozen-LIF checkpoint.")
    parser.add_argument("--connectome_manifest", type=Path, required=True, help="Real-data circuit manifest.")
    parser.add_argument("--task", default="FlyG1-GoalReach-FreePosture-v0")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101, help="Common episode/evaluation seed for every condition.")
    parser.add_argument("--control_seed", type=int, default=17, help="Seed for matched-group sampling.")
    parser.add_argument("--controls", type=int, default=10, help="Number of distinct size/degree-matched random groups.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--neuron_ids", nargs="+", help="Stable MaleCNS source neuron IDs.")
    selection.add_argument("--role", help="Exact model_role annotation, such as sensory_input.")
    selection.add_argument("--annotation", help="Exact annotation selector FIELD=VALUE, such as type=Foo.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "ablation.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write a resolved plan without launching Isaac Sim.")
    parser.add_argument("--resume", action="store_true", help="Resume matching planned conditions from an existing output.")
    args = parser.parse_args()
    if args.episodes < 1 or args.controls < 2:
        parser.error("--episodes must be positive and --controls must be at least two.")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.connectome_manifest = args.connectome_manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint was not found: {args.checkpoint}")
    try:
        circuit = load_connectome(args.connectome_manifest)
        annotation = _annotation(args.annotation) if args.annotation is not None else None
        targets = resolve_target_indices(
            circuit, source_ids=args.neuron_ids, role=args.role, annotation=annotation,
        )
        controls = matched_random_groups(
            circuit.edge_index, targets, seed=args.control_seed, count=args.controls,
            num_neurons=circuit.num_neurons,
        )
    except ValueError as exc:
        parser.error(str(exc))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("policy") != "frozen_lif":
        parser.error("Acute ablation currently requires an original frozen_lif checkpoint.")
    if metadata.get("connectome_checksum") != circuit.checksum:
        parser.error("Checkpoint connectome checksum does not match the supplied manifest.")

    artifact_dir = args.output.parent / args.output.stem
    selected_ids = [circuit.neuron_ids[index] for index in targets]
    jobs = [{"name": "sham", "kind": "sham", "source_neuron_ids": [], "status": "planned"},
            {"name": "target", "kind": "target", "source_neuron_ids": selected_ids, "status": "planned"}]
    for index, group in enumerate(controls):
        jobs.append({
            "name": f"random_{index:02d}", "kind": "matched_random", "status": "planned",
            "source_neuron_ids": [circuit.neuron_ids[neuron] for neuron in group],
            "degree_match": degree_match_report(
                circuit.edge_index, targets, group, num_neurons=circuit.num_neurons,
            ),
        })
    specification = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_training_seed": metadata.get("seed"),
        "circuit_manifest": str(args.connectome_manifest), "circuit_checksum": circuit.checksum,
        "frozen_core_checksum": metadata.get("frozen_core_checksum"),
        "ablation_runner_sha256": file_sha256(Path(__file__).resolve()),
        "ablation_helpers_sha256": file_sha256(
            ROOT / "source" / "g1_fly_control" / "g1_fly_control" / "evaluation" / "ablations.py"
        ),
        "evaluator_sha256": file_sha256(ROOT / "scripts" / "evaluate.py"),
        "evaluation_protocol_sha256": file_sha256(ROOT / "scripts" / "evaluation_protocol.py"),
        "task": args.task, "episodes_per_condition": args.episodes,
        "evaluation_seed": args.seed, "control_selection_seed": args.control_seed,
        "control_count": args.controls, "protocol": "heldout_v1",
        "selector": {"source_ids": args.neuron_ids, "role": args.role, "annotation": annotation},
        "target_source_neuron_ids": selected_ids,
        "target_indices": targets,
        "intervention": "Clamp selected neurons' outgoing spikes to zero at each neural substep; neural voltage and intrinsic spikes still evolve; no weights or adapters change.",
        "matching": "same size, no target overlap, nearest candidate sampling by directed in/out degree; realized profile distances reported per control",
    }
    fingerprint = sha256(json.dumps(specification, sort_keys=True).encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "status": "planned" if args.dry_run else "running",
        "specification": specification, "specification_sha256": fingerprint, "conditions": jobs,
        "pairing": "All conditions request the same heldout_v1 protocol and evaluation seed. Inspect child scenario records for schedule verification.",
    }
    if args.output.exists():
        if not args.resume:
            parser.error(f"Output exists; choose a new --output or use --resume: {args.output}")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("specification_sha256") != fingerprint:
            parser.error("Existing ablation output has a different specification; cannot resume.")
        manifest = existing
    else:
        _write_json(args.output, manifest)
    if args.dry_run:
        print(json.dumps({"status": "planned", "output": str(args.output), "conditions": len(jobs)}, indent=2))
        return 0

    results: dict[str, dict[str, Any]] = {}
    for index, job in enumerate(manifest["conditions"], start=1):
        print(f"[{index}/{len(jobs)}] {job['name']}", flush=True)
        result_file = artifact_dir / "evaluations" / f"{job['name']}.json"
        if job.get("status") == "passed" and result_file.is_file():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result.get("status") != "executed" or result.get("ablation", {}).get("source_neuron_ids") != job["source_neuron_ids"]:
                job["status"] = "failed"
                manifest["status"] = "failed"
                _write_json(args.output, manifest)
                print(f"Existing {job['name']} evaluation does not verify its requested source neuron IDs.", file=sys.stderr)
                return 2
        else:
            try:
                result = _run_condition(job, args=args, artifact_dir=artifact_dir)
            except RuntimeError as exc:
                manifest["status"] = "failed"
                _write_json(args.output, manifest)
                print(str(exc), file=sys.stderr)
                return 2
        results[job["name"]] = result
        if job["name"] != "sham":
            try:
                job["paired_effect_vs_sham"] = _paired_effect(results["sham"], result)
            except ValueError as exc:
                job["status"] = "failed_pairing"
                manifest["status"] = "failed"
                _write_json(args.output, manifest)
                print(str(exc), file=sys.stderr)
                return 2
        _write_json(args.output, manifest)
    manifest["status"] = "executed"
    _write_json(args.output, manifest)
    print(json.dumps({"status": "executed", "output": str(args.output),
                      "conditions": len(manifest["conditions"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
