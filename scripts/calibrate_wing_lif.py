#!/usr/bin/env python3
"""Run the predeclared deterministic activity calibration for the wing LIF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from drone_bootstrap import SOURCE, sha256_file

if str(SOURCE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SOURCE))

from g1_fly_control.connectome import load_connectome  # noqa: E402
from g1_fly_control.policies.lif_core import LIFCore  # noqa: E402


CANDIDATE_THRESHOLDS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
CONTROL_STEPS = 100
BATCH_SIZE = 8
SELECTION_RULE = (
    "Among candidates passing input_mean_rate in [0.05,0.50], motor_mean_rate in "
    "[0.005,0.10], global_mean_rate in [0.005,0.20], dead_neuron_fraction <=0.75, "
    "and saturated_neuron_fraction <=0.05, minimize abs(motor_mean_rate-0.02); "
    "tie-break by higher threshold."
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def calibrate(manifest: Path) -> dict:
    circuit = load_connectome(manifest)
    index_of = {neuron_id: index for index, neuron_id in enumerate(circuit.neuron_ids)}
    input_indices = torch.tensor([index_of[x] for x in circuit.manifest.input_neuron_ids])
    output_indices = torch.tensor([index_of[x] for x in circuit.manifest.output_neuron_ids])
    constants = {
        key: value
        for key, value in circuit.manifest.neuron_model.items()
        if key in {
            "dt", "tau_membrane", "tau_synapse", "reset_value",
            "refractory_steps", "surrogate_beta", "neural_substeps",
        }
    }
    results = []
    for threshold in CANDIDATE_THRESHOLDS:
        core = LIFCore(
            circuit.num_neurons,
            circuit.edge_index,
            circuit.weights,
            threshold=threshold,
            **constants,
        )
        state = core.initial_state(BATCH_SIZE)
        spike_counts = torch.zeros((BATCH_SIZE, circuit.num_neurons))
        for step in range(CONTROL_STEPS):
            current = torch.zeros((BATCH_SIZE, circuit.num_neurons))
            phase = (
                torch.arange(BATCH_SIZE, dtype=torch.float32)[:, None] * 0.37
                + torch.arange(input_indices.numel(), dtype=torch.float32)[None, :] * 0.19
                + step * 0.11
            )
            # Fixed bounded engineering probe, independent of flight outcomes.
            current[:, input_indices] = 0.04 + 0.03 * (torch.sin(phase) + 1.0)
            state = core(current, state)
            spike_counts += state.spikes
        rates = spike_counts / CONTROL_STEPS
        per_neuron = rates.mean(dim=0)
        metrics = {
            "threshold": threshold,
            "global_mean_rate": float(rates.mean()),
            "input_mean_rate": float(rates[:, input_indices].mean()),
            "motor_mean_rate": float(rates[:, output_indices].mean()),
            "dead_neuron_fraction": float((per_neuron == 0).float().mean()),
            "saturated_neuron_fraction": float((per_neuron > 0.50).float().mean()),
        }
        metrics["passed_activity_gates"] = bool(
            0.05 <= metrics["input_mean_rate"] <= 0.50
            and 0.005 <= metrics["motor_mean_rate"] <= 0.10
            and 0.005 <= metrics["global_mean_rate"] <= 0.20
            and metrics["dead_neuron_fraction"] <= 0.75
            and metrics["saturated_neuron_fraction"] <= 0.05
        )
        results.append(metrics)
    passing = [row for row in results if row["passed_activity_gates"]]
    if not passing:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
            "selection_rule": SELECTION_RULE,
            "probe": {"control_steps": CONTROL_STEPS, "batch_size": BATCH_SIZE},
            "results": results,
            "failure_reason": "No candidate passed the predeclared activity gates",
        }
    selected = min(
        passing,
        key=lambda row: (abs(row["motor_mean_rate"] - 0.02), -row["threshold"]),
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
        "selection_rule": SELECTION_RULE,
        "probe": {
            "control_steps": CONTROL_STEPS,
            "batch_size": BATCH_SIZE,
            "input_current_formula": "0.04 + 0.03*(sin(batch*0.37 + input*0.19 + step*0.11)+1)",
            "input_current_bounds": [0.04, 0.10],
        },
        "results": results,
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "interpretation": "Engineering activity calibration only; not fly physiology or flight evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/connectome_wing_precalibration/manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/connectome_wing_calibration.json"),
    )
    args = parser.parse_args()
    report = calibrate(args.manifest.resolve())
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
