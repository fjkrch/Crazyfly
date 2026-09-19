"""Frozen graph and provenance gates for the additive wing circuit."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch

from g1_fly_control.connectome import load_connectome


ROOT = Path(__file__).resolve().parents[2]
WING = ROOT / "data" / "connectome_wing"


def test_frozen_wing_manifest_is_real_path_complete_and_independent() -> None:
    manifest = json.loads((WING / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((WING / "audit.json").read_text(encoding="utf-8"))
    leg_neurons = json.loads(
        (ROOT / "data" / "connectome" / "neurons.json").read_text(encoding="utf-8")
    )
    wing_neurons = json.loads((WING / "neurons.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "real"
    assert manifest["derived_counts"] == {
        "descending_inputs": 12,
        "edges": 8_864,
        "excitatory_edges": 6_726,
        "inhibitory_edges": 2_138,
        "input_path_coverage": 32,
        "intrinsic": 200,
        "motor_outputs": 24,
        "neurons": 256,
        "neurons_on_input_output_paths": 256,
        "output_path_coverage": 24,
        "sensory_inputs": 20,
    }
    assert manifest["role_counts"] == {
        "descending_input": 12,
        "vnc_interneuron": 200,
        "wing_motor_output": 24,
        "wing_sensory_input": 20,
    }
    leg_sensory = {
        item["id"]
        for item in leg_neurons
        if item["annotations"]["model_role"] == "sensory_input"
    }
    wing_sensory = {
        item["id"]
        for item in wing_neurons
        if item["annotations"]["model_role"] == "wing_sensory_input"
    }
    assert len(wing_sensory) == 20
    assert wing_sensory.isdisjoint(leg_sensory)
    assert manifest["selection_settings"]["path_completion_attempts"] == 1
    assert manifest["neuron_model"]["threshold"] == 0.04
    assert manifest["model_calibration"]["status"] == (
        "frozen_after_bounded_activity_pilot"
    )
    assert audit["status"] == "PASS"
    assert audit["derived_counts"] == manifest["derived_counts"]

    for label, filename in (("neurons", "neurons.json"), ("edges", "edges.json")):
        assert sha256((WING / filename).read_bytes()).hexdigest() == manifest["checksums"][label]


def test_frozen_wing_graph_has_declared_signs_and_no_motor_origin_edges() -> None:
    circuit = load_connectome(WING / "manifest.json")
    neurons = json.loads((WING / "neurons.json").read_text(encoding="utf-8"))
    edges = json.loads((WING / "edges.json").read_text(encoding="utf-8"))
    roles = {item["id"]: item["annotations"]["model_role"] for item in neurons}

    assert circuit.num_neurons == 256
    assert circuit.edge_index.shape == (2, 8_864)
    assert torch.isfinite(circuit.weights).all()
    assert torch.count_nonzero(circuit.weights > 0) == 6_726
    assert torch.count_nonzero(circuit.weights < 0) == 2_138
    assert len({(edge["pre"], edge["post"]) for edge in edges}) == len(edges)
    assert all(edge["pre"] != edge["post"] for edge in edges)
    assert all(roles[edge["pre"]] != "wing_motor_output" for edge in edges)
