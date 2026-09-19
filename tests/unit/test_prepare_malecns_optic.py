"""Frozen provenance, biology, topology, and determinism gates for optic MaleCNS."""

from __future__ import annotations

from collections import deque
from hashlib import sha256
import json
import math
from pathlib import Path

import pytest
import torch

from g1_fly_control.connectome import load_connectome
from g1_fly_control.crazyflie.controllers import lif_actor_parameter_target
from scripts.prepare_malecns_optic import (
    EXPECTED_SOURCE_SHA256,
    _balanced_greedy_cover,
)


ROOT = Path(__file__).resolve().parents[2]
OPTIC = ROOT / "data" / "connectome_optic"


def _json(name: str):
    return json.loads((OPTIC / name).read_text(encoding="utf-8"))


def _path_nodes(
    inputs: set[str], outputs: set[str], nodes: set[str], edges: list[dict]
) -> set[str]:
    forward = {node: set() for node in nodes}
    backward = {node: set() for node in nodes}
    for edge in edges:
        forward[edge["pre"]].add(edge["post"])
        backward[edge["post"]].add(edge["pre"])

    def flood(starts: set[str], graph: dict[str, set[str]]) -> set[str]:
        seen = set(starts)
        queue = deque(sorted(starts))
        while queue:
            for target in sorted(graph[queue.popleft()]):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen

    return flood(inputs, forward) & flood(outputs, backward)


def test_frozen_optic_manifest_is_real_balanced_and_parameter_matched() -> None:
    manifest = _json("manifest.json")
    audit = _json("audit.json")
    neurons = _json("neurons.json")

    assert manifest["status"] == "real"
    assert manifest["derived_counts"] == {
        "edges": 1_628,
        "excitatory_edges": 1_220,
        "inhibitory_edges": 408,
        "input_path_coverage": 32,
        "intrinsic": 200,
        "neurons": 256,
        "neurons_on_input_output_paths": 256,
        "output_path_coverage": 24,
        "projection_outputs": 24,
        "sensory_inputs": 32,
    }
    assert manifest["role_counts"] == {
        "optic_intrinsic_interneuron": 200,
        "optic_sensory_input": 32,
        "visual_projection_output": 24,
    }
    assert manifest["side_counts"] == {
        "sensory_root_side": {"L": 16, "R": 16},
        "intrinsic_soma_side": {"L": 100, "R": 100},
        "projection_soma_side": {"L": 12, "R": 12},
    }
    assert manifest["selection_settings"]["selection_audit"][
        "direct_intrinsic_candidate_count"
    ] == 342
    assert lif_actor_parameter_target(12, 4, 32, 24) == 4_776

    roles = {row["id"]: row["annotations"]["model_role"] for row in neurons}
    superclasses = {row["id"]: row["annotations"]["superclass"] for row in neurons}
    transmitters = {row["id"]: row["annotations"]["consensus_nt"] for row in neurons}
    assert {
        superclasses[body]
        for body, role in roles.items()
        if role == "optic_sensory_input"
    } == {"ol_sensory"}
    assert {
        superclasses[body]
        for body, role in roles.items()
        if role == "optic_intrinsic_interneuron"
    } == {"ol_intrinsic"}
    assert {
        superclasses[body]
        for body, role in roles.items()
        if role == "visual_projection_output"
    } == {"visual_projection"}
    assert {
        transmitters[body]
        for body, role in roles.items()
        if role == "optic_sensory_input"
    } == {"histamine"}
    assert {
        transmitters[body]
        for body, role in roles.items()
        if role == "optic_intrinsic_interneuron"
    } <= {"acetylcholine", "gaba"}
    assert set(manifest["input_neuron_ids"]) == {
        body for body, role in roles.items() if role == "optic_sensory_input"
    }
    assert set(manifest["output_neuron_ids"]) == {
        body for body, role in roles.items() if role == "visual_projection_output"
    }

    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert audit["manifest_sha256"] == sha256(
        (OPTIC / "manifest.json").read_bytes()
    ).hexdigest()
    for label, filename in (("neurons", "neurons.json"), ("edges", "edges.json")):
        assert manifest["checksums"][label] == sha256(
            (OPTIC / filename).read_bytes()
        ).hexdigest()


def test_frozen_optic_edges_are_signed_normalized_and_path_complete() -> None:
    manifest = _json("manifest.json")
    neurons = _json("neurons.json")
    edges = _json("edges.json")
    circuit = load_connectome(OPTIC / "manifest.json")

    assert circuit.num_neurons == 256
    assert circuit.edge_index.shape == (2, 1_628)
    assert torch.isfinite(circuit.weights).all()
    assert torch.count_nonzero(circuit.weights > 0).item() == 1_220
    assert torch.count_nonzero(circuit.weights < 0).item() == 408
    assert len({(edge["pre"], edge["post"]) for edge in edges}) == len(edges)

    roles = {row["id"]: row["annotations"]["model_role"] for row in neurons}
    assert all(edge["pre"] != edge["post"] for edge in edges)
    assert all(roles[edge["pre"]] != "visual_projection_output" for edge in edges)
    expected_sign = {"acetylcholine": 1, "gaba": -1, "histamine": -1}
    assert all(edge["sign"] == expected_sign[edge["transmitter"]] for edge in edges)
    assert all(math.copysign(1.0, edge["weight"]) == edge["sign"] for edge in edges)

    incoming_log_mass: dict[str, float] = {}
    for edge in edges:
        incoming_log_mass[edge["post"]] = incoming_log_mass.get(edge["post"], 0.0) + math.log1p(
            edge["raw_synapse_count"]
        )
    for edge in edges:
        expected = (
            edge["sign"]
            * 0.5
            * math.log1p(edge["raw_synapse_count"])
            / incoming_log_mass[edge["post"]]
        )
        assert edge["weight"] == pytest.approx(expected, abs=1e-15)
    absolute_incoming: dict[str, float] = {}
    for edge in edges:
        absolute_incoming[edge["post"]] = absolute_incoming.get(edge["post"], 0.0) + abs(
            edge["weight"]
        )
    assert all(value == pytest.approx(0.5, abs=1e-12) for value in absolute_incoming.values())

    node_ids = set(roles)
    input_ids = set(manifest["input_neuron_ids"])
    output_ids = set(manifest["output_neuron_ids"])
    path_nodes = _path_nodes(input_ids, output_ids, node_ids, edges)
    assert path_nodes == node_ids


def test_balanced_greedy_cover_is_deterministic_and_strict() -> None:
    candidates = [1, 2, 3, 4, 5, 6]
    coverage = {
        1: {"a"},
        2: {"a", "b"},
        3: {"b"},
        4: {"a"},
        5: {"b"},
        6: {"a", "b"},
    }
    score = {1: 1.0, 2: 2.0, 3: 1.5, 4: 4.0, 5: 3.0, 6: 0.5}
    sides = {1: "L", 2: "L", 3: "L", 4: "R", 5: "R", 6: "R"}
    first = _balanced_greedy_cover(
        candidates,
        coverage,
        score,
        sides,
        {"a", "b"},
        per_side=2,
        label="fixture",
    )
    second = _balanced_greedy_cover(
        reversed(candidates),
        coverage,
        score,
        sides,
        {"a", "b"},
        per_side=2,
        label="fixture",
    )
    assert first == second == [2, 3, 4, 5]
    assert {sides[body] for body in first} == {"L", "R"}
    assert sum(sides[body] == "L" for body in first) == 2
    assert sum(sides[body] == "R" for body in first) == 2
    assert set().union(*(coverage[body] for body in first)) == {"a", "b"}

    with pytest.raises(ValueError, match="coverage failed"):
        _balanced_greedy_cover(
            candidates,
            coverage,
            score,
            sides,
            {"missing"},
            per_side=2,
            label="fixture",
        )
