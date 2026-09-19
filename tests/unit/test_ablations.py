"""CPU checks for stable-ID ablations and reproducible control selection."""

from pathlib import Path
import sys

import pytest
import torch

from g1_fly_control.connectome import load_connectome
from g1_fly_control.evaluation.ablations import (
    ablation_mask,
    degree_match_report,
    matched_random_groups,
    resolve_target_indices,
)
from g1_fly_control.policies import LIFCore

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ablate import _paired_effect  # noqa: E402


def test_resolve_source_and_annotation_targets_rejects_invalid_selection():
    circuit = load_connectome("tests/fixtures/synthetic_circuit/manifest.json", allow_synthetic=True)
    assert resolve_target_indices(circuit, source_ids=["n2", "n0"]) == [0, 2]
    assert resolve_target_indices(circuit, annotation=("group", "synthetic_output")) == [2, 3]
    with pytest.raises(ValueError, match="exactly one"):
        resolve_target_indices(circuit, source_ids=["n0"], role="interneuron")
    with pytest.raises(ValueError, match="absent"):
        resolve_target_indices(circuit, source_ids=["unknown"])
    with pytest.raises(ValueError, match="unique"):
        resolve_target_indices(circuit, source_ids=["n0", "n0"])
    with pytest.raises(ValueError, match="no circuit neurons"):
        resolve_target_indices(circuit, role="missing")


def test_size_degree_matched_random_groups_are_distinct_and_seeded():
    edges = torch.tensor([
        [0, 0, 1, 1, 2, 3, 4, 5, 6, 7],
        [2, 3, 2, 3, 4, 5, 6, 7, 0, 1],
    ])
    target = [0, 2]
    first = matched_random_groups(edges, target, seed=19, count=4, num_neurons=8)
    second = matched_random_groups(edges, target, seed=19, count=4, num_neurons=8)
    assert first == second
    assert len({tuple(group) for group in first}) == 4
    assert all(len(group) == 2 and not (set(group) & set(target)) for group in first)
    report = degree_match_report(edges, target, first[0], num_neurons=8)
    assert report["out_degree_l1_distance"] >= 0
    assert report["in_degree_l1_distance"] >= 0
    with pytest.raises(ValueError, match="Insufficient"):
        matched_random_groups(edges, [0, 1, 2, 3, 4], seed=1, count=2, num_neurons=8)
    with pytest.raises(ValueError, match="unique"):
        matched_random_groups(edges, [0, 0], seed=1, count=2, num_neurons=8)


def test_outgoing_clamp_only_suppresses_selected_synaptic_output():
    core = LIFCore(
        3, torch.tensor([[0, 1], [2, 2]]), torch.tensor([0.5, 0.5]),
        dt=0.004, threshold=0.05, neural_substeps=1,
    )
    drive = torch.tensor([[0.3, 0.3, 0.0]])
    sham = core(drive, core.initial_state(1), ablate_outgoing=ablation_mask(3, []))
    ablated = core(drive, core.initial_state(1), ablate_outgoing=ablation_mask(3, [0]))
    assert torch.equal(sham.spikes, ablated.spikes)
    assert sham.synapse[0, 0] == 1 and sham.synapse[0, 1] == 1
    assert ablated.synapse[0, 0] == 0 and ablated.synapse[0, 1] == 1
    with pytest.raises(ValueError, match="repeat"):
        ablation_mask(3, [0, 0])
    with pytest.raises(ValueError, match="valid"):
        ablation_mask(3, [3])


def test_paired_effect_requires_identical_plan_and_initial_state():
    base = {
        "evaluation_seed": 101,
        "scenario": {"schedule": {"sha256": "fixed-schedule"}},
        "episodes": [{
            "episode_id": 0, "paired_plan_sha256": "fixed-plan", "initial_state_sha256": "same-state",
            "success": False, "accumulated_goal_relative_progress_m": 0.2,
            "mechanical_work_proxy": 5.0, "excessive_impact": 1.0,
            "joint_limit_frequency": 0.1, "saturation_frequency": 0.2,
        }],
    }
    other = {
        **base,
        "episodes": [{**base["episodes"][0], "success": True,
                      "accumulated_goal_relative_progress_m": 0.5}],
    }
    effect = _paired_effect(base, other)
    assert effect["pairing_verified"] is True
    assert effect["mean_deltas"]["success"] == 1.0
    assert effect["mean_deltas"]["accumulated_goal_relative_progress_m"] == pytest.approx(0.3)
    other["episodes"] = [{**other["episodes"][0], "initial_state_sha256": "different"}]
    with pytest.raises(ValueError, match="initial_state_sha256"):
        _paired_effect(base, other)
