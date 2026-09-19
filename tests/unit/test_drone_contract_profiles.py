"""CPU-only gates for checkpoint-derived Crazyflie task contracts."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_evaluate as evaluator  # noqa: E402
import drone_play as player  # noqa: E402
import drone_train as trainer  # noqa: E402
from drone_bootstrap import (  # noqa: E402
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_BALANCED_V4,
    CONTRACT_PROFILE_SURVIVAL_V2,
)


RESOLVERS = (
    evaluator._contract_profile_from_resolved_config,
    player._contract_profile_from_resolved_config,
)


@pytest.mark.parametrize("resolve", RESOLVERS)
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("survival_first_contract", CONTRACT_PROFILE_SURVIVAL_V2),
        ("balanced_task_contract", CONTRACT_PROFILE_BALANCED_V3),
        ("balanced_v4_task_contract", CONTRACT_PROFILE_BALANCED_V4),
    ),
)
def test_contract_profile_is_inferred_for_standalone_and_matrix_checkpoints(
    resolve,
    nested: bool,
    field: str,
    expected: str,
) -> None:
    contract = {"version": "unit-contract"}
    resolved = {"matrix": {field: contract}} if nested else {field: contract}
    assert resolve(resolved) == expected


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_matching_duplicate_contract_declarations_are_unambiguous(resolve) -> None:
    contract = {"version": "unit-contract"}
    assert resolve({
        "survival_first_contract": contract,
        "matrix": {"survival_first_contract": dict(contract)},
    }) == CONTRACT_PROFILE_SURVIVAL_V2


@pytest.mark.parametrize("resolve", RESOLVERS)
@pytest.mark.parametrize(
    "resolved",
    (
        {},
        {
            "survival_first_contract": {"version": "survival"},
            "balanced_task_contract": {"version": "balanced"},
        },
        {
            "survival_first_contract": {"version": "survival"},
            "matrix": {"balanced_task_contract": {"version": "balanced"}},
        },
        {
            "balanced_task_contract": {"version": "balanced-v3"},
            "balanced_v4_task_contract": {"version": "balanced-v4"},
        },
    ),
)
def test_contract_profile_rejects_missing_or_ambiguous_provenance(
    resolve,
    resolved: dict,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve(resolved)


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_contract_profile_rejects_malformed_or_conflicting_payloads(resolve) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        resolve({"balanced_task_contract": None})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        resolve({
            "balanced_task_contract": {"version": "one"},
            "matrix": {"balanced_task_contract": {"version": "two"}},
        })
    with pytest.raises(ValueError, match="matrix configuration"):
        resolve({"matrix": []})


def _launch_environment_contract_profile_expression(function) -> ast.expr:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "launch_environment"
    ]
    assert len(calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }
    assert "contract_profile" in keywords
    return keywords["contract_profile"]


def test_evaluation_passes_derived_profile_to_every_scenario_environment() -> None:
    expression = _launch_environment_contract_profile_expression(
        evaluator._evaluate_scenario
    )
    assert isinstance(expression, ast.Name)
    assert expression.id == "contract_profile"


def test_playback_passes_checkpoint_identity_profile_to_environment() -> None:
    expression = _launch_environment_contract_profile_expression(player.main)
    assert isinstance(expression, ast.Subscript)
    assert isinstance(expression.value, ast.Name)
    assert expression.value.id == "identity"
    assert isinstance(expression.slice, ast.Constant)
    assert expression.slice.value == "contract_profile"


def test_playback_declares_all_immutable_checkpoint_protocols() -> None:
    assert player.PLAYBACK_EVALUATION_PROTOCOLS == (
        "integration",
        "lif_proof",
        "main",
    )


@pytest.mark.parametrize("protocol_name", ("integration", "lif_proof", "main"))
def test_playback_resolves_every_immutable_checkpoint_protocol(protocol_name: str) -> None:
    from drone_evaluation_protocol import load_protocol

    expected = load_protocol(protocol_name)
    assert player._evaluation_protocol_from_manifest_id(expected["manifest_id"]) == expected


def test_playback_rejects_unknown_evaluation_manifest() -> None:
    with pytest.raises(ValueError, match="integration/lif_proof/main"):
        player._evaluation_protocol_from_manifest_id("not-an-immutable-manifest")


def test_training_records_v4_under_a_distinct_contract_field() -> None:
    from g1_fly_control.tasks.crazyflie.logic import (
        balanced_v4_switch_target_curriculum_payload,
        balanced_v4_task_contract_payload,
    )

    field, payload = trainer._task_contract_for_profile(CONTRACT_PROFILE_BALANCED_V4)
    assert field == "balanced_v4_task_contract"
    assert payload == balanced_v4_task_contract_payload()
    assert trainer._switch_contract_for_profile(CONTRACT_PROFILE_BALANCED_V4) == (
        balanced_v4_switch_target_curriculum_payload()
    )
