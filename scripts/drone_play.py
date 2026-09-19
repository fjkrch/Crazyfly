#!/usr/bin/env python3
"""Play a validated Crazyflie checkpoint in one project task.

This is a diagnostic rollout, not a held-out evaluation.  Use
``drone_evaluate.py`` for immutable-manifest result rows.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

from drone_bootstrap import (
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_BALANCED_V4,
    CONTRACT_PROFILE_SURVIVAL_V2,
    ROOT,
    canonical_sha256,
    launch_environment,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
)


TASKS = (
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
)
POLICIES = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "gru_matched",
    "mlp_normal",
)
PLAYBACK_EVALUATION_PROTOCOLS = ("integration", "lif_proof", "main")

_CONTRACT_PROFILE_FIELDS = {
    "survival_first_contract": CONTRACT_PROFILE_SURVIVAL_V2,
    "balanced_task_contract": CONTRACT_PROFILE_BALANCED_V3,
    "balanced_v4_task_contract": CONTRACT_PROFILE_BALANCED_V4,
}


def _contract_profile_from_resolved_config(resolved_config: Any) -> str:
    """Infer the one environment contract recorded by a checkpoint/config."""

    if not isinstance(resolved_config, dict):
        raise ValueError("Checkpoint lacks a resolved configuration")
    scopes = [resolved_config]
    matrix_config = resolved_config.get("matrix")
    if matrix_config is not None:
        if not isinstance(matrix_config, dict):
            raise ValueError("Checkpoint matrix configuration is not a JSON object")
        scopes.append(matrix_config)
    declarations = {
        field: [scope[field] for scope in scopes if field in scope]
        for field in _CONTRACT_PROFILE_FIELDS
    }
    present = [field for field, values in declarations.items() if values]
    if len(present) != 1:
        raise ValueError(
            "Checkpoint resolved configuration must contain exactly one of "
            "survival_first_contract, balanced_task_contract, or "
            "balanced_v4_task_contract"
        )
    field = present[0]
    values = declarations[field]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Checkpoint {field} must be a JSON object")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Checkpoint has conflicting duplicate {field} payloads")
    return _CONTRACT_PROFILE_FIELDS[field]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _checkpoint_controller(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    resolved = payload.get("resolved_config", {})
    for candidate in (metadata.get("controller"), resolved.get("controller")):
        if candidate in POLICIES:
            return str(candidate)
    canonical = metadata.get("controller_report", {}).get("controller_kind")
    reverse = {
        "frozen_lif": "frozen_lif_original",
        "frozen_lif_rewired": "frozen_lif_degree_rewired",
        "gru": "gru_matched",
        "mlp": "mlp_normal",
    }
    if canonical in reverse:
        return reverse[canonical]
    raise ValueError("Checkpoint does not identify one of the four declared controllers")


def _evaluation_protocol_from_manifest_id(manifest_id: Any) -> dict[str, Any]:
    """Resolve one immutable evaluation protocol recorded by a checkpoint."""

    from drone_evaluation_protocol import load_protocol

    for protocol_name in PLAYBACK_EVALUATION_PROTOCOLS:
        candidate = load_protocol(protocol_name)
        if candidate["manifest_id"] == manifest_id:
            return candidate
    raise ValueError(
        "Checkpoint evaluation manifest is not one of the immutable "
        "integration/lif_proof/main protocols"
    )


def inspect_checkpoint(
    checkpoint: Path,
    expected_fingerprint: str | None = None,
    *,
    allowed_diagnostic_source_drift: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate checkpoint structure and return the immutable playback identity."""

    from g1_fly_control.crazyflie.checkpoint import checkpoint_sha256, read_checkpoint

    payload = read_checkpoint(checkpoint, map_location="cpu")
    controller = _checkpoint_controller(payload)
    fingerprint = payload.get("fingerprints", {}).get("reproduction")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Checkpoint has no reproduction fingerprint")
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise ValueError(
            f"Expected reproduction fingerprint {expected_fingerprint}, checkpoint contains {fingerprint}"
        )
    controller_report = payload.get("metadata", {}).get("controller_report", {})
    connectome_manifest = controller_report.get("connectome_manifest")
    if not connectome_manifest:
        raise ValueError("Checkpoint metadata does not identify the connectome manifest")
    evaluation_manifest_id = payload.get("evaluation_manifest_id")
    evaluation_manifest = _evaluation_protocol_from_manifest_id(evaluation_manifest_id)
    resolved_config = payload["resolved_config"]
    contract_profile = _contract_profile_from_resolved_config(resolved_config)
    matrix_value = resolved_config.get("matrix")
    if matrix_value is not None and not isinstance(matrix_value, dict):
        raise ValueError("Checkpoint matrix configuration is not a JSON object")
    matrix_config = matrix_value or {}
    rewire_setting = matrix_config.get("rewire_manifest") or resolved_config.get(
        "rewire_manifest"
    )
    if not isinstance(rewire_setting, str) or not rewire_setting:
        raise ValueError("Checkpoint resolved config does not identify the frozen rewire manifest")
    rewire_path = Path(rewire_setting).expanduser()
    if not rewire_path.is_absolute():
        rewire_path = (ROOT / rewire_path).resolve()
    declared_rewire_sha256 = matrix_config.get("rewire_manifest_sha256") or resolved_config.get(
        "rewire_manifest_file_sha256"
    )
    rewire_seed = int(
        matrix_config.get("rewire_seed", resolved_config.get("rewire_seed", 20260916))
    )
    rewire_manifest = load_fingerprint_rewire_manifest(
        rewire_path,
        expected_file_sha256=declared_rewire_sha256,
        expected_seed=rewire_seed,
    )
    current_fingerprint, current_fingerprint_payload = reproduction_fingerprint(
        resolved_config=resolved_config,
        evaluation_manifest=evaluation_manifest,
        connectome_manifest=connectome_manifest,
        rewired_manifest=rewire_manifest,
    )
    diagnostic_source_drift: list[str] = []
    identity_fingerprint_payload = current_fingerprint_payload
    if current_fingerprint != fingerprint:
        allowed = set(allowed_diagnostic_source_drift)
        if not allowed:
            raise ValueError(
                "Checkpoint reproduction fingerprint does not match current source/runtime/config; "
                f"checkpoint={fingerprint}, current={current_fingerprint}"
            )
        manifest_path = checkpoint.parent.parent / "training_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Cannot audit diagnostic-only source drift without the checkpoint training manifest"
            ) from exc
        recorded_payload = manifest.get("fingerprint_payload")
        if (
            manifest.get("fingerprint") != fingerprint
            or not isinstance(recorded_payload, dict)
            or canonical_sha256(recorded_payload) != fingerprint
        ):
            raise ValueError("Training manifest does not authenticate the checkpoint fingerprint payload")
        recorded_sources = recorded_payload.get("source_sha256")
        current_sources = current_fingerprint_payload.get("source_sha256")
        if not isinstance(recorded_sources, dict) or not isinstance(current_sources, dict):
            raise ValueError("Fingerprint payload lacks auditable source hashes")
        differing = {
            path
            for path in set(recorded_sources) | set(current_sources)
            if recorded_sources.get(path) != current_sources.get(path)
        }
        if not differing or not differing.issubset(allowed):
            raise ValueError(
                "Source drift is not limited to the explicitly allowed diagnostic files: "
                f"observed={sorted(differing)}, allowed={sorted(allowed)}"
            )
        recorded_without_diagnostics = deepcopy(recorded_payload)
        current_without_diagnostics = deepcopy(current_fingerprint_payload)
        for payload_value in (recorded_without_diagnostics, current_without_diagnostics):
            for path in allowed:
                payload_value["source_sha256"].pop(path, None)
        if recorded_without_diagnostics != current_without_diagnostics:
            raise ValueError(
                "Checkpoint drift includes runtime, config, contract, or non-diagnostic source changes"
            )
        diagnostic_source_drift = sorted(differing)
        identity_fingerprint_payload = recorded_payload
    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "controller": controller,
        "training_seed": payload.get("metadata", {}).get(
            "seed", payload.get("resolved_config", {}).get("seed")
        ),
        "fingerprint": fingerprint,
        "evaluation_manifest_id": evaluation_manifest_id,
        "connectome_manifest": str(connectome_manifest),
        "rewire_seed": rewire_seed,
        "rewire_manifest": str(rewire_path),
        "rewire_manifest_sha256": sha256_file(rewire_path),
        "source_set_fingerprint": canonical_sha256(
            identity_fingerprint_payload["source_sha256"]
        ),
        "rewire_manifest_fingerprint": current_fingerprint_payload[
            "rewired_manifest_sha256"
        ],
        "diagnostic_source_drift": diagnostic_source_drift,
        "contract_profile": contract_profile,
    }
    return payload, identity


def load_policy_and_normalizer(
    checkpoint: Path,
    payload: dict[str, Any],
    identity: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Construct and strictly load the checkpoint policy and frozen normalizer."""

    from g1_fly_control.crazyflie.checkpoint import load_checkpoint
    from g1_fly_control.crazyflie.controllers import build_controller
    from g1_fly_control.crazyflie.normalization import RunningMeanVariance

    policy, controller_report = build_controller(
        identity["controller"],
        observation_dim=12,
        action_dim=4,
        device=device,
        connectome_manifest=identity["connectome_manifest"],
        rewire_seed=identity["rewire_seed"],
        rewire_manifest_path=(
            identity["rewire_manifest"]
            if identity["controller"] == "frozen_lif_degree_rewired"
            else None
        ),
    )
    loaded = load_checkpoint(
        checkpoint,
        policy=policy,
        map_location=device,
        expected_fingerprints={
            "reproduction": identity["fingerprint"],
            "source_set": identity["source_set_fingerprint"],
            "connectome": controller_report["connectome_checksum"],
            "frozen_core": controller_report["core_checksum"],
            "rewire_manifest": identity["rewire_manifest_fingerprint"],
        },
        expected_evaluation_manifest_id=payload.get("evaluation_manifest_id"),
        materialize_external_history=False,
    )
    if loaded["tainted"]:
        raise RuntimeError("Playback refuses a tainted checkpoint")
    normalizer = RunningMeanVariance.create(12, device=device)
    normalizer.load_state_dict(loaded["normalizers"]["observation"])
    policy.eval()
    return policy, normalizer, controller_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # AppLauncher probes the parser before adding its own options, so required
    # arguments are enforced after the final parse to keep --help functional.
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the saved action distribution; the default is deterministic",
    )
    parser.add_argument("--expected_fingerprint")
    parser.add_argument("--output", type=Path, help="Optional JSON diagnostic report")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if not 1 <= args.num_envs <= 4:
        parser.error("--num_envs must be between 1 and the validated gate maximum of 4")
    if not 1 <= args.steps <= 100_000:
        parser.error("--steps must be between 1 and 100000")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    output = args.output.expanduser().resolve() if args.output else None
    if output is not None and output.exists():
        parser.error(f"Refusing to overwrite existing playback report: {output}")
    try:
        payload, identity = inspect_checkpoint(checkpoint, args.expected_fingerprint)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    resolved = {
        "kind": "diagnostic_playback",
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "deterministic_actions": not args.stochastic,
        **identity,
    }
    print(json.dumps({"resolved_config": resolved, "fingerprint": identity["fingerprint"]}, indent=2))

    app = AppLauncher(args).app
    env = None
    wrapped = None
    try:
        from g1_fly_control.crazyflie.controllers import reset_controller_state
        from g1_fly_control.crazyflie.normalization import NormalizedEnv

        env = launch_environment(
            args.task,
            args.num_envs,
            deterministic_evaluation=False,
            contract_profile=identity["contract_profile"],
        )
        device = torch.device(env.device)
        policy, normalizer, controller_report = load_policy_and_normalizer(
            checkpoint, payload, identity, device
        )
        wrapped = NormalizedEnv(env, normalizer, training=False)
        observation, _ = wrapped.reset(seed=args.seed)
        policy_observation = observation["policy"]
        state = (
            policy.initial_state(args.num_envs, device=device)
            if hasattr(policy, "initial_state")
            else None
        )
        terminated_count = 0
        truncated_count = 0
        completed_episode_count = 0
        reward_sum = 0.0
        maximum_action_abs = 0.0
        maximum_observation_abs = float(policy_observation.abs().max())
        success_count = 0
        executed_steps = 0
        with torch.no_grad():
            for _ in range(args.steps):
                output_policy = policy.act(
                    policy_observation, state, deterministic=not args.stochastic
                )
                if output_policy.action.shape != (args.num_envs, 4):
                    raise RuntimeError(
                        f"Controller returned action shape {tuple(output_policy.action.shape)}, "
                        f"expected {(args.num_envs, 4)}"
                    )
                if not torch.isfinite(output_policy.action).all():
                    raise FloatingPointError("Controller emitted a nonfinite action")
                next_observation, reward, terminated, truncated, _ = wrapped.step(
                    output_policy.action
                )
                if not torch.isfinite(next_observation["policy"]).all() or not torch.isfinite(
                    reward
                ).all():
                    raise FloatingPointError("Playback encountered a nonfinite observation or reward")
                done = terminated | truncated
                state = reset_controller_state(policy, output_policy.state, done)
                policy_observation = next_observation["policy"]
                executed_steps += 1
                reward_sum += float(reward.sum())
                terminated_count += int(terminated.sum())
                truncated_count += int(truncated.sum())
                completed_episode_count += int(done.sum())
                maximum_action_abs = max(
                    maximum_action_abs, float(output_policy.action.abs().max())
                )
                maximum_observation_abs = max(
                    maximum_observation_abs, float(policy_observation.abs().max())
                )
                if bool(done.any()):
                    success_count += int(env.terminal_success_count[done].sum())

        report = {
            "schema_version": 1,
            "status": "completed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scientific_use": "diagnostic_only_not_heldout_evaluation",
            "resolved_config": resolved,
            "controller_report": controller_report,
            "executed_control_steps": executed_steps,
            "environment_interactions": executed_steps * args.num_envs,
            "completed_episode_count": completed_episode_count,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "success_event_count_at_completed_episodes": success_count,
            "reward_sum_all_environments": reward_sum,
            "maximum_action_abs": maximum_action_abs,
            "maximum_normalized_observation_abs": maximum_observation_abs,
        }
        if not math.isfinite(reward_sum) or maximum_action_abs > 1.000001:
            raise RuntimeError("Playback finite/bounded-action invariant failed")
        if output is not None:
            _atomic_json(output, report)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scientific_use": report["scientific_use"],
                    "checkpoint": str(checkpoint),
                    "controller": identity["controller"],
                    "task": args.task,
                    "executed_control_steps": executed_steps,
                    "environment_interactions": executed_steps * args.num_envs,
                    "completed_episode_count": completed_episode_count,
                    "terminated_count": terminated_count,
                    "truncated_count": truncated_count,
                    "output": str(output) if output is not None else None,
                    "fingerprint": identity["fingerprint"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()
        # Preserve the command's process status after the environment closes.


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
