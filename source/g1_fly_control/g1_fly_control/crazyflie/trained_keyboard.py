"""CPU-safe contracts for checkpoint-driven Crazyflie keyboard flight.

The interactive Isaac entry point lives in ``scripts/crazyflie_trained_keyboard.py``.
This module keeps checkpoint identity, command scaling, action arbitration, and
activity extraction independently unit-testable.  A valid trained action is
returned byte-for-byte; the deterministic assist is selected only when an
explicit safety invariant fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


TASK_ID = "FlyCrazyflie-CommandFollow-v0"
WIDE_TASK_ID = "FlyCrazyflie-CommandFollowWide-v0"
WIDE_WIND_TASK_ID = "FlyCrazyflie-CommandFollowWideWind-v0"
TASK_IDS = (TASK_ID, WIDE_TASK_ID, WIDE_WIND_TASK_ID)
COMMAND_FOLLOW_CONTRACT_VERSION = "crazyflie_command_follow_v1"
OBSERVATION_CONTRACT_VERSION = "command_error_12_v1"
OBSERVATION_DIM = 12
ACTION_DIM = 4
MAXIMUM_HORIZONTAL_SPEED_M_S = 0.8
MAXIMUM_VERTICAL_SPEED_M_S = 0.4
MAXIMUM_YAW_RATE_RAD_S = 1.2

PUBLIC_CONTROLLER_KINDS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "wing_lif",
    "leg_wing_lif",
    "optic_lif",
    "gru_matched",
    "mlp_normal",
)
COMBINATION_CONTROLLER_KINDS = (
    "leg_optic_lif",
    "wing_optic_lif",
    "leg_wing_optic_lif",
)
SUPPORTED_CONTROLLER_KINDS = PUBLIC_CONTROLLER_KINDS + COMBINATION_CONTROLLER_KINDS
LIF_CONTROLLER_CORE_LABELS: dict[str, tuple[str, ...]] = {
    "frozen_lif_original": ("leg",),
    "frozen_lif_degree_rewired": ("leg",),
    "wing_lif": ("wing",),
    "leg_wing_lif": ("leg", "wing"),
    "optic_lif": ("optic",),
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
}
_CANONICAL_CONTROLLER_KIND = {
    "frozen_lif_original": "frozen_lif",
    "frozen_lif_degree_rewired": "frozen_lif_rewired",
    "wing_lif": "wing_lif",
    "leg_wing_lif": "leg_wing_lif",
    "optic_lif": "optic_lif",
    "gru_matched": "gru",
    "mlp_normal": "mlp",
    "leg_optic_lif": "leg_optic_lif",
    "wing_optic_lif": "wing_optic_lif",
    "leg_wing_optic_lif": "leg_wing_optic_lif",
}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeSpeeds:
    """Launch-time velocity magnitudes applied to normalized held keys."""

    horizontal_m_s: float = 0.5
    vertical_m_s: float = 0.25
    yaw_rate_rad_s: float = 0.8

    def validate(self, contract: Mapping[str, Any]) -> None:
        limits = (
            (
                "horizontal_speed",
                self.horizontal_m_s,
                contract.get("maximum_horizontal_speed_m_s"),
            ),
            (
                "vertical_speed",
                self.vertical_m_s,
                contract.get("maximum_vertical_speed_m_s"),
            ),
            (
                "yaw_rate",
                self.yaw_rate_rad_s,
                contract.get("maximum_yaw_rate_rad_s"),
            ),
        )
        for name, value, maximum in limits:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            if not isinstance(maximum, (int, float)) or isinstance(maximum, bool):
                raise ValueError(f"checkpoint command contract lacks a numeric {name} limit")
            if not math.isfinite(float(maximum)) or float(maximum) <= 0.0:
                raise ValueError(f"checkpoint command contract has an invalid {name} limit")
            if float(value) > float(maximum) + 1.0e-12:
                raise ValueError(
                    f"{name}={float(value):g} exceeds checkpoint-trained maximum "
                    f"{float(maximum):g}"
                )


def scaled_command_body(
    normalized_command: Sequence[float], speeds: RuntimeSpeeds
) -> tuple[float, float, float, float]:
    """Scale normalized ``forward,left,up,yaw-left`` keyboard coordinates."""

    if len(normalized_command) != ACTION_DIM:
        raise ValueError("normalized keyboard command must contain four values")
    values = tuple(float(value) for value in normalized_command)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("normalized keyboard command must be finite")
    if any(abs(value) > 1.0 + 1.0e-12 for value in values):
        raise ValueError("normalized keyboard command must remain inside [-1, 1]")
    horizontal_norm = math.hypot(values[0], values[1])
    if horizontal_norm > 1.0 + 1.0e-12:
        raise ValueError("normalized horizontal keyboard command exceeds unit length")
    return (
        values[0] * speeds.horizontal_m_s,
        values[1] * speeds.horizontal_m_s,
        values[2] * speeds.vertical_m_s,
        values[3] * speeds.yaw_rate_rad_s,
    )


def command_conditioned_observation(
    physical_observation: torch.Tensor,
    effective_command_body: torch.Tensor,
    target_error_body: torch.Tensor,
) -> torch.Tensor:
    """Construct the exact 12-value command-follow observation.

    ``effective_command_body`` is ordered forward, left, up, yaw-left.  The
    environment owns target integration; callers supply its current body-frame
    target error instead of integrating it here.
    """

    if physical_observation.ndim != 2 or physical_observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("physical observation must have shape [batch, 12]")
    batch = physical_observation.shape[0]
    if effective_command_body.shape != (batch, ACTION_DIM):
        raise ValueError("effective command must have shape [batch, 4]")
    if target_error_body.shape != (batch, 3):
        raise ValueError("target error must have shape [batch, 3]")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (physical_observation, effective_command_body, target_error_body)
    ):
        raise FloatingPointError("command observation inputs must be finite")
    result = physical_observation.clone()
    result[:, 0:3] -= effective_command_body[:, 0:3]
    result[:, 5] -= effective_command_body[:, 3]
    result[:, 9:12] = target_error_body
    return result


def reconstruct_physical_observation(
    command_observation: torch.Tensor, effective_command_body: torch.Tensor
) -> torch.Tensor:
    """Recover physical velocity/rate fields for the deterministic fallback."""

    if command_observation.ndim != 2 or command_observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("command observation must have shape [batch, 12]")
    if effective_command_body.shape != (command_observation.shape[0], ACTION_DIM):
        raise ValueError("effective command must have shape [batch, 4]")
    result = command_observation.clone()
    result[:, 0:3] += effective_command_body[:, 0:3]
    result[:, 5] += effective_command_body[:, 3]
    return result


@dataclass(frozen=True)
class SafetyEnvelope:
    minimum_height_m: float = 0.30
    maximum_height_m: float = 1.70
    maximum_horizontal_offset_m: float = 4.0
    minimum_upright_gravity_z: float = -0.35
    maximum_linear_speed_m_s: float = 2.5
    maximum_angular_speed_rad_s: float = 12.0

    def validate(self) -> None:
        values = tuple(vars(self).values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("safety envelope must be finite")
        if self.minimum_height_m < 0.0 or self.minimum_height_m >= self.maximum_height_m:
            raise ValueError("invalid safety height interval")
        if any(
            value <= 0.0
            for value in (
                self.maximum_horizontal_offset_m,
                self.maximum_linear_speed_m_s,
                self.maximum_angular_speed_rad_s,
            )
        ):
            raise ValueError("safety magnitudes must be positive")
        if not -1.0 <= self.minimum_upright_gravity_z < 0.0:
            raise ValueError("upright gravity threshold must lie in [-1, 0)")


def safety_reason(
    physical_observation: torch.Tensor,
    root_position_w: torch.Tensor,
    origin_w: torch.Tensor,
    envelope: SafetyEnvelope,
) -> str | None:
    """Return the first deterministic fallback reason, or ``None``."""

    envelope.validate()
    if physical_observation.shape != (1, OBSERVATION_DIM):
        return "invalid_observation_shape"
    if root_position_w.shape != (1, 3) or origin_w.shape != (1, 3):
        return "invalid_position_shape"
    if not all(
        bool(torch.isfinite(value).all())
        for value in (physical_observation, root_position_w, origin_w)
    ):
        return "nonfinite_runtime_state"
    height = float(root_position_w[0, 2])
    if height <= envelope.minimum_height_m + 1.0e-6:
        return "below_minimum_height"
    if height >= envelope.maximum_height_m - 1.0e-6:
        return "above_maximum_height"
    horizontal = torch.linalg.vector_norm(root_position_w[:, :2] - origin_w[:, :2])
    if float(horizontal) >= envelope.maximum_horizontal_offset_m - 1.0e-6:
        return "horizontal_workspace_limit"
    if float(physical_observation[0, 8]) >= envelope.minimum_upright_gravity_z:
        return "excessive_tilt"
    if float(torch.linalg.vector_norm(physical_observation[0, 0:3])) >= envelope.maximum_linear_speed_m_s:
        return "linear_overspeed"
    if float(torch.linalg.vector_norm(physical_observation[0, 3:6])) >= envelope.maximum_angular_speed_rad_s:
        return "angular_overspeed"
    return None


def policy_state_is_finite(state: Any) -> bool:
    """Validate recurrent state without importing a specific policy class."""

    if state is None:
        return True
    if isinstance(state, torch.Tensor):
        return state.numel() > 0 and bool(torch.isfinite(state).all())
    names = ("membrane", "spikes", "synapse", "refractory")
    if all(hasattr(state, name) for name in names):
        tensors = tuple(getattr(state, name) for name in names)
        return all(
            isinstance(value, torch.Tensor)
            and value.numel() > 0
            and bool(torch.isfinite(value).all())
            for value in tensors
        )
    return False


@dataclass(frozen=True)
class ActionDecision:
    action: torch.Tensor
    source: str
    fallback_reason: str | None


def arbitrate_action(
    policy_action: torch.Tensor | None,
    policy_state: Any,
    fallback_action: torch.Tensor,
    *,
    runtime_safety_reason: str | None = None,
    policy_error: str | None = None,
) -> ActionDecision:
    """Choose policy output unchanged or a deterministic explicit fallback."""

    if fallback_action.shape != (1, ACTION_DIM):
        raise ValueError("fallback action must have shape [1, 4]")
    if not bool(torch.isfinite(fallback_action).all()) or float(fallback_action.abs().max()) > 1.0:
        raise FloatingPointError("deterministic fallback action is invalid")
    reason = runtime_safety_reason
    if reason is None and policy_error is not None:
        reason = f"policy_exception:{policy_error}"
    if reason is None and not isinstance(policy_action, torch.Tensor):
        reason = "missing_policy_action"
    if reason is None and policy_action.shape != (1, ACTION_DIM):
        reason = "invalid_policy_action_shape"
    if reason is None and not bool(torch.isfinite(policy_action).all()):
        reason = "nonfinite_policy_action"
    if reason is None and float(policy_action.abs().max()) > 1.0 + 1.0e-6:
        reason = "out_of_bounds_policy_action"
    if reason is None and not policy_state_is_finite(policy_state):
        reason = "nonfinite_policy_state"
    if reason is not None:
        return ActionDecision(fallback_action, "deterministic_flight_assist", reason)
    # Preserve exact tensor identity/value: no clamp, blend, cast, or copy.
    return ActionDecision(policy_action, "trained_policy", None)


@dataclass(frozen=True)
class CommandCheckpointSpec:
    path: Path
    sha256: str
    controller: str
    canonical_controller: str
    resolved_config: dict[str, Any]
    controller_report: dict[str, Any]
    command_follow_contract: dict[str, Any]
    fingerprints: dict[str, Any]
    task_manifest_id: str
    evaluation_manifest_id: str
    observation_normalizer_state: dict[str, Any]


def _expected_command_contract(task: str = TASK_ID) -> dict[str, Any]:
    """Import the task-owned contract so training and playback cannot drift."""

    try:
        if task == TASK_ID:
            from g1_fly_control.tasks.crazyflie.command_logic import (
                command_follow_contract_payload,
            )

            value = command_follow_contract_payload()
        elif task in {WIDE_TASK_ID, WIDE_WIND_TASK_ID}:
            from g1_fly_control.tasks.crazyflie.command_wide_logic import (
                command_wide_contract_payload,
            )

            value = command_wide_contract_payload()
        else:
            raise ValueError(f"unsupported command task {task!r}")
    except ImportError as exc:
        raise RuntimeError("command-follow task contract is unavailable") from exc
    if not isinstance(value, dict):
        raise TypeError("command-follow task contract must be a dictionary")
    return value


def inspect_command_checkpoint(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> CommandCheckpointSpec:
    """Authenticate command-follow identity before Isaac is launched."""

    from g1_fly_control.crazyflie.checkpoint import read_checkpoint

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    payload = read_checkpoint(source, map_location="cpu", resolve_external_history=False)
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError("checkpoint lacks a resolved configuration")
    task = resolved.get("task")
    if task not in TASK_IDS:
        raise ValueError(
            f"checkpoint task must be one of {TASK_IDS}, got {task!r}"
        )
    controller = resolved.get("controller")
    if controller not in SUPPORTED_CONTROLLER_KINDS:
        raise ValueError(
            "checkpoint controller must be one of "
            + ", ".join(SUPPORTED_CONTROLLER_KINDS)
        )
    metadata = payload.get("metadata")
    report = metadata.get("controller_report") if isinstance(metadata, Mapping) else None
    if not isinstance(report, dict):
        raise ValueError("checkpoint lacks metadata.controller_report")
    canonical = _CANONICAL_CONTROLLER_KIND[controller]
    if report.get("controller_kind") != canonical:
        raise ValueError(
            "checkpoint controller report disagrees with resolved controller: "
            f"{report.get('controller_kind')!r} != {canonical!r}"
        )
    if report.get("observation_dim") != OBSERVATION_DIM or report.get("action_dim") != ACTION_DIM:
        raise ValueError("checkpoint controller dimensions are not 12 observations / 4 actions")
    contract = resolved.get("command_follow_contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint lacks resolved command_follow_contract")
    expected = (
        dict(expected_contract)
        if expected_contract is not None
        else _expected_command_contract(str(task))
    )
    if contract != expected:
        raise ValueError("checkpoint command-follow contract differs from current task contract")
    if contract.get("version") != expected.get("version"):
        raise ValueError("checkpoint command-follow contract version is incompatible")
    if contract.get("observation_contract_version") != expected.get(
        "observation_contract_version"
    ):
        raise ValueError("checkpoint command observation contract is incompatible")
    normalizers = payload.get("normalizers")
    observation_normalizer = normalizers.get("observation") if isinstance(normalizers, Mapping) else None
    if not isinstance(observation_normalizer, dict):
        raise ValueError("checkpoint lacks its observation normalizer")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("checkpoint lacks fingerprints")
    task_manifest_id = payload.get("task_manifest_id")
    evaluation_manifest_id = payload.get("evaluation_manifest_id")
    if not isinstance(task_manifest_id, str) or not task_manifest_id:
        raise ValueError("checkpoint lacks task manifest identity")
    if not isinstance(evaluation_manifest_id, str) or not evaluation_manifest_id:
        raise ValueError("checkpoint lacks evaluation manifest identity")
    return CommandCheckpointSpec(
        path=source,
        sha256=sha256_file(source),
        controller=controller,
        canonical_controller=canonical,
        resolved_config=dict(resolved),
        controller_report=dict(report),
        command_follow_contract=dict(contract),
        fingerprints=dict(fingerprints),
        task_manifest_id=task_manifest_id,
        evaluation_manifest_id=evaluation_manifest_id,
        observation_normalizer_state=dict(observation_normalizer),
    )


def _controller_build_arguments(spec: CommandCheckpointSpec) -> dict[str, Any]:
    report = spec.controller_report
    raw_widths = report.get("widths", {})
    if not isinstance(raw_widths, Mapping):
        raise ValueError("checkpoint controller widths are malformed")
    widths = {str(key): value for key, value in raw_widths.items() if value is not None}
    manifests = report.get("connectome_manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("checkpoint lacks controller connectome manifests")
    arguments: dict[str, Any] = {"widths": widths}
    core_labels = LIF_CONTROLLER_CORE_LABELS.get(spec.controller)
    if core_labels is not None and len(core_labels) > 1:
        if spec.controller in COMBINATION_CONTROLLER_KINDS:
            reported_labels = report.get("core_labels")
            if not isinstance(reported_labels, (list, tuple)) or tuple(
                reported_labels
            ) != core_labels:
                raise ValueError(
                    "combination checkpoint core ordering differs from its controller"
                )
        missing = [
            label for label in core_labels if not isinstance(manifests.get(label), str)
        ]
        if missing:
            raise ValueError(
                "combined checkpoint lacks connectome paths for " + ", ".join(missing)
            )
        if "leg" in core_labels:
            arguments["connectome_manifest"] = manifests["leg"]
        if "wing" in core_labels:
            arguments["wing_connectome_manifest"] = manifests["wing"]
        if "optic" in core_labels:
            arguments["optic_connectome_manifest"] = manifests["optic"]
    elif spec.controller == "wing_lif":
        primary = manifests.get("primary")
        if not isinstance(primary, str):
            raise ValueError("wing checkpoint lacks its connectome path")
        arguments["wing_connectome_manifest"] = primary
    elif spec.controller == "optic_lif":
        primary = manifests.get("primary")
        if not isinstance(primary, str):
            raise ValueError("optic checkpoint lacks its connectome path")
        arguments["optic_connectome_manifest"] = primary
    else:
        primary = manifests.get("primary")
        if not isinstance(primary, str):
            raise ValueError("checkpoint lacks its primary connectome path")
        arguments["connectome_manifest"] = primary
    if spec.controller == "frozen_lif_degree_rewired":
        path = report.get("rewire_manifest_path")
        seed = report.get("rewire_seed")
        if not isinstance(path, str) or type(seed) is not int:
            raise ValueError("rewired checkpoint lacks exact rewire provenance")
        arguments["rewire_manifest_path"] = path
        arguments["rewire_seed"] = seed
    return arguments


def load_command_controller(
    spec: CommandCheckpointSpec,
    *,
    device: str | torch.device,
    expected_source_set: str | None = None,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Rebuild and strictly load one supported command-follow controller."""

    from g1_fly_control.crazyflie.checkpoint import load_checkpoint
    from g1_fly_control.crazyflie.controllers import build_controller
    from g1_fly_control.crazyflie.normalization import RunningMeanVariance

    if expected_source_set is not None:
        actual = spec.fingerprints.get("source_set")
        if actual != expected_source_set:
            raise ValueError(
                "checkpoint source-set fingerprint differs from current runtime: "
                f"checkpoint={actual!r}, current={expected_source_set!r}"
            )
    policy, report = build_controller(
        spec.controller,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        device=device,
        **_controller_build_arguments(spec),
    )
    loaded = load_checkpoint(
        spec.path,
        policy=policy,
        map_location=device,
        expected_fingerprints=spec.fingerprints,
        expected_config=spec.resolved_config,
        expected_task_manifest_id=spec.task_manifest_id,
        expected_evaluation_manifest_id=spec.evaluation_manifest_id,
        materialize_external_history=False,
    )
    if loaded.get("tainted"):
        raise RuntimeError(f"command controller checkpoint is tainted: {loaded.get('taint_reasons')}")
    normalizer = RunningMeanVariance.create(OBSERVATION_DIM, device=device)
    normalizer.load_state_dict(spec.observation_normalizer_state)
    policy.eval()
    for name, value in policy.state_dict().items():
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"checkpoint policy tensor {name!r} is nonfinite")
    return policy, normalizer, report


def initial_policy_state(
    policy: torch.nn.Module, batch_size: int, device: str | torch.device
) -> Any:
    initializer = getattr(policy, "initial_state", None)
    return initializer(batch_size, device=device) if callable(initializer) else None


def reset_policy_state(policy: torch.nn.Module, state: Any, *, device: str | torch.device) -> Any:
    """Reset the one-row interactive recurrent state after any discontinuity."""

    del state
    return initial_policy_state(policy, 1, device)


class TrainedActivityRecorder:
    """Observe the state/hidden layers used for the actual flight action."""

    def __init__(self, policy: torch.nn.Module, spec: CommandCheckpointSpec) -> None:
        self.policy = policy
        self.spec = spec
        self.role_provenance: dict[str, Any] = {}
        self.unit_ids: tuple[str, ...] = ()
        self.roles = self._roles()
        self.last_activity = torch.zeros(len(self.roles), dtype=torch.float32)
        self.last_summary: dict[str, Any] = {}
        self._mlp_values: list[torch.Tensor] = []
        self._hooks: list[Any] = []
        if spec.controller == "mlp_normal":
            actor = getattr(policy, "actor", None)
            if not isinstance(actor, torch.nn.Sequential):
                raise TypeError("matched MLP actor must be a Sequential module")
            activation_types = (
                torch.nn.ELU,
                torch.nn.GELU,
                torch.nn.LeakyReLU,
                torch.nn.ReLU,
                torch.nn.SiLU,
                torch.nn.Tanh,
            )
            activations = [
                module for module in actor if isinstance(module, activation_types)
            ]
            expected_layers = len(spec.controller_report["widths"]["mlp_hidden_dims"])
            if len(activations) != expected_layers:
                raise ValueError(
                    "MLP actor activation layout differs from its checkpoint widths"
                )
            for activation in activations:
                self._hooks.append(activation.register_forward_hook(self._capture_mlp))

    def _manifest_roles(self, path: str, prefix: str) -> tuple[str, ...]:
        manifest_path = Path(path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError(f"connectome manifest is invalid: {manifest_path}")
        neurons_path = manifest.get("neurons_path")
        if not isinstance(neurons_path, str):
            raise ValueError(f"connectome manifest lacks neurons_path: {manifest_path}")
        resolved_neurons = (manifest_path.parent / neurons_path).resolve()
        expected_neurons_sha = manifest.get("checksums", {}).get("neurons")
        actual_neurons_sha = sha256_file(resolved_neurons)
        if not isinstance(expected_neurons_sha, str) or expected_neurons_sha != actual_neurons_sha:
            raise ValueError(f"connectome neuron checksum mismatch: {resolved_neurons}")
        rows = json.loads(resolved_neurons.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"neuron role file is empty: {resolved_neurons}")
        result = []
        unit_ids = []
        for row in rows:
            role = row.get("annotations", {}).get("model_role") if isinstance(row, dict) else None
            unit_id = str(row.get("id", "")) if isinstance(row, dict) else ""
            if not isinstance(role, str) or not role or not unit_id:
                raise ValueError("connectome neuron lacks a stable ID or model role")
            result.append(prefix + role)
            unit_ids.append(prefix + unit_id)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("connectome stable neuron IDs are duplicated")
        expected_count = manifest.get("derived_counts", {}).get("neurons")
        if expected_count != len(rows):
            raise ValueError("connectome manifest neuron count differs from neurons.json")
        provenance_key = prefix.removesuffix(":") or "primary"
        self.role_provenance[provenance_key] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "neurons": str(resolved_neurons),
            "neurons_sha256": actual_neurons_sha,
            "neuron_count": len(rows),
        }
        self.unit_ids = (*self.unit_ids, *unit_ids)
        return tuple(result)

    def _roles(self) -> tuple[str, ...]:
        manifests = self.spec.controller_report.get("connectome_manifests", {})
        if not isinstance(manifests, Mapping):
            raise ValueError("checkpoint connectome manifests are malformed")
        core_labels = LIF_CONTROLLER_CORE_LABELS.get(self.spec.controller)
        if core_labels is not None:
            if self.spec.controller in COMBINATION_CONTROLLER_KINDS:
                reported_labels = self.spec.controller_report.get("core_labels")
                if not isinstance(reported_labels, (list, tuple)) or tuple(
                    reported_labels
                ) != core_labels:
                    raise ValueError(
                        "combination activity core ordering differs from its controller"
                    )
            roles: tuple[str, ...] = ()
            for label in core_labels:
                manifest_key = label if len(core_labels) > 1 else "primary"
                manifest_path = manifests.get(manifest_key)
                if not isinstance(manifest_path, str):
                    raise ValueError(
                        f"checkpoint lacks the {label} activity connectome path"
                    )
                roles += self._manifest_roles(manifest_path, f"{label}:")
            return roles
        if self.spec.controller == "gru_matched":
            width = int(self.spec.controller_report["widths"]["gru_hidden_dim"])
            self.unit_ids = tuple(f"gru:hidden:{index}" for index in range(width))
            self.role_provenance = {
                "engineering": {
                    "kind": "matched_gru_hidden_state",
                    "biological_roles": False,
                    "unit_count": width,
                }
            }
            return tuple("gru:hidden" for _ in range(width))
        if self.spec.controller == "mlp_normal":
            widths = self.spec.controller_report["widths"]["mlp_hidden_dims"]
            roles = tuple(
                f"mlp:hidden_{layer_index}"
                for layer_index, width in enumerate(widths)
                for _ in range(int(width))
            )
            self.unit_ids = tuple(
                f"mlp:hidden_{layer_index}:{unit_index}"
                for layer_index, width in enumerate(widths)
                for unit_index in range(int(width))
            )
            self.role_provenance = {
                "engineering": {
                    "kind": "matched_mlp_post_activation_hidden_units",
                    "activation_values": "absolute_actual_post_activation_output",
                    "biological_roles": False,
                    "layer_widths": [int(width) for width in widths],
                    "unit_count": len(roles),
                }
            }
            return roles
        raise ValueError(f"unsupported controller {self.spec.controller!r}")

    def _capture_mlp(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        # Capture the actual post-activation tensor used by the next actor
        # layer.  Absolute magnitude gives MLP/GRU a common non-negative
        # activity quantity without substituting ReLU for the actor's ELU.
        self._mlp_values.append(output.detach())

    def begin_step(self) -> None:
        self._mlp_values.clear()

    def activity_batch(self, state: Any) -> torch.Tensor:
        """Return activity from the actual action forward pass as ``[B, units]``."""

        if self.spec.controller in LIF_CONTROLLER_CORE_LABELS:
            activity = state.spikes.detach().float()
        elif self.spec.controller == "gru_matched":
            activity = state.detach().float().abs()
        else:
            if not self._mlp_values:
                raise RuntimeError("MLP activity hooks did not observe the policy actor")
            hidden_layer_count = len(
                self.spec.controller_report["widths"]["mlp_hidden_dims"]
            )
            if len(self._mlp_values) < hidden_layer_count:
                raise RuntimeError("MLP activity hooks observed an incomplete actor pass")
            activity = torch.cat(
                [
                    value.float()
                    for value in self._mlp_values[-hidden_layer_count:]
                ],
                dim=1,
            ).abs()
        if activity.ndim != 2 or activity.shape[1] != len(self.roles):
            raise FloatingPointError("trained activity batch has an incompatible width")
        if activity.shape[0] < 1:
            raise FloatingPointError("trained activity batch must be non-empty")
        return activity

    def update(self, state: Any) -> dict[str, Any]:
        activity_batch = self.activity_batch(state)
        activity = activity_batch[0].to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(activity).all()):
            raise FloatingPointError("trained activity must be finite")
        if self.spec.controller in LIF_CONTROLLER_CORE_LABELS:
            active_mask = activity != 0
            active = int(active_mask.sum())
            membrane = state.membrane[0].detach().to(device="cpu", dtype=torch.float32)
            magnitude = float(torch.linalg.vector_norm(membrane))
            metric_name = "membrane_l2"
        elif self.spec.controller == "gru_matched":
            active_mask = activity > 0.1
            active = int(active_mask.sum())
            magnitude = float(torch.linalg.vector_norm(activity))
            metric_name = "hidden_l2"
        else:
            active_mask = activity > 0.0
            active = int(active_mask.sum())
            magnitude = float(torch.linalg.vector_norm(activity))
            metric_name = "activation_l2"
        self.last_activity = activity
        role_spikes = {
            role: int(active_mask[[i for i, value in enumerate(self.roles) if value == role]].sum())
            for role in dict.fromkeys(self.roles)
        }
        self.last_summary = {
            "active_fraction": float(active_mask.float().mean()),
            "active_neurons": active,
            metric_name: magnitude,
            "activity_l2": magnitude,
            "role_spikes": role_spikes,
        }
        return dict(self.last_summary)

    def brain_packet(self, step: int, keys: Sequence[str]) -> dict[str, Any]:
        if not self.last_summary:
            raise RuntimeError("trained activity has no completed sample")
        return {
            "step": int(step),
            "keys": "+".join(sorted(str(key) for key in keys)) or "-",
            "active_neurons": int(self.last_summary["active_neurons"]),
            "membrane_l2": float(self.last_summary["activity_l2"]),
            "role_spikes": dict(self.last_summary["role_spikes"]),
            "activity": [float(value) for value in self.last_activity],
        }

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


__all__ = [
    "ACTION_DIM",
    "ActionDecision",
    "COMMAND_FOLLOW_CONTRACT_VERSION",
    "COMBINATION_CONTROLLER_KINDS",
    "CommandCheckpointSpec",
    "LIF_CONTROLLER_CORE_LABELS",
    "MAXIMUM_HORIZONTAL_SPEED_M_S",
    "MAXIMUM_VERTICAL_SPEED_M_S",
    "MAXIMUM_YAW_RATE_RAD_S",
    "OBSERVATION_CONTRACT_VERSION",
    "OBSERVATION_DIM",
    "PUBLIC_CONTROLLER_KINDS",
    "SUPPORTED_CONTROLLER_KINDS",
    "RuntimeSpeeds",
    "SafetyEnvelope",
    "TASK_ID",
    "TASK_IDS",
    "WIDE_TASK_ID",
    "WIDE_WIND_TASK_ID",
    "TrainedActivityRecorder",
    "arbitrate_action",
    "command_conditioned_observation",
    "initial_policy_state",
    "inspect_command_checkpoint",
    "load_command_controller",
    "policy_state_is_finite",
    "reconstruct_physical_observation",
    "reset_policy_state",
    "safety_reason",
    "scaled_command_body",
    "sha256_file",
]
