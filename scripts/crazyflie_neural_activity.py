#!/usr/bin/env python3
"""Run the official held-out evaluator while collecting bounded neural activity.

This is an additive wrapper around :mod:`drone_evaluate`.  The simulator,
episode plans, deterministic actions, CUDA-Graph inference, scoring traces,
and memory gates remain owned by the official evaluator.  The wrapper only
accumulates post-policy state for environments whose planned episode is still
active, then adds one ``neural_activity`` object to the otherwise standard
single-scenario evaluation JSON.

For an LIF controller, a spike is the binary state exposed after the final
neural substep of one 20 ms control decision.  Rates are therefore *sampled*
rates, not a count of every spike within the five internal neural substeps.
GRU and MLP values are engineering-unit activations and are deliberately not
reported in hertz or described as biological neurons.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT_S = 0.02
TOP_UNIT_COUNT = 10
ENGINEERING_ACTIVE_ABS_THRESHOLD = 1.0e-6
LEG_MANIFEST = ROOT / "data" / "connectome" / "manifest.json"
WING_MANIFEST = ROOT / "data" / "connectome_wing" / "manifest.json"
LIF_CONTROLLERS = {
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "wing_lif",
    "leg_wing_lif",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_neuron_roles(manifest_path: Path) -> dict[str, Any]:
    """Load and authenticate one connectome's stable IDs and model roles."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Invalid connectome manifest: {manifest_path}")
    neurons_path = (manifest_path.parent / str(manifest["neurons_path"])).resolve()
    expected = manifest.get("checksums", {}).get("neurons")
    actual = _sha256_file(neurons_path)
    if expected != actual:
        raise ValueError(f"Connectome neuron checksum mismatch: {neurons_path}")
    rows = json.loads(neurons_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Connectome neurons are empty: {neurons_path}")
    neuron_ids: list[str] = []
    roles: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Neuron row {index} is not an object")
        neuron_id = str(row.get("id", ""))
        role = str(row.get("annotations", {}).get("model_role", ""))
        if not neuron_id or not role:
            raise ValueError(f"Neuron row {index} lacks a stable ID or model role")
        neuron_ids.append(neuron_id)
        roles.append(role)
    if len(set(neuron_ids)) != len(neuron_ids):
        raise ValueError("Connectome stable neuron IDs are duplicated")
    expected_count = manifest.get("derived_counts", {}).get("neurons")
    if expected_count != len(rows):
        raise ValueError("Connectome manifest neuron count does not match neurons.json")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "neurons": str(neurons_path),
        "neurons_sha256": actual,
        "neuron_ids": neuron_ids,
        "roles": roles,
        "neuron_model": manifest.get("neuron_model"),
    }


def summarize_lif_counts(
    spike_counts: Sequence[int | float],
    *,
    active_control_decisions: int,
    manifest_path: Path,
    top_count: int = TOP_UNIT_COUNT,
) -> dict[str, Any]:
    """Return role-normalized sampled LIF activity from exact per-neuron counts."""

    if type(active_control_decisions) is not int or active_control_decisions <= 0:
        raise ValueError("active_control_decisions must be a positive integer")
    if type(top_count) is not int or top_count <= 0:
        raise ValueError("top_count must be a positive integer")
    identity = _load_neuron_roles(manifest_path)
    values = [_finite_number(value, "spike count") for value in spike_counts]
    if len(values) != len(identity["neuron_ids"]):
        raise ValueError("Spike-count width does not match the connectome")
    if any(value < 0 or value > active_control_decisions or not value.is_integer() for value in values):
        raise ValueError("Per-neuron sampled spike counts are outside their denominator")

    def unit_row(index: int) -> dict[str, Any]:
        count = int(values[index])
        return {
            "index": index,
            "id": identity["neuron_ids"][index],
            "role": identity["roles"][index],
            "sampled_spike_count": count,
            "sampled_spike_rate_hz": count / (active_control_decisions * CONTROL_DT_S),
        }

    ranked = sorted(
        range(len(values)), key=lambda index: (-values[index], identity["neuron_ids"][index])
    )
    role_rows: dict[str, Any] = {}
    for role in sorted(set(identity["roles"])):
        indices = [index for index, value in enumerate(identity["roles"]) if value == role]
        role_spikes = sum(values[index] for index in indices)
        ranked_role = sorted(
            indices, key=lambda index: (-values[index], identity["neuron_ids"][index])
        )
        role_rows[role] = {
            "neuron_count": len(indices),
            "sampled_spike_count": int(role_spikes),
            "sampled_spike_rate_hz_per_neuron": role_spikes
            / (active_control_decisions * CONTROL_DT_S * len(indices)),
            "ever_active_neuron_count": sum(values[index] > 0 for index in indices),
            "ever_active_neuron_fraction": sum(values[index] > 0 for index in indices)
            / len(indices),
            "dead_neuron_count": sum(values[index] == 0 for index in indices),
            "dead_neuron_fraction": sum(values[index] == 0 for index in indices)
            / len(indices),
            "saturated_neuron_count": sum(
                values[index] == active_control_decisions for index in indices
            ),
            "saturated_neuron_fraction": sum(
                values[index] == active_control_decisions for index in indices
            )
            / len(indices),
            "top_neurons": [unit_row(index) for index in ranked_role[:top_count]],
        }
    total_spikes = sum(values)
    return {
        "kind": "biological_connectome_lif",
        "sampling_semantics": "post_final_neural_substep_once_per_active_control_decision",
        "rate_semantics": (
            "sampled_binary_spike_fraction_divided_by_0.02_s_control_dt; "
            "does_not_count_spikes_in_earlier_internal_neural_substeps"
        ),
        "control_dt_s": CONTROL_DT_S,
        "active_control_decisions": active_control_decisions,
        "neuron_count": len(values),
        "neuron_sample_denominator": active_control_decisions * len(values),
        "sampled_spike_count": int(total_spikes),
        "sampled_spike_rate_hz_per_neuron": total_spikes
        / (active_control_decisions * CONTROL_DT_S * len(values)),
        "ever_active_neuron_count": sum(value > 0 for value in values),
        "ever_active_neuron_fraction": sum(value > 0 for value in values) / len(values),
        "dead_neuron_count": sum(value == 0 for value in values),
        "dead_neuron_fraction": sum(value == 0 for value in values) / len(values),
        "saturated_neuron_count": sum(
            value == active_control_decisions for value in values
        ),
        "saturated_neuron_fraction": sum(
            value == active_control_decisions for value in values
        )
        / len(values),
        "roles": role_rows,
        "per_neuron": [unit_row(index) for index in range(len(values))],
        "top_neurons": [unit_row(index) for index in ranked[:top_count]],
        "connectome": {key: value for key, value in identity.items() if key not in {"neuron_ids", "roles"}},
    }


def summarize_engineering_units(
    absolute_sum: Sequence[int | float],
    square_sum: Sequence[int | float],
    active_count: Sequence[int | float],
    *,
    observation_count: int,
    layer: str,
    top_count: int = TOP_UNIT_COUNT,
) -> dict[str, Any]:
    """Summarize GRU/MLP units without assigning biological semantics."""

    if type(observation_count) is not int or observation_count <= 0:
        raise ValueError("observation_count must be a positive integer")
    if not (len(absolute_sum) == len(square_sum) == len(active_count)) or not absolute_sum:
        raise ValueError("Engineering-unit accumulators must be aligned and non-empty")
    abs_values = [_finite_number(value, "absolute activation sum") for value in absolute_sum]
    square_values = [_finite_number(value, "squared activation sum") for value in square_sum]
    active_values = [_finite_number(value, "active sample count") for value in active_count]
    if any(value < 0 for value in abs_values + square_values + active_values):
        raise ValueError("Engineering-unit accumulators cannot be negative")
    if any(value > observation_count or not value.is_integer() for value in active_values):
        raise ValueError("Engineering active counts are outside their denominator")
    unit_rows = [
        {
            "index": index,
            "absolute_activation_sum": abs_values[index],
            "squared_activation_sum": square_values[index],
            "active_sample_count": int(active_values[index]),
            "mean_absolute_activation": abs_values[index] / observation_count,
            "rms_activation": math.sqrt(square_values[index] / observation_count),
            "active_fraction": active_values[index] / observation_count,
        }
        for index in range(len(abs_values))
    ]
    ranked = sorted(
        unit_rows,
        key=lambda row: (-row["mean_absolute_activation"], row["index"]),
    )
    return {
        "kind": "engineering_units_not_biological_neurons",
        "layer": layer,
        "unit_count": len(unit_rows),
        "observation_count": observation_count,
        "activation_definition": (
            f"abs(value)>{ENGINEERING_ACTIVE_ABS_THRESHOLD:g}; values are not spikes or hertz"
        ),
        "mean_absolute_activation_per_unit": sum(abs_values)
        / (observation_count * len(abs_values)),
        "rms_activation_per_unit": math.sqrt(
            sum(square_values) / (observation_count * len(square_values))
        ),
        "mean_active_fraction_per_unit": sum(active_values)
        / (observation_count * len(active_values)),
        "per_unit": unit_rows,
        "top_units": ranked[:top_count],
    }


class ActivityAccumulator:
    """O(number-of-units) device accumulator patched into the official runner."""

    def __init__(self) -> None:
        self.policy: nn.Module | None = None
        self.controller: str | None = None
        self.recording = False
        self.active: torch.Tensor | None = None
        self.sample_count: torch.Tensor | None = None
        self.lif_counts: torch.Tensor | None = None
        self.engineering: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.batch_count = 0

    def attach(self, policy: nn.Module) -> None:
        if self.policy is not None and self.policy is not policy:
            raise RuntimeError("Activity wrapper observed more than one policy")
        if self.policy is not None:
            return
        self.policy = policy
        device = next(policy.parameters()).device
        # Counts stay exactly representable in float32 at the frozen
        # 16x600 upper bound; avoiding FP64 keeps this diagnostic inexpensive
        # on consumer RTX hardware.
        self.sample_count = torch.zeros((), dtype=torch.float32, device=device)
        from g1_fly_control.policies.actor_critic import FrozenLIFActorCritic, MLPActorCritic
        from g1_fly_control.policies.gru import GRUActorCritic

        if isinstance(policy, FrozenLIFActorCritic):
            width = int(policy.initial_state(1, device=device).spikes.shape[1])
            self.lif_counts = torch.zeros(width, dtype=torch.float32, device=device)
        elif isinstance(policy, GRUActorCritic):
            self._ensure_engineering("gru_hidden", int(policy.hidden_dim), device)
        elif isinstance(policy, MLPActorCritic):
            # MLP layer widths are initialized on the first exact observation.
            pass
        else:
            raise TypeError(f"Unsupported policy type for activity analysis: {type(policy).__name__}")

    def _ensure_engineering(self, name: str, width: int, device: torch.device) -> None:
        current = self.engineering.get(name)
        if current is not None:
            if current[0].numel() != width:
                raise RuntimeError(f"Engineering layer width changed for {name}")
            return
        self.engineering[name] = tuple(
            torch.zeros(width, dtype=torch.float32, device=device) for _ in range(3)
        )  # type: ignore[assignment]

    def begin_batch(self, runner: Any) -> None:
        self.attach(runner.policy)
        if self.recording:
            raise RuntimeError("Activity batches cannot overlap")
        self.active = torch.ones(
            runner.batch_size, dtype=torch.bool, device=runner.device
        )
        self.recording = True
        self.batch_count += 1

    def end_batch(self) -> None:
        if not self.recording:
            raise RuntimeError("No activity batch is active")
        self.recording = False
        self.active = None

    def _add_engineering(self, name: str, values: torch.Tensor) -> None:
        if values.ndim != 2:
            raise RuntimeError(f"Engineering activation {name} is not [sample, unit]")
        self._ensure_engineering(name, values.shape[1], values.device)
        absolute, squared, active = self.engineering[name]
        value32 = values.detach().float()
        absolute.add_(value32.abs().sum(dim=0))
        squared.add_(value32.square().sum(dim=0))
        active.add_((value32.abs() > ENGINEERING_ACTIVE_ABS_THRESHOLD).sum(dim=0))

    def observe(
        self,
        runner: Any,
        observation: torch.Tensor,
        reset_before_policy: torch.Tensor,
    ) -> None:
        if not self.recording or self.active is None or self.sample_count is None:
            return
        # A true reset mask means that row terminated on the preceding step.
        # Its terminal policy decision was already counted; exclude the
        # simulator's automatically-reset replacement episode from now on.
        self.active.logical_and_(~reset_before_policy)
        active = self.active
        active_count = active.sum()
        self.sample_count.add_(active_count.float())

        from g1_fly_control.policies.actor_critic import FrozenLIFActorCritic, MLPActorCritic
        from g1_fly_control.policies.gru import GRUActorCritic

        policy = runner.policy
        state = runner._current_state()
        if isinstance(policy, FrozenLIFActorCritic):
            if self.lif_counts is None or state.spikes.shape[1] != self.lif_counts.numel():
                raise RuntimeError("LIF activity state width changed")
            spikes = state.spikes[active]
            # LIFCore constructs this tensor from its hard binary surrogate.
            # The official CUDA-Graph runner already folds recurrent-state
            # finiteness into its fail-closed action validity mask. Avoid an
            # extra device synchronization at every control decision here.
            self.lif_counts.add_(spikes.float().sum(dim=0))
        elif isinstance(policy, GRUActorCritic):
            self._add_engineering("gru_hidden", state[active])
        elif isinstance(policy, MLPActorCritic):
            values = observation[active]
            hidden_index = 0
            # Recompute only the deterministic actor hidden layers from the
            # exact normalized observation. This is diagnostic and does not
            # replace or feed the captured action path.
            for layer in policy.actor:
                values = layer(values)
                if not isinstance(layer, nn.Linear):
                    hidden_index += 1
                    self._add_engineering(f"mlp_hidden_{hidden_index}", values)
        else:  # guarded at attach, retained fail-closed for mutation
            raise TypeError(f"Unsupported policy type: {type(policy).__name__}")

    def finalize(self, evaluation: Mapping[str, Any]) -> dict[str, Any]:
        if self.recording or self.policy is None or self.sample_count is None:
            raise RuntimeError("Neural activity did not reach a complete evaluation boundary")
        controller = evaluation.get("controller")
        if controller not in {
            "frozen_lif_original",
            "frozen_lif_degree_rewired",
            "wing_lif",
            "leg_wing_lif",
            "gru_matched",
            "mlp_normal",
        }:
            raise ValueError(f"Unknown evaluated controller: {controller!r}")
        episodes = evaluation.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("Completed evaluation lacks episode rows")
        expected_samples = sum(
            int(row["completed_steps"])
            for row in episodes
            if isinstance(row, Mapping) and type(row.get("completed_steps")) is int
        )
        packed: list[torch.Tensor] = [self.sample_count.reshape(1)]
        if self.lif_counts is not None:
            packed.append(self.lif_counts)
        for name in sorted(self.engineering):
            packed.extend(self.engineering[name])
        host = torch.cat(packed).detach().cpu()
        cursor = 0
        observed_samples = int(host[cursor].item())
        cursor += 1
        if observed_samples != expected_samples:
            raise RuntimeError(
                "Activity sample count differs from official completed episode steps: "
                f"{observed_samples} != {expected_samples}"
            )

        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "analysis_kind": "crazyflie_heldout_neural_activity_v1",
            "collector": str(Path(__file__).resolve()),
            "collector_sha256": _sha256_file(Path(__file__).resolve()),
            "controller": controller,
            "scenario": evaluation.get("scenario"),
            "training_seed": evaluation.get("training_seed"),
            "evaluation_seed": evaluation.get("evaluation_seed"),
            "evaluation_manifest_id": evaluation.get("evaluation_manifest_id"),
            "checkpoint": evaluation.get("checkpoint"),
            "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
            "episode_ids": [row.get("episode_id") for row in episodes],
            "plan_sha256": [row.get("plan_sha256") for row in episodes],
            "active_control_decisions": observed_samples,
            "batch_count": self.batch_count,
            "official_evaluation_summary_sha256": sha256(
                json.dumps(
                    evaluation.get("summary"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
            "biological_interpretation": (
                "LIF roles retain connectome annotations; GRU/MLP units have no "
                "leg, wing, descending, thoracic, sensory, or motor identity"
            ),
            "cores": {},
            "engineering_layers": {},
        }
        if self.lif_counts is not None:
            width = self.lif_counts.numel()
            counts = host[cursor : cursor + width].tolist()
            cursor += width
            if controller == "leg_wing_lif":
                if width != 512:
                    raise ValueError("Combined LIF state must contain exactly 512 neurons")
                payload["cores"] = {
                    "leg": summarize_lif_counts(
                        counts[:256],
                        active_control_decisions=observed_samples,
                        manifest_path=LEG_MANIFEST,
                    ),
                    "wing": summarize_lif_counts(
                        counts[256:],
                        active_control_decisions=observed_samples,
                        manifest_path=WING_MANIFEST,
                    ),
                }
            else:
                manifest = WING_MANIFEST if controller == "wing_lif" else LEG_MANIFEST
                core_name = "wing" if controller == "wing_lif" else "leg"
                payload["cores"] = {
                    core_name: summarize_lif_counts(
                        counts,
                        active_control_decisions=observed_samples,
                        manifest_path=manifest,
                    )
                }
        for name in sorted(self.engineering):
            width = self.engineering[name][0].numel()
            arrays = []
            for _ in range(3):
                arrays.append(host[cursor : cursor + width].tolist())
                cursor += width
            payload["engineering_layers"][name] = summarize_engineering_units(
                arrays[0],
                arrays[1],
                arrays[2],
                observation_count=observed_samples,
                layer=name,
            )
        if cursor != host.numel():
            raise RuntimeError("Neural activity host payload was not consumed exactly")
        if controller in LIF_CONTROLLERS and not payload["cores"]:
            raise RuntimeError("LIF controller produced no biological-core analysis")
        if controller not in LIF_CONTROLLERS and not payload["engineering_layers"]:
            raise RuntimeError("Engineering baseline produced no unit analysis")
        return payload


def install_activity_hooks(evaluator: Any, accumulator: ActivityAccumulator) -> None:
    """Patch only evaluator extension points; keep official control logic intact."""

    original_step = evaluator._CudaGraphPolicyRunner.step
    original_batch = evaluator._evaluate_batch
    original_atomic_json = evaluator._atomic_json

    def step(runner: Any, observation: torch.Tensor, reset: torch.Tensor) -> Any:
        result = original_step(runner, observation, reset)
        accumulator.observe(runner, observation, reset)
        return result

    def evaluate_batch(*args: Any, **kwargs: Any) -> Any:
        runner = args[2] if len(args) >= 3 else kwargs["policy_runner"]
        accumulator.begin_batch(runner)
        try:
            return original_batch(*args, **kwargs)
        finally:
            accumulator.end_batch()

    def atomic_json(path: Path, value: dict[str, Any]) -> None:
        if value.get("status") == "completed":
            if value.get("scenario") == "all":
                raise ValueError("Neural activity wrapper requires one explicit --scenario")
            value = dict(value)
            value["neural_activity"] = accumulator.finalize(value)
        original_atomic_json(path, value)

    evaluator._CudaGraphPolicyRunner.step = step
    evaluator._evaluate_batch = evaluate_batch
    evaluator._atomic_json = atomic_json


def main() -> int:
    if "--all_scenarios" in sys.argv:
        raise SystemExit(
            "Use one --scenario per process; the activity denominator is authenticated per task"
        )
    scripts = str(ROOT / "scripts")
    source = str(ROOT / "source" / "g1_fly_control")
    for path in (scripts, source):
        if path not in sys.path:
            sys.path.insert(0, path)
    import drone_evaluate as evaluator

    accumulator = ActivityAccumulator()
    install_activity_hooks(evaluator, accumulator)
    return int(evaluator.main())


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
