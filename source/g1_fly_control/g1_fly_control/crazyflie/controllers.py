"""Matched Crazyflie controllers built around the frozen project primitives.

The public builder in this module is CPU safe: importing it never imports
Isaac Lab or launches Isaac Sim.  All four actors use a 12-value observation,
a four-value tanh-bounded action, the same action distribution, and the same
critic architecture.  The recurrent actors expose one independent state row
per environment and support masked row resets.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from g1_fly_control.connectome import load_connectome
from g1_fly_control.connectome.rewire import degree_preserving_rewire
from g1_fly_control.connectome.schema import CircuitData, file_sha256
from g1_fly_control.policies.actor_critic import (
    FrozenLIFActorCritic,
    MLPActorCritic,
    PolicyOutput,
    TanhDiagonalGaussian,
    _mlp,
)
from g1_fly_control.policies.gru import GRUActorCritic
from g1_fly_control.policies.lif_core import LIFCore, LIFState
from g1_fly_control.crazyflie.stabilization import (
    CRAZYFLIE_RESIDUAL_LATENT_SCALE,
    compose_crazyflie_residual_mean,
    stabilization_contract_payload,
)


OBSERVATION_DIM = 12
ACTION_DIM = 4
DEFAULT_ADAPTER_HIDDEN_DIM = 64
DEFAULT_CRITIC_HIDDEN_DIM = 128
DEFAULT_REWIRE_SEED = 20260916
DEFAULT_REWIRE_MANIFEST_FILE_SHA256 = "6c2a10b879d22741b0d17110d052953adc4f081e9ecac636fabdf0ba68d57b14"
DEFAULT_PARAMETER_TOLERANCE = 0.10
CONTROLLER_KINDS = ("frozen_lif", "frozen_lif_rewired", "optic_lif", "gru", "mlp")
WING_EXTENSION_CONTROLLER_KINDS = ("wing_lif", "leg_wing_lif")
COMBINATION_CONTROLLER_KINDS = (
    "leg_optic_lif",
    "wing_optic_lif",
    "leg_wing_optic_lif",
)

# The installed action mapping produces vehicle weight when
# (action[0] + 1) / 2 * 1.9 == 1.  The shared fixed stabilization prior now
# supplies this command.  Every trainable head is initialized as a zero-bias
# residual around that prior.
CRAZYFLIE_THRUST_TO_WEIGHT = 1.9
CRAZYFLIE_HOVER_ACTION = (2.0 / CRAZYFLIE_THRUST_TO_WEIGHT - 1.0, 0.0, 0.0, 0.0)
# Collective-thrust exploration retains the validated low-amplitude 0.03
# scale.  The three native moment coordinates are ten times more sensitive in
# the near-hover regime: the bounded pre-main LIF pilot showed that 0.03
# moment noise overwhelmed the roughly 0.004 normalized damping commands used
# by the live stabilizing diagnostic and drove every episode into the ground.
# Keep the smaller moment scale identical across all four comparison actors.
CRAZYFLIE_INITIAL_LATENT_STD = (0.03, 0.003, 0.003, 0.003)

# Keep the final actor weights nonzero so gradients can cross the LIF decoder
# and frozen differentiable core into the encoder.  Fan-in scaling makes the
# perturbation around the hover bias comparable across the differently sized
# LIF, GRU, and MLP heads.  Four non-constant rows from an order-eight Walsh
# matrix keep the four action directions independent without consuming RNG.
# The 0.03 thrust standard deviation remains inside the plan's predeclared
# low-amplitude smoke envelope with roughly 99.9% probability before tanh;
# moment exploration is narrower to preserve recoverable near-hover data.
CRAZYFLIE_INITIAL_HEAD_WEIGHT_SCALE = 1.0e-3

_KIND_ALIASES = {
    "frozen_lif": "frozen_lif",
    "original_lif": "frozen_lif",
    "original_frozen_lif": "frozen_lif",
    "frozen_lif_original": "frozen_lif",
    "lif": "frozen_lif",
    "frozen_lif_rewired": "frozen_lif_rewired",
    "rewired_lif": "frozen_lif_rewired",
    "rewired_frozen_lif": "frozen_lif_rewired",
    "frozen_lif_degree_rewired": "frozen_lif_rewired",
    "gru": "gru",
    "matched_gru": "gru",
    "gru_matched": "gru",
    "mlp": "mlp",
    "normal_mlp": "mlp",
    "mlp_normal": "mlp",
    "wing_lif": "wing_lif",
    "frozen_lif_wing": "wing_lif",
    "leg_wing_lif": "leg_wing_lif",
    "combined_leg_wing_lif": "leg_wing_lif",
    "optic_lif": "optic_lif",
    "frozen_lif_optic": "optic_lif",
    "leg_optic_lif": "leg_optic_lif",
    "combined_leg_optic_lif": "leg_optic_lif",
    "frozen_lif_leg_optic": "leg_optic_lif",
    "wing_optic_lif": "wing_optic_lif",
    "combined_wing_optic_lif": "wing_optic_lif",
    "frozen_lif_wing_optic": "wing_optic_lif",
    "leg_wing_optic_lif": "leg_wing_optic_lif",
    "combined_leg_wing_optic_lif": "leg_wing_optic_lif",
    "frozen_lif_leg_wing_optic": "leg_wing_optic_lif",
    "all_connectome_lif": "leg_wing_optic_lif",
}

_COMBINATION_CORE_LABELS = {
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
}


def _default_connectome_manifest() -> Path:
    # controllers.py -> crazyflie -> g1_fly_control -> g1_fly_control -> source -> repository
    return Path(__file__).resolve().parents[4] / "data" / "connectome" / "manifest.json"


def default_wing_connectome_manifest() -> Path:
    """Return the independently extracted primary wing-circuit manifest."""

    return Path(__file__).resolve().parents[4] / "data" / "connectome_wing" / "manifest.json"


def default_optic_connectome_manifest() -> Path:
    """Return the independently extracted primary optic-circuit manifest."""

    return Path(__file__).resolve().parents[4] / "data" / "connectome_optic" / "manifest.json"


def default_rewire_manifest_path() -> Path:
    """Return the checked-in primary rewire artifact path."""

    return (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "experiments"
        / "crazyflie_rewire_seed_20260916.json"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _tensor_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_json(list(value.shape)))
    digest.update(value.numpy().tobytes())


def _lif_core_checksum(core: LIFCore) -> str:
    """Hash all fixed graph and neuron-model values for one LIF core."""

    digest = sha256()
    digest.update(
        _canonical_json(
            {
                "num_neurons": core.num_neurons,
                "dt": core.dt,
                "tau_membrane": core.tau_membrane,
                "tau_synapse": core.tau_synapse,
                "threshold": core.threshold,
                "reset_value": core.reset_value,
                "refractory_steps": core.refractory_steps,
                "surrogate_beta": core.surrogate_beta,
                "neural_substeps": core.neural_substeps,
            }
        )
    )
    _tensor_hash(digest, "edge_index", core.edge_index)
    _tensor_hash(digest, "weights", core.weights)
    _tensor_hash(digest, "membrane_decay", core.membrane_decay)
    _tensor_hash(digest, "synapse_decay", core.synapse_decay)
    return digest.hexdigest()


def controller_core_checksum(policy: nn.Module) -> str | None:
    """Hash every fixed value that defines a LIF core, including topology.

    The legacy ``LIFCore.frozen_checksum`` remains available, but this stronger
    Crazyflie checksum also covers the threshold, reset, refractory, surrogate,
    and neural-substep settings.  Derived dense weights are intentionally
    excluded because they are deterministically reconstructed from the graph.
    """

    if isinstance(policy, CrazyflieCombinedConnectomeActorCritic):
        return sha256(
            _canonical_json(
                {
                    "composition": policy.fusion_contract,
                    "cores": {
                        label: _lif_core_checksum(core)
                        for label, core in policy.named_lif_cores()
                    },
                }
            )
        ).hexdigest()
    if isinstance(policy, CrazyflieCombinedLegWingActorCritic):
        return sha256(
            _canonical_json(
                {
                    "composition": "independent_leg_and_wing_cores_concat_motor_readouts_v1",
                    "leg": _lif_core_checksum(policy.core),
                    "wing": _lif_core_checksum(policy.wing_core),
                }
            )
        ).hexdigest()
    core = getattr(policy, "core", None)
    if not isinstance(core, LIFCore):
        return None
    return _lif_core_checksum(core)


def controller_core_checksums(policy: nn.Module) -> dict[str, str]:
    """Return deterministic per-core checksums in recurrent-state order."""

    return {
        label: _lif_core_checksum(core)
        for label, core in named_lif_cores(policy)
    }


def verify_frozen_core(policy: nn.Module, expected_checksum: str | None) -> None:
    """Raise if a LIF core differs from a previously recorded checksum."""

    actual = controller_core_checksum(policy)
    if actual != expected_checksum:
        raise RuntimeError(
            f"Frozen LIF core checksum changed: expected {expected_checksum!r}, got {actual!r}."
        )


class _CrazyflieBoundedResidualMixin:
    """Shared latent-prior composition without trainable prior parameters."""

    crazyflie_residual_latent_scale: torch.Tensor

    def _install_crazyflie_residual_contract(self) -> None:
        self.register_buffer(
            "crazyflie_residual_latent_scale",
            torch.tensor(CRAZYFLIE_RESIDUAL_LATENT_SCALE, dtype=torch.float32),
        )

    def compose_residual_mean(
        self, observations: torch.Tensor, residual_logits: torch.Tensor
    ) -> torch.Tensor:
        return compose_crazyflie_residual_mean(
            observations,
            residual_logits,
            residual_scale=self.crazyflie_residual_latent_scale,
        )


class CrazyflieFrozenLIFActorCritic(
    _CrazyflieBoundedResidualMixin, FrozenLIFActorCritic
):
    """Frozen LIF actor with an explicit per-environment reset interface."""

    is_recurrent = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._install_crazyflie_residual_contract()

    def _mean_and_state(
        self,
        observations: torch.Tensor,
        state: LIFState,
        *,
        ablate_outgoing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LIFState]:
        residual_logits, next_state = super()._mean_and_state(
            observations, state, ablate_outgoing=ablate_outgoing
        )
        return self.compose_residual_mean(observations, residual_logits), next_state

    def reset_state(self, state: LIFState, done: torch.Tensor) -> LIFState:
        if done.ndim != 1 or done.shape[0] != state.membrane.shape[0]:
            raise ValueError("done must contain one Boolean value per environment.")
        return state.masked_reset(done.to(device=state.membrane.device, dtype=torch.bool))

    def named_lif_cores(self) -> tuple[tuple[str, LIFCore], ...]:
        """Return the deterministic label/core sequence used by this actor."""

        return ((getattr(self, "_lif_core_label", "primary"), self.core),)

    def named_lif_core_indices(
        self,
    ) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
        """Return input/readout indices in the same order as the LIF cores."""

        label = self.named_lif_cores()[0][0]
        return ((label, self.input_indices, self.output_indices),)

    def named_lif_encoders(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return each observation encoder in deterministic core order."""

        label = self.named_lif_cores()[0][0]
        return ((label, self.encoder),)


class CrazyflieCombinedLegWingActorCritic(CrazyflieFrozenLIFActorCritic):
    """Two independent frozen LIF cores fused only at their motor readouts.

    The inherited ``core`` is the untouched leg core so the existing recurrent
    PPO recognizes this as a frozen-LIF actor.  ``wing_core`` is a second fixed
    graph.  State is represented as one concatenated :class:`LIFState`, which
    preserves the existing rollout/checkpoint/reset interface without adding
    any recurrent edge between the biological circuits.
    """

    fusion_contract = "independent_leg_and_wing_cores_concat_motor_readouts_v1"

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        leg_core: LIFCore,
        wing_core: LIFCore,
        *,
        leg_input_indices: list[int] | torch.Tensor,
        leg_output_indices: list[int] | torch.Tensor,
        wing_input_indices: list[int] | torch.Tensor,
        wing_output_indices: list[int] | torch.Tensor,
        adapter_hidden_dim: int = DEFAULT_ADAPTER_HIDDEN_DIM,
        critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM,
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            leg_core,
            input_indices=leg_input_indices,
            output_indices=leg_output_indices,
            adapter_hidden_dim=adapter_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
        )
        self.wing_core = wing_core
        self.register_buffer(
            "wing_input_indices", torch.as_tensor(wing_input_indices, dtype=torch.long)
        )
        self.register_buffer(
            "wing_output_indices", torch.as_tensor(wing_output_indices, dtype=torch.long)
        )
        if self.wing_input_indices.numel() == 0 or self.wing_output_indices.numel() == 0:
            raise ValueError("Combined controller requires non-empty wing input/output populations.")
        if self.wing_input_indices.min() < 0 or self.wing_input_indices.max() >= wing_core.num_neurons:
            raise ValueError("Wing input subset is outside the wing core.")
        if self.wing_output_indices.min() < 0 or self.wing_output_indices.max() >= wing_core.num_neurons:
            raise ValueError("Wing output subset is outside the wing core.")
        self.wing_encoder = _mlp(
            [observation_dim, adapter_hidden_dim, self.wing_input_indices.numel()]
        )
        self.decoder = _mlp(
            [
                self.output_indices.numel() + self.wing_output_indices.numel(),
                adapter_hidden_dim,
                action_dim,
            ]
        )

    @property
    def combined_neuron_count(self) -> int:
        return self.core.num_neurons + self.wing_core.num_neurons

    def named_lif_cores(self) -> tuple[tuple[str, LIFCore], ...]:
        return (("leg", self.core), ("wing", self.wing_core))

    def named_lif_core_indices(
        self,
    ) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
        return (
            ("leg", self.input_indices, self.output_indices),
            ("wing", self.wing_input_indices, self.wing_output_indices),
        )

    def named_lif_encoders(self) -> tuple[tuple[str, nn.Module], ...]:
        return (("leg", self.encoder), ("wing", self.wing_encoder))

    def initial_state(
        self, batch_size: int, *, device: torch.device | str | None = None
    ) -> LIFState:
        leg = self.core.initial_state(batch_size, device=device)
        wing = self.wing_core.initial_state(batch_size, device=device)
        return self._join_state(leg, wing)

    def _split_state(self, state: LIFState) -> tuple[LIFState, LIFState]:
        if state.membrane.shape[1] != self.combined_neuron_count:
            raise ValueError(
                f"Combined state width must be {self.combined_neuron_count}, "
                f"got {state.membrane.shape[1]}."
            )
        boundary = self.core.num_neurons
        values = (state.membrane, state.spikes, state.synapse, state.refractory)
        leg = LIFState(*(value[:, :boundary] for value in values))
        wing = LIFState(*(value[:, boundary:] for value in values))
        return leg, wing

    @staticmethod
    def _join_state(leg: LIFState, wing: LIFState) -> LIFState:
        return LIFState(*(
            torch.cat((leg_value, wing_value), dim=1)
            for leg_value, wing_value in zip(
                (leg.membrane, leg.spikes, leg.synapse, leg.refractory),
                (wing.membrane, wing.spikes, wing.synapse, wing.refractory),
                strict=True,
            )
        ))

    def _mean_and_state(
        self,
        observations: torch.Tensor,
        state: LIFState,
        *,
        ablate_outgoing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LIFState]:
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f"Expected observations [batch, {self.observation_dim}].")
        if ablate_outgoing is not None:
            raise ValueError("Combined-core ablation requires an explicitly named per-core mask.")
        leg_state, wing_state = self._split_state(state)
        leg_current = torch.zeros(
            (observations.shape[0], self.core.num_neurons),
            dtype=observations.dtype,
            device=observations.device,
        )
        wing_current = torch.zeros(
            (observations.shape[0], self.wing_core.num_neurons),
            dtype=observations.dtype,
            device=observations.device,
        )
        leg_current[:, self.input_indices] = self.encoder(observations)
        wing_current[:, self.wing_input_indices] = self.wing_encoder(observations)
        next_leg = self.core(leg_current, leg_state)
        next_wing = self.wing_core(wing_current, wing_state)
        motor_readouts = torch.cat(
            (
                next_leg.synapse[:, self.output_indices],
                next_wing.synapse[:, self.wing_output_indices],
            ),
            dim=1,
        )
        residual_logits = self.decoder(motor_readouts)
        return (
            self.compose_residual_mean(observations, residual_logits),
            self._join_state(next_leg, next_wing),
        )


class CrazyflieCombinedConnectomeActorCritic(CrazyflieFrozenLIFActorCritic):
    """Fuse two or three independent authenticated LIF connectomes.

    Each core has its own observation encoder and recurrent state.  Only the
    selected readout populations are concatenated, immediately before one
    shared trainable action decoder.  Keeping the cores in separate modules
    makes it structurally impossible to introduce inter-core recurrent edges.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cores: Mapping[str, LIFCore],
        *,
        input_indices: Mapping[str, list[int] | torch.Tensor],
        output_indices: Mapping[str, list[int] | torch.Tensor],
        adapter_hidden_dim: int = DEFAULT_ADAPTER_HIDDEN_DIM,
        critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM,
    ) -> None:
        ordered_cores = tuple(cores.items())
        labels = tuple(label for label, _core in ordered_cores)
        if len(labels) not in {2, 3}:
            raise ValueError("Combined connectome actors require exactly two or three LIF cores.")
        if len(set(labels)) != len(labels) or any(
            not label or not label.replace("_", "").isalnum() for label in labels
        ):
            raise ValueError("Combined connectome core labels must be unique identifiers.")
        if set(input_indices) != set(labels) or set(output_indices) != set(labels):
            raise ValueError("Every named LIF core requires one input and one output index set.")

        primary_label, primary_core = ordered_cores[0]
        super().__init__(
            observation_dim,
            action_dim,
            primary_core,
            input_indices=input_indices[primary_label],
            output_indices=output_indices[primary_label],
            adapter_hidden_dim=adapter_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
        )
        self._lif_core_label = primary_label
        self.core_labels = labels
        self.additional_cores = nn.ModuleDict()
        self.additional_encoders = nn.ModuleDict()
        for label, core in ordered_cores[1:]:
            encoded_inputs = torch.as_tensor(input_indices[label], dtype=torch.long)
            readout_outputs = torch.as_tensor(output_indices[label], dtype=torch.long)
            if encoded_inputs.numel() == 0 or readout_outputs.numel() == 0:
                raise ValueError(f"Combined core {label!r} has an empty input/output population.")
            if encoded_inputs.min() < 0 or encoded_inputs.max() >= core.num_neurons:
                raise ValueError(f"Input subset for combined core {label!r} is outside its core.")
            if readout_outputs.min() < 0 or readout_outputs.max() >= core.num_neurons:
                raise ValueError(f"Output subset for combined core {label!r} is outside its core.")
            self.additional_cores[label] = core
            self.additional_encoders[label] = _mlp(
                [observation_dim, adapter_hidden_dim, encoded_inputs.numel()]
            )
            self.register_buffer(f"_lif_{label}_input_indices", encoded_inputs)
            self.register_buffer(f"_lif_{label}_output_indices", readout_outputs)

        total_readouts = sum(
            int(torch.as_tensor(indices).numel()) for indices in output_indices.values()
        )
        self.decoder = _mlp([total_readouts, adapter_hidden_dim, action_dim])
        self.fusion_contract = (
            "independent_"
            + "_and_".join(labels)
            + "_cores_concat_motor_readouts_v1"
        )

    @property
    def combined_neuron_count(self) -> int:
        return sum(core.num_neurons for _label, core in self.named_lif_cores())

    def named_lif_cores(self) -> tuple[tuple[str, LIFCore], ...]:
        return (
            (self.core_labels[0], self.core),
            *((label, self.additional_cores[label]) for label in self.core_labels[1:]),
        )

    def named_lif_core_indices(
        self,
    ) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
        values: list[tuple[str, torch.Tensor, torch.Tensor]] = [
            (self.core_labels[0], self.input_indices, self.output_indices)
        ]
        values.extend(
            (
                label,
                getattr(self, f"_lif_{label}_input_indices"),
                getattr(self, f"_lif_{label}_output_indices"),
            )
            for label in self.core_labels[1:]
        )
        return tuple(values)

    def named_lif_encoders(self) -> tuple[tuple[str, nn.Module], ...]:
        return (
            (self.core_labels[0], self.encoder),
            *((label, self.additional_encoders[label]) for label in self.core_labels[1:]),
        )

    def initial_state(
        self, batch_size: int, *, device: torch.device | str | None = None
    ) -> LIFState:
        return self._join_states(
            tuple(
                core.initial_state(batch_size, device=device)
                for _label, core in self.named_lif_cores()
            )
        )

    def _split_state(self, state: LIFState) -> tuple[LIFState, ...]:
        if state.membrane.shape[1] != self.combined_neuron_count:
            raise ValueError(
                f"Combined state width must be {self.combined_neuron_count}, "
                f"got {state.membrane.shape[1]}."
            )
        fields = (state.membrane, state.spikes, state.synapse, state.refractory)
        result = []
        start = 0
        for _label, core in self.named_lif_cores():
            stop = start + core.num_neurons
            result.append(LIFState(*(field[:, start:stop] for field in fields)))
            start = stop
        return tuple(result)

    @staticmethod
    def _join_states(states: tuple[LIFState, ...]) -> LIFState:
        if not states:
            raise ValueError("At least one LIF state is required.")
        return LIFState(
            torch.cat(tuple(state.membrane for state in states), dim=1),
            torch.cat(tuple(state.spikes for state in states), dim=1),
            torch.cat(tuple(state.synapse for state in states), dim=1),
            torch.cat(tuple(state.refractory for state in states), dim=1),
        )

    def _mean_and_state(
        self,
        observations: torch.Tensor,
        state: LIFState,
        *,
        ablate_outgoing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LIFState]:
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f"Expected observations [batch, {self.observation_dim}].")
        if ablate_outgoing is not None:
            raise ValueError("Combined-core ablation requires an explicitly named per-core mask.")

        next_states = []
        readouts = []
        for core_item, index_item, encoder_item, core_state in zip(
            self.named_lif_cores(),
            self.named_lif_core_indices(),
            self.named_lif_encoders(),
            self._split_state(state),
            strict=True,
        ):
            label, core = core_item
            index_label, inputs, outputs = index_item
            encoder_label, encoder = encoder_item
            if label != index_label or label != encoder_label:
                raise RuntimeError("Named LIF core metadata order is inconsistent.")
            current = torch.zeros(
                (observations.shape[0], core.num_neurons),
                dtype=observations.dtype,
                device=observations.device,
            )
            current[:, inputs] = encoder(observations)
            next_state = core(current, core_state)
            next_states.append(next_state)
            readouts.append(next_state.synapse[:, outputs])
        residual_logits = self.decoder(torch.cat(readouts, dim=1))
        return (
            self.compose_residual_mean(observations, residual_logits),
            self._join_states(tuple(next_states)),
        )


class MatchedGRUActorCritic(_CrazyflieBoundedResidualMixin, GRUActorCritic):
    """GRU actor whose critic is independent of the actor matching width."""

    is_recurrent = True

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dim: int,
        critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM,
    ) -> None:
        if hidden_dim < 1 or critic_hidden_dim < 1:
            raise ValueError("GRU and critic hidden dimensions must be positive.")
        super().__init__(observation_dim, action_dim, hidden_dim=hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)
        self.critic = _mlp([observation_dim, critic_hidden_dim, critic_hidden_dim, 1])
        self._install_crazyflie_residual_contract()

    def _mean_and_state(
        self, observations: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = self.gru(observations, state)
        residual_logits = self.actor(next_state)
        return self.compose_residual_mean(observations, residual_logits), next_state

    def act(
        self,
        observations: torch.Tensor,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
        **_: object,
    ) -> PolicyOutput:
        mean, next_state = self._mean_and_state(observations, state)
        action, log_prob = TanhDiagonalGaussian(mean, self.log_std).sample(
            deterministic
        )
        return PolicyOutput(
            action,
            log_prob,
            self.critic(observations).squeeze(-1),
            next_state,
            mean,
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, next_state = self._mean_and_state(observations, state)
        distribution = TanhDiagonalGaussian(mean, self.log_std)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(observations).squeeze(-1),
            next_state,
        )

    def reset_state(self, state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or done.ndim != 1 or done.shape[0] != state.shape[0]:
            raise ValueError("done must contain one Boolean value per environment.")
        return state.masked_fill(done.to(device=state.device, dtype=torch.bool)[:, None], 0.0)


class MatchedMLPActorCritic(_CrazyflieBoundedResidualMixin, MLPActorCritic):
    """A genuine single-step feed-forward baseline with the common critic."""

    is_recurrent = False

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: Iterable[int],
        critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM,
    ) -> None:
        dimensions = tuple(int(width) for width in hidden_dims)
        if not dimensions or any(width < 1 for width in dimensions) or critic_hidden_dim < 1:
            raise ValueError("MLP and critic hidden dimensions must be positive.")
        # The parent establishes the shared action API.  Both networks are then
        # replaced so actor matching never changes the common critic.
        super().__init__(observation_dim, action_dim, hidden_dim=dimensions[0])
        self.hidden_dims = dimensions
        self.critic_hidden_dim = int(critic_hidden_dim)
        self.actor = _mlp([observation_dim, *dimensions, action_dim])
        self.critic = _mlp([observation_dim, critic_hidden_dim, critic_hidden_dim, 1])
        self._install_crazyflie_residual_contract()

    def _mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.compose_residual_mean(observations, self.actor(observations))

    def act(
        self,
        observations: torch.Tensor,
        state: None = None,
        *,
        deterministic: bool = False,
        **_: object,
    ) -> PolicyOutput:
        del state
        mean = self._mean(observations)
        action, log_prob = TanhDiagonalGaussian(mean, self.log_std).sample(
            deterministic
        )
        return PolicyOutput(
            action,
            log_prob,
            self.critic(observations).squeeze(-1),
            None,
            mean,
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state: None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        del state
        distribution = TanhDiagonalGaussian(self._mean(observations), self.log_std)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(observations).squeeze(-1),
            None,
        )

    def reset_state(self, state: None, done: torch.Tensor) -> None:
        del state, done
        return None


def _actor_output_layer(policy: nn.Module) -> nn.Linear:
    """Return the final latent-mean layer for a supported Crazyflie actor."""

    if isinstance(policy, FrozenLIFActorCritic):
        layer = policy.decoder[-1]
    elif isinstance(policy, GRUActorCritic):
        layer = policy.actor
    elif isinstance(policy, MLPActorCritic):
        layer = policy.actor[-1]
    else:
        raise TypeError(f"Unsupported Crazyflie actor class: {type(policy).__name__}")
    if not isinstance(layer, nn.Linear):
        raise TypeError("Crazyflie actor output head must end in a Linear layer.")
    return layer


def initialize_crazyflie_actor(policy: nn.Module) -> dict[str, Any]:
    """Apply the shared bounded-residual initialization to one actor.

    Only the trainable action head and distribution scale are changed.  The
    LIF core, encoder, recurrent body, feed-forward body, and critic retain
    their ordinary initialization.  A deterministic full-row-rank Walsh sign
    pattern avoids consuming RNG state while keeping every final weight
    nonzero.  The fixed stabilizer supplies hover, so the residual bias is
    exactly zero for all four conditions.
    """

    action_dim = int(getattr(policy, "action_dim", -1))
    if action_dim != ACTION_DIM:
        raise ValueError(
            f"Crazyflie hover initialization requires action_dim={ACTION_DIM}, got {action_dim}."
        )
    log_std = getattr(policy, "log_std", None)
    if not isinstance(log_std, nn.Parameter) or tuple(log_std.shape) != (ACTION_DIM,):
        raise TypeError("Crazyflie actor must expose a four-value trainable log_std parameter.")

    head = _actor_output_layer(policy)
    if head.out_features != ACTION_DIM or head.bias is None:
        raise ValueError("Crazyflie actor output head must have four outputs and a bias.")
    if head.in_features < ACTION_DIM:
        raise ValueError(
            "Crazyflie actor output head needs at least four input features for "
            "the full-row-rank shared initialization."
        )

    with torch.no_grad():
        # These are Walsh rows 1, 2, 3, and 4 from the order-eight Hadamard
        # matrix.  Each is balanced and the first four columns already have
        # rank four, so repeating the eight-column block preserves full row
        # rank for every supported fan-in (all are at least four).
        walsh_rows = torch.tensor(
            (
                (1, -1, 1, -1, 1, -1, 1, -1),
                (1, 1, -1, -1, 1, 1, -1, -1),
                (1, -1, -1, 1, 1, -1, -1, 1),
                (1, 1, 1, 1, -1, -1, -1, -1),
            ),
            device=head.weight.device,
            dtype=head.weight.dtype,
        )
        column_index = torch.arange(head.in_features, device=head.weight.device)
        signs = walsh_rows[:, column_index.remainder(walsh_rows.shape[1])]
        element_magnitude = CRAZYFLIE_INITIAL_HEAD_WEIGHT_SCALE / math.sqrt(head.in_features)
        head.weight.copy_(signs * element_magnitude)

        head.bias.zero_()
        initial_std = torch.tensor(
            CRAZYFLIE_INITIAL_LATENT_STD, device=log_std.device, dtype=log_std.dtype
        )
        log_std.copy_(initial_std.log())

        head_rank = int(
            torch.linalg.matrix_rank(
                head.weight.detach().to(device="cpu", dtype=torch.float64)
            ).item()
        )
        if head_rank != ACTION_DIM:
            raise RuntimeError(
                f"Crazyflie actor output initialization has rank {head_rank}, expected {ACTION_DIM}."
            )

    return {
        "scheme": (
            "shared_stabilization_bounded_residual_v2_full_rank_walsh"
        ),
        "target_deterministic_action": list(CRAZYFLIE_HOVER_ACTION),
        "target_residual_logits": [0.0] * ACTION_DIM,
        "residual_latent_scale": list(CRAZYFLIE_RESIDUAL_LATENT_SCALE),
        "initial_latent_std": list(CRAZYFLIE_INITIAL_LATENT_STD),
        "initial_log_std": [math.log(value) for value in CRAZYFLIE_INITIAL_LATENT_STD],
        "final_head_weight_scale": CRAZYFLIE_INITIAL_HEAD_WEIGHT_SCALE,
        "final_head_element_magnitude": element_magnitude,
        "final_head_sign_pattern": "walsh_h8_rows_1_2_3_4_repeated",
        "final_head_weight_rank": head_rank,
        "final_head_all_weights_nonzero": bool(
            torch.count_nonzero(head.weight) == head.weight.numel()
        ),
    }


def reset_controller_state(
    policy: nn.Module,
    state: LIFState | torch.Tensor | None,
    done: torch.Tensor,
) -> LIFState | torch.Tensor | None:
    """Reset only completed environment rows for any comparison controller."""

    reset = getattr(policy, "reset_state", None)
    if callable(reset):
        return reset(state, done)
    if isinstance(state, LIFState):
        return state.masked_reset(done)
    if isinstance(state, torch.Tensor):
        return state.masked_fill(done.to(device=state.device, dtype=torch.bool)[:, None], 0.0)
    return None


def named_lif_cores(policy: nn.Module) -> tuple[tuple[str, LIFCore], ...]:
    """Return all frozen cores in their canonical recurrent-state order."""

    accessor = getattr(policy, "named_lif_cores", None)
    if not callable(accessor):
        return ()
    values = tuple(accessor())
    labels = tuple(label for label, _core in values)
    if len(set(labels)) != len(labels) or any(
        not isinstance(core, LIFCore) for _label, core in values
    ):
        raise RuntimeError("Invalid named LIF core metadata exposed by controller.")
    return values


def named_lif_core_indices(
    policy: nn.Module,
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    """Return named input/readout indices in canonical core order."""

    accessor = getattr(policy, "named_lif_core_indices", None)
    if not callable(accessor):
        return ()
    values = tuple(accessor())
    if tuple(label for label, _inputs, _outputs in values) != tuple(
        label for label, _core in named_lif_cores(policy)
    ):
        raise RuntimeError("Named LIF index metadata does not match core order.")
    return values


def named_lif_encoders(policy: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Return named trainable encoders in canonical core order."""

    accessor = getattr(policy, "named_lif_encoders", None)
    if not callable(accessor):
        return ()
    values = tuple(accessor())
    if tuple(label for label, _encoder in values) != tuple(
        label for label, _core in named_lif_cores(policy)
    ):
        raise RuntimeError("Named LIF encoder metadata does not match core order.")
    return values


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _actor_parameter_count(policy: nn.Module) -> tuple[int, int]:
    """Return actor count including, and distribution count within, that total."""

    distribution = int(policy.log_std.numel()) if hasattr(policy, "log_std") else 0
    if isinstance(policy, CrazyflieCombinedConnectomeActorCritic):
        count = (
            sum(
                _module_parameter_count(encoder)
                for _label, encoder in named_lif_encoders(policy)
            )
            + _module_parameter_count(policy.decoder)
            + distribution
        )
    elif isinstance(policy, CrazyflieCombinedLegWingActorCritic):
        count = (
            _module_parameter_count(policy.encoder)
            + _module_parameter_count(policy.wing_encoder)
            + _module_parameter_count(policy.decoder)
            + distribution
        )
    elif isinstance(policy, FrozenLIFActorCritic):
        count = (
            _module_parameter_count(policy.encoder)
            + _module_parameter_count(policy.decoder)
            + distribution
        )
    elif isinstance(policy, GRUActorCritic):
        count = _module_parameter_count(policy.gru) + _module_parameter_count(policy.actor) + distribution
    elif isinstance(policy, MLPActorCritic):
        count = _module_parameter_count(policy.actor) + distribution
    else:
        raise TypeError(f"Unsupported controller class for parameter accounting: {type(policy).__name__}")
    return count, distribution


def lif_actor_parameter_target(
    observation_dim: int,
    action_dim: int,
    input_population_size: int,
    output_population_size: int,
    adapter_hidden_dim: int = DEFAULT_ADAPTER_HIDDEN_DIM,
) -> int:
    """Exact LIF adapter plus action-distribution trainable parameter count."""

    dimensions = (
        observation_dim,
        action_dim,
        input_population_size,
        output_population_size,
        adapter_hidden_dim,
    )
    if min(dimensions) < 1:
        raise ValueError("Controller dimensions and population sizes must be positive.")
    # encoder obs->H->input, decoder output->H->action, plus log_std[action]
    return (
        observation_dim * adapter_hidden_dim
        + adapter_hidden_dim
        + adapter_hidden_dim * input_population_size
        + input_population_size
        + output_population_size * adapter_hidden_dim
        + adapter_hidden_dim
        + adapter_hidden_dim * action_dim
        + action_dim
        + action_dim
    )


def _nearest_width(target: int, count_for_width: Any, *, search_limit: int = 4096) -> int:
    if target < 1:
        raise ValueError("Actor parameter target must be positive.")
    # Parameter counts are monotone for these networks; a bounded exact search
    # avoids depending on a rounded quadratic solution.
    best_width = min(range(1, search_limit + 1), key=lambda width: abs(count_for_width(width) - target))
    if best_width == search_limit and count_for_width(best_width) < target:
        raise ValueError("Actor matching width exceeded the supported search range.")
    return best_width


def _gru_actor_count(observation_dim: int, action_dim: int, hidden_dim: int) -> int:
    # GRUCell: 3H*obs + 3H*H + two 3H biases; actor H->action; log_std[action].
    return (
        3 * hidden_dim * observation_dim
        + 3 * hidden_dim * hidden_dim
        + 6 * hidden_dim
        + hidden_dim * action_dim
        + action_dim
        + action_dim
    )


def _two_layer_mlp_actor_count(observation_dim: int, action_dim: int, hidden_dim: int) -> int:
    return (
        observation_dim * hidden_dim
        + hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim
        + hidden_dim * action_dim
        + action_dim
        + action_dim
    )


def _dynamic_state_report(policy: nn.Module) -> dict[str, int]:
    if isinstance(
        policy,
        (CrazyflieCombinedConnectomeActorCritic, CrazyflieCombinedLegWingActorCritic),
    ):
        neurons = int(policy.combined_neuron_count)
        return {
            "continuous_dynamic_state_per_environment": 3 * neurons,
            "discrete_dynamic_state_per_environment": neurons,
            "total_dynamic_state_per_environment": 4 * neurons,
        }
    if isinstance(policy, FrozenLIFActorCritic):
        neurons = int(policy.core.num_neurons)
        return {
            "continuous_dynamic_state_per_environment": 3 * neurons,
            "discrete_dynamic_state_per_environment": neurons,
            "total_dynamic_state_per_environment": 4 * neurons,
        }
    if isinstance(policy, GRUActorCritic):
        hidden = int(policy.hidden_dim)
        return {
            "continuous_dynamic_state_per_environment": hidden,
            "discrete_dynamic_state_per_environment": 0,
            "total_dynamic_state_per_environment": hidden,
        }
    return {
        "continuous_dynamic_state_per_environment": 0,
        "discrete_dynamic_state_per_environment": 0,
        "total_dynamic_state_per_environment": 0,
    }


def controller_parameter_report(
    policy: nn.Module,
    *,
    reference_actor_parameters: int,
    parameter_tolerance: float = DEFAULT_PARAMETER_TOLERANCE,
) -> dict[str, Any]:
    """Report disjoint trainable/frozen counts and recurrent state size."""

    if reference_actor_parameters < 1:
        raise ValueError("reference_actor_parameters must be positive.")
    if not math.isfinite(parameter_tolerance) or parameter_tolerance < 0:
        raise ValueError("parameter_tolerance must be finite and non-negative.")
    actor, distribution = _actor_parameter_count(policy)
    critic = _module_parameter_count(policy.critic)
    total_trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    unassigned = total_trainable - actor - critic
    if unassigned != 0:
        raise RuntimeError(f"Parameter accounting left {unassigned} trainable values unassigned.")
    cores = [core for _label, core in named_lif_cores(policy)]
    frozen_synapses = sum(int(item.weights.numel()) for item in cores)
    graph_index_values = sum(int(item.edge_index.numel()) for item in cores)
    fixed_scalar_buffers = 2 * len(cores)
    population_index_values = sum(
        int(inputs.numel() + outputs.numel())
        for _label, inputs, outputs in named_lif_core_indices(policy)
    )
    derived_dense_values = sum(int(item.dense_recurrent_weights.numel()) for item in cores)
    encoder = sum(
        _module_parameter_count(item) for _label, item in named_lif_encoders(policy)
    )
    decoder = _module_parameter_count(policy.decoder) if isinstance(policy, FrozenLIFActorCritic) else 0
    recurrent_actor = _module_parameter_count(policy.gru) if isinstance(policy, GRUActorCritic) else 0
    action_head = _module_parameter_count(policy.actor) if isinstance(policy, GRUActorCritic) else 0
    feedforward_actor = _module_parameter_count(policy.actor) if isinstance(policy, MLPActorCritic) else 0
    deviation = abs(actor - reference_actor_parameters) / reference_actor_parameters
    return {
        "actor_trainable_parameters": actor,
        "actor_trainable_excluding_distribution": actor - distribution,
        "distribution_trainable_parameters": distribution,
        "encoder_trainable_parameters": encoder,
        "decoder_trainable_parameters": decoder,
        "recurrent_actor_trainable_parameters": recurrent_actor,
        "action_head_trainable_parameters": action_head,
        "feedforward_actor_trainable_parameters": feedforward_actor,
        "critic_trainable_parameters": critic,
        "total_trainable_parameters": total_trainable,
        # Fixed synaptic weights are scientific model values stored as buffers,
        # not optimizer parameters.  Indices and derived dense matrices are
        # reported separately rather than inflating the frozen-model count.
        "frozen_parameters": frozen_synapses,
        "model_total_parameters": total_trainable + frozen_synapses,
        "frozen_synaptic_weights": frozen_synapses,
        "fixed_core_scalar_buffers": fixed_scalar_buffers,
        "graph_index_values": graph_index_values,
        "population_index_values": population_index_values,
        "derived_dense_recurrent_buffer_values": derived_dense_values,
        "total_registered_buffer_values": sum(buffer.numel() for buffer in policy.buffers()),
        "reference_actor_parameters": int(reference_actor_parameters),
        "actor_parameter_deviation_fraction": deviation,
        "actor_parameter_tolerance_fraction": parameter_tolerance,
        "actor_parameter_match_passed": deviation <= parameter_tolerance,
        **_dynamic_state_report(policy),
    }


def _validate_rewire(
    original_edge_index: torch.Tensor,
    original_weights: torch.Tensor,
    rewired_edge_index: torch.Tensor,
    rewired_weights: torch.Tensor,
    *,
    num_neurons: int,
) -> dict[str, bool]:
    if (
        rewired_edge_index.shape != original_edge_index.shape
        or rewired_weights.shape != original_weights.shape
    ):
        raise ValueError("Rewired graph dimensions differ from the source graph.")
    # The project algorithm swaps only destinations.  Keeping source and weight
    # positions paired preserves every source neuron's outgoing sign/weight
    # multiset in addition to the global multiset.
    source_preserved = torch.equal(rewired_edge_index[0].cpu(), original_edge_index[0].cpu())
    weights_preserved = torch.equal(rewired_weights.cpu(), original_weights.cpu())
    out_degree_preserved = torch.equal(
        rewired_edge_index[0].cpu().bincount(minlength=num_neurons),
        original_edge_index[0].cpu().bincount(minlength=num_neurons),
    )
    in_degree_preserved = torch.equal(
        rewired_edge_index[1].cpu().bincount(minlength=num_neurons),
        original_edge_index[1].cpu().bincount(minlength=num_neurons),
    )
    pairs = list(zip(rewired_edge_index[0].cpu().tolist(), rewired_edge_index[1].cpu().tolist(), strict=True))
    no_duplicates = len(pairs) == len(set(pairs))
    no_self_loops = all(source != target for source, target in pairs)
    topology_changed = not torch.equal(rewired_edge_index.cpu(), original_edge_index.cpu())
    invariants = {
        "source_indices_preserved": source_preserved,
        "directed_out_degree_preserved": out_degree_preserved,
        "directed_in_degree_preserved": in_degree_preserved,
        "weight_multiset_preserved": weights_preserved,
        "per_source_weight_and_sign_multisets_preserved": source_preserved and weights_preserved,
        "no_duplicate_edges": no_duplicates,
        "no_self_loops": no_self_loops,
        "topology_changed": topology_changed,
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"Degree-preserving rewire failed invariants: {', '.join(failed)}")
    return invariants


def _rewire_manifest(
    circuit: CircuitData,
    rewired_edge_index: torch.Tensor,
    rewired_weights: torch.Tensor,
    report: Any,
    invariants: Mapping[str, bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "directed_destination_double_edge_swap_v1",
        "source_connectome_checksum": circuit.checksum,
        "source_manifest_fingerprint": circuit.manifest.fingerprint,
        "seed": int(report.seed),
        "requested_swaps": int(report.requested_swaps),
        "completed_swaps": int(report.completed_swaps),
        "attempts": int(report.attempts),
        "allow_self_loops": False,
        "allow_duplicates": False,
        "num_neurons": circuit.num_neurons,
        "input_population_size": len(circuit.manifest.input_neuron_ids),
        "output_population_size": len(circuit.manifest.output_neuron_ids),
        "invariants": dict(invariants),
        # Store the exact result, not just the seed, so the primary comparison
        # can freeze and audit one concrete rewire across all five seeds.
        "edge_index": rewired_edge_index.detach().cpu().tolist(),
        "weights": [float(value) for value in rewired_weights.detach().cpu().tolist()],
    }
    payload["checksum"] = sha256(_canonical_json(payload)).hexdigest()
    return payload


def rewire_manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Return the canonical checksum of a manifest, excluding its checksum field."""

    payload = dict(manifest)
    payload.pop("checksum", None)
    return sha256(_canonical_json(payload)).hexdigest()


def generate_rewire_manifest(
    circuit: CircuitData,
    *,
    seed: int,
    swaps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate and fully validate one deterministic directed rewire."""

    edge_index, weights, report = degree_preserving_rewire(
        circuit.edge_index,
        circuit.weights,
        seed=seed,
        swaps=swaps,
        allow_self_loops=False,
        allow_duplicates=False,
    )
    if report.completed_swaps != report.requested_swaps:
        raise RuntimeError(
            "Degree-preserving rewiring did not complete its predeclared swap count: "
            f"{report.completed_swaps}/{report.requested_swaps}."
        )
    invariants = _validate_rewire(
        circuit.edge_index,
        circuit.weights,
        edge_index,
        weights,
        num_neurons=circuit.num_neurons,
    )
    return edge_index, weights, _rewire_manifest(circuit, edge_index, weights, report, invariants)


def validate_rewire_manifest(
    manifest: Mapping[str, Any],
    circuit: CircuitData,
    *,
    expected_seed: int | None = None,
    expected_swaps: int | None = None,
    verify_fresh_generation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Validate a frozen rewire artifact and return its exact tensors.

    Normal training validates the content checksum and every graph invariant
    without regenerating 51,030 swaps.  Acceptance tests additionally set
    ``verify_fresh_generation=True`` to compare the entire checked-in payload
    with a new deterministic generation.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("Rewire manifest must be a JSON object.")
    required = {
        "schema_version",
        "algorithm",
        "source_connectome_checksum",
        "source_manifest_fingerprint",
        "seed",
        "requested_swaps",
        "completed_swaps",
        "attempts",
        "allow_self_loops",
        "allow_duplicates",
        "num_neurons",
        "input_population_size",
        "output_population_size",
        "invariants",
        "edge_index",
        "weights",
        "checksum",
    }
    missing = sorted(required - manifest.keys())
    unknown = sorted(set(manifest) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError("Invalid rewire manifest fields: " + ", ".join(details))
    if manifest["schema_version"] != 1:
        raise ValueError(f"Unsupported rewire schema version: {manifest['schema_version']!r}")
    if manifest["algorithm"] != "directed_destination_double_edge_swap_v1":
        raise ValueError(f"Unsupported rewire algorithm: {manifest['algorithm']!r}")
    stored_checksum = manifest["checksum"]
    actual_checksum = rewire_manifest_checksum(manifest)
    if not isinstance(stored_checksum, str) or stored_checksum != actual_checksum:
        raise ValueError(
            f"Rewire manifest checksum mismatch: stored={stored_checksum!r}, actual={actual_checksum!r}."
        )
    if manifest["source_connectome_checksum"] != circuit.checksum:
        raise ValueError("Rewire manifest source connectome checksum does not match the loaded circuit.")
    if manifest["source_manifest_fingerprint"] != circuit.manifest.fingerprint:
        raise ValueError("Rewire manifest source provenance fingerprint does not match the loaded circuit.")
    integer_fields = (
        "seed",
        "requested_swaps",
        "completed_swaps",
        "attempts",
        "num_neurons",
        "input_population_size",
        "output_population_size",
    )
    if any(type(manifest[field]) is not int for field in integer_fields):
        raise ValueError("Rewire manifest count and seed fields must be integers.")
    if expected_seed is not None and manifest["seed"] != expected_seed:
        raise ValueError(
            f"Rewire manifest seed {manifest['seed']} does not match requested seed {expected_seed}."
        )
    if expected_swaps is not None and manifest["requested_swaps"] != expected_swaps:
        raise ValueError(
            "Rewire manifest requested swap count does not match the configured count: "
            f"{manifest['requested_swaps']} != {expected_swaps}."
        )
    if (
        manifest["requested_swaps"] < 1
        or manifest["completed_swaps"] != manifest["requested_swaps"]
        or manifest["attempts"] < manifest["completed_swaps"]
    ):
        raise ValueError("Rewire manifest records an incomplete or impossible swap run.")
    if manifest["allow_self_loops"] is not False or manifest["allow_duplicates"] is not False:
        raise ValueError("Primary rewire manifest must disallow self-loops and duplicate edges.")
    if manifest["num_neurons"] != circuit.num_neurons:
        raise ValueError("Rewire manifest neuron count does not match the loaded circuit.")
    if manifest["input_population_size"] != len(circuit.manifest.input_neuron_ids):
        raise ValueError("Rewire manifest input population size does not match the loaded circuit.")
    if manifest["output_population_size"] != len(circuit.manifest.output_neuron_ids):
        raise ValueError("Rewire manifest output population size does not match the loaded circuit.")

    raw_edges = manifest["edge_index"]
    raw_weights = manifest["weights"]
    if (
        not isinstance(raw_edges, list)
        or len(raw_edges) != 2
        or not all(isinstance(row, list) for row in raw_edges)
        or not isinstance(raw_weights, list)
    ):
        raise ValueError("Rewire manifest edge_index must be two lists and weights must be a list.")
    if not all(type(value) is int for row in raw_edges for value in row):
        raise ValueError("Rewire manifest edge indices must be integers.")
    if not all(type(value) in {int, float} for value in raw_weights):
        raise ValueError("Rewire manifest weights must be numeric.")
    try:
        edge_index = torch.tensor(raw_edges, dtype=torch.long)
        weights = torch.tensor(raw_weights, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Rewire manifest tensors could not be decoded.") from exc
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] != weights.numel():
        raise ValueError("Rewire manifest edge and weight dimensions are inconsistent.")
    if not torch.isfinite(weights).all():
        raise ValueError("Rewire manifest contains nonfinite weights.")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= circuit.num_neurons):
        raise ValueError("Rewire manifest contains an out-of-range neuron index.")
    try:
        invariants = _validate_rewire(
            circuit.edge_index,
            circuit.weights,
            edge_index,
            weights,
            num_neurons=circuit.num_neurons,
        )
    except RuntimeError as exc:
        raise ValueError(f"Frozen rewire graph failed validation: {exc}") from exc
    if not isinstance(manifest["invariants"], Mapping) or dict(manifest["invariants"]) != invariants:
        raise ValueError("Rewire manifest invariant declarations do not match recomputed invariants.")

    normalized = dict(manifest)
    if verify_fresh_generation:
        fresh_edges, fresh_weights, fresh_manifest = generate_rewire_manifest(
            circuit,
            seed=manifest["seed"],
            swaps=manifest["requested_swaps"],
        )
        if not torch.equal(edge_index, fresh_edges) or not torch.equal(weights, fresh_weights):
            raise ValueError("Frozen rewire tensors differ from fresh deterministic generation.")
        if _canonical_json(normalized) != _canonical_json(fresh_manifest):
            raise ValueError("Frozen rewire manifest differs from fresh deterministic generation.")
    return edge_index, weights, normalized


def load_rewire_manifest(
    path: str | Path,
    circuit: CircuitData,
    *,
    expected_seed: int | None = None,
    expected_swaps: int | None = None,
    expected_file_sha256: str | None = None,
    verify_fresh_generation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load a frozen rewire JSON artifact with file and content validation."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Frozen rewire manifest does not exist: {source}")
    if expected_file_sha256 is not None:
        actual_file_sha256 = file_sha256(source)
        if actual_file_sha256 != expected_file_sha256:
            raise ValueError(
                "Frozen rewire file checksum mismatch: "
                f"expected={expected_file_sha256}, actual={actual_file_sha256}."
            )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in frozen rewire manifest {source}: {exc}") from exc
    return validate_rewire_manifest(
        raw,
        circuit,
        expected_seed=expected_seed,
        expected_swaps=expected_swaps,
        verify_fresh_generation=verify_fresh_generation,
    )


def _lif_constants(circuit: CircuitData) -> dict[str, Any]:
    allowed = {
        "dt",
        "tau_membrane",
        "tau_synapse",
        "threshold",
        "reset_value",
        "refractory_steps",
        "surrogate_beta",
        "neural_substeps",
    }
    return {key: value for key, value in circuit.manifest.neuron_model.items() if key in allowed}


def build_controller(
    kind: str,
    observation_dim: int = OBSERVATION_DIM,
    action_dim: int = ACTION_DIM,
    device: torch.device | str | None = None,
    connectome_manifest: str | Path | None = None,
    rewire_seed: int = DEFAULT_REWIRE_SEED,
    widths: Mapping[str, Any] | None = None,
    *,
    wing_connectome_manifest: str | Path | None = None,
    optic_connectome_manifest: str | Path | None = None,
    rewire_swaps: int | None = None,
    rewire_manifest_path: str | Path | None = None,
    parameter_tolerance: float = DEFAULT_PARAMETER_TOLERANCE,
    enforce_parameter_match: bool = True,
    allow_synthetic: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a baseline comparison policy or an additive wing extension.

    ``widths`` may contain ``adapter_hidden_dim``, ``critic_hidden_dim``,
    ``gru_hidden_dim``, or ``mlp_hidden_dims``.  When baseline actor widths are
    omitted, the nearest integer width to the LIF actor budget is selected.
    Synthetic circuit manifests remain rejected unless a unit test explicitly
    opts in with ``allow_synthetic=True``.  The primary real-graph rewire loads
    its checked-in exact manifest by default; alternate seeds or test graphs
    are generated only when no frozen manifest path applies.
    """

    try:
        canonical_kind = _KIND_ALIASES[kind.strip().lower()]
    except (AttributeError, KeyError) as exc:
        choices = CONTROLLER_KINDS + WING_EXTENSION_CONTROLLER_KINDS + COMBINATION_CONTROLLER_KINDS
        raise ValueError(f"Unknown controller kind {kind!r}; choose one of {choices}.") from exc
    if observation_dim < 1 or action_dim < 1:
        raise ValueError("observation_dim and action_dim must be positive.")
    if not isinstance(rewire_seed, int) or isinstance(rewire_seed, bool):
        raise TypeError("rewire_seed must be an integer.")
    if rewire_manifest_path is not None and canonical_kind != "frozen_lif_rewired":
        raise ValueError("rewire_manifest_path is only valid for the rewired frozen-LIF controller.")
    settings = dict(widths or {})
    allowed_widths = {"adapter_hidden_dim", "critic_hidden_dim", "gru_hidden_dim", "mlp_hidden_dims"}
    unknown_widths = sorted(set(settings) - allowed_widths)
    if unknown_widths:
        raise ValueError(f"Unknown controller width settings: {', '.join(unknown_widths)}")
    adapter_hidden_dim = int(settings.get("adapter_hidden_dim", DEFAULT_ADAPTER_HIDDEN_DIM))
    critic_hidden_dim = int(settings.get("critic_hidden_dim", DEFAULT_CRITIC_HIDDEN_DIM))
    if adapter_hidden_dim < 1 or critic_hidden_dim < 1:
        raise ValueError("Adapter and critic hidden dimensions must be positive.")

    leg_manifest_path = (
        Path(connectome_manifest) if connectome_manifest is not None else _default_connectome_manifest()
    )
    wing_manifest_path = (
        Path(wing_connectome_manifest)
        if wing_connectome_manifest is not None
        else default_wing_connectome_manifest()
    )
    optic_manifest_path = (
        Path(optic_connectome_manifest)
        if optic_connectome_manifest is not None
        else default_optic_connectome_manifest()
    )
    core_manifest_paths = {
        "leg": leg_manifest_path,
        "wing": wing_manifest_path,
        "optic": optic_manifest_path,
    }
    combination_labels = _COMBINATION_CORE_LABELS.get(canonical_kind)
    if combination_labels is not None:
        combination_circuits = {
            label: load_connectome(core_manifest_paths[label], allow_synthetic=allow_synthetic)
            for label in combination_labels
        }
        circuit = combination_circuits[combination_labels[0]]
    else:
        manifest_path = (
            wing_manifest_path
            if canonical_kind == "wing_lif"
            else optic_manifest_path
            if canonical_kind == "optic_lif"
            else leg_manifest_path
        )
        circuit = load_connectome(manifest_path, allow_synthetic=allow_synthetic)
        combination_circuits = {}
    wing_circuit = (
        load_connectome(wing_manifest_path, allow_synthetic=allow_synthetic)
        if canonical_kind == "leg_wing_lif"
        else None
    )
    input_size = len(circuit.manifest.input_neuron_ids)
    output_size = len(circuit.manifest.output_neuron_ids)
    reference_actor_parameters = lif_actor_parameter_target(
        observation_dim, action_dim, input_size, output_size, adapter_hidden_dim
    )
    target_device = torch.device("cpu") if device is None else torch.device(device)
    rewire_manifest: dict[str, Any] | None = None
    resolved_rewire_manifest_path: Path | None = None
    rewire_manifest_source: str | None = None
    rewire_manifest_file_sha256: str | None = None

    lif_controller_kinds = {
        "frozen_lif",
        "frozen_lif_rewired",
        "wing_lif",
        "leg_wing_lif",
        "optic_lif",
        *COMBINATION_CONTROLLER_KINDS,
    }
    if canonical_kind in lif_controller_kinds:
        edge_index = circuit.edge_index
        weights = circuit.weights
        if canonical_kind == "frozen_lif_rewired":
            if rewire_manifest_path is not None:
                resolved_rewire_manifest_path = Path(rewire_manifest_path).expanduser().resolve()
            elif (
                rewire_seed == DEFAULT_REWIRE_SEED
                and rewire_swaps is None
                and circuit.manifest.path == _default_connectome_manifest().resolve()
            ):
                resolved_rewire_manifest_path = default_rewire_manifest_path().resolve()
            if resolved_rewire_manifest_path is not None:
                expected_file_checksum = (
                    DEFAULT_REWIRE_MANIFEST_FILE_SHA256
                    if resolved_rewire_manifest_path == default_rewire_manifest_path().resolve()
                    else None
                )
                edge_index, weights, rewire_manifest = load_rewire_manifest(
                    resolved_rewire_manifest_path,
                    circuit,
                    expected_seed=rewire_seed,
                    expected_swaps=rewire_swaps,
                    expected_file_sha256=expected_file_checksum,
                )
                rewire_manifest_source = "frozen_artifact"
                rewire_manifest_file_sha256 = file_sha256(resolved_rewire_manifest_path)
            else:
                edge_index, weights, rewire_manifest = generate_rewire_manifest(
                    circuit,
                    seed=rewire_seed,
                    swaps=rewire_swaps,
                )
                rewire_manifest_source = "generated"
        core = LIFCore(
            circuit.num_neurons,
            edge_index.to(target_device),
            weights.to(target_device),
            **_lif_constants(circuit),
        ).to(target_device)
        index_of = {neuron_id: index for index, neuron_id in enumerate(circuit.neuron_ids)}
        if combination_labels is not None:
            named_cores: dict[str, LIFCore] = {}
            named_inputs: dict[str, list[int]] = {}
            named_outputs: dict[str, list[int]] = {}
            input_size = 0
            output_size = 0
            for label in combination_labels:
                named_circuit = combination_circuits[label]
                named_core = (
                    core
                    if label == combination_labels[0]
                    else LIFCore(
                        named_circuit.num_neurons,
                        named_circuit.edge_index.to(target_device),
                        named_circuit.weights.to(target_device),
                        **_lif_constants(named_circuit),
                    ).to(target_device)
                )
                named_index_of = {
                    neuron_id: index
                    for index, neuron_id in enumerate(named_circuit.neuron_ids)
                }
                named_cores[label] = named_core
                named_inputs[label] = [
                    named_index_of[neuron_id]
                    for neuron_id in named_circuit.manifest.input_neuron_ids
                ]
                named_outputs[label] = [
                    named_index_of[neuron_id]
                    for neuron_id in named_circuit.manifest.output_neuron_ids
                ]
                input_size += len(named_inputs[label])
                output_size += len(named_outputs[label])
            policy = CrazyflieCombinedConnectomeActorCritic(
                observation_dim,
                action_dim,
                named_cores,
                input_indices=named_inputs,
                output_indices=named_outputs,
                adapter_hidden_dim=adapter_hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(target_device)
        elif canonical_kind == "leg_wing_lif":
            if wing_circuit is None:
                raise RuntimeError("Combined controller is missing its wing circuit")
            wing_core = LIFCore(
                wing_circuit.num_neurons,
                wing_circuit.edge_index.to(target_device),
                wing_circuit.weights.to(target_device),
                **_lif_constants(wing_circuit),
            ).to(target_device)
            wing_index_of = {
                neuron_id: index for index, neuron_id in enumerate(wing_circuit.neuron_ids)
            }
            policy = CrazyflieCombinedLegWingActorCritic(
                observation_dim,
                action_dim,
                core,
                wing_core,
                leg_input_indices=[
                    index_of[neuron_id] for neuron_id in circuit.manifest.input_neuron_ids
                ],
                leg_output_indices=[
                    index_of[neuron_id] for neuron_id in circuit.manifest.output_neuron_ids
                ],
                wing_input_indices=[
                    wing_index_of[neuron_id]
                    for neuron_id in wing_circuit.manifest.input_neuron_ids
                ],
                wing_output_indices=[
                    wing_index_of[neuron_id]
                    for neuron_id in wing_circuit.manifest.output_neuron_ids
                ],
                adapter_hidden_dim=adapter_hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(target_device)
            input_size += len(wing_circuit.manifest.input_neuron_ids)
            output_size += len(wing_circuit.manifest.output_neuron_ids)
        else:
            policy = CrazyflieFrozenLIFActorCritic(
                observation_dim,
                action_dim,
                core,
                input_indices=[index_of[neuron_id] for neuron_id in circuit.manifest.input_neuron_ids],
                output_indices=[index_of[neuron_id] for neuron_id in circuit.manifest.output_neuron_ids],
                adapter_hidden_dim=adapter_hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(target_device)
    elif canonical_kind == "gru":
        configured = settings.get("gru_hidden_dim")
        hidden_dim = int(configured) if configured is not None else _nearest_width(
            reference_actor_parameters,
            lambda width: _gru_actor_count(observation_dim, action_dim, width),
        )
        policy = MatchedGRUActorCritic(
            observation_dim,
            action_dim,
            hidden_dim=hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
        ).to(target_device)
    else:
        configured = settings.get("mlp_hidden_dims")
        if configured is None:
            hidden_dim = _nearest_width(
                reference_actor_parameters,
                lambda width: _two_layer_mlp_actor_count(observation_dim, action_dim, width),
            )
            hidden_dims = (hidden_dim, hidden_dim)
        elif isinstance(configured, int):
            hidden_dims = (configured, configured)
        else:
            hidden_dims = tuple(int(width) for width in configured)
        policy = MatchedMLPActorCritic(
            observation_dim,
            action_dim,
            hidden_dims=hidden_dims,
            critic_hidden_dim=critic_hidden_dim,
        ).to(target_device)

    actor_initialization = initialize_crazyflie_actor(policy)
    counts = controller_parameter_report(
        policy,
        reference_actor_parameters=reference_actor_parameters,
        parameter_tolerance=parameter_tolerance,
    )
    if enforce_parameter_match and canonical_kind in CONTROLLER_KINDS and not counts["actor_parameter_match_passed"]:
        raise ValueError(
            f"{canonical_kind} actor has {counts['actor_trainable_parameters']} trainable parameters, "
            f"outside {parameter_tolerance:.1%} of the LIF reference {reference_actor_parameters}."
        )
    core_checksum = controller_core_checksum(policy)
    if combination_labels is not None:
        connectome_manifests = {
            label: str(combination_circuits[label].manifest.path)
            for label in combination_labels
        }
        connectome_checksums = {
            label: combination_circuits[label].checksum for label in combination_labels
        }
        connectome_fingerprints = {
            label: combination_circuits[label].manifest.fingerprint
            for label in combination_labels
        }
        combined_connectome_checksum = sha256(
            _canonical_json(connectome_checksums)
        ).hexdigest()
    elif wing_circuit is not None:
        connectome_manifests = {
            "leg": str(circuit.manifest.path),
            "wing": str(wing_circuit.manifest.path),
        }
        connectome_checksums = {"leg": circuit.checksum, "wing": wing_circuit.checksum}
        connectome_fingerprints = {
            "leg": circuit.manifest.fingerprint,
            "wing": wing_circuit.manifest.fingerprint,
        }
        combined_connectome_checksum = sha256(
            _canonical_json(connectome_checksums)
        ).hexdigest()
    else:
        connectome_manifests = {"primary": str(circuit.manifest.path)}
        connectome_checksums = {"primary": circuit.checksum}
        connectome_fingerprints = {"primary": circuit.manifest.fingerprint}
        combined_connectome_checksum = circuit.checksum
    policy_cores = named_lif_cores(policy)
    policy_core_indices = named_lif_core_indices(policy)
    report: dict[str, Any] = {
        "controller_kind": canonical_kind,
        "observation_dim": int(observation_dim),
        "action_dim": int(action_dim),
        "action_bounds": [-1.0, 1.0],
        "stabilization_and_residual_contract": stabilization_contract_payload(),
        "actor_initialization": actor_initialization,
        "recurrent": bool(getattr(policy, "is_recurrent", False)),
        "widths": {
            "adapter_hidden_dim": (
                adapter_hidden_dim
                if canonical_kind in lif_controller_kinds
                else None
            ),
            "critic_hidden_dim": critic_hidden_dim,
            "gru_hidden_dim": int(policy.hidden_dim) if isinstance(policy, GRUActorCritic) else None,
            "mlp_hidden_dims": (
                list(policy.hidden_dims) if isinstance(policy, MatchedMLPActorCritic) else None
            ),
        },
        "connectome_manifest": str(circuit.manifest.path),
        "connectome_checksum": combined_connectome_checksum,
        "connectome_manifest_fingerprint": circuit.manifest.fingerprint,
        "connectome_manifests": connectome_manifests,
        "connectome_checksums": connectome_checksums,
        "connectome_manifest_fingerprints": connectome_fingerprints,
        "input_population_size": input_size,
        "output_population_size": output_size,
        "core_checksum": core_checksum,
        "legacy_core_checksum": (
            policy.core.frozen_checksum
            if isinstance(policy, FrozenLIFActorCritic)
            and len(policy_cores) == 1
            else None
        ),
        "core_labels": [label for label, _core in policy_cores],
        "per_core_checksums": {
            label: _lif_core_checksum(item) for label, item in policy_cores
        },
        "per_core_population_indices": {
            label: {
                "input_indices": inputs.detach().cpu().tolist(),
                "output_indices": outputs.detach().cpu().tolist(),
            }
            for label, inputs, outputs in policy_core_indices
        },
        "fusion_contract": getattr(policy, "fusion_contract", None),
        "parameter_matching_required": canonical_kind in CONTROLLER_KINDS,
        "rewire_seed": rewire_seed if canonical_kind == "frozen_lif_rewired" else None,
        "rewire_manifest_checksum": rewire_manifest["checksum"] if rewire_manifest is not None else None,
        "rewire_manifest_path": (
            str(resolved_rewire_manifest_path) if resolved_rewire_manifest_path is not None else None
        ),
        "rewire_manifest_file_sha256": rewire_manifest_file_sha256,
        "rewire_manifest_source": rewire_manifest_source,
        "rewire_manifest": rewire_manifest,
        **counts,
    }
    # These are fixed buffers, never optimizer parameters.  Assert this at the
    # construction boundary instead of relying only on a reported count.
    if isinstance(policy, FrozenLIFActorCritic):
        for label, item in policy_cores:
            if any(True for _ in item.parameters()):
                raise RuntimeError(
                    f"Frozen {label} LIF core unexpectedly exposes optimizer parameters."
                )
            if item.weights.requires_grad:
                raise RuntimeError(
                    f"Frozen {label} LIF weights unexpectedly require gradients."
                )
    return policy, report


__all__ = [
    "ACTION_DIM",
    "COMBINATION_CONTROLLER_KINDS",
    "CONTROLLER_KINDS",
    "WING_EXTENSION_CONTROLLER_KINDS",
    "CRAZYFLIE_HOVER_ACTION",
    "CRAZYFLIE_INITIAL_HEAD_WEIGHT_SCALE",
    "CRAZYFLIE_INITIAL_LATENT_STD",
    "CRAZYFLIE_RESIDUAL_LATENT_SCALE",
    "CRAZYFLIE_THRUST_TO_WEIGHT",
    "DEFAULT_PARAMETER_TOLERANCE",
    "DEFAULT_REWIRE_MANIFEST_FILE_SHA256",
    "DEFAULT_REWIRE_SEED",
    "OBSERVATION_DIM",
    "CrazyflieFrozenLIFActorCritic",
    "CrazyflieCombinedConnectomeActorCritic",
    "CrazyflieCombinedLegWingActorCritic",
    "MatchedGRUActorCritic",
    "MatchedMLPActorCritic",
    "build_controller",
    "controller_core_checksum",
    "controller_parameter_report",
    "default_rewire_manifest_path",
    "default_optic_connectome_manifest",
    "default_wing_connectome_manifest",
    "generate_rewire_manifest",
    "initialize_crazyflie_actor",
    "lif_actor_parameter_target",
    "load_rewire_manifest",
    "named_lif_core_indices",
    "named_lif_cores",
    "named_lif_encoders",
    "reset_controller_state",
    "rewire_manifest_checksum",
    "validate_rewire_manifest",
    "verify_frozen_core",
]
