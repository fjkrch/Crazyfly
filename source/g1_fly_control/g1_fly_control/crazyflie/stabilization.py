"""Fixed Crazyflie stabilization prior and bounded residual composition.

The policy-visible observation remains exactly twelve values.  It is scaled by
the fixed physics scales declared below, so the stabilizer can reconstruct the
corresponding physical values without mutable running statistics.  The prior
uses only velocity, angular velocity, and projected gravity; changing the goal
coordinates can therefore never change the fixed command.

The learned controller supplies a bounded residual in *latent* action space.
Adding it to the fixed prior before the existing tanh-Gaussian transform
preserves an exact, replayable log probability while retaining the native
four-action Crazyflie wrench contract.
"""

from __future__ import annotations

from typing import Any

import torch


OBSERVATION_WIDTH = 12
ACTION_WIDTH = 4
FIXED_OBSERVATION_SCALE = (
    2.0,
    2.0,
    2.0,
    5.0,
    5.0,
    5.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
)
FIXED_OBSERVATION_CLIP = 5.0

CRAZYFLIE_THRUST_TO_WEIGHT = 1.9
CRAZYFLIE_HOVER_ACTION = 2.0 / CRAZYFLIE_THRUST_TO_WEIGHT - 1.0
CRAZYFLIE_RESIDUAL_LATENT_SCALE = (0.35, 0.08, 0.08, 0.05)

# The native action range is closed, but atanh is not finite at +/-1.  This is
# only an inverse-transform guard: ordinary stabilizer commands are unchanged.
PRIOR_ACTION_ATANH_EPS = 1.0e-4


def _check_last_dimension(values: torch.Tensor, width: int, name: str) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.ndim < 1 or values.shape[-1] != width:
        raise ValueError(f"{name} must end in exactly {width} values")


def _constant_like(values: torch.Tensor, constants: tuple[float, ...]) -> torch.Tensor:
    return values.new_tensor(constants)


def normalize_physical_observation(observation: torch.Tensor) -> torch.Tensor:
    """Apply the immutable physics scaling and symmetric clip."""

    _check_last_dimension(observation, OBSERVATION_WIDTH, "observation")
    scale = _constant_like(observation, FIXED_OBSERVATION_SCALE)
    return (observation / scale).clamp(
        -FIXED_OBSERVATION_CLIP, FIXED_OBSERVATION_CLIP
    )


def reconstruct_physical_observation(normalized_observation: torch.Tensor) -> torch.Tensor:
    """Invert the fixed scaling for observations already inside the clip."""

    _check_last_dimension(
        normalized_observation, OBSERVATION_WIDTH, "normalized_observation"
    )
    scale = _constant_like(normalized_observation, FIXED_OBSERVATION_SCALE)
    return normalized_observation * scale


def crazyflie_stabilization_prior_action(
    normalized_observation: torch.Tensor,
) -> torch.Tensor:
    """Return the shared goal-independent normalized-wrench prior.

    Observation indices are the installed task contract:
    ``v_b[0:3], omega_b[3:6], projected_gravity_b[6:9], goal_b[9:12]``.
    The final three values are deliberately not read.
    """

    _check_last_dimension(
        normalized_observation, OBSERVATION_WIDTH, "normalized_observation"
    )
    # Spell out the three homogeneous scale groups instead of constructing a
    # device tensor here.  Besides documenting the physical reconstruction,
    # this keeps deterministic evaluation CUDA-Graph-capturable (creating a
    # new CUDA constant from host values during capture is forbidden).
    velocity_b = normalized_observation[..., 0:3] * 2.0
    angular_velocity_b = normalized_observation[..., 3:6] * 5.0
    gravity_b = normalized_observation[..., 6:9]

    collective = (
        CRAZYFLIE_HOVER_ACTION
        - 0.18 * velocity_b[..., 2]
        + 0.20 * (1.0 + gravity_b[..., 2])
    )
    roll_moment = (
        0.08 * gravity_b[..., 1]
        - 0.02 * angular_velocity_b[..., 0]
        + 0.015 * velocity_b[..., 1]
    )
    pitch_moment = (
        -0.08 * gravity_b[..., 0]
        - 0.02 * angular_velocity_b[..., 1]
        - 0.015 * velocity_b[..., 0]
    )
    yaw_moment = -0.01 * angular_velocity_b[..., 2]
    prior = torch.stack(
        (collective, roll_moment, pitch_moment, yaw_moment), dim=-1
    )
    limit = 1.0 - PRIOR_ACTION_ATANH_EPS
    return prior.clamp(-limit, limit)


def crazyflie_stabilization_prior_latent(
    normalized_observation: torch.Tensor,
) -> torch.Tensor:
    """Return the finite inverse-tanh latent of the fixed prior action."""

    return torch.atanh(crazyflie_stabilization_prior_action(normalized_observation))


def compose_crazyflie_residual_mean(
    normalized_observation: torch.Tensor,
    residual_logits: torch.Tensor,
    *,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose the fixed flight prior and bounded residual into a latent mean."""

    _check_last_dimension(
        normalized_observation, OBSERVATION_WIDTH, "normalized_observation"
    )
    _check_last_dimension(residual_logits, ACTION_WIDTH, "residual_logits")
    if residual_logits.shape[:-1] != normalized_observation.shape[:-1]:
        raise ValueError("observation and residual batch dimensions must match")
    if residual_scale is None:
        residual_scale = _constant_like(
            residual_logits, CRAZYFLIE_RESIDUAL_LATENT_SCALE
        )
    elif residual_scale.shape != (ACTION_WIDTH,):
        raise ValueError("residual_scale must contain exactly four values")
    scale = residual_scale.to(
        device=residual_logits.device, dtype=residual_logits.dtype
    )
    return crazyflie_stabilization_prior_latent(normalized_observation) + (
        scale * torch.tanh(residual_logits)
    )


def stabilization_contract_payload() -> dict[str, Any]:
    """Return the canonical JSON-safe controller/normalization contract."""

    return {
        "version": "crazyflie_shared_stabilization_bounded_residual_v2",
        "observation_normalization": {
            "kind": "fixed_physics_scale",
            "scale": list(FIXED_OBSERVATION_SCALE),
            "clip": FIXED_OBSERVATION_CLIP,
            "updates": False,
        },
        "prior": {
            "goal_independent": True,
            "used_observation_indices": list(range(9)),
            "ignored_goal_indices": [9, 10, 11],
            "collective": "hover - 0.18*v_z + 0.20*(1 + gravity_z)",
            "roll_moment": "0.08*gravity_y - 0.02*omega_x + 0.015*v_y",
            "pitch_moment": "-0.08*gravity_x - 0.02*omega_y - 0.015*v_x",
            "yaw_moment": "-0.01*omega_z",
            "action_clamp": [
                -1.0 + PRIOR_ACTION_ATANH_EPS,
                1.0 - PRIOR_ACTION_ATANH_EPS,
            ],
            "hover_action": CRAZYFLIE_HOVER_ACTION,
        },
        "residual": {
            "latent_scale": list(CRAZYFLIE_RESIDUAL_LATENT_SCALE),
            "composition": "atanh(prior_action) + scale*tanh(residual_logits)",
            "final_action": "tanh(Normal(composed_latent_mean, exp(log_std)))",
        },
    }


__all__ = [
    "ACTION_WIDTH",
    "CRAZYFLIE_HOVER_ACTION",
    "CRAZYFLIE_RESIDUAL_LATENT_SCALE",
    "CRAZYFLIE_THRUST_TO_WEIGHT",
    "FIXED_OBSERVATION_CLIP",
    "FIXED_OBSERVATION_SCALE",
    "OBSERVATION_WIDTH",
    "PRIOR_ACTION_ATANH_EPS",
    "compose_crazyflie_residual_mean",
    "crazyflie_stabilization_prior_action",
    "crazyflie_stabilization_prior_latent",
    "normalize_physical_observation",
    "reconstruct_physical_observation",
    "stabilization_contract_payload",
]
