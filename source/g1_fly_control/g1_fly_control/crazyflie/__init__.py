"""Crazyflie-specific controller and training-state interfaces.

This package is deliberately separate from the paused Unitree G1 task.  It may
reuse frozen, tested primitives from :mod:`g1_fly_control.policies`, but it does
not alter their behavior.
"""

from .controllers import (
    CONTROLLER_KINDS,
    DEFAULT_REWIRE_MANIFEST_FILE_SHA256,
    DEFAULT_REWIRE_SEED,
    CrazyflieFrozenLIFActorCritic,
    MatchedGRUActorCritic,
    MatchedMLPActorCritic,
    build_controller,
    controller_core_checksum,
    controller_parameter_report,
    default_rewire_manifest_path,
    generate_rewire_manifest,
    load_rewire_manifest,
    reset_controller_state,
    rewire_manifest_checksum,
    validate_rewire_manifest,
    verify_frozen_core,
)

__all__ = [
    "CONTROLLER_KINDS",
    "DEFAULT_REWIRE_MANIFEST_FILE_SHA256",
    "DEFAULT_REWIRE_SEED",
    "CrazyflieFrozenLIFActorCritic",
    "MatchedGRUActorCritic",
    "MatchedMLPActorCritic",
    "build_controller",
    "controller_core_checksum",
    "controller_parameter_report",
    "default_rewire_manifest_path",
    "generate_rewire_manifest",
    "load_rewire_manifest",
    "reset_controller_state",
    "rewire_manifest_checksum",
    "validate_rewire_manifest",
    "verify_frozen_core",
]
