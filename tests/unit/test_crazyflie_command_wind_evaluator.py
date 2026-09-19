from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from g1_fly_control.tasks.crazyflie.command_wide_logic import (
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    HELD_OUT_EPISODES,
    HELD_OUT_EPISODE_STEPS,
    HELD_OUT_WIND_SEED,
    WIND_CATEGORIES,
    WIND_REFERENCE_ARM_M,
    held_out_wind_at_step,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "crazyflie_command_evaluate.py"
spec = importlib.util.spec_from_file_location(
    "crazyflie_command_wind_evaluate_test", SCRIPT
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

VEHICLE_WEIGHT_N = 0.27
VECTOR_SHAPE = (HELD_OUT_EPISODE_STEPS, HELD_OUT_EPISODES, 3)
SCALAR_SHAPE = (HELD_OUT_EPISODE_STEPS, HELD_OUT_EPISODES)


def _wind_traces() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    force = torch.zeros(VECTOR_SHAPE)
    torque = torch.zeros(VECTOR_SHAPE)
    categories = torch.zeros(SCALAR_SHAPE, dtype=torch.long)
    category_to_code = {name: index for index, name in enumerate(WIND_CATEGORIES)}
    for episode_index in range(HELD_OUT_EPISODES):
        for step in range(HELD_OUT_EPISODE_STEPS):
            value = held_out_wind_at_step(
                episode_index,
                step,
                seed=HELD_OUT_WIND_SEED,
            )
            force[step, episode_index] = (
                torch.tensor(value.force_ratio_world) * VEHICLE_WEIGHT_N
            )
            torque[step, episode_index] = (
                torch.tensor(value.torque_ratio_world)
                * VEHICLE_WEIGHT_N
                * WIND_REFERENCE_ARM_M
            )
            categories[step, episode_index] = category_to_code[value.category]
    return force, torque, categories


def test_still_air_requires_exact_zero_physical_wrench() -> None:
    report = module.physical_wind_telemetry_report(
        task=COMMAND_FOLLOW_WIDE_TASK_ID,
        force_world=torch.zeros(VECTOR_SHAPE),
        torque_world=torch.zeros(VECTOR_SHAPE),
        category_code=torch.zeros(SCALAR_SHAPE, dtype=torch.long),
        interval_observed=torch.ones(SCALAR_SHAPE, dtype=torch.bool),
        vehicle_weight_n=VEHICLE_WEIGHT_N,
    )
    assert report["condition"] == "still_air"
    assert report["samples"]["planned_intervals"] == 16 * 600
    assert report["samples"]["planned_pulse_intervals"] == 0
    assert report["integrity"]["still_air_exact_zero_all_simulated_intervals"]
    assert report["integrity"]["passed"]

    force = torch.zeros(VECTOR_SHAPE)
    force[0, 0, 0] = 1.0e-5
    altered = module.physical_wind_telemetry_report(
        task=COMMAND_FOLLOW_WIDE_TASK_ID,
        force_world=force,
        torque_world=torch.zeros(VECTOR_SHAPE),
        category_code=torch.zeros(SCALAR_SHAPE, dtype=torch.long),
        interval_observed=torch.ones(SCALAR_SHAPE, dtype=torch.bool),
        vehicle_weight_n=VEHICLE_WEIGHT_N,
    )
    assert not altered["integrity"]["still_air_exact_zero_all_simulated_intervals"]
    assert not altered["integrity"]["passed"]


def test_wind_telemetry_matches_exact_heldout_world_frame_pulses() -> None:
    force, torque, categories = _wind_traces()
    report = module.physical_wind_telemetry_report(
        task=COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
        force_world=force,
        torque_world=torque,
        category_code=categories,
        interval_observed=torch.ones(SCALAR_SHAPE, dtype=torch.bool),
        vehicle_weight_n=VEHICLE_WEIGHT_N,
    )
    assert report["frame"] == "world"
    assert report["application_point"] == "body_center_of_mass"
    assert report["protocol"]["pulse_start_steps"] == [75, 175, 275, 375, 475]
    assert report["protocol"]["pulse_duration_steps"] == 25
    assert report["samples"]["planned_pulse_intervals"] == 16 * 5 * 25
    assert report["samples"]["observed_nonzero_wrench_intervals"] == 16 * 5 * 25
    assert report["integrity"]["expected_force_matches_on_observed_intervals"]
    assert report["integrity"]["expected_torque_matches_on_observed_intervals"]
    assert report["integrity"]["expected_category_matches_on_observed_intervals"]
    assert report["integrity"]["pulse_window_matches_on_observed_intervals"]
    assert report["integrity"]["wind_nonzero_wrench_observed"]
    assert report["integrity"]["passed"]


def test_schedule_equality_stops_at_first_done_but_bounds_cover_all_steps() -> None:
    force, torque, categories = _wind_traces()
    observed = torch.zeros(SCALAR_SHAPE, dtype=torch.bool)
    observed[:80] = True
    # Model Isaac Lab auto-reset activity after the first terminal interval.
    # It is not attributed to the original episode but remains finite/bounded.
    force[80:] = 0.0
    torque[80:] = 0.0
    categories[80:] = 0
    report = module.physical_wind_telemetry_report(
        task=COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
        force_world=force,
        torque_world=torque,
        category_code=categories,
        interval_observed=observed,
        vehicle_weight_n=VEHICLE_WEIGHT_N,
    )
    assert report["samples"]["observed_intervals_through_first_done"] == 16 * 80
    assert not report["integrity"][
        "complete_600_step_protocol_observed_for_all_episodes"
    ]
    assert report["integrity"]["passed"]

    force[100, 0, 0] = 1.0
    out_of_bounds = module.physical_wind_telemetry_report(
        task=COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
        force_world=force,
        torque_world=torque,
        category_code=categories,
        interval_observed=observed,
        vehicle_weight_n=VEHICLE_WEIGHT_N,
    )
    assert not out_of_bounds["integrity"]["force_within_declared_bound"]
    assert not out_of_bounds["integrity"]["passed"]


def test_shifted_or_missing_observed_pulse_is_detected() -> None:
    force, torque, categories = _wind_traces()
    force[75, 0] = 0.0
    torque[75, 0] = 0.0
    categories[75, 0] = 0
    report = module.physical_wind_telemetry_report(
        task=COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
        force_world=force,
        torque_world=torque,
        category_code=categories,
        interval_observed=torch.ones(SCALAR_SHAPE, dtype=torch.bool),
        vehicle_weight_n=VEHICLE_WEIGHT_N,
    )
    assert not report["integrity"]["pulse_window_matches_on_observed_intervals"]
    assert not report["integrity"]["passed"]


def test_nonfinite_wind_telemetry_is_rejected() -> None:
    force = torch.zeros(VECTOR_SHAPE)
    force[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        module.physical_wind_telemetry_report(
            task=COMMAND_FOLLOW_WIDE_TASK_ID,
            force_world=force,
            torque_world=torch.zeros(VECTOR_SHAPE),
            category_code=torch.zeros(SCALAR_SHAPE, dtype=torch.long),
            interval_observed=torch.ones(SCALAR_SHAPE, dtype=torch.bool),
            vehicle_weight_n=VEHICLE_WEIGHT_N,
        )
