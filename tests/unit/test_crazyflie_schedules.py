import math

import pytest
import torch

from g1_fly_control.tasks.crazyflie.logic import (
    CONTROL_DT_S,
    DEFAULT_GUST_STEPS,
    DEFAULT_SWITCH_STEPS,
    GUST_DURATION_STEPS,
    RECOVERY_WINDOW_STEPS,
    constant_force_impulse_n_s,
    gust_force_n,
    gust_impulse_n_s,
    horizontal_gust_force_vector_n,
    recovery_window,
    schedule_active_mask,
    schedule_steps,
    scheduled_event_index,
    seconds_to_control_steps,
)


def test_frozen_schedules_land_exactly_on_the_50_hz_grid():
    assert schedule_steps((3.0, 6.0, 9.0)) == (150, 300, 450)
    assert DEFAULT_SWITCH_STEPS == (150, 300, 450)
    assert DEFAULT_GUST_STEPS == (150, 300, 450)
    assert seconds_to_control_steps(0.10) == GUST_DURATION_STEPS == 5
    assert seconds_to_control_steps(2.0) == RECOVERY_WINDOW_STEPS == 100
    with pytest.raises(ValueError, match="not aligned"):
        seconds_to_control_steps(0.011)


def test_gust_interval_is_half_open_and_exactly_five_steps():
    probe = torch.tensor([149, 150, 154, 155, 299, 300, 304, 305])
    assert schedule_active_mask(probe, DEFAULT_GUST_STEPS, GUST_DURATION_STEPS).tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
    ]
    assert scheduled_event_index(probe, DEFAULT_GUST_STEPS, GUST_DURATION_STEPS).tolist() == [
        -1,
        0,
        0,
        -1,
        -1,
        1,
        1,
        -1,
    ]


def test_recovery_window_begins_after_gust_and_has_100_half_open_steps():
    window = recovery_window(150)
    assert window.gust_end_step == 155
    assert (window.start_step, window.stop_step) == (155, 255)
    assert window.duration_steps == 100
    assert not window.contains(154)
    assert window.contains(155)
    assert window.contains(254)
    assert not window.contains(255)


def test_gust_force_realizes_mass_normalized_delta_v_over_point_one_seconds():
    mass_kg = 0.027
    expected_impulse = mass_kg * 0.75
    expected_force = expected_impulse / 0.10
    assert gust_impulse_n_s(mass_kg) == pytest.approx(expected_impulse)
    assert gust_force_n(mass_kg) == pytest.approx(expected_force)
    assert constant_force_impulse_n_s(expected_force, GUST_DURATION_STEPS) == pytest.approx(
        expected_impulse
    )
    assert GUST_DURATION_STEPS * CONTROL_DT_S == pytest.approx(0.10)


def test_gust_direction_is_normalized_horizontal_world_frame_vector():
    vector = horizontal_gust_force_vector_n(0.027, (3.0, 4.0))
    assert vector[2] == 0.0
    assert math.hypot(vector[0], vector[1]) == pytest.approx(gust_force_n(0.027))
    assert vector[0] / vector[1] == pytest.approx(3.0 / 4.0)
    with pytest.raises(ValueError, match="horizontal"):
        horizontal_gust_force_vector_n(0.027, (1.0, 0.0, 0.1))
    with pytest.raises(ValueError, match="non-zero"):
        horizontal_gust_force_vector_n(0.027, (0.0, 0.0))
