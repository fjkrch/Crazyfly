"""Paired evaluation draws must be stable and separated from training RNG."""

import math

from scripts.evaluation_protocol import push_direction, schedule_manifest, target_offset


def test_heldout_schedule_is_deterministic_and_within_training_support():
    first = schedule_manifest(101, 16, "FlyG1-PushRecovery-FreePosture-v0")
    second = schedule_manifest(101, 16, "FlyG1-PushRecovery-FreePosture-v0")
    changed = schedule_manifest(102, 16, "FlyG1-PushRecovery-FreePosture-v0")
    assert first == second
    assert first["sha256"] != changed["sha256"]
    for env_id in range(16):
        for ordinal in range(4):
            x, y = target_offset(101, env_id, ordinal)
            assert 1.0 <= math.hypot(x, y) < 3.0
            assert max(abs(x), abs(y)) < 3.0
        for ordinal in range(3):
            x, y = push_direction(101, env_id, ordinal)
            assert math.hypot(x, y) == math.hypot(*push_direction(101, env_id, ordinal))
            assert math.isclose(math.hypot(x, y), 1.0)
