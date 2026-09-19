import math

import pytest

from g1_fly_control.crazyflie import memory
from g1_fly_control.crazyflie.memory import assess


def _sample(
    stage: str,
    rss: float,
    *,
    ram: float = 40.0,
    gpu: float | None = 1000.0,
    swap_out: float = 0.0,
    device_type: str = "cuda",
):
    return {
        "timestamp_utc": "2026-09-17T00:00:00+00:00",
        "stage": stage,
        "step": 1,
        "process_rss_mib": rss,
        "system_ram_total_gib": 24.0,
        "system_ram_used_gib": 10.0,
        "system_ram_available_gib": 14.0,
        "system_ram_percent": ram,
        "system_swap_total_gib": 8.0,
        "system_swap_used_gib": 0.0,
        "system_swap_percent": 0.0,
        "system_swap_in_mib": 0.0,
        "system_swap_out_mib": swap_out,
        "compute_device_type": device_type,
        "gpu_devices": [] if gpu is None else [{
            "index": 0,
            "name": "Unit Test GPU",
            "driver_version": "test",
            "total_mib": 8151.0,
            "used_mib": gpu,
        }],
        "torch_allocated_mib": 0.0,
        "torch_reserved_mib": 0.0,
        "torch_peak_allocated_mib": 0.0,
        "torch_peak_reserved_mib": 0.0,
    }


def test_memory_growth_uses_only_like_for_like_steady_state_samples():
    samples = [
        _sample("environment_loaded", 1000.0),
        _sample("steady_state", 1500.000),
        _sample("steady_state", 1500.004),
        _sample("steady_state", 1500.008),
        _sample("steady_state", 1500.012),
        _sample("ppo_forward_backward_update", 2000.0),
    ]
    report = assess(samples)
    assert report["passed"]
    assert not report["monotonic_process_growth_detected"]
    assert report["steady_state_growth_window_mib"] == [1500.0, 1500.004, 1500.008, 1500.012]
    assert report["warnings"] == []
    assert report["policy_version"] == "crazyflie_memory_acceptance_v2"


def test_memory_growth_warns_but_passes_for_material_steady_state_rise():
    samples = [
        _sample("steady_state", 1000.0),
        _sample("steady_state", 1050.0),
        _sample("steady_state", 1100.0),
        _sample("steady_state", 1150.0),
    ]
    report = assess(samples)
    assert report["passed"]
    assert report["monotonic_process_growth_detected"]
    assert report["failures"] == []
    assert any("process RSS rose monotonically" in warning for warning in report["warnings"])
    assert report["limits"]["rss_growth_disposition"] == "warning_only"


def test_memory_growth_warns_for_a_sustained_rise_above_declared_noise_allowance():
    samples = [
        _sample("steady_state", 1000.0),
        _sample("steady_state", 1000.4),
        _sample("steady_state", 1000.8),
        _sample("steady_state", 1001.2),
    ]
    report = assess(samples)
    assert report["passed"]
    assert report["monotonic_process_growth_detected"]
    assert report["warnings"]
    assert report["limits"]["rss_growth_tolerance_mib"] == 1.0


def test_training_optimizer_samples_form_a_steady_state_window():
    samples = [
        _sample("rollout", 900.0),
        _sample("optimizer_update", 1000.0),
        _sample("optimizer_update", 1050.0),
        _sample("optimizer_update", 1100.0),
        _sample("optimizer_update", 1150.0),
        _sample("training_complete", 1200.0),
    ]
    report = assess(samples)
    assert report["passed"]
    assert report["monotonic_process_growth_detected"]


def test_system_ram_guard_is_strictly_below_90_percent():
    assert assess([_sample("steady_state", 1000.0, ram=89.999)])["passed"]
    report = assess([_sample("steady_state", 1000.0, ram=90.0)])
    assert not report["passed"]
    assert report["limits"]["system_ram_percent_exclusive"] == 90.0
    assert "90.00%" in report["failures"][0]


def test_device_wide_gpu_guard_is_strictly_below_6_8_gib():
    assert assess([_sample("steady_state", 1000.0, gpu=6963.1)])["passed"]
    report = assess([_sample("steady_state", 1000.0, gpu=6963.2)])
    assert not report["passed"]
    assert "6963.2 MiB" in report["failures"][0]


def test_cuda_gpu_telemetry_is_required_instead_of_failing_open():
    report = assess([_sample("steady_state", 1000.0, gpu=None)])
    assert not report["passed"]
    assert not report["device_gpu_telemetry_complete"]
    assert any("telemetry was unavailable" in failure for failure in report["failures"])


def test_sustained_swap_out_growth_fails_memory_gate():
    samples = [
        _sample("steady_state", 1000.0, swap_out=100.0),
        _sample("steady_state", 1000.0, swap_out=100.4),
        _sample("steady_state", 1000.0, swap_out=100.8),
        _sample("steady_state", 1000.0, swap_out=101.2),
    ]
    report = assess(samples)
    assert not report["passed"]
    assert report["sustained_paging_detected"]
    assert report["limits"]["swap_out_growth_tolerance_mib"] == 1.0


def test_rss_warning_does_not_mask_sustained_paging_failure():
    samples = [
        _sample("steady_state", 1000.0, swap_out=100.0),
        _sample("steady_state", 1050.0, swap_out=100.4),
        _sample("steady_state", 1100.0, swap_out=100.8),
        _sample("steady_state", 1150.0, swap_out=101.2),
    ]
    report = assess(samples)
    assert not report["passed"]
    assert report["monotonic_process_growth_detected"]
    assert report["sustained_paging_detected"]
    assert any("process RSS rose monotonically" in warning for warning in report["warnings"])
    assert any("system swap-out rose continuously" in failure for failure in report["failures"])
    assert not any("process RSS rose monotonically" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("system_ram_total_gib", math.nan),
        ("system_ram_used_gib", math.inf),
        ("system_ram_available_gib", -math.inf),
        ("system_ram_percent", math.nan),
        ("process_rss_mib", math.inf),
        ("system_swap_total_gib", math.nan),
        ("system_swap_used_gib", math.inf),
        ("system_swap_percent", -math.inf),
        ("system_swap_in_mib", math.nan),
        ("system_swap_out_mib", -math.inf),
        ("torch_allocated_mib", math.inf),
        ("torch_reserved_mib", -math.inf),
        ("torch_peak_allocated_mib", math.nan),
        ("torch_peak_reserved_mib", math.inf),
        ("gpu_total_mib", math.inf),
        ("gpu_used_mib", math.nan),
    ],
)
def test_nonfinite_numeric_telemetry_fails_closed(field, value):
    sample = _sample("steady_state", 1000.0)
    if field.startswith("gpu_"):
        sample["gpu_devices"][0][field.removeprefix("gpu_")] = value
    else:
        sample[field] = value

    report = assess([sample])

    assert not report["passed"]
    assert any("finite non-Boolean number" in failure for failure in report["failures"])
    assert math.isfinite(report["max_system_ram_percent"])
    assert math.isfinite(report["max_device_gpu_used_mib"])
    assert math.isfinite(report["max_process_rss_mib"])
    assert math.isfinite(report["max_torch_allocated_mib"])
    assert math.isfinite(report["max_torch_reserved_mib"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process_rss_mib", -0.001),
        ("system_ram_total_gib", 0.0),
        ("system_ram_used_gib", -0.001),
        ("system_ram_available_gib", 24.001),
        ("system_ram_percent", -0.001),
        ("system_ram_percent", 100.001),
        ("system_swap_total_gib", -0.001),
        ("system_swap_used_gib", 8.001),
        ("system_swap_percent", 100.001),
        ("system_swap_in_mib", -0.001),
        ("system_swap_out_mib", -0.001),
        ("torch_allocated_mib", -0.001),
        ("torch_reserved_mib", -0.001),
        ("torch_peak_allocated_mib", -0.001),
        ("torch_peak_reserved_mib", -0.001),
        ("gpu_total_mib", 0.0),
        ("gpu_used_mib", -0.001),
        ("gpu_used_mib", 8151.001),
    ],
)
def test_physically_invalid_numeric_telemetry_fails_closed(field, value):
    sample = _sample("steady_state", 1000.0)
    if field.startswith("gpu_"):
        sample["gpu_devices"][0][field.removeprefix("gpu_")] = value
    else:
        sample[field] = value

    report = assess([sample])

    assert not report["passed"]
    assert any("invalid memory telemetry" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    "field",
    [
        "timestamp_utc",
        "stage",
        "step",
        "process_rss_mib",
        "system_ram_total_gib",
        "system_ram_used_gib",
        "system_ram_available_gib",
        "system_ram_percent",
        "system_swap_total_gib",
        "system_swap_used_gib",
        "system_swap_percent",
        "system_swap_in_mib",
        "system_swap_out_mib",
        "compute_device_type",
        "gpu_devices",
        "torch_allocated_mib",
        "torch_reserved_mib",
        "torch_peak_allocated_mib",
        "torch_peak_reserved_mib",
    ],
)
def test_missing_standard_snapshot_fields_fail_closed(field):
    sample = _sample("steady_state", 1000.0)
    del sample[field]

    report = assess([sample])

    assert not report["passed"]
    assert any(field in failure for failure in report["failures"])


@pytest.mark.parametrize("field", ["index", "name", "driver_version", "total_mib", "used_mib"])
def test_missing_gpu_device_fields_fail_closed(field):
    sample = _sample("steady_state", 1000.0)
    del sample["gpu_devices"][0][field]

    report = assess([sample])

    assert not report["passed"]
    assert not report["device_gpu_telemetry_complete"]
    assert any(field in failure for failure in report["failures"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process_rss_mib", True),
        ("system_ram_percent", False),
        ("system_swap_out_mib", True),
        ("torch_peak_allocated_mib", False),
        ("gpu_total_mib", True),
        ("gpu_used_mib", False),
    ],
)
def test_numeric_booleans_fail_closed(field, value):
    sample = _sample("steady_state", 1000.0)
    if field.startswith("gpu_"):
        sample["gpu_devices"][0][field.removeprefix("gpu_")] = value
    else:
        sample[field] = value

    report = assess([sample])

    assert not report["passed"]
    assert any("non-Boolean" in failure for failure in report["failures"])


@pytest.mark.parametrize("devices", [None, {}, [None], ["not-a-device"]])
def test_malformed_gpu_device_collections_fail_closed(devices):
    sample = _sample("steady_state", 1000.0)
    sample["gpu_devices"] = devices

    report = assess([sample])

    assert not report["passed"]
    assert not report["device_gpu_telemetry_complete"]


def test_checkpoint_heap_release_collects_and_trims_without_weakening_gate(monkeypatch):
    calls: list[object] = []

    class FakeLibc:
        @staticmethod
        def malloc_trim(pad):
            calls.append(pad)
            return 1

    monkeypatch.setattr(memory.gc, "collect", lambda: 7)
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _name: FakeLibc())

    report = memory.release_checkpoint_serialization_heap()

    assert report == {
        "python_objects_collected": 7,
        "allocator_trim_supported": True,
        "allocator_trim_released": True,
    }
    assert calls == [0]
    # The acceptance threshold remains exactly the declared 1 MiB.
    assert memory.RSS_GROWTH_TOLERANCE_MIB == 1.0


def test_checkpoint_commit_frame_unwinds_before_heap_release(monkeypatch):
    events: list[str] = []

    class CommitLocal:
        def __del__(self):
            events.append("commit_local_released")

    def commit():
        local = CommitLocal()
        assert local is not None
        return "checkpoint"

    monkeypatch.setattr(
        memory,
        "release_checkpoint_serialization_heap",
        lambda: events.append("heap_release"),
    )

    assert memory.commit_checkpoint_then_release_heap(commit) == "checkpoint"
    assert events == ["commit_local_released", "heap_release"]
    assert memory.RSS_GROWTH_TOLERANCE_MIB == 1.0


def test_snapshot_releases_unreachable_heap_before_measurement(monkeypatch):
    events: list[object] = []
    monkeypatch.setattr(
        memory,
        "release_checkpoint_serialization_heap",
        lambda: events.append("heap_release"),
    )
    monkeypatch.setattr(
        memory,
        "snapshot",
        lambda stage, device, *, step=None: events.append(
            ("snapshot", stage, str(device), step)
        ) or {"stage": stage, "step": step},
    )

    result = memory.snapshot_after_heap_release("optimizer_update", "cpu", step=100)

    assert result == {"stage": "optimizer_update", "step": 100}
    assert events == ["heap_release", ("snapshot", "optimizer_update", "cpu", 100)]
    assert memory.RSS_GROWTH_TOLERANCE_MIB == 1.0
