"""Declared disturbance schedules used for paired push-recovery evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PushSchedule:
    start_s: float
    duration_s: float
    force_newton: tuple[float, float, float]
    body: str = "torso_link"
    frame: str = "world"

    @property
    def impulse_newton_seconds(self) -> tuple[float, float, float]:
        return tuple(component * self.duration_s for component in self.force_newton)

    def manifest(self) -> dict[str, object]:
        return {**asdict(self), "impulse_newton_seconds": self.impulse_newton_seconds}

