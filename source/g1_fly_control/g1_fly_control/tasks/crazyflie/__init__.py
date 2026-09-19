"""Crazyflie direct-task registrations and CPU-safe public contract."""

from .adapter import ACTION_NAMES, NATIVE_TASK_ID, OBSERVATION_NAMES
from .registration import (
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    COMMAND_TASK_IDS,
    MIXED_TRAINING_TASK_ID,
    REGISTERED_TASK_IDS,
    TASK_IDS,
    TASK_TO_CFG,
    register_tasks,
)

__all__ = [
    "ACTION_NAMES",
    "COMMAND_FOLLOW_TASK_ID",
    "COMMAND_FOLLOW_WIDE_TASK_ID",
    "COMMAND_FOLLOW_WIDE_WIND_TASK_ID",
    "COMMAND_TASK_IDS",
    "NATIVE_TASK_ID",
    "OBSERVATION_NAMES",
    "MIXED_TRAINING_TASK_ID",
    "REGISTERED_TASK_IDS",
    "TASK_IDS",
    "TASK_TO_CFG",
    "register_tasks",
]
