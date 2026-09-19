"""Bounded position action definition and audited G1 joint allowlist."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import JointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import JointPositionToLimitsAction

from .logic import default_centered_targets

# Arms, legs, and torso remain actuated. Finger joints are deliberately excluded in
# this initial task and are reported by inspect_asset.py rather than silently inferred.
ACTUATED_JOINT_ALLOWLIST = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "torso_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
]


class DefaultCenteredJointPositionAction(JointPositionToLimitsAction):
    """Keep zero command at the asset's default pose while obeying resolved limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        default = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._processed_actions[:] = default_centered_targets(actions, limits[..., 0], limits[..., 1], default)


def bounded_joint_position_action() -> JointPositionToLimitsActionCfg:
    """Map [-1, 0, 1] to lower/default/upper resolved simulator soft limits."""
    return JointPositionToLimitsActionCfg(
        class_type=DefaultCenteredJointPositionAction,
        asset_name="robot",
        joint_names=ACTUATED_JOINT_ALLOWLIST,
        scale=1.0,
        rescale_to_limits=True,
        preserve_order=True,
    )
