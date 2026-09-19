"""Independent manager-based G1 configurations; never inherit stock walking rewards."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import events, observations, rewards, terminations
from .actions import bounded_joint_position_action
from .scene import FreePostureG1SceneCfg


@configclass
class ActionsCfg:
    joint_pos = bounded_joint_position_action()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Explicit ordering documented in docs/task_spec.md.
        joint_pos = ObsTerm(func=base_mdp.joint_pos_limit_normalized, clip=(-1.0, 1.0))
        joint_vel = ObsTerm(func=observations.joint_velocity_normalized, clip=(-1.0, 1.0))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, clip=(-10.0, 10.0), scale=0.1)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, clip=(-1.0, 1.0))
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, clip=(-5.0, 5.0), scale=0.2)
        target_relative_body = ObsTerm(func=observations.target_relative_body, clip=(-3.0, 3.0), scale=1.0 / 3.0)
        previous_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_goal = EventTerm(func=events.reset_goal, mode="reset", params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class RewardsCfg:
    # Each term returns a rate and RewardManager multiplies by env.step_dt exactly once.
    goal_progress = RewTerm(func=rewards.goal_progress, weight=1.0)
    goal_success = RewTerm(func=rewards.goal_success, weight=2.0)
    mechanical_work_proxy = RewTerm(func=rewards.mechanical_work_proxy, weight=-2.0e-5)
    action_change = RewTerm(func=rewards.action_change, weight=-0.01)
    joint_limit_violation = RewTerm(func=rewards.joint_limit_violation, weight=-1.0)
    excessive_impact = RewTerm(func=rewards.excessive_impact, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    invalid_state = DoneTerm(func=terminations.invalid_state)
    escaped_workspace = DoneTerm(func=terminations.escaped_workspace, params={"radius": 8.0})


@configclass
class FreePostureG1EnvCfg(ManagerBasedRLEnvCfg):
    scene: FreePostureG1SceneCfg = FreePostureG1SceneCfg(num_envs=16, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation


@configclass
class GoalReachFreePostureEnvCfg(FreePostureG1EnvCfg):
    pass


@configclass
class GoalSwitchFreePostureEnvCfg(FreePostureG1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.switch_goal = EventTerm(func=events.switch_goal, mode="interval", interval_range_s=(5.0, 5.0))


@configclass
class PushRecoveryFreePostureEnvCfg(FreePostureG1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.push = EventTerm(
            func=events.start_push,
            mode="interval",
            interval_range_s=(5.0, 5.0),
            params={"force_newton": 200.0, "duration_s": 0.10, "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )
