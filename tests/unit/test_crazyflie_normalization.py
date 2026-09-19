import pytest
import torch

from g1_fly_control.crazyflie.normalization import (
    COMMAND_TASK_ID,
    FixedPhysicsScaleNormalizer,
    NormalizedEnv,
    RunningMeanVariance,
)
from g1_fly_control.crazyflie.stabilization import (
    FIXED_OBSERVATION_CLIP,
    FIXED_OBSERVATION_SCALE,
)


class _MetricEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.drone_terminal_observation = torch.zeros(2, 12)
        self.terminal_success_count = torch.tensor([1, 0])
        self.terminal_scenario_code = torch.tensor([0, 2])
        self.terminal_failure_cause = torch.tensor([0, 2])
        self.terminal_distance_m = torch.tensor([0.1, 3.0])
        self.terminal_speed_mps = torch.tensor([0.05, 2.0])
        self.reward_components = {"progress": torch.tensor([1.0, 2.0])}
        self._calls = 0

    def reset(self):
        return {"policy": torch.zeros(2, 12)}, {}

    def step(self, action):
        del action
        self._calls += 1
        if self._calls == 1:
            terminated = torch.tensor([False, True])
            truncated = torch.tensor([False, False])
            reward = torch.tensor([1.0, -3.0])
        else:
            terminated = torch.tensor([False, False])
            truncated = torch.tensor([True, False])
            reward = torch.tensor([2.0, 4.0])
        return {"policy": torch.ones(2, 12)}, reward, terminated, truncated, {}

    def close(self):
        pass


def test_completed_episode_metrics_are_consumed_once_and_keep_partial_returns():
    wrapped = NormalizedEnv(_MetricEnv(), RunningMeanVariance.create(12), training=True)
    wrapped.reset()
    wrapped.step(torch.zeros(2, 4))
    first = wrapped.consume_training_metrics()
    assert first["completed_episode_count"] == 1
    assert first["episodic_return_mean"] == -3.0
    assert first["failure_termination_count"] == 1
    assert first["failure_cause_counts"]["2"] == 1
    assert first["episodic_reward_component_means"]["progress"] == 2.0
    assert wrapped.consume_training_metrics()["completed_episode_count"] == 0

    wrapped.step(torch.zeros(2, 4))
    second = wrapped.consume_training_metrics()
    assert second["completed_episode_count"] == 1
    assert second["episodic_return_mean"] == 3.0  # row 0 retained 1.0 from the prior step
    assert second["successful_episode_count"] == 1
    assert second["target_success_count"] == 1
    assert second["time_limit_truncation_count"] == 1
    assert second["episodic_reward_component_means"]["progress"] == 2.0


class _MixedMetricEnv:
    num_envs = 8
    device = "cpu"

    def __init__(self):
        self.drone_terminal_observation = torch.zeros(self.num_envs, 12)
        # Reach: fail/succeed; Switch: partial 1/3 then exact 4; Gust:
        # fail/succeed.  The last two rows prove that scenario masks, rather
        # than the task-wide maximum or a generic >0 rule, select thresholds.
        self.terminal_scenario_code = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
        self.terminal_success_count = torch.tensor([0, 1, 1, 3, 4, 0, 1, 2])
        self.terminal_failure_cause = torch.zeros(self.num_envs, dtype=torch.long)
        self.terminal_distance_m = torch.ones(self.num_envs)
        self.terminal_speed_mps = torch.zeros(self.num_envs)
        self.reward_components = {"progress": torch.zeros(self.num_envs)}

    def reset(self):
        return {"policy": torch.zeros(self.num_envs, 12)}, {}

    def step(self, action):
        del action
        return (
            {"policy": torch.zeros(self.num_envs, 12)},
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.ones(self.num_envs, dtype=torch.bool),
            {},
        )

    def close(self):
        pass


def test_mixed_training_success_uses_scenario_specific_target_requirements() -> None:
    wrapped = NormalizedEnv(
        _MixedMetricEnv(), RunningMeanVariance.create(12), training=True
    )
    wrapped.reset()
    wrapped.step(torch.zeros(8, 4))

    metrics = wrapped.consume_training_metrics()
    assert metrics["completed_episode_count"] == 8
    assert metrics["target_success_count"] == 12
    # Reach count=1, Switch count=4, and both Gust counts 1/2 succeed.
    assert metrics["successful_episode_count"] == 4


def test_training_success_metrics_reject_unknown_scenario_codes() -> None:
    env = _MixedMetricEnv()
    env.terminal_scenario_code[0] = 3
    wrapped = NormalizedEnv(env, RunningMeanVariance.create(12), training=True)
    wrapped.reset()

    with pytest.raises(RuntimeError, match="Reach=0, Switch=1, or Gust=2"):
        wrapped.step(torch.zeros(8, 4))


class _CommandMetricEnv:
    num_envs = 2
    device = "cpu"
    REWARD_COMPONENT_NAMES = (
        "linear_tracking",
        "yaw_tracking",
        "tracking_progress",
        "wrong_direction_acceleration",
        "jerk",
        "target_retention",
        "attitude_stability",
        "angular_stability",
        "control_effort",
        "action_smoothness",
        "survival",
        "failure",
        "total",
    )

    def __init__(self):
        self.command_tracking_contract = {"version": "unit-command-v1"}
        self.drone_terminal_observation = torch.zeros(self.num_envs, 12)
        self.requested_command_body = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
        )
        self.effective_command_body = torch.tensor(
            [[0.5, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
        )
        self.tracking_error_body = torch.zeros(self.num_envs, 4)
        self.terminal_tracking_error_body = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
        )
        self.terminal_failure_cause = torch.tensor([0, 2])
        self.reward_components = {
            name: torch.full((self.num_envs,), float(index + 1))
            for index, name in enumerate(self.REWARD_COMPONENT_NAMES)
        }
        self._calls = 0

    def reset(self):
        return {"policy": torch.zeros(self.num_envs, 12)}, {}

    def step(self, action):
        del action
        self._calls += 1
        if self._calls == 1:
            self.tracking_error_body.copy_(
                torch.tensor([[1.0, 2.0, 2.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
            )
            terminated = torch.tensor([False, True])
            truncated = torch.tensor([False, False])
        else:
            self.tracking_error_body.zero_()
            terminated = torch.tensor([False, False])
            truncated = torch.tensor([True, False])
        return (
            {"policy": torch.zeros(self.num_envs, 12)},
            torch.ones(self.num_envs),
            terminated,
            truncated,
            {},
        )

    def close(self):
        pass


def test_command_training_metrics_are_continuous_and_goal_free() -> None:
    env = _CommandMetricEnv()
    wrapped = NormalizedEnv(
        env,
        RunningMeanVariance.create(12),
        training=True,
        task=COMMAND_TASK_ID,
    )
    wrapped.reset()
    wrapped.step(torch.zeros(2, 4))
    wrapped.step(torch.zeros(2, 4))

    metrics = wrapped.consume_training_metrics()

    assert metrics["completed_episode_count"] == 2
    assert metrics["failure_termination_count"] == 1
    assert metrics["time_limit_truncation_count"] == 1
    assert metrics["failure_cause_counts"]["2"] == 1
    assert metrics["command_tracking_sample_count"] == 4
    assert metrics["command_tracking_rmse_linear_mps"] == pytest.approx(
        (18.0 / 4.0) ** 0.5
    )
    assert metrics["command_tracking_rmse_yaw_radps"] == pytest.approx(
        (4.25 / 4.0) ** 0.5
    )
    assert metrics["requested_command_mean_body"] == pytest.approx(
        [1.0, 0.5, 0.0, 0.0]
    )
    assert metrics["effective_command_mean_body"] == pytest.approx(
        [0.75, 0.5, 0.0, 0.0]
    )
    assert metrics["effective_command_max_abs_body"] == pytest.approx(
        [1.0, 1.0, 0.0, 0.0]
    )
    assert metrics["command_projection_fraction"] == pytest.approx(0.5)
    assert set(metrics["rollout_reward_component_means"]) == set(
        env.REWARD_COMPONENT_NAMES
    )
    assert metrics["rollout_reward_component_means"]["tracking_progress"] == 3.0
    assert metrics["rollout_reward_component_means"][
        "wrong_direction_acceleration"
    ] == 4.0
    assert metrics["rollout_reward_component_means"]["jerk"] == 5.0
    assert metrics["rollout_reward_component_means"]["attitude_stability"] == 7.0
    assert metrics["rollout_reward_component_means"]["angular_stability"] == 8.0
    assert metrics["rollout_reward_component_means"]["survival"] == 11.0
    assert metrics["rollout_reward_component_means"]["failure"] == 12.0
    assert "successful_episode_count" not in metrics
    assert "target_success_count" not in metrics
    assert "final_goal_distance_mean_m" not in metrics


def test_command_training_metrics_fail_closed_on_missing_terminal_error() -> None:
    env = _CommandMetricEnv()
    del env.terminal_tracking_error_body
    wrapped = NormalizedEnv(
        env,
        RunningMeanVariance.create(12),
        training=True,
        task=COMMAND_TASK_ID,
    )
    wrapped.reset()

    with pytest.raises(RuntimeError, match="terminal_tracking_error_body"):
        wrapped.step(torch.zeros(2, 4))


def test_fixed_normalizer_has_no_mutable_training_statistics() -> None:
    normalizer = RunningMeanVariance.create(12)
    assert isinstance(normalizer, FixedPhysicsScaleNormalizer)
    before = normalizer.state_dict()
    normalizer.update(torch.randn(100, 12) * 100.0)
    after = normalizer.state_dict()
    assert before["kind"] == after["kind"] == "fixed_physics_scale_v1"
    assert before["update_rule"] == after["update_rule"] == "immutable_no_updates"
    assert before["clip"] == after["clip"] == FIXED_OBSERVATION_CLIP
    torch.testing.assert_close(before["scale"], after["scale"], rtol=0.0, atol=0.0)


def test_fixed_normalizer_scaling_clipping_and_serialization() -> None:
    normalizer = RunningMeanVariance.create(12)
    scale = torch.tensor(FIXED_OBSERVATION_SCALE)
    raw = torch.stack((scale, 10.0 * scale, -10.0 * scale))
    normalized = normalizer.normalize(raw)
    torch.testing.assert_close(normalized[0], torch.ones(12))
    torch.testing.assert_close(
        normalized[1], torch.full((12,), FIXED_OBSERVATION_CLIP)
    )
    torch.testing.assert_close(
        normalized[2], torch.full((12,), -FIXED_OBSERVATION_CLIP)
    )

    restored = RunningMeanVariance.create(12)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored.normalize(raw), normalized)
    torch.testing.assert_close(restored.denormalize(normalized[0]), scale)


def test_fixed_normalizer_rejects_old_or_tampered_state() -> None:
    normalizer = RunningMeanVariance.create(12)
    with pytest.raises(ValueError, match="state fields changed"):
        normalizer.load_state_dict(
            {
                "mean": torch.zeros(12),
                "variance": torch.ones(12),
                "count": torch.tensor(10),
            }
        )
    tampered = normalizer.state_dict()
    tampered["scale"] = tampered["scale"].clone()
    tampered["scale"][0] = 3.0
    with pytest.raises(ValueError, match="scales changed"):
        normalizer.load_state_dict(tampered)


def test_fixed_normalizer_requires_exact_crazyflie_width() -> None:
    with pytest.raises(ValueError, match="requires width 12"):
        RunningMeanVariance.create(11)
