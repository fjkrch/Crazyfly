# Crazyflie installed asset audit

Generated: `2026-09-17T08:41:04+00:00`  
Overall status: **PASS**

This is a live audit of the installed `Isaac-Quadcopter-Direct-v0` environment. It does not infer a physical sensor suite or individual-rotor interface.

## Contract checks

| Check | Result |
| --- | --- |
| native_task_registered | PASS |
| project_task_ids_registered_without_shadowing_native | PASS |
| observation_width_12 | PASS |
| action_width_4 | PASS |
| physics_dt_0.01_s | PASS |
| decimation_2 | PASS |
| control_dt_0.02_s | PASS |
| native_horizon_10_s | PASS |
| thrust_to_weight_1.9 | PASS |
| moment_scale_0.01_n_m | PASS |
| crazyflie_cf2x_asset | PASS |
| positive_finite_mass | PASS |
| live_step_finite | PASS |
| instantaneous_gust_wrench_api_available | PASS |
| gust_world_frame_center_of_mass_contract | PASS |
| gust_duration_5_control_10_physics_steps_0.10_s | PASS |
| gust_physical_response_tolerance_predeclared | PASS |
| native_goal_bounds_pinned | PASS |
| native_height_termination_pinned | PASS |

## Live values

- Observation shape: `[1, 12]` (12 values, in the pinned upstream order)
- Action shape: `[4]` (aggregate thrust and three body moments)
- Physics timestep: `0.01` s
- Control timestep: `0.02` s (`50.0` Hz)
- Native episode horizon: `10.0` s
- Crazyflie mass: `0.028200002387166023` kg
- Vehicle weight: `0.2766420245170593` N
- Thrust-to-weight scale: `1.9`
- Moment scale: `0.01` N m
- Body name/index: `body` / `0`
- USD identifier: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd`
- Height failure bounds: `[0.1, 2.0]` m
- Native goal bounds relative to environment origin: x `[-2.0, 2.0]` m; y `[-2.0, 2.0]` m; z `[0.5, 1.5]` m

The native controller uses `Articulation.permanent_wrench_composer.set_forces_and_torques` for its
aggregate body-frame thrust/moments. Project gusts instead use
`Articulation.instantaneous_wrench_composer.set_forces_and_torques` with `is_global=True`: a world-frame force on
body `body` at its center of mass, with no application-point
offset. Each gust spans `5` control decisions / 
`10` physics steps (`0.1` s).

Gate C reports three distinct quantities: the mass-derived expected impulse,
the force-time integral submitted to the wrench composer, and the measured
horizontal `mass * delta-velocity` response. The physical response is isolated
by `paired_identical_action_rollouts_same_live_environment_v1` and must be within the predeclared maximum of
`0.0005` N s absolute or
`0.1` relative error. This asset audit records
the contract; the custom-task smoke artifact records the live measurements.

## Observation ordering

`body linear velocity (3), body angular velocity (3), projected gravity (3), body-frame goal displacement (3)`

## Action mapping

`a0 -> (a0 + 1) / 2 * 1.9 * vehicle weight` along local +Z; `a1:a4 -> +/-0.01 N m` body moments. Policies emit bounded values in `[-1, 1]`.

## Installed versions

| Package | Version |
| --- | --- |
| isaaclab | `0.54.4` |
| isaaclab-assets | `0.2.4` |
| isaaclab-tasks | `0.11.16` |
| isaacsim | `5.1.0.0` |
| torch | `2.7.0+cu128` |
| gymnasium | `1.2.1` |
| numpy | `1.26.0` |
| python | `3.11.15` |

## Source identities

- Reproduction fingerprint: `0066e7dc07201e36c5c43342beeb16560e5f630f888a7a85b0022197ee9ef58f`
- Isaac Lab commit: `b4c321024792976150ca55fddb26fa34480d974e`
- Upstream environment SHA-256: `b78a48cdb04f4a215dded24c195f14818fd5fb43f6e682eeaa1a057fc6c77f23`
- DirectRLEnv SHA-256: `3a7573303d83ec8c816d11ca6e8c8997c92f6765ae5d2e0d2a369e65d2cec3c0`
- Isaac Lab math helpers SHA-256: `881ab27758816475e26d6721bab22c934fd70d0c064392266a90b06f5bbe4149`
- Upstream registration SHA-256: `1cd4d4087adfd9ba2cf8b7968fa492f9c35235463d469b5e703f393c7809f8c2`
- Asset configuration SHA-256: `b36ebdc4a75c670cf503496734c022ade078c77f9d4c68dcd2e1fe6d8a620c72`
