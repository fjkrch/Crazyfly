# Free-posture task specification

All three IDs are independent manager-based registrations:

- `FlyG1-GoalReach-FreePosture-v0`: approach an XY target on flat ground.
- `FlyG1-GoalSwitch-FreePosture-v0`: resample/log a target at a fixed 5 s interval.
- `FlyG1-PushRecovery-FreePosture-v0`: retain goal reaching while a 200 N horizontal force acts on the torso for 0.10 s every 5 s. Direction is randomized per environment, and the world-frame applied impulse is 20 N·s.

The initial horizon is 20 s, with goal success after staying within 0.30 m XY distance for 0.50 s (25 control steps). Target distance is sampled uniformly from 1–3 m around the resolved G1 root. The reference is `root_pos_w[:2]`. Goal target localization and base velocity are simulator-state observations, not claimed onboard estimates.

Policy observation order, concatenated with no noise or vision, is:

1. joint position normalized to resolved soft limits and clipped to [-1,1]
2. joint velocity divided by resolved velocity limits and clipped to [-1,1]
3. base angular velocity in root frame, clipped to ±10 rad/s and divided by 10
4. projected unit gravity direction in root frame, clipped to [-1,1]
5. base linear velocity in root frame, clipped to ±5 m/s and divided by 5
6. yaw-aligned target-relative XY, clipped to ±3 m and divided by 3
7. previous normalized policy action [unit]

The action is one `[-1, 1]` value for every resolved allowlisted leg, ankle, torso, shoulder, and elbow joint. The project's action term maps `-1/0/+1` to each joint's resolved soft lower limit/default pose/soft upper limit; the simulator's configured effort/velocity limits remain in force. Finger joints are explicitly excluded. `inspect_asset.py` exports the final names/order/limits and actuator configuration.

Reward terms are rates and Isaac Lab multiplies them by `step_dt` exactly once:

- positive XY progress and a one-time success bonus;
- mechanical-work proxy `sum over physics substeps of abs(applied_torque * joint_velocity) * physics_dt`, explicitly not electrical consumption;
- penalties for action change, joint-limit violation, and force exceeding 350 N.

There is no uprightness, root-height, orientation, foot-air-time, foot-alternation, motion-reference, or posture reward. Torso, hand, knee, and ordinary support contacts do not terminate an episode. Termination is only the 20 s time limit (truncation), workspace escape (8 m), or non-finite simulated state. Goal/reset paths clear prior-distance and success state so target teleportation cannot cause artificial progress.
