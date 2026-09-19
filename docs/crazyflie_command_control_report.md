# Crazyflie learned keyboard-command control report

Report state: **COMPLETE — all six jobs independently verified**

Generated (UTC): `2026-09-18T05:00:36+00:00`  
Task: `FlyCrazyflie-CommandFollow-v0`  
Config SHA-256: `9120a52eb24ceb456ede809f12cdfff89497429680ce9a0932d734285492b830`  
Queue snapshot SHA-256: `648ad8b8476a1b4873e23eedacefeb803e3d7887d73d171e3e1512245ef5b99c`

A verified row requires a completed 500,000-interaction training manifest, an authenticated immutable history, a matching checkpoint hash, a passing 16 × 600 command evaluation, passing memory gates, and measured activity from the same action-producing forward pass. Missing or inconsistent evidence is shown as N/A; task quality is not inferred from PPO loss.

## Controller score and capacity

| Controller | Evidence state | Actor trainable | Total trainable | Frozen weights | Parameter match | Total /100 | Acceleration /100 | Response /100 | Stability /100 | Survival /100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_lif_original | VERIFIED COMPLETE | 4776 | 23081 | 5103 | PASS | 51.449 | 98.178 | 63.267 | 83.099 | 100.000 |
| frozen_lif_degree_rewired | VERIFIED COMPLETE | 4776 | 23081 | 5103 | PASS | 51.195 | 98.390 | 63.763 | 83.410 | 100.000 |
| wing_lif | VERIFIED COMPLETE | 4776 | 23081 | 8864 | not required | 49.849 | 98.564 | 64.527 | 84.464 | 100.000 |
| leg_wing_lif | VERIFIED COMPLETE | 9224 | 27529 | 13967 | not required | 52.392 | 98.408 | 64.540 | 84.402 | 100.000 |
| gru_matched | VERIFIED COMPLETE | 4793 | 23098 | 0 | PASS | 52.814 | 98.532 | 65.224 | 83.258 | 100.000 |
| mlp_normal | VERIFIED COMPLETE | 4827 | 23132 | 0 | PASS | 52.364 | 98.417 | 64.944 | 84.001 | 100.000 |

![Score comparison](../runs/crazyflie_command_seed0_500k/command_score_comparison.png)

## Raw held-out control measurements

| Measurement | frozen_lif_original | frozen_lif_degree_rewired | wing_lif | leg_wing_lif | gru_matched | mlp_normal |
| --- | --- | --- | --- | --- | --- | --- |
| Linear tracking RMSE (m/s) | 0.440 | 0.449 | 0.500 | 0.444 | 0.445 | 0.448 |
| Yaw tracking RMSE (rad/s) | 0.338 | 0.323 | 0.269 | 0.280 | 0.264 | 0.264 |
| Wrong-direction fraction | 0.110 | 0.132 | 0.165 | 0.124 | 0.128 | 0.134 |
| Mean command projection ratio | 0.463 | 0.468 | 0.487 | 0.464 | 0.476 | 0.466 |
| Response latency (s) | 0.366 | 0.360 | 0.350 | 0.350 | 0.342 | 0.345 |
| Overshoot ratio | 0.029 | 0.043 | 0.081 | 0.026 | 0.024 | 0.026 |
| Brake settling (s) | 0.980 | 0.962 | 1.000 | 0.934 | 0.895 | 0.951 |
| Hover drift (m) | 0.272 | 0.283 | 0.329 | 0.275 | 0.279 | 0.285 |
| Survival fraction | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Invalid states | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Action effort RMS | 0.023 | 0.023 | 0.023 | 0.023 | 0.023 | 0.024 |
| Action delta RMS | 0.007 | 0.007 | 0.007 | 0.007 | 0.007 | 0.007 |

## Evaluation reward components

| Mean per 16-episode evaluation | frozen_lif_original | frozen_lif_degree_rewired | wing_lif | leg_wing_lif | gru_matched | mlp_normal |
| --- | --- | --- | --- | --- | --- | --- |
| action_smoothness | -5.175e-05 | -5.239e-05 | -5.638e-05 | -5.604e-05 | -5.335e-05 | -5.664e-05 |
| angular_stability | -0.017 | -0.017 | -0.017 | -0.017 | -0.020 | -0.018 |
| attitude_stability | -0.012 | -0.012 | -0.012 | -0.012 | -0.014 | -0.013 |
| control_effort | -7.611e-04 | -7.619e-04 | -7.638e-04 | -7.618e-04 | -8.069e-04 | -8.337e-04 |
| failure | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| jerk | -1.342e-04 | -1.355e-04 | -1.380e-04 | -1.355e-04 | -1.468e-04 | -1.520e-04 |
| linear_tracking | 13.806 | 13.652 | 12.830 | 13.734 | 13.805 | 13.722 |
| survival | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 |
| target_retention | 1.897 | 1.374 | 0.748 | 1.575 | 1.648 | 1.337 |
| total | 21.258 | 20.496 | 18.820 | 20.909 | 21.118 | 20.641 |
| tracking_progress | 1.638 | 1.546 | 1.280 | 1.603 | 1.669 | 1.585 |
| wrong_direction_acceleration | -0.034 | -0.048 | -0.083 | -0.034 | -0.049 | -0.051 |
| yaw_tracking | 3.381 | 3.403 | 3.475 | 3.461 | 3.480 | 3.480 |

![Training reward](../runs/crazyflie_command_seed0_500k/command_training_reward.png)

![Training loss](../runs/crazyflie_command_seed0_500k/command_training_loss.png)

## Frozen-core integrity

| Controller | Frozen check | Before | After | Authenticated per-core identities |
| --- | --- | --- | --- | --- |
| frozen_lif_original | PASS | cf2c0b8f9fe4319eeebabeefaebd6125084012abd5b8f891422dea2eede4fc54 | cf2c0b8f9fe4319eeebabeefaebd6125084012abd5b8f891422dea2eede4fc54 | primary=cf2c0b8f9fe4319eeebabeefaebd6125084012abd5b8f891422dea2eede4fc54 |
| frozen_lif_degree_rewired | PASS | 923cb1ad72c4507d1cddb939934774260bb013b0bf42e3d908fd653d084d1fe9 | 923cb1ad72c4507d1cddb939934774260bb013b0bf42e3d908fd653d084d1fe9 | primary=923cb1ad72c4507d1cddb939934774260bb013b0bf42e3d908fd653d084d1fe9 |
| wing_lif | PASS | 461ff516c0d7e3feb79c8aac86dbc3f5242940564b84de6a18cac9edb27719eb | 461ff516c0d7e3feb79c8aac86dbc3f5242940564b84de6a18cac9edb27719eb | primary=461ff516c0d7e3feb79c8aac86dbc3f5242940564b84de6a18cac9edb27719eb |
| leg_wing_lif | PASS | 333dfc4caccb83a6911eeb38b3a0e620f93b070c1605e473cd81e9d00d614443 | 333dfc4caccb83a6911eeb38b3a0e620f93b070c1605e473cd81e9d00d614443 | leg=cf2c0b8f9fe4319eeebabeefaebd6125084012abd5b8f891422dea2eede4fc54; wing=461ff516c0d7e3feb79c8aac86dbc3f5242940564b84de6a18cac9edb27719eb |
| gru_matched | N/A (non-LIF) | N/A | N/A | N/A |
| mlp_normal | N/A (non-LIF) | N/A | N/A | N/A |

## Measured controller activity

| Controller | Activity type | Units | Mean \|activity\| | RMS | Active fraction |
| --- | --- | --- | --- | --- | --- |
| frozen_lif_original | sampled_lif_spikes | 256 | 0.040 | 0.201 | 0.040 |
| frozen_lif_degree_rewired | sampled_lif_spikes | 256 | 0.022 | 0.149 | 0.022 |
| wing_lif | sampled_lif_spikes | 256 | 0.167 | 0.409 | 0.167 |
| leg_wing_lif | sampled_lif_spikes | 512 | 0.208 | 0.456 | 0.208 |
| gru_matched | engineering_absolute_activations | 33 | 0.077 | 0.105 | 1.000 |
| mlp_normal | engineering_absolute_activations | 122 | 0.118 | 0.165 | 1.000 |

LIF values are sampled spikes; GRU/MLP values are absolute hidden activations. They are descriptive within-controller measurements, not causal evidence and not directly equivalent in physical units.

| Role | frozen_lif_original | frozen_lif_degree_rewired | wing_lif | leg_wing_lif | gru_matched | mlp_normal |
| --- | --- | --- | --- | --- | --- | --- |
| gru:hidden | N/A | N/A | N/A | N/A | 0.077 (1.000 active) | N/A |
| leg:descending_input | 0.123 (0.123 active) | 0.081 (0.081 active) | N/A | 0.109 (0.109 active) | N/A | N/A |
| leg:motor_output | 0.002 (0.002 active) | 0.010 (0.010 active) | N/A | 0.025 (0.025 active) | N/A | N/A |
| leg:sensory_input | 0.151 (0.151 active) | 0.087 (0.087 active) | N/A | 0.127 (0.127 active) | N/A | N/A |
| leg:vnc_interneuron | 0.028 (0.028 active) | 0.014 (0.014 active) | N/A | 0.037 (0.037 active) | N/A | N/A |
| mlp:hidden_0 | N/A | N/A | N/A | N/A | N/A | 0.189 (1.000 active) |
| mlp:hidden_1 | N/A | N/A | N/A | N/A | N/A | 0.047 (1.000 active) |
| wing:descending_input | N/A | N/A | 0.101 (0.101 active) | 0.309 (0.309 active) | N/A | N/A |
| wing:vnc_interneuron | N/A | N/A | 0.145 (0.145 active) | 0.364 (0.364 active) | N/A | N/A |
| wing:wing_motor_output | N/A | N/A | 0.200 (0.200 active) | 0.399 (0.399 active) | N/A | N/A |
| wing:wing_sensory_input | N/A | N/A | 0.394 (0.394 active) | 0.420 (0.420 active) | N/A | N/A |

![Activity comparison](../runs/crazyflie_command_seed0_500k/command_activity_comparison.png)

## Per-job validation

| Queue job | Controller | Validation result | Checkpoint | History rows |
| --- | --- | --- | --- | --- |
| 01__frozen_lif_original__command__seed-0 | frozen_lif_original | VERIFIED COMPLETE | 6f9663e7c6dc… | 125 |
| 02__frozen_lif_degree_rewired__command__seed-0 | frozen_lif_degree_rewired | VERIFIED COMPLETE | 1ec1d4b1177b… | 125 |
| 03__wing_lif__command__seed-0 | wing_lif | VERIFIED COMPLETE | 4d7b980c33b7… | 125 |
| 04__leg_wing_lif__command__seed-0 | leg_wing_lif | VERIFIED COMPLETE | 5ef167c813e8… | 125 |
| 05__gru_matched__command__seed-0 | gru_matched | VERIFIED COMPLETE | 89da461141af… | 125 |
| 06__mlp_normal__command__seed-0 | mlp_normal | VERIFIED COMPLETE | fbb04f19217a… | 125 |

## Input boundary

The reporter read only the selected command config, its queue, and paths declared by those six queue jobs. Paused waypoint/gust queues and Unitree G1 artifacts are outside this report's input boundary.

