# Crazyflie wide-command optic/wind comparison report

Report state: **COMPLETE — all 14 jobs independently verified**

Generated (UTC): `2026-09-18T11:16:52+00:00`  
Tasks: `FlyCrazyflie-CommandFollowWide-v0` and `FlyCrazyflie-CommandFollowWideWind-v0`  
Config SHA-256: `bf7369d74d119784eb6a932cbeb57734fa93b7fdda5457e3e46decfb6f140c44`  
Queue snapshot SHA-256: `68f4fee83d659edb439691aa79c5f4b6decc610c1e35e869b1caddd04cf5bddc`

Every numeric row below comes from an authenticated completed 1,000,000-interaction checkpoint and its matching deterministic 16 × 600 held-out evaluation. Pending, failed, missing, hash-mismatched, nonfinite, or memory-gate-failing cells remain N/A. Still and wind jobs are independently trained but use the same seed, PPO budget, command schedule seed, and per-controller architecture.

## Controller/task scores

| Controller | Condition | Evidence state | Actor params | Total /100 | Linear tracking /100 | Linear RMSE m/s | Yaw RMSE rad/s | Acceleration /100 | Response /100 | Stability /100 | Survival /100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_lif | still | VERIFIED COMPLETE | 4776 | 47.996 | 8.647 | 0.548 | 0.344 | 97.917 | 64.401 | 77.179 | 100.000 |
| original_lif | wind | VERIFIED COMPLETE | 4776 | 33.400 | 7.736e-04 | 1.201 | 1.370 | 63.673 | 64.880 | 8.144 | 68.708 |
| rewired_lif | still | VERIFIED COMPLETE | 4776 | 47.238 | 9.189 | 0.541 | 0.392 | 97.839 | 63.800 | 75.845 | 100.000 |
| rewired_lif | wind | VERIFIED COMPLETE | 4776 | 33.501 | 5.526e-04 | 1.218 | 1.364 | 63.237 | 64.761 | 8.145 | 68.875 |
| wing_lif | still | VERIFIED COMPLETE | 4776 | 46.966 | 8.393 | 0.551 | 0.387 | 97.765 | 64.351 | 76.104 | 100.000 |
| wing_lif | wind | VERIFIED COMPLETE | 4776 | 32.528 | 4.661e-04 | 1.226 | 1.441 | 61.436 | 60.523 | 7.297 | 66.979 |
| leg_wing_lif | still | VERIFIED COMPLETE | 9224 | 48.288 | 9.081 | 0.542 | 0.343 | 97.787 | 64.603 | 77.101 | 100.000 |
| leg_wing_lif | wind | VERIFIED COMPLETE | 9224 | 33.544 | 7.121e-04 | 1.205 | 1.393 | 64.550 | 62.644 | 7.972 | 70.177 |
| optic_lif | still | VERIFIED COMPLETE | 4776 | 46.212 | 9.001 | 0.543 | 0.443 | 96.477 | 62.530 | 74.524 | 100.000 |
| optic_lif | wind | VERIFIED COMPLETE | 4776 | 33.609 | 2.510e-04 | 1.257 | 1.385 | 61.321 | 62.650 | 8.254 | 74.854 |
| gru_matched | still | VERIFIED COMPLETE | 4793 | 48.522 | 9.400 | 0.538 | 0.343 | 98.045 | 65.211 | 76.018 | 100.000 |
| gru_matched | wind | VERIFIED COMPLETE | 4793 | 31.990 | 9.660e-04 | 1.189 | 1.373 | 65.980 | 61.573 | 8.978 | 67.188 |
| mlp_normal | still | VERIFIED COMPLETE | 4827 | 48.960 | 9.665 | 0.535 | 0.329 | 97.838 | 66.186 | 75.402 | 100.000 |
| mlp_normal | wind | VERIFIED COMPLETE | 4827 | 31.279 | 3.793e-04 | 1.237 | 1.654 | 63.072 | 52.357 | 5.536 | 74.094 |

![Score by condition](../runs/crazyflie_command_optic_wind_seed0_1m/command_v2_score_by_condition.png)

## Wind effect (wind minus still)

| Controller | Pair state | Δ total | Δ tracking score | Δ linear RMSE | Δ acceleration | Δ stability | Δ survival |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_lif | paired | -14.595 | -8.646 | 0.653 | -34.243 | -69.036 | -31.292 |
| rewired_lif | paired | -13.737 | -9.189 | 0.677 | -34.602 | -67.700 | -31.125 |
| wing_lif | paired | -14.438 | -8.393 | 0.675 | -36.329 | -68.808 | -33.021 |
| leg_wing_lif | paired | -14.744 | -9.080 | 0.663 | -33.237 | -69.129 | -29.823 |
| optic_lif | paired | -12.603 | -9.001 | 0.714 | -35.156 | -66.271 | -25.146 |
| gru_matched | paired | -16.532 | -9.399 | 0.651 | -32.065 | -67.040 | -32.812 |
| mlp_normal | paired | -17.681 | -9.665 | 0.702 | -34.767 | -69.866 | -25.906 |

For score columns, negative means wind reduced performance. For RMSE, positive means wind increased error. These are descriptive paired-condition differences, not uncertainty-adjusted causal estimates.

![Wind deltas](../runs/crazyflie_command_optic_wind_seed0_1m/command_v2_wind_delta.png)

## Capacity and frozen-core contract

| Controller | Actor trainable | Total trainable | Frozen weights | Parameter match | Frozen-core check | Core SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| original_lif | 4776 | 23081 | 5103 | PASS | PASS in both conditions | cf2c0b8f9fe4… |
| rewired_lif | 4776 | 23081 | 5103 | PASS | PASS in both conditions | 923cb1ad72c4… |
| wing_lif | 4776 | 23081 | 8864 | not required | PASS in both conditions | 461ff516c0d7… |
| leg_wing_lif | 9224 | 27529 | 13967 | not required | PASS in both conditions | 333dfc4caccb… |
| optic_lif | 4776 | 23081 | 1628 | PASS | PASS in both conditions | 268ac6eb120e… |
| gru_matched | 4793 | 23098 | 0 | PASS | N/A (non-LIF) | N/A |
| mlp_normal | 4827 | 23132 | 0 | PASS | N/A (non-LIF) | N/A |

## LIF role activity

Activity is taken from the exact forward pass that produced each evaluated action. The table reports sampled spike activity; it does not imply causal importance.

| Controller | Condition | Role | Units | Mean \|spike\| | Active fraction |
| --- | --- | --- | --- | --- | --- |
| original_lif | still | leg:descending_input | 8 | 0.256 | 0.256 |
| original_lif | still | leg:motor_output | 24 | 0.204 | 0.204 |
| original_lif | still | leg:sensory_input | 24 | 0.235 | 0.235 |
| original_lif | still | leg:vnc_interneuron | 200 | 0.206 | 0.206 |
| original_lif | wind | leg:descending_input | 8 | 0.156 | 0.156 |
| original_lif | wind | leg:motor_output | 24 | 0.165 | 0.165 |
| original_lif | wind | leg:sensory_input | 24 | 0.191 | 0.191 |
| original_lif | wind | leg:vnc_interneuron | 200 | 0.165 | 0.165 |
| rewired_lif | still | leg:descending_input | 8 | 0.152 | 0.152 |
| rewired_lif | still | leg:motor_output | 24 | 0.241 | 0.241 |
| rewired_lif | still | leg:sensory_input | 24 | 0.102 | 0.102 |
| rewired_lif | still | leg:vnc_interneuron | 200 | 0.153 | 0.153 |
| rewired_lif | wind | leg:descending_input | 8 | 0.167 | 0.167 |
| rewired_lif | wind | leg:motor_output | 24 | 0.244 | 0.244 |
| rewired_lif | wind | leg:sensory_input | 24 | 0.104 | 0.104 |
| rewired_lif | wind | leg:vnc_interneuron | 200 | 0.144 | 0.144 |
| wing_lif | still | wing:descending_input | 12 | 0.207 | 0.207 |
| wing_lif | still | wing:vnc_interneuron | 200 | 0.292 | 0.292 |
| wing_lif | still | wing:wing_motor_output | 24 | 0.355 | 0.355 |
| wing_lif | still | wing:wing_sensory_input | 20 | 0.400 | 0.400 |
| wing_lif | wind | wing:descending_input | 12 | 0.234 | 0.234 |
| wing_lif | wind | wing:vnc_interneuron | 200 | 0.364 | 0.364 |
| wing_lif | wind | wing:wing_motor_output | 24 | 0.445 | 0.445 |
| wing_lif | wind | wing:wing_sensory_input | 20 | 0.394 | 0.394 |
| leg_wing_lif | still | leg:descending_input | 8 | 0.192 | 0.192 |
| leg_wing_lif | still | leg:motor_output | 24 | 0.171 | 0.171 |
| leg_wing_lif | still | leg:sensory_input | 24 | 0.212 | 0.212 |
| leg_wing_lif | still | leg:vnc_interneuron | 200 | 0.164 | 0.164 |
| leg_wing_lif | still | wing:descending_input | 12 | 0.264 | 0.264 |
| leg_wing_lif | still | wing:vnc_interneuron | 200 | 0.379 | 0.379 |
| leg_wing_lif | still | wing:wing_motor_output | 24 | 0.455 | 0.455 |
| leg_wing_lif | still | wing:wing_sensory_input | 20 | 0.393 | 0.393 |
| leg_wing_lif | wind | leg:descending_input | 8 | 0.116 | 0.116 |
| leg_wing_lif | wind | leg:motor_output | 24 | 0.056 | 0.056 |
| leg_wing_lif | wind | leg:sensory_input | 24 | 0.176 | 0.176 |
| leg_wing_lif | wind | leg:vnc_interneuron | 200 | 0.075 | 0.075 |
| leg_wing_lif | wind | wing:descending_input | 12 | 0.285 | 0.285 |
| leg_wing_lif | wind | wing:vnc_interneuron | 200 | 0.375 | 0.375 |
| leg_wing_lif | wind | wing:wing_motor_output | 24 | 0.417 | 0.417 |
| leg_wing_lif | wind | wing:wing_sensory_input | 20 | 0.478 | 0.478 |
| optic_lif | still | optic:optic_intrinsic_interneuron | 200 | 0.000 | 0.000 |
| optic_lif | still | optic:optic_sensory_input | 32 | 0.090 | 0.090 |
| optic_lif | still | optic:visual_projection_output | 24 | 0.000 | 0.000 |
| optic_lif | wind | optic:optic_intrinsic_interneuron | 200 | 0.000 | 0.000 |
| optic_lif | wind | optic:optic_sensory_input | 32 | 0.074 | 0.074 |
| optic_lif | wind | optic:visual_projection_output | 24 | 0.000 | 0.000 |

![LIF activity](../runs/crazyflie_command_optic_wind_seed0_1m/command_v2_lif_activity.png)

## Most active LIF neurons

| Controller | Condition | Rank | Neuron ID | Role | Mean \|activity\| | RMS | Active fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_lif | still | 1 | leg:804589 | leg:vnc_interneuron | 0.501 | 0.707 | 0.501 |
| original_lif | still | 2 | leg:800501 | leg:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| original_lif | still | 3 | leg:806080 | leg:sensory_input | 0.500 | 0.707 | 0.500 |
| original_lif | still | 4 | leg:903735 | leg:sensory_input | 0.500 | 0.707 | 0.500 |
| original_lif | still | 5 | leg:816782 | leg:sensory_input | 0.499 | 0.707 | 0.499 |
| original_lif | still | 6 | leg:806212 | leg:sensory_input | 0.498 | 0.706 | 0.498 |
| original_lif | still | 7 | leg:811713 | leg:sensory_input | 0.498 | 0.706 | 0.498 |
| original_lif | still | 8 | leg:908258 | leg:sensory_input | 0.498 | 0.706 | 0.498 |
| original_lif | still | 9 | leg:904022 | leg:vnc_interneuron | 0.497 | 0.705 | 0.497 |
| original_lif | still | 10 | leg:800233 | leg:vnc_interneuron | 0.497 | 0.705 | 0.497 |
| original_lif | wind | 1 | leg:820164 | leg:sensory_input | 0.501 | 0.708 | 0.501 |
| original_lif | wind | 2 | leg:800125 | leg:vnc_interneuron | 0.499 | 0.707 | 0.499 |
| original_lif | wind | 3 | leg:811713 | leg:sensory_input | 0.499 | 0.707 | 0.499 |
| original_lif | wind | 4 | leg:908258 | leg:sensory_input | 0.498 | 0.706 | 0.498 |
| original_lif | wind | 5 | leg:806212 | leg:sensory_input | 0.498 | 0.706 | 0.498 |
| original_lif | wind | 6 | leg:804589 | leg:vnc_interneuron | 0.498 | 0.705 | 0.498 |
| original_lif | wind | 7 | leg:804402 | leg:vnc_interneuron | 0.497 | 0.705 | 0.497 |
| original_lif | wind | 8 | leg:800111 | leg:vnc_interneuron | 0.491 | 0.701 | 0.491 |
| original_lif | wind | 9 | leg:800642 | leg:vnc_interneuron | 0.486 | 0.697 | 0.486 |
| original_lif | wind | 10 | leg:800856 | leg:vnc_interneuron | 0.485 | 0.696 | 0.485 |
| rewired_lif | still | 1 | leg:904657 | leg:motor_output | 0.485 | 0.696 | 0.485 |
| rewired_lif | still | 2 | leg:800461 | leg:vnc_interneuron | 0.484 | 0.696 | 0.484 |
| rewired_lif | still | 3 | leg:800003 | leg:vnc_interneuron | 0.461 | 0.679 | 0.461 |
| rewired_lif | still | 4 | leg:800211 | leg:vnc_interneuron | 0.458 | 0.676 | 0.458 |
| rewired_lif | still | 5 | leg:800175 | leg:motor_output | 0.451 | 0.671 | 0.451 |
| rewired_lif | still | 6 | leg:801552 | leg:vnc_interneuron | 0.443 | 0.666 | 0.443 |
| rewired_lif | still | 7 | leg:801768 | leg:motor_output | 0.443 | 0.665 | 0.443 |
| rewired_lif | still | 8 | leg:802143 | leg:vnc_interneuron | 0.441 | 0.664 | 0.441 |
| rewired_lif | still | 9 | leg:801057 | leg:motor_output | 0.435 | 0.660 | 0.435 |
| rewired_lif | still | 10 | leg:801058 | leg:motor_output | 0.433 | 0.658 | 0.433 |
| rewired_lif | wind | 1 | leg:801768 | leg:motor_output | 0.471 | 0.687 | 0.471 |
| rewired_lif | wind | 2 | leg:801171 | leg:vnc_interneuron | 0.467 | 0.684 | 0.467 |
| rewired_lif | wind | 3 | leg:800461 | leg:vnc_interneuron | 0.466 | 0.683 | 0.466 |
| rewired_lif | wind | 4 | leg:904657 | leg:motor_output | 0.459 | 0.678 | 0.459 |
| rewired_lif | wind | 5 | leg:10045 | leg:descending_input | 0.450 | 0.671 | 0.450 |
| rewired_lif | wind | 6 | leg:800003 | leg:vnc_interneuron | 0.449 | 0.670 | 0.449 |
| rewired_lif | wind | 7 | leg:800175 | leg:motor_output | 0.433 | 0.658 | 0.433 |
| rewired_lif | wind | 8 | leg:820164 | leg:sensory_input | 0.424 | 0.651 | 0.424 |
| rewired_lif | wind | 9 | leg:801057 | leg:motor_output | 0.420 | 0.648 | 0.420 |
| rewired_lif | wind | 10 | leg:800856 | leg:vnc_interneuron | 0.419 | 0.648 | 0.419 |
| wing_lif | still | 1 | wing:803702 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 2 | wing:804373 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 3 | wing:806465 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 4 | wing:807093 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 5 | wing:807458 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 6 | wing:807510 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 7 | wing:807554 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 8 | wing:807561 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 9 | wing:807714 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | still | 10 | wing:807817 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 1 | wing:803702 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 2 | wing:804373 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 3 | wing:807093 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 4 | wing:807458 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 5 | wing:807890 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 6 | wing:909349 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 7 | wing:806465 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 8 | wing:807561 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 9 | wing:808022 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| wing_lif | wind | 10 | wing:903423 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 1 | wing:909760 | wing:vnc_interneuron | 0.503 | 0.709 | 0.503 |
| leg_wing_lif | still | 2 | wing:806787 | wing:vnc_interneuron | 0.502 | 0.708 | 0.502 |
| leg_wing_lif | still | 3 | leg:908258 | leg:sensory_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 4 | wing:10056 | wing:descending_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 5 | wing:801590 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 6 | wing:803702 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 7 | wing:804373 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 8 | wing:806465 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 9 | wing:807093 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | still | 10 | wing:807458 | wing:wing_sensory_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | wind | 1 | wing:801653 | wing:vnc_interneuron | 0.502 | 0.708 | 0.502 |
| leg_wing_lif | wind | 2 | leg:806212 | leg:sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 3 | wing:807093 | wing:wing_sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 4 | wing:807458 | wing:wing_sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 5 | wing:808075 | wing:wing_sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 6 | wing:808461 | wing:wing_sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 7 | wing:808625 | wing:wing_sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 8 | wing:903423 | wing:wing_sensory_input | 0.501 | 0.707 | 0.501 |
| leg_wing_lif | wind | 9 | wing:10223 | wing:descending_input | 0.500 | 0.707 | 0.500 |
| leg_wing_lif | wind | 10 | wing:807230 | wing:vnc_interneuron | 0.500 | 0.707 | 0.500 |
| optic_lif | still | 1 | optic:35426 | optic:optic_sensory_input | 0.494 | 0.703 | 0.494 |
| optic_lif | still | 2 | optic:285891 | optic:optic_sensory_input | 0.487 | 0.698 | 0.487 |
| optic_lif | still | 3 | optic:173575 | optic:optic_sensory_input | 0.467 | 0.683 | 0.467 |
| optic_lif | still | 4 | optic:197943 | optic:optic_sensory_input | 0.289 | 0.538 | 0.289 |
| optic_lif | still | 5 | optic:198133 | optic:optic_sensory_input | 0.253 | 0.503 | 0.253 |
| optic_lif | still | 6 | optic:218571 | optic:optic_sensory_input | 0.248 | 0.498 | 0.248 |
| optic_lif | still | 7 | optic:192209 | optic:optic_sensory_input | 0.172 | 0.415 | 0.172 |
| optic_lif | still | 8 | optic:186286 | optic:optic_sensory_input | 0.125 | 0.353 | 0.125 |
| optic_lif | still | 9 | optic:167204 | optic:optic_sensory_input | 0.083 | 0.288 | 0.083 |
| optic_lif | still | 10 | optic:182558 | optic:optic_sensory_input | 0.071 | 0.266 | 0.071 |
| optic_lif | wind | 1 | optic:35426 | optic:optic_sensory_input | 0.449 | 0.670 | 0.449 |
| optic_lif | wind | 2 | optic:218571 | optic:optic_sensory_input | 0.422 | 0.650 | 0.422 |
| optic_lif | wind | 3 | optic:173575 | optic:optic_sensory_input | 0.361 | 0.601 | 0.361 |
| optic_lif | wind | 4 | optic:285891 | optic:optic_sensory_input | 0.287 | 0.536 | 0.287 |
| optic_lif | wind | 5 | optic:186286 | optic:optic_sensory_input | 0.283 | 0.532 | 0.283 |
| optic_lif | wind | 6 | optic:197943 | optic:optic_sensory_input | 0.154 | 0.392 | 0.154 |
| optic_lif | wind | 7 | optic:170342 | optic:optic_sensory_input | 0.100 | 0.316 | 0.100 |
| optic_lif | wind | 8 | optic:172592 | optic:optic_sensory_input | 0.060 | 0.245 | 0.060 |
| optic_lif | wind | 9 | optic:182558 | optic:optic_sensory_input | 0.045 | 0.211 | 0.045 |
| optic_lif | wind | 10 | optic:447720 | optic:optic_sensory_input | 0.026 | 0.161 | 0.026 |

## Physical wind evidence

| Controller | Condition | Evidence state | Recorded telemetry summary | Physical telemetry integrity | Actual-interval source |
| --- | --- | --- | --- | --- | --- |
| original_lif | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| original_lif | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010541348718106747,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.0003326537844259292},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":6609,"observed_nonzero_wrench_intervals":1483,"observed_planned_pulse_intervals":1483,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| rewired_lif | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| rewired_lif | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010524827055633068,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.0003317219379823655},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":6625,"observed_nonzero_wrench_intervals":1484,"observed_planned_pulse_intervals":1484,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| wing_lif | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| wing_lif | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010552360676229,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.0003315550566185266},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":6443,"observed_nonzero_wrench_intervals":1447,"observed_planned_pulse_intervals":1447,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| leg_wing_lif | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| leg_wing_lif | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010401501320302486,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.0003299604868516326},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":6749,"observed_nonzero_wrench_intervals":1475,"observed_planned_pulse_intervals":1475,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| optic_lif | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| optic_lif | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010330302640795708,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.00031975682941265404},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":7197,"observed_nonzero_wrench_intervals":1568,"observed_planned_pulse_intervals":1568,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| gru_matched | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| gru_matched | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010407192632555962,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.0003302834811620414},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":6461,"observed_nonzero_wrench_intervals":1428,"observed_planned_pulse_intervals":1428,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| mlp_normal | still | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"still_air","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":true,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":true,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":null},"limits":{"force_to_weight_ratio":0.0,"maximum_force_norm_n":0.0,"maximum_torque_norm_nm":0.0,"torque_to_weight_arm_ratio":0.0},"measured":{"force_rms_n_observed":0.0,"maximum_force_norm_n_all_simulated_intervals":0.0,"maximum_torque_norm_nm_all_simulated_intervals":0.0,"torque_rms_nm_observed":0.0},"protocol":{"pulse_duration_steps":0,"pulse_start_steps":[],"seed":null},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":9600,"observed_nonzero_wrench_intervals":0,"observed_planned_pulse_intervals":0,"planned_intervals":9600,"planned_pulse_intervals":0},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |
| mlp_normal | wind | VERIFIED COMPLETE | {"application_point":"body_center_of_mass","condition":"wind","frame":"world","integrity":{"all_values_finite":true,"category_codes_valid_all_simulated_intervals":true,"complete_600_step_protocol_observed_for_all_episodes":false,"expected_category_matches_on_observed_intervals":true,"expected_force_matches_on_observed_intervals":true,"expected_torque_matches_on_observed_intervals":true,"force_within_declared_bound":true,"passed":true,"pulse_window_matches_on_observed_intervals":true,"still_air_exact_zero_all_simulated_intervals":null,"torque_within_declared_bound":true,"wind_nonzero_wrench_observed":true},"limits":{"force_to_weight_ratio":0.15,"maximum_force_norm_n":0.041496303677558896,"maximum_torque_norm_nm":0.001272553312778473,"torque_to_weight_arm_ratio":0.1},"measured":{"force_rms_n_observed":0.010441379621624947,"maximum_force_norm_n_all_simulated_intervals":0.041381482034921646,"maximum_torque_norm_nm_all_simulated_intervals":0.0012636833125725389,"torque_rms_nm_observed":0.00032859755447134376},"protocol":{"pulse_duration_steps":25,"pulse_start_steps":[75,175,275,375,475],"seed":20260919},"reference_arm_m":0.046,"samples":{"observed_intervals_through_first_done":7124,"observed_nonzero_wrench_intervals":1598,"observed_planned_pulse_intervals":1598,"planned_intervals":9600,"planned_pulse_intervals":2000},"source":"terminal_applied_wind_force_world/torque_world captured immediately after env.step for the physical interval just simulated","vehicle_weight_n":0.2766420245170593} | True | True |

![Training reward](../runs/crazyflie_command_optic_wind_seed0_1m/command_v2_training_reward.png)

![Training loss](../runs/crazyflie_command_optic_wind_seed0_1m/command_v2_training_loss.png)

## Per-job validation

| Queue job | Controller | Condition | Validation | Checkpoint | History rows |
| --- | --- | --- | --- | --- | --- |
| 01__original_lif__still__seed-0 | original_lif | still | VERIFIED COMPLETE | af888f5bfa94… | 250 |
| 02__original_lif__wind__seed-0 | original_lif | wind | VERIFIED COMPLETE | 4eb7bc7a6a5f… | 250 |
| 03__rewired_lif__still__seed-0 | rewired_lif | still | VERIFIED COMPLETE | 02cf5587bcca… | 250 |
| 04__rewired_lif__wind__seed-0 | rewired_lif | wind | VERIFIED COMPLETE | 6869579fce11… | 250 |
| 05__wing_lif__still__seed-0 | wing_lif | still | VERIFIED COMPLETE | 0f7f9bc38f01… | 250 |
| 06__wing_lif__wind__seed-0 | wing_lif | wind | VERIFIED COMPLETE | 624fa993e992… | 250 |
| 07__leg_wing_lif__still__seed-0 | leg_wing_lif | still | VERIFIED COMPLETE | 9b15d145ed4f… | 250 |
| 08__leg_wing_lif__wind__seed-0 | leg_wing_lif | wind | VERIFIED COMPLETE | 6bb4bc96e600… | 250 |
| 09__optic_lif__still__seed-0 | optic_lif | still | VERIFIED COMPLETE | 8c2ed357e2da… | 250 |
| 10__optic_lif__wind__seed-0 | optic_lif | wind | VERIFIED COMPLETE | 08d79b01d29b… | 250 |
| 11__gru_matched__still__seed-0 | gru_matched | still | VERIFIED COMPLETE | 05f11d27cfcb… | 250 |
| 12__gru_matched__wind__seed-0 | gru_matched | wind | VERIFIED COMPLETE | 1a5f1a56c42e… | 250 |
| 13__mlp_normal__still__seed-0 | mlp_normal | still | VERIFIED COMPLETE | 14011bf5cf2c… | 250 |
| 14__mlp_normal__wind__seed-0 | mlp_normal | wind | VERIFIED COMPLETE | 86b9690b2f9b… | 250 |

## Missing or rejected evidence

None — all 14 cells passed artifact validation.

## Input boundary

The reporter read only the selected v2 config, its queue, queue-authenticated runner/trainer/evaluator files, and artifact paths declared by those 14 jobs. It did not read or modify paused Unitree G1, legacy waypoint/gust, or v1 run artifacts.

