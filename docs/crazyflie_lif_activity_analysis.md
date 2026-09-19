# Crazyflie LIF neuron-activity analysis

## Status and scope

This report analyzes the completed seed-0, 500,000-interaction command-control
checkpoints. It compares the original leg/VNC LIF, the standalone
wing-thoracic LIF, and the independent leg + wing-thoracic LIF architecture.
It does not use training loss as a proxy for flight quality and does not claim
that correlated activity proves a biological or causal mechanism.

The held-out protocol contains 16 episodes of 600 control decisions, or 9,600
action-producing forward passes per model. All activity below came from the
same forward passes that produced the evaluated deterministic actions. The
evaluation records assert matching source, task, evaluation protocol, and
checkpoint identities, finite bounded actions, and actual-forward activity.

The word `dead` below has a narrow measurement definition: no sampled spike
was observed at the final LIF substep in any of the 9,600 control decisions.
It does not mean that a biological neuron is dead. The sampled-rate column is
`sampled active fraction / 0.02 s`; it is not a record of every internal spike
across all five neural substeps.

## Architecture-level result

| Architecture/core | Active neurons | Dead neurons | Mean sampled activity | Sampled rate |
|---|---:|---:|---:|---:|
| Original leg/VNC LIF | 99/256 | 157 | 4.026530% | 2.013265 Hz |
| Wing-thoracic LIF | 228/256 | 28 | 16.714030% | 8.357015 Hz |
| Combined model: leg core | 190/256 | 66 | 4.651815% | 2.325907 Hz |
| Combined model: wing-thoracic core | 240/256 | 16 | 36.902466% | 18.451233 Hz |
| Combined model: both cores | 430/512 | 82 | 20.777140% | 10.388570 Hz |

Within the combined controller, the wing-thoracic core's mean sampled
activity was 7.93 times the leg core's activity. High activity is not itself a
quality metric: the wing-only controller was more active than the leg-only
controller but had a lower command-control score.

## Activity by authenticated population

| Architecture/core | Sensory | Descending | Interneuron / thoracic | Motor |
|---|---:|---:|---:|---:|
| Original leg/VNC | 15.135851% | 12.333333% | 2.819010% | 0.210938% |
| Combined leg | 12.655382% | 10.884115% | 3.703802% | 2.470920% |
| Wing-thoracic only | 39.388542% | 10.064236% | 14.453073% | 19.984809% |
| Combined wing-thoracic | **42.043750%** | 30.888021% | 36.393333% | 39.868056% |

The extracted wing manifest does not contain a separate `torso` model role.
Thoracic circuit cells are represented primarily by `wing:vnc_interneuron`;
the report therefore retains the authenticated manifest label instead of
inventing a torso classification.

## Most active stable MaleCNS neuron IDs

Ties are ordered by stable neuron ID. A fraction of 0.5 means the neuron was
sampled active in 4,800 of 9,600 action-producing control decisions.

| Architecture/core | ID | Role | Active samples | Fraction |
|---|---|---|---:|---:|
| Leg-only | `804402` | VNC interneuron | 4,784 | 0.498333 |
| Leg-only | `804589` | VNC interneuron | 4,784 | 0.498333 |
| Leg-only | `908258` | sensory input | 4,784 | 0.498333 |
| Leg-only | `816782` | sensory input | 4,744 | 0.494167 |
| Leg-only | `817245` | sensory input | 4,740 | 0.493750 |
| Wing-only | `803702` | VNC interneuron | 4,800 | 0.500000 |
| Wing-only | `804373` | wing sensory input | 4,800 | 0.500000 |
| Wing-only | `806465` | VNC interneuron | 4,800 | 0.500000 |
| Wing-only | `807093` | wing sensory input | 4,800 | 0.500000 |
| Wing-only | `807458` | wing sensory input | 4,800 | 0.500000 |
| Combined leg | `811713` | sensory input | 4,800 | 0.500000 |
| Combined leg | `806212` | sensory input | 4,784 | 0.498333 |
| Combined leg | `804589` | VNC interneuron | 4,432 | 0.461667 |
| Combined leg | `908258` | sensory input | 4,432 | 0.461667 |
| Combined leg | `804402` | VNC interneuron | 4,428 | 0.461250 |
| Combined wing | `10056` | descending input | 4,800 | 0.500000 |
| Combined wing | `800574` | VNC interneuron | 4,800 | 0.500000 |
| Combined wing | `801590` | VNC interneuron | 4,800 | 0.500000 |
| Combined wing | `803702` | VNC interneuron | 4,800 | 0.500000 |
| Combined wing | `806465` | VNC interneuron | 4,800 | 0.500000 |

## Command-segment activity

Each row contains the mean sampled activity across the indicated 256-neuron
core. Each segment contains 800 samples: 16 episodes times 50 control steps.

| Segment | Wing-only | Combined leg | Combined wing |
|---|---:|---:|---:|
| Initial hover | 6.3750% | 2.8125% | 21.0234% |
| Forward | 11.9375% | 3.5469% | 37.4121% |
| Brake after forward | 17.5020% | 3.5469% | 37.7988% |
| Lateral | 17.0020% | 6.5469% | 38.0937% |
| Horizontal diagonal | 17.5742% | 4.2637% | 38.1660% |
| Vertical | 17.3564% | 4.5049% | 38.5264% |
| Yaw | 17.2959% | 3.8457% | 38.5049% |
| Full simultaneous | 17.0132% | 5.0381% | 38.5874% |
| Reverse full simultaneous | 18.0898% | 7.3701% | 38.6411% |
| Brake after full | 19.9067% | 4.8970% | 38.6929% |
| Reverse horizontal | 20.3335% | 4.3511% | 38.7207% |
| Final hover | 20.1821% | 5.0981% | 38.6621% |

The wing-only and combined recurrent states retain command history, so final
hover activity need not return to initial-hover activity. This observation is
descriptive; it is not proof of memory causing better control.

## Same frozen core in different architectures

The standalone and combined wing conditions use the same frozen wing topology
and synaptic-weight checksum, `461ff516c0d7e3feb79c8aac86dbc3f5242940564b84de6a18cac9edb27719eb`.
The original and combined leg conditions likewise share leg-core checksum
`cf2c0b8f9fe4319eeebabeefaebd6125084012abd5b8f891422dea2eede4fc54`.
Their trainable encoders/decoders and observed state trajectories differ.

| Comparison | Pearson | Spearman | Became active | Became unsampled |
|---|---:|---:|---:|---:|
| Leg-only vs combined leg | 0.767745 | 0.703314 | 93 | 2 |
| Wing-only vs combined wing | 0.259765 | 0.451050 | 16 | 4 |

Largest wing activity increases in the combined architecture were:

| ID | Wing-only | Combined wing | Change |
|---|---:|---:|---:|
| `808625` | 0.000000 | 0.499479 | +0.499479 |
| `807230` | 0.000208 | 0.499375 | +0.499167 |
| `805737` | 0.000000 | 0.498854 | +0.498854 |
| `806337` | 0.000000 | 0.498542 | +0.498542 |
| `807751` | 0.001979 | 0.498542 | +0.496563 |

The largest decreases included wing neuron `814621`
(`0.380729 -> 0.000313`) and descending neuron `11737`
(`0.326771 -> 0.001458`). The combined model therefore did not simply increase
every neuron's firing; it changed the observed activity pattern substantially.

## Flight result and capacity caveat

| Model | Command-control score | Evaluation reward mean | Full-horizon survival | Actor parameters |
|---|---:|---:|---:|---:|
| Original leg/VNC LIF | 51.448654 | **21.257893** | 16/16 | 4,776 |
| Wing-thoracic LIF | 49.848517 | 18.819551 | 16/16 | 4,776 |
| Leg + wing-thoracic LIF | **52.392087** | 20.908831 | 16/16 | 9,224 |

The combined model leads the transparent control score by 2.543570 points
over wing-only and 0.943433 over leg-only. Leg-only has the highest mean task
reward. The combined actor has 93.1% more trainable actor parameters than
wing-only, so these results cannot isolate topology from capacity.

## Continuous viewer validation

The trained viewer now supports `--continuous`. It leaves the authenticated
600-step task configuration unchanged and suppresses only the viewer's timeout
signal. Hard-safety and nonfinite termination remain enabled. A 650-step
headless validation crossed the original 600-step boundary and reported:

- status `PASS`;
- 650/650 trained-policy actions;
- zero fallback actions;
- zero truncation resets;
- zero termination resets;
- zero invalid-state resets; and
- finite final state.

The scripted smoke intentionally exercised one manual reset at step 40. The
interactive command contains no scripted reset; it continues until `Esc`
unless the user presses `R` or a real hard-safety/nonfinite termination occurs.

```bash
cd /home/chayanin/Desktop/flyg1
/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python \
  scripts/crazyflie_trained_keyboard.py \
  --checkpoint runs/crazyflie_command_seed0_500k/jobs/04__leg_wing_lif__command__seed-0/checkpoints/latest.pt \
  --horizontal_speed 0.50 --vertical_speed 0.25 --yaw_rate 0.80 \
  --trail_points 300 --continuous --offline_only
```

## Recommended next gate

Before interpreting the activity as causal, run an evaluation-only ablation
matrix on this fixed checkpoint and held-out protocol:

1. unmodified leg + wing controller;
2. leg-core output suppressed;
3. wing-core output suppressed;
4. top-10 active leg neurons suppressed;
5. top-10 active wing neurons suppressed; and
6. stable-ID-matched random-10 controls for each core.

Report score and raw tracking/safety deltas, and do not retrain during this
gate. After that, complete the additive optic and physical-wind integration,
run bounded 14-cell smoke/resume/memory gates, produce the 14-job 1M dry run,
and only then launch the sequential 14-million-interaction matrix.

## Authenticated inputs

| Artifact | SHA-256 |
|---|---|
| Original LIF evaluation | `ef1f6a749b67277586cde2131f85e8e58f51399d0e6d5bcd5125c6b26f65b3da` |
| Wing LIF evaluation | `706b83c384e9eb2850359d9d02dd076b829a4a902631a63cd6d9bba36af34b76` |
| Leg + wing LIF evaluation | `636a173510f1408846a1976d775b7da3d112bd9e6d0bd16849f3e1e93875aebd` |
| Leg + wing checkpoint | `5ef167c813e8ed91238560604ad22d43398893d938e3ecba797a1181cc48b17c` |

