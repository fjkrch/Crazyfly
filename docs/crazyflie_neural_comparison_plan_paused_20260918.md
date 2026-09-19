# PAUSED — Crazyflie manual flight and neural-controller comparison plan

## Historical objective — paused by user on 2026-09-18

This comparison is preserved for provenance only. Do not resume its queue or
launch any Reach, Switch, or Gust training/evaluation unless the user gives a
new explicit instruction to do so. The active manual-only plan is `../plan.md`.

This document is the only active Crazyflie execution plan. It supersedes the
controller-comparison plan archived at
`docs/crazyflie_controller_comparison_plan_paused_20260918.md`.  The manual
flight implementation remains complete and preserved.  The user subsequently
authorized the separate training comparison in **Phase 2** below; that newer
authorization supersedes the earlier manual-only prohibition where the two
conflict.

Fly one installed Isaac Lab Crazyflie manually in the visible Isaac Sim viewer.
The operator's held keyboard keys are the sole navigation command. A
deterministic, non-learning flight-assist mixer converts those commands into the
installed four-value aggregate-wrench action while maintaining hover and
attitude stability.

Manual control never uses PPO, rewards, a checkpoint, or neural output. Rewards
returned internally by the Gym environment are discarded by the manual path.
Do not resume or relabel an old queue as Phase 2 evidence. Preserve every
existing run, checkpoint, queue, manifest, log, and report unchanged.

## Keyboard contract

| Held key | Body-relative command |
|---|---|
| `W` | forward |
| `S` | backward |
| `A` | left |
| `D` | right |
| `I` or `E` | up |
| `Q` | down |
| `J` | yaw left |
| `L` | yaw right |
| `H` or `Space` | clear movement keys and hover |
| `R` | reset the drone, set a new hover point, and clear held keys |
| `Esc` | exit cleanly |

Commands remain active only while keys are held. Releasing one key removes only
that key's contribution. Opposite keys cancel. Key-repeat events never
accumulate. Multiple axes may be commanded simultaneously: `W+A` flies
diagonally forward-left, `W+I` flies forward-up, and `W+A+I` combines all three.
The horizontal vector is normalized so a diagonal command is not faster than a
single horizontal direction; vertical and yaw commands remain independent.

Run one environment at the installed 50 Hz control rate. Movement is relative
to the drone's current heading. With no movement key held, the flight assist
brakes velocity and holds the current commanded position. Actions must remain
finite and inside `[-1, 1]`. Height and horizontal setpoints remain inside
explicit safety bounds, and an invalid state or safety termination triggers a
visible reset rather than continuing with stale inputs.

## Wing + thoracic neural monitor

The existing `data/connectome_wing/manifest.json` is a 256-cell wing-thoracic
VNC circuit: 20 wing sensory inputs, 12 descending inputs, 200 thoracic
intrinsic neurons, and 24 wing motor readouts. The manual viewer may run this
frozen circuit, or the existing independent-core leg-plus-wing-thoracic
combination, as a real-time activity monitor. In visible mode the monitor opens
a separate role-colored neural activity window showing the wing sensory,
descending, thoracic intrinsic, and wing motor populations. The neural input's
target component is the manual flight-assist setpoint, never the native task's
unused random waypoint. `wing_thoracic` is the default; `--neural_monitor none`
may explicitly disable it, and `leg_wing_thoracic` selects the larger optional
independent-core combination.

The monitor is diagnostic only. Its untrained adapter/decoder output must never
be mixed into the flight action, and the UI and documentation must state that
the deterministic flight assist—not the neural monitor—controls the vehicle.
This separation guarantees that the Crazyflie follows the operator's command
without pretending that an unvalidated biological circuit learned flight.

## Installed simulator contract

Use `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python` and the
installed `Isaac-Quadcopter-Direct-v0` environment. Preserve its Crazyflie
body, 0.01 s physics timestep, decimation 2, 12-value state observation, and
four actions:

1. normalized collective thrust, mapped to `[0, 1.9 x vehicle weight]`;
2. normalized body roll moment;
3. normalized body pitch moment;
4. normalized body yaw moment.

Never send XYZ keyboard values directly as these four actions. Use the fixed
hover/attitude/velocity flight-assist conversion. No checkpoint is required or
loaded.

Use the pinned Crazyflie `cf2x.usd`. Build a local offline mirror from the two
verified Omniverse cache files (the top-level asset and its relative robot
schema dependency) before Isaac starts. Do not edit the installed Isaac Lab or
Unitree G1 packages. If neither the verified mirror nor verified cache sources
exist, fail clearly; an optional online fetch may populate the mirror only
after matching the pinned SHA-256 values. Replace the native cloud-backed grid
USD with an equivalent locally generated 20 m flat collision mesh so the full
manual scene works with `--offline_only`.

## Additive implementation boundary

Implement teleoperation in a standalone non-`drone_*.py` entry point so the
historical checkpoint reproduction source set does not change. Do not repurpose
`scripts/drone_play.py` or any training/evaluation launcher. The implementation
may read existing fixed stabilization and connectome modules but must not write
any model, optimizer, checkpoint, reward, evaluation, or queue artifact.

The paused Unitree G1 work is immutable. The before-manifest plus supplement
defines 37 frozen G1 files; recompute the after-manifest and require an exact
37/37 match after implementation. Never edit, resume, rename, or delete a G1
run, checkpoint, queue, log, or pause file.

## Acceptance gates

1. Pure unit tests cover press/hold/release, multi-key combination, horizontal
   diagonal normalization, opposing-key cancellation, hover/reset clearing,
   finite bounded actions, and the declared direction-to-moment signs.
2. A headless scripted smoke uses the exact interactive control path for idle,
   `W`, `W+A`, up, down, release-to-hover, and reset. It reports displacement,
   termination/reset count, action bounds, and finite-state checks. It is a
   control test, not training or evaluation.
3. A visible viewer launch presents the key map, close camera, command/state
   telemetry, and optional wing-thoracic activity. Closing the viewer or
   pressing `Esc` exits cleanly with no Isaac process left behind.
4. The local asset mirror works without network access after preparation.
5. The manual acceptance run does not implicitly launch training or evaluation;
   all prior artifacts remain untouched and all 37 frozen G1 hashes match.

## Phase 2 — seed-0 leg/wing/baseline comparison

### Question and controller conditions

Train a new task-separated comparison that asks how the frozen biological
circuits perform relative to conventional neural controllers.  It contains
six conditions in this LIF-first order:

1. `frozen_lif_original`: the original 256-cell leg/VNC core;
2. `frozen_lif_degree_rewired`: the fixed degree-preserving rewire of the
   original leg/VNC core;
3. `wing_lif`: the 256-cell wing-thoracic core (20 wing sensory, 12
   descending, 200 thoracic intrinsic, and 24 wing motor cells);
4. `leg_wing_lif`: the independently identifiable leg and wing-thoracic cores,
   concatenated only at the trainable decoder;
5. `gru_matched`: the recurrent non-biological baseline;
6. `mlp_normal`: the feed-forward non-biological baseline.

The single-core LIF, GRU, and MLP actors are parameter matched within the
existing ten-percent gate.  The combined leg-plus-wing actor is intentionally
larger and must be reported as an exploratory capacity-unmatched condition,
never as a parameter-matched winner.  GRU and MLP hidden units have no
anatomical role and must not be labelled as wing, leg, thoracic, or motor
neurons.

### Frozen experiment contract

Train Reach, Switch, and Gust independently from scratch.  A task receives a
fresh policy, optimizer, recurrent state, RNG lineage, run directory, and
checkpoint lineage; no warm start or cross-task transfer is allowed.

```text
6 controllers x 3 independently trained tasks x seed 0 = 18 fresh jobs
18 jobs x 500,000 interactions/job = 9,000,000 interactions
18 checkpoints x 16 matching held-out episodes = 288 evaluation episodes
```

Use the public tasks:

- `FlyCrazyflie-WaypointReach-v0`;
- `FlyCrazyflie-WaypointSwitch-v0`;
- `FlyCrazyflie-GustRecovery-v0`.

These tasks subclass the installed `Isaac-Quadcopter-Direct-v0` implementation
and preserve its Crazyflie body, physics step, 12 observations, and four
aggregate-wrench actions.  The current training stack deliberately does not
support training the raw native task because its curriculum clock, detailed
metrics, and fixed held-out plans are project-task contracts.  Do not describe
the project reward or curriculum as NVIDIA's native default.

Freeze these settings for every cell:

| Setting | Value |
|---|---:|
| Contract profile | `balanced_v4` |
| Training seed | `0` |
| Environments | `40` |
| Horizon | `100` |
| Interactions/update | `4,000` |
| PPO updates/job | `125` |
| PPO epochs | `2` |
| Full-vector microbatch | `40` |
| Learning rate | `3e-4` |
| Gamma / GAE lambda | `0.99 / 0.95` |
| Clip ratio | `0.2` |
| Value / entropy coefficients | `0.5 / 0.002` |
| Max gradient norm / target KL | `1.0 / 0.05` |
| Checkpoint cadence | `100` updates |
| Held-out protocol | `main`, seed `101` |
| Held-out episodes | `16` on the matching task |

The balanced-v4 curriculum reaches its full distribution at 250,000
interactions, so each 500,000-interaction job spends its latter half at the
full stage.  All policies use the same deterministic fixed flight stabilizer
and learn only bounded residual control on top of it.

### Offline, queue, resource, and integrity requirements

Training and evaluation must use the verified local `cf2x.usd` mirror and a
local procedural flat ground so internet loss cannot stop the queue.  Use only
the declared Isaac Python interpreter.  Every job is resumable only from its
own authenticated `latest.pt`; a restart must skip a genuinely completed cell,
resume a valid incomplete cell, and never overwrite or borrow an older run.

One or two Isaac jobs may run concurrently.  Two are allowed only after a
paired bounded start demonstrates device-wide VRAM strictly below `6963.2 MiB`,
system RAM strictly below `90%`, no new sustained swap-out, finite state, and
isolated checkpoint directories.  GPU compute utilization may reach 100%.
Any hard-memory, OOM, nonfinite, fingerprint, checkpoint, or isolation failure
falls back to one process and remains visible in the queue/report.  LIF jobs
have priority: original leg, degree-preserving rewired leg, wing-only, and
combined complete before GRU and MLP jobs are launched.  The rewire must use
the immutable checked-in manifest and pass topology-changed, directed in/out
degree, weight/sign multiset, no-duplicate, and no-self-loop invariants.

Before the long queue, require focused unit tests, one short train/checkpoint/
resume test, and a bounded start for all six controller interfaces.  Preserve
all prior Crazyflie and G1 artifacts.  Recompute the 37-file G1 manifest after
all implementation and results are complete.

### Evaluation, score, and neural activity

Run the official deterministic held-out evaluator on each checkpoint's
matching task and fixed 16-plan main manifest.  Report training reward and
held-out task performance separately.  At minimum report score out of 100,
event count, strict success, survival, crash, out-of-bounds, invalid-state,
latency, final and time-averaged goal error, action effort/smoothness, work
proxy, parameter counts, checkpoint SHA-256, RAM, and VRAM.  An incomplete or
invalid cell is `N/A`, never an invented zero or success.

For LIF checkpoints, record bounded per-neuron sampled spike counts and group
them by manifest truth.  Report normalized sampled Hz per neuron, active/dead/
saturated fractions, and the highest-activity authenticated neuron IDs for:

- leg sensory, descending, VNC intrinsic, and leg motor roles;
- wing sensory, descending, thoracic intrinsic, and wing motor roles;
- leg and wing cores separately in the combined condition.

The activity sample is the post-final-neural-substep state once per 20 ms
control decision; do not mislabel it as a full-substep biological firing rate.
Compare wing versus thoracic activity using per-neuron rates, not raw group
totals.  GRU hidden state and MLP layer activations may be summarized with
engineering statistics only.  Activation is correlational; do not claim that a
more active role caused better reward without a separately predeclared
ablation.

Write the final evidence-backed result to
`docs/crazyflie_neural_comparison_seed0_500k_summary.md`, with its supporting
machine-readable JSON, authenticated loss/reward-curve figure, exact commands,
hashes, failures, and limitations.

## Canonical commands

```bash
cd /home/chayanin/Desktop/flyg1
ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

# Pure logic tests
"$ISAAC_PYTHON" -m pytest tests/unit/test_crazyflie_keyboard.py -q

# Finite scripted simulator smoke using the same flight-assist path
"$ISAAC_PYTHON" scripts/crazyflie_keyboard.py \
  --scripted_smoke --headless --offline_only --steps 800

# Interactive manual flight; no checkpoint and no training
"$ISAAC_PYTHON" scripts/crazyflie_keyboard.py \
  --offline_only --neural_monitor wing_thoracic

# Optional combined independent leg + wing-thoracic diagnostic monitor
"$ISAAC_PYTHON" scripts/crazyflie_keyboard.py \
  --offline_only --neural_monitor leg_wing_thoracic
```

The Phase-2 queue and analysis commands are generated from
`configs/experiments/crazyflie_neural_comparison_seed0_500k.json`; the queue must
first pass a dry run that prints all 18 train/evaluation cells and their output
paths without launching Isaac.

Completion means both phases' gates pass, all 18 cells either complete or
retain an explicit failure, and the final Markdown does not infer learned
flight from process success, low loss, or neural activity alone.
