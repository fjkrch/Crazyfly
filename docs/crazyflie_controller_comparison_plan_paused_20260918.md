# Crazyflie controller-comparison plan for Isaac Lab

## User-authorized seed-0 comparison run (2026-09-17)

The next executable experiment is a descriptive, task-separated balanced-v4
comparison, not the official five-seed main matrix.  It is authorized despite
the interrupted seed-5 Reach proof, but it cannot satisfy, unlock, or be
reported as the official multi-seed main result.  Preserve the interrupted
proof directory and never warm-start from it or from either learning-rate
screen.

Run exactly:

```text
4 controllers x 3 independently trained tasks x seed 0 = 12 fresh jobs
12 jobs x 1,000,000 interactions/job = 12,000,000 training interactions
12 matched checkpoints x 16 seed-101 episodes = 192 evaluation episodes
```

Use balanced-v4, the selected common learning rate `3e-4`, four environments,
horizon 100, full-vector microbatch 4, two PPO epochs, and a checkpoint every
100 updates.  Every job therefore ends at exactly 2,500 updates.  Train and
evaluate in controller-first order: original
frozen LIF on Reach, Switch, and Gust; degree-preserving rewired frozen LIF on
the same three tasks; matched GRU; then normal MLP.  Each job starts from a
fresh initialization and is evaluated only on its matching task.  Internet
loss must not stop the local queue, and a queue restart must validate and
resume the existing checkpoint rather than overwrite it.

GPU compute utilization may reach 100%; it is telemetry, not an acceptance
gate for this run.  Device VRAM must remain below 90% of the installed device,
while the existing stricter plan gate of `<6963.2 MiB` remains in force, so the
effective limit is still `<6963.2 MiB`.  System RAM remains `<90%`; sustained
paging, OOM, nonfinite state, stale fingerprints, or invalid checkpoints fail
the affected cell.  To retain the exact four-environment optimization
semantics while using the GPU more fully, a maximum of two independent Isaac
jobs may run concurrently only after a paired original-LIF smoke demonstrates
that aggregate device memory, RAM, paging, finite-value, and process-isolation
gates pass.  Both processes must use disjoint run directories and checkpoint
lineages.  If the paired smoke fails any gate, the queue must fall back to one
Isaac process; environment count, microbatch size, PPO epochs, and update count
must not be changed merely to increase utilization.

Freeze the descriptive 100-point score before observing any 1M evaluation:

- 25 points for the fraction of episodes without termination;
- 25 points for task-event completion (Reach target event, four Switch target
  events, or three authenticated Gust recoveries);
- 15 points for strict task-complete episodes;
- 5 points for fixed-horizon-censored event latency, with the contractual
  0.5-second dwell as the best possible value;
- 20 points for the mean of `exp(-mean_error_m)` computed from time-averaged
  integrated goal error and final goal error;
- 10 points for bounded command effort and smoothness (`48` action-squared
  seconds and `4` mean L2 delta are the immutable worst-case normalizers).

Report the raw numerator, denominator, crash, out-of-bounds, invalid-state,
latency, final error, integrated error, command effort, smoothness, work proxy,
RAM, VRAM, parameter count, and checkpoint identity beside every score.  An
incomplete or unauthenticated cell receives `N/A`, never an invented zero.
The controller headline is the unweighted mean of its three task scores and
must also show its worst-task score.  Rank by macro score, then worst-task
score, then event completion; exact ties remain ties.  With only seed 0 there
is no confidence interval or claim of training-seed generalization.

## Latest user-authorized amendment: task-separated-v1 (2026-09-17)

This amendment is the active execution protocol. It supersedes every
conflicting statement later in this document that calls for Mixed training,
one shared training task, a 20-job main matrix, or evaluation of every trained
checkpoint on all three scenarios. Those passages are retained as historical
context and must not be deleted or used to construct a new queue. All prior
runs, checkpoints, queue files, reports, manifests, and G1 artifacts remain
preserved and unchanged; none is silently relabeled as task-separated-v1
evidence.

Train each public task independently from scratch:

- `FlyCrazyflie-WaypointReach-v0`;
- `FlyCrazyflie-WaypointSwitch-v0`;
- `FlyCrazyflie-GustRecovery-v0`.

Each task gets a fresh controller initialization, optimizer, run directory,
checkpoint lineage, recurrent/LIF state, and random-seed lineage. There is no
Mixed-task training, warm-start, task chaining, or checkpoint transfer between
Reach, Switch, and Gust. The primary held-out result for a checkpoint is the
matching task only: Reach checkpoints are evaluated on Reach, Switch on
Switch, and Gust on Gust. Any future cross-task evaluation is supplementary
and must not be substituted for the matched-task result.

The task-separated-v1 main matrix is:

```text
4 controllers x 5 seeds x 3 independently trained tasks = 60 training jobs
60 jobs x 5,000,000 interactions/job = 300,000,000 training interactions
60 checkpoints x 16 matched held-out episodes/checkpoint = 960 evaluation episodes
```

The controllers remain original frozen LIF, degree-preserving rewired frozen
LIF, matched GRU, and normal MLP. The seeds remain `0, 1, 2, 3, 4`. Main
configuration, queue, dry-run report, fingerprints, and output paths must use
new task-separated-v1 names; the historical balanced-v3/Mixed 20-job files
must not be overwritten or executed as though they implement this amendment.

Before the main queue is eligible to launch, run a bounded task-separated-v1
integration matrix consisting of one job for each controller/task pair:

```text
4 controllers x 3 tasks = 12 integration jobs
12 checkpoints x 2 matched held-out episodes/checkpoint = 24 evaluation episodes
```

Every integration job must start from scratch and use its matching task for
evaluation. It remains subject to the finite-value, crash, out-of-bounds,
checkpoint/resume, fingerprint, parameter-matching, determinism, and memory
gates elsewhere in this plan wherever those requirements do not conflict with
this amendment.

LIF is the first-priority launch gate. For each of the three tasks, complete a
task-specific 500,000-interaction proof run from scratch for each frozen LIF
condition (original and degree-preserving rewired), followed by the matching
held-out evaluation. Report event score, strict full-episode success, crashes,
out-of-bounds terminations, invalid values, RAM, VRAM, and checkpoint/resume
evidence separately for every controller/task gate. A result from a Mixed run
or from a different task cannot satisfy one of these six gates. A gate that
does not meet the active empirical-success criterion must be reported as a
failure and blocks that controller/task main cell; it must not be presented as
passed by averaging with another task.

Resource selection is also task-separated. Tune the environment count with
only one Isaac process at a time and target measured GPU compute utilization
in the `50%` to `90%` band. A measurement above `90%` requires scaling down
and re-testing before a long run. The hard acceptance limits remain
device-wide GPU memory strictly below `6.8 GiB` (`6963.2 MiB`) and system RAM
strictly below `90%`, with sustained paging, OOM, or nonfinite evidence fatal.
The environment count may differ by controller/task only when each choice is
measured, recorded, fingerprinted, and preserves the declared interaction
budget and optimization semantics.

The currently authorized pilot is normal MLP on
`FlyCrazyflie-WaypointReach-v0`, seed `0`, trained from scratch for exactly
`1,000,000` interactions and then evaluated only on held-out Reach. This is a
pilot, not one of the 5,000,000-interaction main jobs, and it does not waive or
replace any LIF-first gate. Switch and Gust pilots must use independent fresh
runs when authorized; the Reach pilot checkpoint must not initialize them.

### Reach baseline result and predeclared balanced-v4 follow-up

The authorized balanced-v3 MLP Reach pilot completed on 2026-09-17. It is a
preserved failed empirical baseline, not a launch gate: training completed
exactly 1,000,000 interactions and 250 updates with ten training success
events, but deterministic held-out Reach scored `0/5` event successes and
`0/5` strict successes. All five episodes ran for 600 control intervals with
zero crashes, zero out-of-bounds terminations, and zero invalid states; mean
final distance was 1.1437 m and mean final/initial distance ratio was 0.7156.
The checkpoint SHA-256 is
`e643fe82b714d7f7b586a83429236da94d337bb04f0a2a7e3c9ce420fda1f7d3`
and the evaluation artifact SHA-256 is
`7fbc0f0de9bfea62da0b36b3a8ef757f23bad53f113ce4baebd782725caa176f`.
Balanced-v3 activates its full distribution only at exactly 1,000,000
interactions, so this run completed no full-stage training episode before its
held-out full-distribution evaluation. Preserve this evidence and never
reinterpret evaluator exit status `PASS` as empirical Reach success.

Before any further result is observed, freeze an additive `balanced_v4`
candidate. It must not mutate balanced-v3 constants, hashes, checkpoints, or
reports. Its four reset distributions remain numerically identical to
balanced-v3, but their inclusive start interactions are `0`, `50,000`,
`125,000`, and `250,000`; consequently an exact 500,000-interaction proof has
substantial full-distribution exposure. The reward changes are limited to:

- progress scale `4.0` with the existing `2*tanh(distance/2)` potential;
- proximity `0.150*exp(-0.5*(distance/0.80)^2)` per interval;
- the existing braking scale and cap evaluated with the same `0.80 m`
  proximity factor;
- a one-shot `+3.0` Gust recovery term only on a new recovery edge whose
  submitted world-frame impulse is finite, nonzero, and agrees componentwise
  within `1e-6 N*s` with the predeclared direction times audited vehicle mass
  times `0.75 m/s`.

All other balanced-v3 reward coefficients, the exact 0.20 m / 0.25 m/s /
25-interval success tube, failure bounds, action mapping, observation, fixed
stabilizer, controller architectures, and held-out plans remain unchanged.
Reach and Switch receive exactly zero Gust-recovery reward. The new contract
field is `balanced_v4_task_contract`; the profile, reward, curriculum, and
switch versions must be separately hashed and embedded in every new artifact.
Balanced-v3 remains the CLI default until bounded-v4 validation passes.

LIF retains first priority. Select only the learning rate with two fresh
original-LIF Reach screens at pilot seed `20260917`, exactly 40,000
interactions (100 updates at four environments and horizon 100), candidate
rates `1e-4` and `3e-4`, and otherwise identical PPO settings. Do not use the
seed-101 held-out plans for this selection. The fail-closed ordering is:
memory/nonfinite/crash gates, then higher training target-success event count,
then lower final goal distance over completed pilot episodes, then higher
cumulative progress reward, then the lower learning rate. Freeze the selected
rate before starting a fresh canonical Reach proof at training seed `5` for
exactly 500,000 interactions and five seed-101 held-out Reach episodes. The
canonical proof must score at least `1/5` genuine Reach event success with all
hard gates passing before balanced-v4 can replace balanced-v3 in any
task-separated integration or main config. A screen checkpoint is never
warm-started into the canonical proof or a main job.

**Status:** implementation in progress.  The historical survival-v2 execution
was stopped on 2026-09-16 at the user's direction; its Crazyflie run, queue,
checkpoint, and log artifacts were removed after their hashes and zero-success
outcome were recorded in `docs/crazyflie_survival_v2_removal_20260916.md`.
Balanced-v3 is the active protocol.  No balanced-v3 main job may start until
the LIF-first empirical-success gate below passes.

**User-authorized acceptance update (2026-09-17):** memory acceptance is
`crazyflie_memory_acceptance_v2`: device-wide GPU remains strictly below
6.8 GiB, system RAM is strictly below 90%, sustained paging/OOM/nonfinite
evidence remains fatal, and monotonic RSS growth is a reported warning rather
than a failure. The LIF proof uses five episodes per held-out scenario, reports
the detailed score out of five, and requires at least one success-event-scored
episode in each scenario. The launch score is event-based: an episode scores when it
contains at least one genuine target-dwell/recovery success event. Strict
full-episode success remains reported separately but is not the launch gate.
Main evaluation remains 16 episodes per scenario/job.

**Prepared:** 2026-09-16  
**Project:** `/home/chayanin/Desktop/flyg1`  
**Simulator installation inspected:** `/home/chayanin/Downloads/IsaacLab` at commit `b4c321024792976150ca55fddb26fa34480d974e`

The previous Unitree G1 work is paused. Its former plan is preserved at
`docs/g1_plan_paused_20260913.md`, and its pause state remains under
`runs/main_matrix_pause_20260913T230933Z/`. This plan must not resume, delete,
or rewrite those runs.

## 1. Decision and research question

Build a new Crazyflie experiment beside the paused G1 implementation. Use the
Crazyflie model already available through Isaac Lab's
`Isaac-Quadcopter-Direct-v0` task. The experiment asks whether a frozen
MaleCNS-derived LIF controller can learn goal-directed flight through small
trainable input/output adapters, and how it compares with matched controller
baselines.

Under the active balanced-v3 protocol, this question is specifically about
goal-directed **residual** learning on top of one shared fixed flight
stabilizer. It is not an end-to-end comparison of unassisted flight; the
stabilizer and its contribution must remain explicit in every result.

A drone and an MLP are different kinds of things. The drone is the simulated
body; the MLP is one controller for that body. Therefore the drone experiment
keeps all four controller conditions:

1. **original frozen LIF** — the selected MaleCNS-derived graph is fixed;
   only the input adapter, output adapter, critic, and declared distribution
   parameters train;
2. **rewired frozen LIF** — a predeclared degree-preserving rewiring of the
   same graph, with the same trainable interface and budget;
3. **GRU** — a conventional recurrent controller matched as closely as
   practical on trainable actor parameters;
4. **normal MLP** — a conventional feed-forward controller using the same
   observation and action spaces.

Run five training seeds per condition, for a complete main matrix of
`4 conditions x 5 seeds = 20 training jobs`. Do not replace the MLP with the
drone, and do not report a result table that silently omits any planned cell.

The current circuit is a leg/VNC-derived model. It is not a known biological
flight circuit. Results may describe simulated controller performance and
activity only; they must not claim that the model explains insect flight,
dopamine concentration, or causal biological function.

## 2. Scope and non-goals

The first version covers:

- one Crazyflie body;
- flat, obstacle-free airspace;
- position-goal flight, goal switching, and horizontal gust recovery;
- aggregate thrust and body-moment control through the installed task;
- PPO training on one task and fixed held-out evaluation on three scenarios;
- the four controller conditions and five seeds listed above;
- reproducible checkpoints, fingerprints, memory measurements, and complete
  result tables.

The first version does not cover cameras, lidar, obstacle avoidance, rotor
failure, ROS, real-hardware deployment, real onboard state estimation,
multi-drone control, or individual motor commands. Do not present simulator
state observations as measurements available on a physical vehicle.

## 3. Protect the paused G1 work

Implement the drone path as a parallel execution island. Before making code
changes, the implementation agent must create a SHA-256 manifest for the
existing G1 execution inputs and verify it again after implementation. At
minimum, freeze these existing files and all existing Python files in the
listed packages:

```text
scripts/_bootstrap.py
scripts/smoke_env.py
scripts/train.py
scripts/evaluate.py
scripts/evaluation_protocol.py
scripts/run_matrix.py
scripts/memory_watch.py
source/g1_fly_control/g1_fly_control/connectome/*.py
source/g1_fly_control/g1_fly_control/policies/*.py
source/g1_fly_control/g1_fly_control/training/*.py
source/g1_fly_control/g1_fly_control/evaluation/*.py
source/g1_fly_control/g1_fly_control/tasks/g1/*.py
configs/experiments/main*.json
```

Do not edit, resume, rename, or delete existing G1 runs, checkpoints, queue
files, logs, summaries, or pause metadata. New shared behavior should be
implemented through new drone modules or thin drone-specific wrappers. If a
truly shared bug must be fixed, stop and document the proposed change and its
effect on the G1 fingerprint before applying it.

The directory is not currently a Git worktree, so content hashes are the
primary change guard. Store the pre-change and post-change manifests in
`docs/` and require an exact match for the frozen set.

## 4. Authoritative installed Crazyflie contract

Use the installed source as the execution authority:

```text
/home/chayanin/Downloads/IsaacLab/source/isaaclab_tasks/
  isaaclab_tasks/direct/quadcopter/quadcopter_env.py
/home/chayanin/Downloads/IsaacLab/source/isaaclab_tasks/
  isaaclab_tasks/direct/quadcopter/__init__.py
/home/chayanin/Downloads/IsaacLab/source/isaaclab_assets/
  isaaclab_assets/robots/quadcopter.py
```

The inspected installation provides:

- native task ID: `Isaac-Quadcopter-Direct-v0`;
- asset configuration: `CRAZYFLIE_CFG`, using `cf2x.usd`;
- direct environment API, not the manager-based G1 API;
- observation width: 12;
- action width: 4;
- physics timestep: 0.01 s (100 Hz);
- control decimation: 2, giving a 0.02 s control step (50 Hz);
- native episode horizon: 10 s;
- native goals: x/y in `[-2, 2]` m around each environment origin and z in
  `[0.5, 1.5]` m;
- native height termination below 0.1 m or above 2.0 m.

The 12 observations, in order, are:

1. body-frame linear velocity, 3 values;
2. body-frame angular velocity, 3 values;
3. projected gravity, 3 values;
4. body-frame displacement to the goal, 3 values.

The four actions are:

- action 0: normalized collective thrust, mapped by the native task from
  `[-1, 1]` to `[0, 1.9 x vehicle weight]` along local +Z;
- actions 1-3: normalized body moments with limits of approximately
  `+/-0.01 N m` on each axis in the inspected configuration.

The first project version must keep this 12-value observation contract and
must not add previous action. All conditions receive exactly the same values,
ordering, clipping, and normalization. The custom environment may change
goals, rewards, schedules, episode length, and metrics, but must not silently
change the asset or action mapping.

The built-in task randomizes `episode_length_buf` during a full reset. Fixed
evaluation must explicitly zero or otherwise deterministically initialize it,
then test that every evaluation episode starts at step zero.

Record the installed versions in every run. The inspected environment had
Isaac Lab 0.54.4, Isaac Lab Assets 0.2.4, Isaac Lab Tasks 0.11.16,
Isaac Sim 5.1.0, and PyTorch 2.7.0+cu128; the implementation agent must verify
them again rather than assuming they remain unchanged.

### 4.1 Native task versus project defaults

There are two distinct meanings of "default" in this experiment. The native
embodiment and physics authority is the installed
`Isaac-Quadcopter-Direct-v0` registration. It supplies the Crazyflie asset,
direct-environment dynamics, 12-value observation ordering, four-value
aggregate-wrench action mapping, physics timing, and native bounds. It is used
unchanged for the native smoke test and as the upstream base class for every
project task; it is not a training cell in the controller comparison.

The active project task profile is `balanced_v3`.  It never changes the native
body or simulator contract: every project task still uses the installed
Crazyflie asset, 0.01 s physics step, decimation 2, 12 observations, four
aggregate-wrench actions, and the same failure bounds.  A profile selector
changes only the explicitly fingerprinted reward, reset curriculum, and
training scenario schedule.

`FlyCrazyflie-WaypointReach-v0` remains the first LIF proof task so target
acquisition can be checked without switch/gust confounds.  The active balanced
comparison trains on the separately registered training-only
`FlyCrazyflie-Mixed-v0`, which deterministically mixes Reach, Switch, and Gust
episodes.  Official held-out evaluation still uses only the three public task
IDs and their immutable episode plans; evaluation never fine-tunes weights.
The historical survival-v2 Reach-only behavior remains documented below for
provenance but is retired from execution.

## 5. Planned additive layout

Add drone-specific files and leave the frozen G1 files unchanged:

```text
source/g1_fly_control/g1_fly_control/tasks/crazyflie/
  __init__.py
  adapter.py
  env.py
  env_cfg.py
  logic.py
  metrics.py
  registration.py

scripts/
  drone_bootstrap.py
  drone_inspect_asset.py
  drone_smoke_env.py
  drone_train.py
  drone_play.py
  drone_evaluation_protocol.py
  drone_evaluate.py
  drone_record.py
  drone_run_matrix.py
  drone_summarize_matrix.py
  drone_monitor_matrix.py
  execute_drone_integration.sh
  execute_drone_matrix.sh

configs/experiments/
  crazyflie_integration.json
  crazyflie_main.json
  crazyflie_balanced_v3_integration.json
  crazyflie_balanced_v3_main.json

tests/unit/
  test_crazyflie_logic.py
  test_crazyflie_schedules.py
  test_crazyflie_metrics.py
  test_crazyflie_checkpoint.py
  test_crazyflie_matrix.py

docs/
  crazyflie_task_spec.md
  crazyflie_asset_audit.json
  crazyflie_asset_audit.md
  g1_frozen_sha256_before.json
  g1_frozen_sha256_after.json
```

Reuse the existing policy, connectome, PPO, storage, and checkpoint concepts
through imports or drone wrappers only when their interfaces actually match.
The native quadcopter task is a `DirectRLEnv`, so the existing G1 scripts
cannot simply receive a new task ID. The adapter may expose the legacy
`flyg1_terminal_observation` key as a documented compatibility alias while
also providing a correctly named drone terminal observation. Do not emit G1
joint, contact, or locomotion-work metrics for a vehicle that has none.

Register three public evaluation task IDs and one training-only task ID:

```text
FlyCrazyflie-WaypointReach-v0
FlyCrazyflie-WaypointSwitch-v0
FlyCrazyflie-GustRecovery-v0
FlyCrazyflie-Mixed-v0            # training only; never used as a held-out scenario
```

Task registration must be idempotent, must not shadow NVIDIA's native task,
and must fail with a clear message if the pinned upstream task contract has
changed.

## 6. Environment and scenario definitions

Use a 12 s episode horizon, or 600 control decisions at 50 Hz, for all three
public tasks and the training-only Mixed task. Use the same physics, asset,
observation, action, termination, and reset conventions across conditions.

### 6.1 WaypointReach (training and evaluation)

- Spawn from a valid near-hover state with bounded, predeclared perturbations
  to yaw, position, linear velocity, and angular velocity.
- Sample a reachable 3D target from the validated workspace, initially using
  the native x/y and z limits.
- Require a minimum start-to-target separation so trivial episodes are not
  counted as successes.
- Keep one target for the full episode.

Historically this was the only survival-v2 training task.  Under active
balanced-v3 it is the isolated LIF proof task, while the comparison checkpoint
is trained on the deterministic Mixed schedule.  Switch and Gust remain
separate immutable held-out evaluations even though their event types are now
also represented in the training mixture.

### 6.2 WaypointSwitch (held-out evaluation)

- Use fixed target changes at 3, 6, and 9 s.
- Each episode's complete target sequence is generated before evaluation and
  stored in a manifest.
- Enforce a declared minimum distance between consecutive targets.
- Reset success dwell state, distance history, and target-specific timers at
  every switch so the switch itself cannot create a false progress reward or
  success.

### 6.3 GustRecovery (held-out evaluation)

- Keep the waypoint objective active throughout the episode.
- Apply fixed horizontal gust events beginning at 3, 6, and 9 s.
- Each gust lasts 0.10 s.
- Express gust strength as a desired mass-normalized horizontal velocity
  change of 0.75 m/s: `impulse = mass x 0.75 m/s` and
  `force = impulse / 0.10 s`.
- Predeclare and store each gust direction. Apply the same schedule to every
  policy evaluated on that episode.
- Validate the force API, reference frame, application point, duration, and
  actual measured impulse before freezing the protocol.

Every gust counts in the unconditional recovery metric. Also report a
conditional metric for gusts where the vehicle was stable immediately before
the disturbance; never substitute the conditional number for the
unconditional result.

### 6.4 Reward and termination

Use one frozen reward for training all four conditions. Begin with:

- signed reduction in 3D goal distance per control step;
- a one-time success bonus;
- a small control-effort penalty;
- a small action-change penalty;
- explicit penalties for invalid simulation, crash, or workspace escape.

Do not reward a prescribed attitude, path, or control style. Freeze reward
coefficients after bounded pilot validation and before any main seed. State
whether each term is a rate or an interval quantity and test that it is not
multiplied by the timestep twice.

Keep the installed height bounds initially. Treat ground contact, height
escape, nonfinite state, and other declared invalid conditions as failures.
Distinguish time-limit truncation from failure termination in the stored data.

Because the native model accepts aggregate wrench rather than individual
rotor commands, label control costs accordingly. A possible mechanical-work
proxy is the time integral of absolute aggregate force/velocity and
moment/angular-velocity products. It is a simulator proxy, not battery energy.

### 6.5 Historical retired survival-v2 training protocol

This subsection is retained only to authenticate the removed survival-v2
attempt and its old fingerprints.  It is not an active training configuration.

The initial coefficient-agnostic reward proposal above was frozen, before any
main seed, as `crazyflie_survival_first_reward_v2`. Let
`h = 2 / 1.9 - 1 = 0.05263158...` be the normalized collective command that
maps to vehicle weight. For each 0.02 s control interval the exact terms are:

```text
progress = 0.5 * clamp(previous_distance - current_distance, -0.05, +0.05)
success = +5.0 once for each newly completed 25-step target dwell
survival = +0.02 if and only if the interval is not a failure

effort_metric = clamp(
    (action[0] - h)^2 + sum((action[1:4] / 0.005)^2), maximum=4.0)
control_effort = -0.01 * effort_metric

change_metric = clamp(
    delta_action[0]^2 + sum((delta_action[1:4] / 0.005)^2), maximum=4.0)
action_change = -0.001 * change_metric

failure = -20.0 once on failure termination
```

Progress, survival, effort, and action change are interval quantities and are
not multiplied by `dt` again. Success and failure are one-time event
quantities. A valid time-limit interval earns survival credit; a true failure
interval does not. There is no attitude, path, or privileged-state reward.
The frozen reward payload SHA-256 is
`8a17e7d55b87550c68690a080749e3accbb49321354ecc0b5cd28ef0128f9ba7`.

The shared training reset curriculum was frozen as
`crazyflie_survival_first_curriculum_v2`. A stage is selected only when an
episode resets; crossing a boundary never mutates an episode in flight.
Deterministic evaluation and an installed held-out episode plan always bypass
the curriculum and use the full distribution.

| Start interactions | Stage | Spawn Z | Spawn X/Y half-range | Spawn Z half-range | Yaw half-range | Linear/angular velocity half-range | Goal X/Y | Goal Z | Minimum separation |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 0 | `vertical_lift` | 1.00 m | 0.01 m | 0.005 m | 0.02 rad | 0.00 m/s, 0.00 rad/s | `[-0.02, 0.02]` m | `[1.45, 1.50]` m | 0.40 m |
| 200,000 | `near` | 0.85 m | 0.025 m | 0.01 m | 0.05 rad | 0.02 m/s, 0.03 rad/s | `[-0.50, 0.50]` m | `[1.00, 1.50]` m | 0.50 m |
| 500,000 | `mid` | 0.70 m | 0.05 m | 0.02 m | 0.10 rad | 0.04 m/s, 0.07 rad/s | `[-1.00, 1.00]` m | `[0.75, 1.50]` m | 0.65 m |
| 1,000,000 | `full` | 0.50 m | 0.10 m | 0.05 m | 0.25 rad | 0.10 m/s, 0.20 rad/s | `[-2.00, 2.00]` m | `[0.50, 1.50]` m | 0.75 m |

The frozen curriculum payload SHA-256 is
`8ea8c1f8c52205ac028b0a493855764727b59243d45a2a4901d0596f2f91920d`.
Changing any coefficient, boundary, or distribution requires a new versioned
protocol and fingerprint; a resulting checkpoint cannot enter the current
primary comparison.

### 6.6 Active balanced-v3 training protocol

The user retired survival-v2 after the original frozen-LIF attempt reached
449,200 interactions and 756 completed episodes with zero target successes.
The replacement is the separately versioned
`crazyflie_balanced_task_reward_v3`; it is not a continuation of, and cannot
resume from, the removed survival checkpoint.  Let
`h = 2 / 1.9 - 1` and `phi(d) = 2*tanh(d/2)`.  For every 0.02 s control
interval the exact terms are:

```text
raw_progress = phi(previous_distance) - phi(current_distance)
progress = raw_progress                         on a nonfailure interval
progress = min(raw_progress, 0)                on a failure interval

q = exp(-0.5 * (current_distance / 0.50)^2)
proximity = +0.015*q                           on a nonfailure interval
dwell = +0.020                                 when distance <= 0.20 m and
                                                speed <= 0.25 m/s, nonfailure
braking = -0.010*q*min((speed / 0.50)^2, 4)
success = +3.0 once for each newly completed 25-step target dwell
retention = -0.040*latched*clamp((distance - 0.20) / 0.50, 0, 1)
survival = +0.010                              on a nonfailure interval

effort_metric = min(
    ((action[0] - h) / 0.15)^2
    + sum((action[1:4] / 0.020)^2), 4)
control_effort = -0.0025*effort_metric

change_metric = min(
    (delta_action[0] / 0.10)^2
    + sum((delta_action[1:4] / 0.010)^2), 4)
action_change = -0.0005*change_metric

B = min(
    relu((0.30 - world_z) / 0.20)^2
    + relu((world_z - 1.70) / 0.20)^2
    + relu((abs(local_x) - 2.25) / 0.25)^2
    + relu((abs(local_y) - 2.25) / 0.25)^2, 4)
boundary = -0.050*B
failure = -25.0 once on failure termination
```

On a failure interval, proximity, dwell, success, survival, and positive
progress are exactly zero; negative progress, braking, retention, effort,
change, boundary, and failure penalties remain.  A nonfinite position receives
the capped boundary metric.  Reset and target-switch baselines make progress
exactly zero, success latching does not terminate an episode, and every term
above is an interval or event quantity that is not multiplied by `dt` again.
Hard termination remains below 0.10 m, above 2.00 m, outside +/-2.75 m in
environment-local X/Y, or on a nonfinite state.  The balanced reward payload
SHA-256 is
`bfbba45d2f19ddf8368da6a1318f1dba446d27e49fec9f8b4ec09abf2da611e9`.

The reset curriculum is `crazyflie_balanced_task_curriculum_v3`.  A stage is
selected only at reset; fixed held-out evaluation bypasses it.  Switch targets
1--3 use the same active stage as target 0 so early curriculum episodes cannot
silently sample the full distribution.

| Start interactions | Stage | Spawn Z | Spawn X/Y half-range | Spawn Z half-range | Yaw half-range | Linear/angular velocity half-range | Goal X/Y | Goal Z | Minimum separation |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 0 | `near_3d` | 1.00 m | 0.01 m | 0.005 m | 0.02 rad | 0.00 m/s, 0.00 rad/s | `[-0.35, 0.35]` m | `[0.80, 1.20]` m | 0.40 m |
| 200,000 | `local_3d` | 0.85 m | 0.025 m | 0.01 m | 0.05 rad | 0.02 m/s, 0.03 rad/s | `[-0.75, 0.75]` m | `[0.65, 1.35]` m | 0.50 m |
| 500,000 | `mid_3d` | 0.70 m | 0.05 m | 0.02 m | 0.10 rad | 0.04 m/s, 0.07 rad/s | `[-1.25, 1.25]` m | `[0.60, 1.50]` m | 0.65 m |
| 1,000,000 | `full` | 0.50 m | 0.10 m | 0.05 m | 0.25 rad | 0.10 m/s, 0.20 rad/s | `[-2.00, 2.00]` m | `[0.50, 1.50]` m | 0.75 m |

Its payload SHA-256 is
`5aefd508cb3a489db630506c33392f4ad7cf40f39851b0194231a661731f3f0c`.
`FlyCrazyflie-Mixed-v0` is training-only and assigns each environment episode
by `(environment_id + per_environment_episode_index + seed) mod 3` over Reach,
Switch, and Gust, giving exact per-environment balance every three episodes.
All four controllers use this exact task, reward, curriculum, PPO settings,
and interaction budget.

## 7. Success, recovery, and evaluation protocol

A target is successful only when both conditions hold continuously for 0.50 s
(25 control steps):

- 3D distance to target is at most 0.20 m; and
- vehicle speed is at most 0.25 m/s.

For a switched target, start a new dwell counter after the switch. For a gust,
define recovery as returning to the same distance-and-speed tube for a
continuous 0.50 s within 2.0 s after the gust ends. A crash or termination
before recovery is a failed recovery.

Create one immutable evaluation manifest with 16 held-out episode plans per
scenario and evaluation seed 101. The manifest contains initial state, target
coordinates, switch schedule, gust directions, and all scenario seeds. Every
checkpoint uses the same manifest and deterministic evaluation action rule.

The LIF-first gate uses a separate deterministic `lif_proof` protocol at the
same evaluation seed with exactly five held-out plans per scenario. It reports
an event score as the number of episodes containing at least one genuine
task-specific event, divided by five: a Reach target dwell, any of Switch's
four target dwells including the initial target, or a post-gust recovery. For
Switch, completing all four targets remains the separately reported strict
episode-success measure. The
minimum launch prerequisite is an event score of at least `1/5` in each of
Reach, Switch, and Gust. This does not change the 16-episode main protocol.

The final evaluation volume is:

```text
20 trained checkpoints x 3 scenarios x 16 episodes = 960 episodes
```

Store both episode-level records and aggregate summaries. An evaluation
failure must remain visible as a failed matrix cell; it must not be discarded
from the denominator.

## 8. Matched controller comparison

All four conditions must share:

- the same 12 actor observations and four action outputs;
- the same observation normalization data and update rule;
- the same PPO implementation, critic architecture, rollout length,
  minibatch schedule, optimizer family, discount, GAE settings, action
  distribution, and interaction budget;
- the same task distribution, environment count, evaluation manifest, and
  checkpoint-selection rule;
- five seeds numbered 0, 1, 2, 3, and 4;
- exactly 5,000,000 environment interactions per main job unless a later
  versioned plan changes the budget before the matrix begins.

Match trainable actor parameter counts within a predeclared tolerance, with a
starting target of +/-10%. Report exact trainable and frozen parameter counts
for every condition. Do not hide the LIF core's total state size by reporting
only trainable parameters; report both trainable parameters and total dynamic
state dimensions.

### Shared fixed stabilizer, residual, and estimand

After a bounded pre-main LIF attempt exposed immediate-survival failure, the
controller interface was revised before any main seed.  Balanced-v3 retains
this already validated engineering interface unchanged. Every condition now
receives the same fixed, non-trainable, goal-independent Crazyflie stabilizer.
The fingerprinted contract version is
`crazyflie_shared_stabilization_bounded_residual_v2`.
Raw policy observations are divided elementwise by the immutable physics
scales

```text
[2, 2, 2, 5, 5, 5, 1, 1, 1, 2, 2, 2]
```

and clipped to `[-5, 5]`. The first three scales apply to body linear velocity,
the next three to body angular velocity, the next three to projected gravity,
and the final three to body-frame goal displacement. There is no running mean,
variance, or rollout-dependent update in training or evaluation. Checkpoints
store and validate this fixed contract rather than learned observation
statistics.

Using reconstructed physical values, the shared normalized-wrench prior is:

```text
collective = h - 0.18*v_z + 0.20*(1 + gravity_z)
roll       = 0.08*gravity_y - 0.02*omega_x + 0.015*v_y
pitch      = -0.08*gravity_x - 0.02*omega_y - 0.015*v_x
yaw        = -0.01*omega_z
```

where `h = 2 / 1.9 - 1`. The prior reads observation indices 0--8 only and
deliberately ignores all three goal coordinates at indices 9--11. Its action
is clamped to `[-0.9999, 0.9999]` solely to keep the inverse tanh finite. This
fixed block supplies basic vertical, tilt, rate, and horizontal-velocity
damping; it does not know the waypoint and therefore cannot solve goal
navigation on its own.

Each trainable actor still sees all 12 normalized observations and emits four
residual logits. The exact latent mean supplied to the common transformed
Gaussian is

```text
mean = atanh(prior_action)
       + [0.35, 0.08, 0.08, 0.05] * tanh(residual_logits)
action = tanh(Normal(mean, exp(log_std)))
```

The residual scales are immutable and bound how far a learned policy can move
each latent coordinate away from the prior. All four output heads start with
zero bias and the same deterministic nonzero full-rank Walsh initialization,
with element magnitude `1e-3 / sqrt(fan_in)`. The trainable initial latent
standard deviations are exactly `[0.03, 0.003, 0.003, 0.003]`: collective
exploration remains larger, while the three sensitive moment channels use the
smaller value. Deterministic evaluation uses `tanh(mean)`.

Consequently, the primary estimand is no longer end-to-end flight from an
unassisted controller. It compares how the original frozen LIF, rewired frozen
LIF, matched GRU, and normal MLP learn goal-directed residual control on top of
the identical fixed flight stabilizer. The original and rewired LIF cores
remain frozen exactly as declared; their trainable adapters learn the residual
interface. The stabilizer is shared fixed engineering structure, must be
fingerprinted in every artifact, and must not be described as LIF output or as
learned flight skill.

### 8.1 Original frozen LIF

- Load the provenance-tracked MaleCNS-derived manifest already used by the
  project.
- Preserve topology and core weights exactly during training.
- Train only the declared observation encoder, action decoder, critic, and
  action-distribution parameters.
- Keep one independent recurrent state per environment and reset only the
  environments that end.
- Keep the differentiable path through the fixed core when updating the
  encoder; freezing core parameters must not detach encoder gradients.
- Store and verify the core checksum before and after each job.

### 8.2 Rewired frozen LIF

- Generate one degree-preserving rewire from a predeclared rewire seed before
  main training.
- Preserve neuron count, directed in-degree and out-degree, weight multiset,
  declared sign constraints, input/output population sizes, and adapter
  budgets as supported by the source data.
- Freeze and hash the rewired manifest before all five training seeds.
- Use the same rewire for the 20-job primary matrix. Additional rewires form a
  separately labeled secondary experiment and cannot be mixed into the main
  five-seed summary.

### 8.3 GRU

- Use a small recurrent actor with hidden state reset on episode end.
- Tune width only during the integration stage to meet the declared parameter
  matching rule.
- Use the same critic and PPO path as the other conditions.

### 8.4 Normal MLP

- Use a conventional feed-forward actor with no recurrent hidden state.
- Give it the same single-step 12-value observation; do not grant a stacked
  history unless every other condition receives the same history.
- Tune hidden widths only during integration to meet the parameter rule.

### 8.5 Main matrix

The immutable main matrix must contain these 20 cells:

| Controller | Seeds | Jobs | Train task | Interactions/job |
|---|---:|---:|---|---:|
| Original frozen LIF | 0-4 | 5 | Mixed balanced-v3 | 5,000,000 |
| Rewired frozen LIF | 0-4 | 5 | Mixed balanced-v3 | 5,000,000 |
| GRU | 0-4 | 5 | Mixed balanced-v3 | 5,000,000 |
| Normal MLP | 0-4 | 5 | Mixed balanced-v3 | 5,000,000 |
| **Total** | | **20** | | **100,000,000** |

The queue must run one training job at a time on this machine. It may evaluate
a completed checkpoint before moving to the next job, but it must never start
two Isaac processes concurrently. A dry run must print all 20 cells, their
commands, seeds, output paths, and expected fingerprints without launching
them.

## 9. Measurements and result tables

### 9.1 Training measurements

Record at least:

- environment interactions and PPO updates;
- episodic return and each reward component;
- success and failure counts;
- goal distance and speed;
- policy/value/entropy losses and KL diagnostics;
- action mean, saturation fraction, and action-change magnitude;
- frames or interactions per second;
- peak process RAM, total system RAM use, PyTorch allocated/reserved VRAM,
  and total GPU memory from `nvidia-smi` where available;
- NaN, invalid-state, crash, and out-of-bounds counts;
- checkpoint and fingerprint identifiers.

For LIF conditions also record bounded summaries of firing/activity,
dead/saturated fractions, recurrent-state norms, encoder/decoder gradient
norms, and the before/after frozen-core checksum. Do not retain every neuron's
full trace during training. Detailed traces belong in short, separate
evaluation recordings written in chunks to disk.

### 9.2 Evaluation measurements

For every scenario report:

- success rate with the fixed denominator of 16 episodes per checkpoint;
- time to first success, with failures handled by a declared censoring rule;
- final and time-integrated 3D goal error;
- speed inside the target region;
- crash and out-of-bounds rates;
- command effort and command smoothness;
- aggregate-wrench mechanical-work proxy, clearly labeled;
- for switches: success and latency after each switch;
- for gusts: unconditional recovery rate, conditional recovery rate,
  recovery latency, maximum displacement, and post-gust error integral.

### 9.3 Required reports

Generate:

1. one row per episode;
2. one summary per checkpoint and scenario;
3. one seed table with all 20 jobs;
4. one controller comparison with per-seed values, mean, median, standard
   deviation, and a paired-seed bootstrap 95% interval;
5. one execution-status table containing every planned job, including
   `pending`, `running`, `completed`, `failed`, `paused`, or `cancelled`;
6. a machine-readable JSON and human-readable Markdown report.

Seed 0 is an ordinary member of the five-seed matrix. It may be reported as an
early result, but it must be labeled `1/5 seeds` for that controller and must
not be described as the final comparison. Partial runs must state the exact
completed fraction, for example `2/20 main jobs`, and keep missing cells
visible. Never use the best seed as the headline result.

## 10. Training state, checkpoints, pause, and resume

Write atomic periodic checkpoints at a declared cadence, initially every 100
PPO updates, and at clean termination. Write to a temporary file, flush it,
then rename it into place. A checkpoint must contain:

- actor, critic, adapters, and action-distribution state;
- optimizer and learning-rate scheduler state;
- observation/reward normalization state;
- PPO counters, update number, and exact interaction count;
- controller recurrent state needed by the saved boundary;
- Python, NumPy, PyTorch CPU/CUDA, environment, and task-schedule RNG states;
- complete resolved configuration and command;
- task/evaluation manifest IDs;
- policy/connectome/rewire/code/environment fingerprints;
- training history needed to append without duplicate samples.

Prefer resume at a completed rollout boundary. If Isaac state cannot be
restored exactly, restart environments and recurrent states together, mark
the event as `resume_reset=true`, and preserve the interaction counter. This
is a valid practical resume but not bit-exact continuation. Test that a resume
does not repeat or skip counted updates, overwrite an earlier checkpoint, or
combine histories from different fingerprints.

The queue must support a requested pause by letting the current update reach
a checkpoint boundary, writing queue state, and exiting. A later resume must
continue the same cell or explicitly mark it restarted. Do not silently
restart a 5,000,000-interaction job from zero.

## 11. Reproduction fingerprints

Every training and evaluation artifact must identify a content-addressed
fingerprint that includes:

- all new drone task, script, config, policy-interface, training, checkpoint,
  and evaluation code;
- every reused existing module imported by the drone path;
- connectome and rewired-manifest content hashes;
- the fully resolved task and PPO configuration;
- the held-out evaluation manifest;
- installed package versions, Python, PyTorch, CUDA/runtime, driver, GPU, OS,
  and Isaac Lab commit;
- the installed upstream quadcopter environment source;
- the Crazyflie asset configuration and resolvable USD identifier/version;
- observation/action names, ordering, units, scaling, and control frequency.

Training, evaluation, summary, and resume must reject incompatible
fingerprints by default. Any override must create a visibly tainted result and
must not enter the primary comparison.

## 12. RAM and VRAM gates

The target machine has limited memory, so scale only after measurement:

1. start Isaac headless with one environment and no cameras or recording;
2. run 1,000 native task steps after warm-up;
3. run 1,000 custom task steps with each policy interface;
4. run one real forward/backward PPO update at one environment;
5. repeat at two environments, then four environments;
6. select the largest environment count that passes every controller and use
   that same count for the matched matrix.

Initial gates:

- keep sampled device-wide GPU memory strictly below 6.8 GiB
  (6963.2 MiB) on the 8 GiB GPU; a sample at the boundary fails;
- under `crazyflie_memory_acceptance_v2`, keep total system RAM strictly below
  90% of 24 GiB; a sample at 90% fails;
- no CUDA OOM, host OOM, nonfinite state, or sustained paging;
- retain the existing final-four-sample monotonic RSS detector and its 1.0 MiB
  tolerance, but record a detected rise as a visible warning rather than a
  pass/fail condition;
- `num_workers=0` initially;
- FP32 for LIF dynamics initially;
- microbatch 1 and a short recurrent sequence initially;
- one Isaac process and one matrix job at a time;
- cameras, viewport, video, ROS, notebooks, and graph preprocessing off during
  training.

Use both process and system measurements. PyTorch's allocated/reserved values
do not include all Isaac/driver allocations. `empty_cache()` is not a remedy
for live tensors. If one environment plus one real update fails the gates,
stop and report the measured blocker instead of shrinking physics fidelity or
changing the circuit without a new scientific plan.

## 13. Acceptance gates before the full matrix

The implementation agent must pass these gates in order.

### Gate A — preservation and pure logic

- The archived G1 hash manifest exists and matches after the drone changes.
- Imports that do not need Isaac remain CPU-testable.
- Unit tests cover schedules, success dwell, switch reset, gust timing and
  impulse, metrics, matrix enumeration, checkpoint metadata, and no duplicate
  interaction counting on resume.

### Gate B — installed native task

- The asset audit records the real observation/action shapes, scaling,
  timestep, mass, frames, bounds, and resolved asset source.
- One native Crazyflie environment runs headless for 1,000 steps with zero or
  low-amplitude bounded actions and finite state.

### Gate C — custom task

- Each of the three public evaluation task IDs loads and runs 1,000 steps; the
  training-only Mixed registration also passes its separate runtime smoke.
- Reset isolation, deterministic evaluation episode length, goal switches,
  gust schedule, truncation, failure termination, and terminal observations
  are verified.
- One, two, and four environment memory measurements are stored, stopping at
  the first failed hard gate. RSS-growth warnings remain visible.

### Gate D — controllers and learning path

- All four actors emit finite, bounded four-value actions.
- All four actors use the exact same immutable physics scaling, goal-independent
  stabilization prior, bounded residual composition, and per-axis initial
  action-distribution standard deviation.
- A controlled test proves that changing only goal observation indices 9--11
  cannot change the fixed prior, while the trainable residual path still sees
  and receives gradients from all 12 observations.
- Original and rewired LIF cores retain identical before/after checksums.
- LIF encoder and decoder receive finite nonzero gradients in a controlled
  update.
- GRU state and LIF state reset only for finished environments.
- MLP is a genuine feed-forward baseline.
- Exact parameter counts and matching deviations are in the report.

### Gate E — checkpoint and evaluation

- A short training job saves, loads in a fresh process, and continues without
  double counting interactions.
- The same checkpoint evaluates all three scenario types against a small
  fixed integration manifest.
- Episode-level and aggregate outputs agree on denominators and failures.

### Gate E2 — LIF-first empirical flight proof

- Run the original frozen-LIF controller first under `balanced_v3`; unit tests
  and reward-shape inspection alone do not prove flight success.
- Exercise WaypointReach in isolation, then run a bounded Mixed pilot to an
  exact 500,000-interaction checkpoint with the same simulator and PPO
  settings intended for the comparison.
- Load that exact checkpoint in fresh deterministic evaluation processes for
  Reach, Switch, and Gust using the five-episode `lif_proof` manifest.
- Report each scenario as a detailed event score out of five and retain every
  episode record. Each scenario must have at least one episode containing a
  genuine target-dwell/recovery success event (`>=1/5`). Strict full-episode
  success is reported separately and is not this launch gate. Every action/
  state must remain finite, and the RAM/VRAM/paging hard gates must pass. A
  monotonic RSS rise is reported as a warning under
  `crazyflie_memory_acceptance_v2` and does not fail the gate by itself. Merely
  surviving, receiving positive return, or completing the script is not a
  success event.
- A fail-closed verifier must parse the exact three-scenario evaluation
  artifact and establish these success and memory conditions before the main
  launcher may spawn its first training process.  An evaluator exit code of
  zero or an artifact execution status of `PASS` is not itself flight success.
- If any scenario still has a zero event score, preserve the balanced checkpoint
  and evidence, do not launch the 20-job main queue, and create a new
  versioned protocol rather than tuning this frozen contract in place.

### Gate F — bounded integration matrix

Run a small four-condition, seed-0 integration matrix with a predeclared short
budget. It verifies orchestration and artifact completeness; it is not a
scientific result. Summaries must label it `integration` and keep it separate
from the 20-job main matrix.

### Gate G — main dry run and human review

- The main dry run enumerates exactly 20 jobs and 60 checkpoint/scenario
  evaluation bundles.
- It predicts 960 held-out episodes.
- All commands, output paths, resource settings, seeds, budgets, and
  fingerprints are visible.
- No duplicate cell or output path exists.
- The full matrix is not launched until the user reviews these artifacts and
  explicitly asks to run it.

## 14. Planned commands after implementation

These are the canonical acceptance interfaces for the implemented drone path.
Run Isaac commands one process at a time, preserve every generated report, and
stop at the first failed acceptance or memory gate.

Set the verified interpreter:

```bash
cd /home/chayanin/Desktop/flyg1
ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python
```

Run focused pure tests, then the existing unit suite:

```bash
"$ISAAC_PYTHON" -m pytest \
  tests/unit/test_crazyflie_logic.py \
  tests/unit/test_crazyflie_schedules.py \
  tests/unit/test_crazyflie_metrics.py \
  tests/unit/test_crazyflie_checkpoint.py \
  tests/unit/test_crazyflie_matrix.py -q
"$ISAAC_PYTHON" -m pytest tests/unit -q
```

Audit the installed asset and task:

```bash
"$ISAAC_PYTHON" scripts/drone_inspect_asset.py --headless \
  --json docs/crazyflie_asset_audit.json \
  --markdown docs/crazyflie_asset_audit.md
```

Run the native one-environment smoke test:

```bash
"$ISAAC_PYTHON" scripts/drone_smoke_env.py \
  --task Isaac-Quadcopter-Direct-v0 \
  --num_envs 1 --steps 1000 \
  --hover_centered_random_actions --random_action_scale 0.1 \
  --headless \
  --output_report runs/crazyflie-native-smoke-balanced-v3-1x1000.json
```

Run the three public custom-task smoke tests at one environment before scaling.
The outer environment-count loop makes the 1, 2, and 4 environment gates
explicit, gives every report a unique balanced-v3 path, and stops before a
larger count if any smaller gate fails:

```bash
for num_envs in 1 2 4
do
  for task in \
    FlyCrazyflie-WaypointReach-v0 \
    FlyCrazyflie-WaypointSwitch-v0 \
    FlyCrazyflie-GustRecovery-v0
  do
    "$ISAAC_PYTHON" scripts/drone_smoke_env.py \
      --task "$task" --contract_profile balanced_v3 \
      --num_envs "$num_envs" --steps 1000 --headless \
      --output_report \
        "runs/crazyflie-balanced-v3-${task}-${num_envs}x1000.json" || exit 1
  done
done
```

Smoke the training-only Mixed registration separately before any Mixed
learning job.  It checks finite runtime, deterministic per-environment scenario
assignment, curriculum selection, and memory without pretending that Mixed is
a held-out evaluation task:

```bash
"$ISAAC_PYTHON" scripts/drone_smoke_env.py \
  --task FlyCrazyflie-Mixed-v0 --contract_profile balanced_v3 \
  --num_envs 1 --steps 1000 --headless \
  --output_report \
    runs/crazyflie-balanced-v3-FlyCrazyflie-Mixed-v0-1x1000.json
```

After the task-only gates pass, run the four controller interfaces in the
declared comparison order, completing original LIF first.  Each command
executes 1,000 live Reach steps and one real PPO forward/backward update.  The
inner environment-count loop enforces 1 then 2 then 4 separately for each
controller, and `exit 1` prevents scaling that controller past its first
failure:

```bash
for policy in \
  frozen_lif_original \
  frozen_lif_degree_rewired \
  gru_matched \
  mlp_normal
do
  for num_envs in 1 2 4
  do
    "$ISAAC_PYTHON" scripts/drone_smoke_env.py \
      --task FlyCrazyflie-WaypointReach-v0 \
      --contract_profile balanced_v3 \
      --policy "$policy" --ppo_update \
      --num_envs "$num_envs" --steps 1000 --headless \
      --output_report \
        "runs/crazyflie-balanced-v3-controller-${policy}-${num_envs}x1000-update.json" \
      || exit 1
  done
done
```

Run one short normal-MLP pause/checkpoint/resume/evaluation path.  The first
trainer exit code is intentionally `3` at exactly 100 updates; the second
invocation is a fresh process with the same fingerprinted arguments and must
resume to exactly 200 updates without double counting:

```bash
MLP_GATE_RUN=runs/crazyflie-balanced-v3-mlp-checkpoint-gate-v1
test ! -e "$MLP_GATE_RUN"
pause_status=0
"$ISAAC_PYTHON" scripts/drone_train.py \
  --task FlyCrazyflie-WaypointReach-v0 \
  --contract_profile balanced_v3 \
  --policy mlp_normal --seed 0 --num_envs 1 \
  --total_interactions 20000 --horizon 100 --microbatch_size 1 \
  --ppo_epochs 2 --learning_rate 3e-5 \
  --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.2 \
  --value_coefficient 0.5 --entropy_coefficient 0.002 \
  --max_grad_norm 1.0 --target_kl 0.05 \
  --checkpoint_every_updates 100 --pause_after_updates 100 \
  --evaluation_protocol integration \
  --run_dir "$MLP_GATE_RUN" --headless || pause_status=$?
test "$pause_status" -eq 3

"$ISAAC_PYTHON" scripts/drone_train.py \
  --task FlyCrazyflie-WaypointReach-v0 \
  --contract_profile balanced_v3 \
  --policy mlp_normal --seed 0 --num_envs 1 \
  --total_interactions 20000 --horizon 100 --microbatch_size 1 \
  --ppo_epochs 2 --learning_rate 3e-5 \
  --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.2 \
  --value_coefficient 0.5 --entropy_coefficient 0.002 \
  --max_grad_norm 1.0 --target_kl 0.05 \
  --checkpoint_every_updates 100 \
  --evaluation_protocol integration \
  --run_dir "$MLP_GATE_RUN" --resume --headless

"$ISAAC_PYTHON" scripts/drone_evaluate.py \
  --checkpoint "$MLP_GATE_RUN/checkpoints/latest.pt" \
  --protocol integration --all_scenarios --headless \
  --output "$MLP_GATE_RUN/evaluation-integration-all.json"
```

Run original frozen LIF on isolated Reach first.  This is an independent,
bounded learning-path diagnostic; its weights are not warm-started into the
Mixed proof or any comparison job:

```bash
LIF_REACH_RUN=runs/crazyflie-balanced-v3-lif-reach-isolation-20k-v1
test ! -e "$LIF_REACH_RUN"
"$ISAAC_PYTHON" scripts/drone_train.py \
  --task FlyCrazyflie-WaypointReach-v0 \
  --contract_profile balanced_v3 \
  --policy frozen_lif_original --seed 0 --num_envs 4 \
  --total_interactions 20000 --horizon 100 --microbatch_size 4 \
  --ppo_epochs 2 --learning_rate 3e-5 \
  --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.2 \
  --value_coefficient 0.5 --entropy_coefficient 0.002 \
  --max_grad_norm 1.0 --target_kl 0.05 \
  --checkpoint_every_updates 100 \
  --evaluation_protocol integration \
  --run_dir "$LIF_REACH_RUN" --headless

"$ISAAC_PYTHON" scripts/drone_evaluate.py \
  --checkpoint "$LIF_REACH_RUN/checkpoints/latest.pt" \
  --protocol integration --scenario FlyCrazyflie-WaypointReach-v0 --headless \
  --output "$LIF_REACH_RUN/evaluation-reach-integration.json"
```

Then run the mandatory LIF-first Mixed proof.  The 500,000-interaction job uses
the same four-environment, 100-step, two-epoch PPO shape as the main matrix;
the main queue remains locked if any held-out scenario has a zero event score:

```bash
LIF_PROOF_RUN=runs/crazyflie-balanced-v3-lif-proof
test ! -e "$LIF_PROOF_RUN"
"$ISAAC_PYTHON" scripts/drone_train.py \
  --task FlyCrazyflie-Mixed-v0 \
  --contract_profile balanced_v3 \
  --policy frozen_lif_original --seed 0 --num_envs 4 \
  --total_interactions 500000 --horizon 100 --microbatch_size 4 \
  --ppo_epochs 2 --learning_rate 3e-5 \
  --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.2 \
  --value_coefficient 0.5 --entropy_coefficient 0.002 \
  --max_grad_norm 1.0 --target_kl 0.05 \
  --checkpoint_every_updates 100 \
  --evaluation_protocol lif_proof \
  --run_dir "$LIF_PROOF_RUN" --headless

"$ISAAC_PYTHON" scripts/drone_evaluate.py \
  --checkpoint "$LIF_PROOF_RUN/checkpoints/latest.pt" \
  --protocol lif_proof --all_scenarios --headless \
  --output "$LIF_PROOF_RUN/evaluation.json"

"$ISAAC_PYTHON" scripts/drone_verify_lif_proof.py
```

Preview and run the bounded integration matrix:

```bash
"$ISAAC_PYTHON" scripts/drone_run_matrix.py \
  --config configs/experiments/crazyflie_balanced_v3_integration.json \
  --dry_run --output runs/crazyflie_balanced_v3_integration_v1.json
bash scripts/execute_drone_integration.sh
```

Preview the complete main matrix without launching it:

```bash
"$ISAAC_PYTHON" scripts/drone_run_matrix.py \
  --config configs/experiments/crazyflie_balanced_v3_main.json \
  --dry_run --output runs/crazyflie_balanced_v3_main_v1.json
```

After every acceptance gate passes and the user explicitly authorizes the full
run, the launcher must first execute a fail-closed empirical-proof verifier.
That verifier must reject a missing or stale proof checkpoint/evaluation,
anything other than the three exact five-episode `lif_proof` scenarios, zero
genuine success events in any scenario, nonfinite evidence, a failed hard-memory
gate, or a
fingerprint/profile mismatch.  It must check episode rows and recomputed
aggregates rather than treating evaluator process success as flight success.
Until this verifier is implemented, tested, and invoked by
`execute_drone_matrix.sh` before its first child process, the following launch
command remains disabled even if `--authorize_main` is supplied:

```bash
bash scripts/execute_drone_matrix.sh --authorize_main
"$ISAAC_PYTHON" scripts/drone_monitor_matrix.py \
  --queue runs/crazyflie_balanced_v3_main_v1.json
```

The implementation agent must make each script provide `--help`, validate
arguments before starting Isaac, print the resolved config and fingerprint,
and exit nonzero on failed gates. It may adjust a proposed argument name only
if the final documentation and implementation agree exactly.

## 15. Additive post-acceptance wing-circuit experiment

This is an exploratory extension requested after the baseline plan was written.
It must not replace, modify, or delay the interpretation of the frozen leg-VNC
baseline, the four-controller 20-job main matrix, or any G1 preservation gate.
It begins only after Gates A--G above pass, uses separate artifact names and
queues, and does not authorize the full main matrix.

The locally pinned MaleCNS raw Feather files contain viable wing candidates.
A read-only feasibility scan found:

- 20 annotated wing mechanosensory cells (`superclass=vnc_sensory`,
  `subclass=wing`, proprioceptive or tactile class, `entryNerve=ADMN`), all
  with acetylcholine under the current sign rule;
- 67 annotated wing motor cells (`superclass=vnc_motor`, `subclass=wm`),
  spanning ADMN, MesoAN, PDMNa, and PDMNp exits;
- directed paths from all 20 wing sensory candidates through intrinsic VNC
  cells to all 67 wing motor candidates; and
- an optional, separately labelled haltere-feedback pool containing 195
  acetylcholine proprioceptive sensory candidates and four DMetaN motor cells.

These connectivity facts establish extraction feasibility only. They are not
evidence of drone control, biological flight equivalence, or a validated
model. The existing 256-cell circuit is explicitly a leg circuit and must
continue to be reported as such.

### 15.1 Leg-versus-wing circuit boundary

The baseline comparison and its 20-job/960-episode main matrix use only the
existing frozen leg-VNC circuit. Its declared biological path is:

```text
leg sensory inputs (ProLN, MesoLN, MetaLN; left and right)
  + declared descending inputs
  -> selected intrinsic VNC neurons
  -> selected leg motor-neuron readouts
  -> trainable decoder
  -> four Crazyflie aggregate-wrench actions
```

No wing sensory neuron, wing motor neuron, wing nerve, or haltere neuron is
present in that baseline circuit. The degree-preserving rewired baseline is a
rewire of the same leg circuit and is not a wing circuit.

The proposed wing experiment must instead use a separately extracted and
named biological path:

```text
wing sensory inputs (primary candidates: ADMN)
  + any deterministically selected descending inputs required for coverage
  -> separately selected intrinsic VNC neurons
  -> wing motor-neuron readouts selected from ADMN, MesoAN, PDMNa, and PDMNp
  -> separately trained decoder
  -> four Crazyflie aggregate-wrench actions
```

Leg and wing neurons must not be silently combined into one controller. This
extension must implement a third, explicitly named combined leg-plus-wing LIF
controller in addition to the untouched leg-only baseline and the new
wing-only controller. The combined controller is a required experimental
condition, not an optional future ablation. It must have its own deterministic
extraction/composition rule, immutable manifest, parameter accounting,
fingerprint, memory gates, pilot, evaluation artifacts, and dry-run queue. It
must be compared directly against both the untouched leg-only and wing-only
circuits and must not enter the baseline main matrix without an explicit plan
revision and user approval.

The required combined biological path is:

```text
leg sensory inputs (ProLN, MesoLN, MetaLN; left and right)
  + wing sensory inputs (primary candidates: ADMN)
  + deterministically selected descending inputs
  -> declared leg/wing-shared or separately partitioned intrinsic VNC neurons
  -> declared leg motor and wing motor readout populations
  -> one trainable fusion/decoder interface
  -> four Crazyflie aggregate-wrench actions
```

The combined design must state exactly where fusion occurs. The default
required design keeps the leg and wing frozen recurrent cores independently
identifiable, concatenates their declared motor-readout activities, and uses
one trainable decoder to produce the four actions. Cross-core recurrent edges
are forbidden unless a later plan revision predeclares their biological source
and extraction rule. Report activity, gradients, reset behavior, and frozen
checksums for both cores independently.

Haltere feedback is also separate from both the primary wing-only circuit and
the required combined circuit. Its inclusion must be explicit in the controller
name and provenance manifest.

Implement the wing experiment additively:

1. Add a dedicated deterministic extractor (for example,
   `scripts/prepare_malecns_wing.py`) and write only to a new
   `data/connectome_wing/` tree. Never overwrite `data/connectome/` or reuse its
   manifest/checksums.
2. Freeze and document exact selection, ranking, tie-breaking, transmitter,
   duplicate-edge, motor-origin-edge, path-repair, and disconnected-node rules.
   Use a compact path-complete circuit with an explicit size rationale. For a
   size-matched 256-cell primary variant, predeclare 32 inputs, 200 intrinsic
   cells, and 24 motor readouts; select any required descending inputs and the
   24-of-67 motor subset by deterministic graph criteria, not manual outcome
   tuning.
3. Treat haltere feedback as a separate named ablation/variant. Do not silently
   mix it into the primary wing circuit. Exclude the six serotonin-labelled
   haltere sensory candidates unless a new signed-weight rule is declared and
   justified before testing.
4. Emit an independent provenance manifest, source hashes, graph checksum,
   role/type/side/nerve counts, path-coverage audit, and immutable neuron/edge
   files. Verify the raw MaleCNS files against their pinned hashes.
5. Build a separately named wing-LIF controller while preserving the same
   interface: 12 drone observations -> trainable encoder -> declared wing
   inputs -> frozen LIF core -> declared wing motor readouts -> trainable
   decoder -> four bounded Crazyflie actions. Do not imply that a biological
   motor neuron maps one-to-one to a rotor.
6. Calibrate the LIF threshold and related engineering constants with a
   predeclared bounded activity pilot, then freeze them before flight-result
   comparisons. The leg-circuit threshold `0.047` must not be copied without
   validation. Report dead/saturated/spike fractions and frozen-core hashes.
7. If a degree-preserving wing rewire is tested, generate a new immutable
   rewire manifest and seed for the wing graph. Never apply the leg rewire
   artifact to the wing graph.
8. Add graph/provenance/unit tests, parameter accounting, recurrent reset and
   gradient checks, then run the same strict one-, two-, and four-environment
   RAM/VRAM/paging hard gates with RSS warnings retained. Use only one Isaac
   process at a time.
9. Run a bounded wing-LIF-first pilot on WaypointReach, checkpoint/resume it,
   and evaluate its exact checkpoint on Reach, Switch, and Gust. Report actual
   success, crash, censoring, activity, timing, and memory results; execution
   alone is not successful flight.
10. Build and test the required combined leg-plus-wing controller using two
    independently identifiable frozen recurrent cores. Feed the same 12-value
    Crazyflie observation to separately trainable leg and wing encoders,
    concatenate the declared leg and wing motor-readout activities, and train
    one fusion/decoder head for the same four bounded actions. Preserve and
    verify each core's checksum before and after every update. Run the same
   gradient, recurrent-reset, checkpoint/resume, one-/two-/four-environment
   hard-memory gates plus visible RSS warnings, Reach pilot, and
   Reach/Switch/Gust evaluation gates as wing-only.
11. Produce a bounded three-condition extension comparison containing the
    untouched original leg-LIF baseline, wing-only LIF, and combined
    leg-plus-wing LIF under matched task, seed, budget, evaluation, and resource
    settings. Parameter differences caused by the second frozen core must be
    reported explicitly; do not mislabel this as parameter matched. Generate a
    reviewable extension dry run, but do not launch a full extension matrix
    without explicit user authorization.
12. Keep every wing and combined artifact, config, run directory, queue, and report under a
   distinct `wing` label. Do not add wing jobs to the baseline 20-job/960-
   episode matrix without a new dry run and explicit user authorization.

The review-only extension queue is deliberately separate from the baseline
main queue. It contains the three required circuit conditions (leg-only,
wing-only, and independent-core leg-plus-wing), five seeds per condition,
5,000,000 interactions per job, and 16 fixed episodes for each of the three
held-out scenarios: 15 jobs, 45 checkpoint/scenario bundles, and 720 episodes.
Generate it without starting Isaac or training with:

```bash
"$ISAAC_PYTHON" scripts/drone_wing_run_matrix.py \
  --config configs/experiments/crazyflie_wing_main.json \
  --dry_run --output runs/crazyflie_wing_main_v2_verified.json
```

This extension planner intentionally exposes no `--execute` mode. A future
launch requires a reviewed executor and new explicit user authorization. The
baseline queue remains exactly four controllers, 20 jobs, and 960 evaluation
episodes.

Stop this exploratory branch only after the bounded wing-only and required
combined leg-plus-wing smoke/evaluation paths and their reviewable
three-condition extension dry run are complete. Do not launch a full wing or
combined comparison automatically.

## 16. Completion definition

Implementation is complete only when the additive drone stack, tests, asset
audit, memory gates, controller integration, checkpoint/resume path, bounded
integration matrix, full main dry run, and reproduction documentation all
pass while the frozen G1 hashes remain unchanged.

The research comparison is complete only when all 20 authorized main jobs and
all 960 fixed evaluation episodes finish or retain an explicit failed status,
and the report includes every controller and seed. No result may be invented,
backfilled, or called final from a smoke test, integration run, or single
seed.

The current implementation request authorizes balanced-v3 implementation,
acceptance tests, the LIF-first proof, integration, and preparation of the
reviewable main queue.  Earlier authorization to run a main matrix does not
waive the newly requested empirical-success or memory gates: the balanced-v3
20-job executor remains locked until those gates pass and its dry-run queue is
verified.
