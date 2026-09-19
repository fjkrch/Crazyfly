# Crazyflie learned keyboard-command control plan

## Active all-fair restart override — 2026-09-18

The user superseded the revision-3-only launch before its dry run or any new
1M training job started.  Preserve revision 1, revision 2, and the unlaunched
revision-3 implementation and artifacts, but do not use their results as an
exact fair comparison.  The active execution is a new additive revision-4
matrix in its own config, queue, output root, logs, checkpoints, evaluations,
plots, and report.

Revision 4 trains every engineering baseline and every declared non-empty
combination of the independently authenticated leg, wing-thoracic, and optic
connectomes from scratch under one reset-invariant command/wind schedule:

1. original leg frozen LIF,
2. degree-preserving rewired leg frozen LIF,
3. wing-thoracic frozen LIF,
4. optic-lobe frozen LIF,
5. independent leg + wing-thoracic frozen LIF,
6. independent leg + optic frozen LIF,
7. independent wing-thoracic + optic frozen LIF,
8. independent leg + wing-thoracic + optic frozen LIF,
9. matched GRU, and
10. normal MLP.

Each controller is trained once in still air and once in physical wind for
each of seeds 0, 1, and 2, with exactly 1,000,000 interactions, 40
environments, a 100-step horizon, two PPO epochs, and the same remaining PPO
hyperparameters.  The matrix is
therefore exactly 60 jobs and 60,000,000 new interactions.  Each completed job
receives 16 held-out 600-step episodes, for exactly 320 evaluation episodes
per seed and exactly 960 evaluation episodes overall.  All eight LIF
controllers at all three seeds run before GRU and MLP.  Main training is
strictly sequential (`max_parallel=1`), GPU utilization may reach 100%, VRAM
must remain strictly below 6,963.2 MiB, RAM strictly below 90%, and sustained
paging triggers a clean checkpoint and stop.

The revision-4 supervisor is failure-isolated.  A failed training,
evaluation, provenance, numerical, or per-job memory gate records an immutable
attempt receipt and clean checkpoint when available, marks only that cell
failed/paused, and proceeds to later eligible cells.  It never represents a
partial matrix as complete.  `--execute --resume` retries incomplete cells.
A currently unsafe global resource precheck launches no child; it records and
defers work until a later safe precheck rather than bypassing the gate.

For fairness, the requested command cursor and (for wind jobs) physical-wind
cursor advance once for every vector control interval for every environment,
including environments that terminate on that interval.  Runtime resets reset
physical state and the integrated target but do not resample, advance, or
rewind either schedule.  The realized command trace is consequently a pure
function of schedule seed, environment index, global vector-control interval,
and curriculum clock—not survival, reset timing, controller, or condition.
The still/wind member of every controller pair sees the same command trace;
all wind cells see the same wind trace.  This semantic contract must have a
new version/hash, must round-trip at update 0 and later checkpoint boundaries,
and must be tested against deliberately different reset masks.

The exact fair report is built only from these 60 freshly trained revision-4
cells.  Prior revision-2 and revision-3 artifacts may be cited as historical
context only and may not be merged into the revision-4 20-cell/320-episode
per-seed table or the aggregate 60-cell/960-episode score table.  Parameter
counts, dynamic state, frozen per-core checksums,
actual action-producing inference latency, training/evaluation RAM and VRAM,
score per 1,000 actor parameters, still/wind deltas, reward/loss curves, and
per-core activity are mandatory.  Cross-capacity comparisons remain
descriptive; topology claims are permitted only within a genuinely matched
capacity/contract tier.

Before revision-4 execution: run the complete unit suite; native, still, and
wind environment smokes; bounded PPO still/wind smokes for all ten controller
types; checkpoint/pause/resume coverage for every controller type including
the update-0 boundary; strict preservation/source manifests; and a verified
60-job dry run.  No older smoke or resume artifact generated before the
reset-invariant contract change counts toward these gates.  After execution,
all 60 training manifests and 60 evaluations must pass the reporter before the
aggregate report is written.  At final handoff no Isaac, trainer, evaluator,
queue, or NVIDIA compute process may remain.

## Active objective — 2026-09-18

The completed LIF neuron-activity result, continuous-viewer validation, and
recommended causal ablation gate are recorded in
`docs/crazyflie_lif_activity_analysis.md`. That report is descriptive evidence
from authenticated action-producing forward passes; it does not replace the
new optic/wind 1M matrix specified below.

Train and compare neural controllers only for direct keyboard-commanded
Crazyflie flight. Revision 2 extends the completed still-air comparison with
one matched physical-wind condition and one independently extracted optic-lobe
LIF controller. The active work is additive and contains exactly two new
command-v2 control tasks; the completed command-v1 task and its checkpoints
remain untouched:

`FlyCrazyflie-CommandFollowWide-v0`

`FlyCrazyflie-CommandFollowWideWind-v0`

The wind task is not the old autonomous GustRecovery task. It receives the
same held-key velocity/yaw commands, 12-value observation, four native wrench
actions, reward, PPO hyperparameters, seed, and evaluation command script as
the wide still-air task; only a deterministic, versioned external-force and
torque field is added. Wind is applied physically through Isaac's articulation
API, never through observation or reward edits, and its complete schedule is
checkpointed and restored.

Do not train, evaluate, score, resume, delete, rename, or overwrite the old
WaypointReach, WaypointSwitch, GustRecovery, or Mixed jobs. Their historical
plan is preserved at
`docs/crazyflie_neural_comparison_plan_paused_20260918.md`, and all old queues,
runs, logs, checkpoints, evaluations, activity files, and Unitree G1 files
remain paused and immutable.

## What is learned

The policy learns to track body-relative velocity commands generated by held
keyboard keys. It is not trained to autonomously seek a waypoint, switch a
target, or recover from a scripted gust.

| Held key | Requested command |
|---|---|
| `W` / `S` | forward / backward |
| `A` / `D` | left / right |
| `I` or `E` / `Q` | up / down |
| `J` / `L` | yaw left / right |
| `H` or `Space` | release movement and hover |
| `R` | reset, clear keys, and set a new hover point |
| `Esc` | clean exit |

Key state is held, not accumulated. Opposite keys cancel. Multiple axes,
including `W+A`, `W+I`, and `W+A+I+J`, work simultaneously. Horizontal
diagonals are normalized so they are not faster than a single horizontal
axis.

## Installed Isaac and command-v2 wind contract

Use the installed `Isaac-Quadcopter-Direct-v0` Crazyflie implementation as the
physics/action base. Preserve its Crazyflie asset, 0.01 s physics step,
decimation 2 (50 Hz control), four normalized aggregate-wrench actions, and
12-value policy observation width. Do not edit installed Isaac Lab.

Neither task exposes wind, camera pixels, a privileged target, or the desired
action in the observation. This deliberately tests closed-loop compensation
from the same onboard-style kinematic/state error vector. The optic-lobe graph
therefore supplies a different frozen recurrent topology, not an extra sensor.

The `command_v2` task's versioned 12-value observation is:

1. body linear-velocity tracking error, `actual - effective command` (3),
2. body angular velocity x/y and yaw-rate tracking error (3),
3. projected gravity in the body frame (3), and
4. integrated safe target-position error in the body frame (3).

This representation exposes forward, lateral, vertical, and yaw commands
without changing any controller width. The shared fixed stabilization prior
therefore acts on command error, while each neural controller learns the
bounded residual. In trained viewer mode the policy's four-value action is
the primary action; deterministic flight assist is allowed only as an
explicit, counted safety fallback for invalid output/state or a breached hard
safety envelope. A valid policy action must never be blended with fallback.

## Runtime speed parameters and trained envelope

The visible command accepts these launch-time parameters:

- `--horizontal_speed`: `0.0` through `1.00` m/s, default `0.50`
- `--vertical_speed`: `0.0` through `0.50` m/s, default `0.25`
- `--yaw_rate`: `0.0` through `1.50` rad/s, default `0.80`

The checkpoint records the same trained maxima. Runtime values above the
checkpoint envelope are rejected instead of being described as validated.
Near the floor, ceiling, or horizontal workspace edge, requested commands are
converted to safe effective commands before reaching the policy. The viewer
shows both requested and effective commands.

## Command curriculum and reward

Each command is held for a seeded random 25–100 control steps, with explicit
hover/release, cardinal, diagonal, vertical, yaw, and fully simultaneous
segments. Opposite directions are symmetric. The fixed interaction
curriculum is:

| Interactions | Max horizontal | Max vertical | Max yaw rate |
|---:|---:|---:|---:|
| `0–99,999` | 0.25 m/s | 0.15 m/s | 0.40 rad/s |
| `100,000–249,999` | 0.50 m/s | 0.25 m/s | 0.80 rad/s |
| `250,000–499,999` | 0.80 m/s | 0.40 m/s | 1.20 rad/s |
| `500,000–1,000,000` | 1.00 m/s | 0.50 m/s | 1.50 rad/s |

For the wind condition, a deterministic curriculum is layered on the identical
command stages. Training seed `20260918` produces a stateless, reproducible
per-environment sequence cycling through calm, force-only, torque-only, and
combined segments. Active directions are sampled uniformly on the unit sphere;
their magnitude is 25–100% of the active stage limit. Active pulses last
10–30 control steps and calm intervals last 25–75 steps.

Wind is an actual world-frame wrench applied at the Crazyflie body center of
mass with Isaac's articulation `permanent_wrench_composer`: its pose cache is
reset at each physics substep, then wind is set with
`set_forces_and_torques(..., is_global=True)` before the native body-frame
motor wrench is added. This avoids the deprecated wrapper and preserves the
same physical external-wrench semantics without warning-log flooding.
Force equals the dimensionless force ratio times vehicle weight. Torque equals
the dimensionless torque ratio times vehicle weight times the fixed `0.046 m`
reference arm. The curriculum is:

| Interactions | Max force / weight | Max torque / (weight x 0.046 m) |
|---:|---:|---:|
| `0–99,999` | 0.03 | 0.02 |
| `100,000–249,999` | 0.05 | 0.03 |
| `250,000–499,999` | 0.08 | 0.05 |
| `500,000–749,999` | 0.12 | 0.075 |
| `750,000–1,000,000` | 0.15 | 0.10 |

Held-out wind uses the independent seed `20260919`, never seen during
training. Each 600-step episode has 25-step pulses beginning at steps 75, 175,
275, 375, and 475, cycling force-only, torque-only, and combined cases at the
full 0.15/0.10 envelope. Record the complete contract and SHA-256. No
aerodynamic effect may be simulated by editing observations, rewards, actions,
or command targets.

The reward is continuous, not a waypoint success flag. It rewards linear and
yaw command tracking, acceleration/tracking-error progress in the requested
direction, stable attitude, and safe survival. It penalizes acceleration in
the wrong direction, jerk, tilt, unwanted angular rate, effort, action change,
drift during hover, boundary pressure, and hard failure. Raw acceleration
magnitude is never rewarded by itself because that would encourage
oscillation. Every component and the exact contract hash must be reported.

The implemented `command_v2` weights are explicit and immutable in the task
contract. They retain the command-v1 reward design and widen only the declared
tracking scales. Tracking, target retention, stability, effort, smoothness, jerk, and
survival rates are multiplied by the 0.02 s control interval; progress and the
bounded wrong-direction acceleration term are transition terms; failure is a
one-time terminal penalty.

| Reward component | Weight / meaning |
|---|---:|
| linear velocity tracking | `+1.50 * exp(-normalized_error²)` |
| yaw-rate tracking | `+0.30 * exp(-normalized_error²)` |
| tracking-error progress | `+0.20 * clipped(previous_error-current_error)` |
| acceleration that increases tracking error | `-0.03 * bounded_projection` |
| normalized jerk squared | `-0.002` |
| integrated command-target retention | `+0.20 * exp(-normalized_error²)` |
| attitude / angular stability | `-0.20` / `-0.03` |
| control effort / action change | `-0.01` / `-0.005` |
| alive inside the hard safety envelope | `+0.05` per second |
| low/high/workspace/nonfinite hard failure | `-5.00` once |

## Controller comparison and main queue

Train the following seven controllers under both conditions with the exact
same command stream, PPO settings, seed, interaction budget, evaluation
scripts, and safety gates:

1. original frozen LIF,
2. degree-preserving rewired frozen LIF,
3. wing-thoracic frozen LIF,
4. independent leg + wing-thoracic frozen LIF,
5. optic-lobe frozen LIF,
6. matched GRU, and
7. normal MLP.

Use seed `0` and exactly `1,000,000` interactions per task/controller cell:
fourteen jobs and `14,000,000` interactions total. Every one of the ten LIF
cells has scheduling priority over every GRU/MLP cell. The completed six-job,
500,000-interaction still-air run is preserved as historical evidence and is
never overwritten or counted toward this new budget. The
leg+wing extension may remain larger, but every parameter count must be
reported; original/rewired/wing/optic LIF, GRU, and MLP must retain the
existing matched actor-parameter gate. Preserve every frozen LIF weight and
verify each checksum before and after training. The optic graph must come from
the locally pinned MaleCNS release through a deterministic audited extractor;
synthetic or randomly generated optic graphs are forbidden.

The optic controller is a real frozen 256-neuron bilateral MaleCNS optic-lobe
graph with exactly 32 optic sensory inputs, 200 optic intrinsic interneurons,
and 24 visual-projection readouts. Its 1,628 signed recurrent edges and roles
come from the pinned `MaleCNS v1.0 flat connectome, minconf 0.5` artifacts. Its
trainable observation/action adapter has exactly `4,776` actor parameters,
matching the original, rewired, and wing LIF target; the frozen recurrent
weights do not train. GRU and MLP use the existing accepted near-match gate,
while the explicitly unmatched independent leg+wing fusion is reported as
such instead of being presented as capacity matched.

Fairness is paired and predeclared: every controller receives the same seed-0
command samples, 40 environments, 100-step horizon, two PPO epochs, optimizer
settings, 1M interaction budget, 12 observations, four actions, reward,
termination logic, and held-out command episodes. The still-air and wind
members of each pair receive the identical command stream. All wind jobs also
receive the same training wind seed/schedule, and all wind evaluations receive
the same independent held-out wind protocol. Controller identity and physical
wind are the only experimental factors; neither observation width nor scoring
weights may vary by controller or condition.

Run one training process at a time. The prior exact two-process gate detected
sustained paging, so it is negative evidence and must not authorize parallel
main training. GPU utilization may reach 100%, but device memory must stay
strictly below 6,963.2 MiB (the user-approved 6.8 GiB ceiling and below 90% of
the 8,151 MiB reported device) and system RAM strictly below 90%.
Missing/nonfinite GPU telemetry, sustained paging, or a hard memory violation
stops the affected job only after writing a clean resumable checkpoint.
Network access is not required after verified local assets are prepared.

## Additive three-combination LIF extension

The running fourteen-job revision-2 queue is immutable. Do not edit any file
included in its source-set fingerprint, change its configuration, or insert a
new cell into it. After that queue and its evaluations finish, run a separate
revision-3 follow-on queue that completes the three missing non-empty
combinations of the independently authenticated leg, wing-thoracic, and optic
connectomes.
The four combinations already represented by revision 2 are leg, wing,
optic, and leg+wing. Revision 3 adds:

1. independent leg + optic frozen LIF,
2. independent wing-thoracic + optic frozen LIF, and
3. independent leg + wing-thoracic + optic frozen LIF.

Natural multi-core fusion keeps one observation encoder per frozen core,
concatenates only the authenticated motor/readout populations, and uses one
shared action decoder. No recurrent edge may be invented between biological
graphs. Every graph remains frozen and every per-core checksum is verified
before and after training. With the existing 64-unit adapters, the declared
trainable-actor capacity targets are approximately `4,776` for one core,
`9,224` for two cores, and `13,672` for three cores; the implementation must
record exact counts and reject unreported drift.

Per the user's narrowed execution scope, revision 3 contains no newly trained
GRU or MLP controls. The critic remains exactly `18,305` trainable parameters
for every cell. Thus the follow-on queue has three LIF controller labels, two
physical conditions, six jobs, and exactly `6,000,000` new training
interactions. Existing revision-2 GRU and MLP results remain immutable and may
appear only as clearly identified historical context, not as capacity-matched
controls for the larger multi-core actors.

Every follow-on cell uses seed 0, 40 environments, a 100-step horizon, two PPO
epochs, the identical command and wind streams, exactly 1,000,000
interactions, and the unchanged 16-by-600-step held-out protocol. Compare
leg+wing, leg+optic, and wing+optic descriptively as the two-core tier. Report
leg+wing+optic as the three-core tier without claiming a within-tier winner.
Single-core claims continue to use the completed revision-2 results. Cross-tier
tables are descriptive only
and must include score per 1,000 trainable actor parameters, inference
latency, dynamic-state size, peak RAM, and peak VRAM; they may not claim a
topology advantage from raw score alone.

Revision 3 must be implemented only after revision 2 reaches a terminal clean
state because the reproduction fingerprint intentionally hashes every active
Crazyflie source module. It gets its own before/after source manifest, config,
dry run, pause/resume gate, bounded still/wind integration matrix, queue file,
logs, checkpoints, evaluations, and report. It must never rewrite or merge a
revision-2 artifact.

## Evaluation and score

Every completed checkpoint runs 16 episodes of the same deterministic held-out
command script containing idle, both signs of each axis, horizontal diagonals,
horizontal+vertical combinations, yaw combinations, full simultaneous input,
release/braking, and hover. Report raw measurements and a transparent
0–100 score derived from:

- velocity and yaw-rate tracking error,
- command projection and wrong-direction motion,
- response latency and overshoot,
- release/braking settling time,
- hover drift,
- survival and invalid/crash/boundary events,
- action effort and smoothness, and
- real controller activity by command segment.

Activity means observed activation/correlation, not causal proof. For LIF,
report authenticated neuron groups including leg, wing, thoracic, sensory,
optic sensory, optic intrinsic, visual projection, interneuron, and motor
populations as applicable. Report GRU hidden activity and MLP hidden-layer
activity using comparable descriptive statistics. Revision 2 evaluates
`14 * 16 = 224` held-out episodes and revision 3 evaluates another
`6 * 16 = 96`. The final combined study therefore contains 320 held-out
episodes, reported by cell, condition, topology tier, and in a single 20-cell
table. Also report each controller's wind score delta relative to its
still-air score; do not present correlation or activation as proof that a
named biological region caused performance.

## Required acceptance gates

1. Create immutable before/after SHA-256 manifests and prove all 37 frozen G1
   files match exactly; never include generated run files as frozen source.
2. Pure unit tests cover command state, simultaneous axes, normalization,
   deterministic schedule/resume, curriculum bounds, safety clamping,
   observation ordering/signs, reward monotonicity, failure classification,
   checkpoint/task/envelope rejection, recurrent reset, exact action
   selection, fallback accounting, and activity grouping.
3. Native `Isaac-Quadcopter-Direct-v0` offline smoke passes using the pinned
   local asset and procedural ground.
4. Custom `FlyCrazyflie-CommandFollowWide-v0` and
   `FlyCrazyflie-CommandFollowWideWind-v0` smokes pass under `command_v2` with
   finite 12-value observations, finite bounded four-value actions, command
   changes, reward components, safety resets, deterministic seeded command
   scheduling, and an authenticated physical-wrench response in the wind task.
5. All seven controllers complete bounded PPO smoke runs in still air; every
   controller also completes a bounded wind-path smoke. Report parameter
   counts, frozen-core checksums, finite gradients/losses, RAM, and VRAM.
6. A short checkpoint/pause/resume test reproduces counters, command schedule,
   recurrent state, normalization, RNG state, and immutable history.
7. Preserve the prior failed two-process paging receipt and force the new
   queue to `max_parallel=1`; no concurrency override is accepted.
8. Generate and validate a fresh 14-job dry run. It must reference only the
   two command-follow task IDs, contain exactly 1,000,000 interactions per
   cell, and order all LIF jobs before GRU/MLP.
9. Run the fourteen 1M jobs, 224 held-out command evaluations, activity
   collection, large score table, plots, and verified trained-keyboard smoke.
   Never represent an incomplete or failed job as successful.
10. A visible viewer command loads one selected command-task checkpoint,
    clearly states which trained controller owns the action, shows a close
    follow camera and live command/state/action/fallback telemetry, and opens
    a separate real-time activity window when requested.
11. At completion, no Isaac, trainer, evaluator, queue, or NVIDIA compute
    process may remain. Old paused queues and every G1 artifact must remain
    unchanged.
12. Only after revision 2 completes, implement and unit-test the three missing
    connectome combinations. Prove
    exact per-core provenance, frozen checksums, actor/critic counts, recurrent
    reset, checkpoint reconstruction, and activity labels for all cores.
13. Run bounded still/wind PPO and pause/resume gates for all three revision-3
    controllers, then produce a verified sequential six-job dry run. Do
    not launch it if any capacity, source-identity, memory, or preservation
    gate fails.
14. Run revision 3 only as a separate follow-on queue; complete its six 1M
    jobs and 96 held-out episodes, then generate the topology and efficiency
    efficiency report without altering any revision-2 artifact.

## Canonical interpreter and intended commands

```bash
cd /home/chayanin/Desktop/flyg1
ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

# Unit suite
"$ISAAC_PYTHON" -m pytest tests/unit -q

# Native and both command-v2 task offline smokes
"$ISAAC_PYTHON" scripts/drone_smoke_env.py \
  --task Isaac-Quadcopter-Direct-v0 --num_envs 1 --steps 100 \
  --headless --offline_only
"$ISAAC_PYTHON" scripts/drone_smoke_env.py \
  --task FlyCrazyflie-CommandFollowWide-v0 --contract_profile command_v2 \
  --num_envs 4 --steps 500 --headless --offline_only
"$ISAAC_PYTHON" scripts/drone_smoke_env.py \
  --task FlyCrazyflie-CommandFollowWideWind-v0 --contract_profile command_v2 \
  --num_envs 4 --steps 500 \
  --headless --offline_only

# Fresh command-only 7 x 2 matrix: dry run first, then sequential execution
# only after every gate above passes. The old six-job queue remains immutable.
"$ISAAC_PYTHON" scripts/crazyflie_command_queue_v2.py \
  --config configs/experiments/crazyflie_command_optic_wind_seed0_1m.json --dry_run
"$ISAAC_PYTHON" scripts/crazyflie_command_queue_v2.py \
  --config configs/experiments/crazyflie_command_optic_wind_seed0_1m.json --execute

# Read-only status, clean pause, and checkpoint resume for that verified queue.
"$ISAAC_PYTHON" scripts/crazyflie_command_queue_v2.py \
  --config configs/experiments/crazyflie_command_optic_wind_seed0_1m.json --status
"$ISAAC_PYTHON" scripts/crazyflie_command_queue_v2.py \
  --config configs/experiments/crazyflie_command_optic_wind_seed0_1m.json --pause
"$ISAAC_PYTHON" scripts/crazyflie_command_queue_v2.py \
  --config configs/experiments/crazyflie_command_optic_wind_seed0_1m.json \
  --execute --resume

# Visible learned keyboard flight; choose a completed command-task checkpoint.
"$ISAAC_PYTHON" scripts/crazyflie_trained_keyboard.py \
  --checkpoint runs/<command-job>/checkpoints/latest.pt \
  --horizontal_speed 1.00 --vertical_speed 0.50 --yaw_rate 1.50

# After revision 2 is terminal and all revision-3 gates pass: create the
# separate three-combination LIF dry run. The revision-3 script/config
# are not permitted to replace the revision-2 queue or output directory.
"$ISAAC_PYTHON" scripts/crazyflie_command_combinations_queue_v3.py \
  --config configs/experiments/crazyflie_command_combinations_seed0_1m.json \
  --dry_run

# Execute only the three requested combinations, sequentially, after the
# verified dry run and all revision-3 controller gates pass.  This is six
# jobs (still + wind for each controller), not another GRU/MLP matrix.
"$ISAAC_PYTHON" scripts/crazyflie_command_combinations_queue_v3.py \
  --config configs/experiments/crazyflie_command_combinations_seed0_1m.json \
  --execute

# Read-only status, clean pause, and exact checkpoint resume for revision 3.
"$ISAAC_PYTHON" scripts/crazyflie_command_combinations_queue_v3.py \
  --config configs/experiments/crazyflie_command_combinations_seed0_1m.json \
  --status
"$ISAAC_PYTHON" scripts/crazyflie_command_combinations_queue_v3.py \
  --config configs/experiments/crazyflie_command_combinations_seed0_1m.json \
  --pause
"$ISAAC_PYTHON" scripts/crazyflie_command_combinations_queue_v3.py \
  --config configs/experiments/crazyflie_command_combinations_seed0_1m.json \
  --execute --resume

# Generate the authenticated six-cell topology/efficiency report and its
# combined 20-cell historical-context table only after all evaluations pass.
"$ISAAC_PYTHON" scripts/crazyflie_command_combinations_report.py \
  --config configs/experiments/crazyflie_command_combinations_seed0_1m.json

# Active revision-4 all-fair restart: ten controllers x still/wind x seeds
# 0, 1, and 2.  This is exactly 60 sequential 1M-interaction jobs and 960
# held-out episodes.  The dry run must be created and verified before execute.
"$ISAAC_PYTHON" scripts/crazyflie_command_all_fair_queue_v4.py \
  --config configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json \
  --dry_run
"$ISAAC_PYTHON" scripts/crazyflie_command_all_fair_queue_v4.py \
  --config configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json \
  --execute

# Read-only status and clean operator pause/resume.  Per-cell training,
# evaluation, provenance, numerical, and job-memory failures are retained as
# immutable attempts and skipped so later cells can run.  Resume retries only
# failed/paused/incomplete cells.  An unsafe global precheck launches nothing.
"$ISAAC_PYTHON" scripts/crazyflie_command_all_fair_queue_v4.py \
  --config configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json \
  --status
"$ISAAC_PYTHON" scripts/crazyflie_command_all_fair_queue_v4.py \
  --config configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json \
  --pause
"$ISAAC_PYTHON" scripts/crazyflie_command_all_fair_queue_v4.py \
  --config configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json \
  --execute --resume

# Fail-closed publication: refuses to write the fair result unless all 60
# fresh cells and all 960 fresh held-out episodes pass their gates.
"$ISAAC_PYTHON" scripts/crazyflie_command_all_fair_report_v4.py \
  --config configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json
```

The old three-task matrix commands are intentionally absent.  Revision 2 and
revision 3 remain preserved historical work; the active completion target is
the fresh 60-cell revision-4 command-control comparison, not autonomous
waypoint performance.  The historical 500k, revision-2, and revision-3 runs
remain inspectable but are never silently merged into the revision-4 table.

## Additive stock G1/Go1 walking-control queue — 2026-09-19

The user's latest scoped request adds a separate, dry-run-only native Isaac Lab
walking queue.  It does not resume, replace, edit, or score the paused custom
Unitree G1 matrix above, and it does not alter any Crazyflie source, queue,
checkpoint, or run.  New outputs live only below
`stock_isaaclab_runs/default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1/`;
the frozen `runs/` tree remains byte-identical.

Use the installed training tasks and their registered default environments,
rewards, observations, joint-position actions, PPO settings, and native RSL-RL
MLP agent:

1. `Isaac-Velocity-Flat-G1-v0`,
2. `Isaac-Velocity-Rough-G1-v0`,
3. `Isaac-Velocity-Flat-Unitree-Go1-v0`, and
4. `Isaac-Velocity-Rough-Unitree-Go1-v0`.

"Flat" is the installed Isaac Lab name for the requested plain terrain.  The
`-Play-v0` variants are forbidden for training because they change scene,
randomization, command, and terrain behavior.  Every task is planar velocity
following only: `lin_vel_x`, `lin_vel_y`, and yaw (`ang_vel_z`).  There is no
vertical velocity, base-height, or up/down command.  Rough terrain can move a
body vertically through contact physics, but the policy is never commanded to
move up or down.  Preserve the exact registered command ranges:

| Robot/task | `vx` m/s | `vy` m/s | yaw rad/s |
|---|---:|---:|---:|
| G1 Flat | `[0, 1]` | `[-0.5, 0.5]` | `[-1, 1]` |
| G1 Rough | `[0, 1]` | `[0, 0]` | `[-1, 1]` |
| Go1 Flat | `[-1, 1]` | `[-1, 1]` | `[-1, 1]` |
| Go1 Rough | `[-1, 1]` | `[-1, 1]` | `[-1, 1]` |

Therefore the default G1 Rough task deliberately cannot strafe.  Do not widen
that range while calling the environment default.  Commands resample every 10
seconds and are stochastic training commands, not live keyboard input.

Use seeds 0, 1, and 2.  The matrix has one native stock MLP controller, four
tasks, and three seeds: exactly 12 jobs.  G1 and Go1 must be physically
separated:

- `.../g1/queue.json` contains six G1 jobs and only G1 logs/checkpoints.
- `.../go1/queue.json` contains six Go1 jobs and only Go1 logs/checkpoints.
- `matrix_index.json` is the read-only aggregate index.
- Each job uses `{robot}/{flat|rough}/seed_N/attempt_001/` as its working
  directory, so the stock trainer's relative `logs/rsl_rl/...` output cannot
  cross robot folders.

The installed task default is 4,096 environments, but that value is not yet
authorized under the user's strict 6.8 GiB device-memory limit.  Use a common
provisional safety override of 1,024 environments, while retaining 24 steps
per environment and every task's native iteration count and PPO parameters.
Before execution, run one measured update for all four tasks and require every
one to remain strictly below 6,963.2 MiB VRAM and 90% system RAM.  Missing or
nonfinite telemetry fails closed.  Main execution remains sequential
(`max_parallel=1`); the prior real paging event forbids parallel training.

| Job type | Native iterations | Interactions/job at 1,024 envs | Actor trainable parameters | Critic parameters |
|---|---:|---:|---:|---:|
| G1 Flat | 1,500 | 36,864,000 | 85,962 | 81,281 |
| G1 Rough | 3,000 | 73,728,000 | 328,266 | 323,585 |
| Go1 Flat | 300 | 7,372,800 | 40,856 | 39,425 |
| Go1 Rough | 1,500 | 36,864,000 | 286,616 | 285,185 |

The actor totals include the native trainable state-independent Gaussian
action-standard-deviation vector (37 values for G1 and 12 for Go1).  The MLP
weights alone are respectively 85,925, 328,229, 40,844, and 286,604; both
figures are recorded in every queue job so they cannot be conflated.

Across three seeds this is 331,776,000 G1 interactions, 132,710,400 Go1
interactions, and 464,486,400 total predicted interactions.  These are native
task budgets, not a capacity-matched cross-robot experiment: rewards,
observations, action widths, MLP widths, and iteration budgets differ.  Never
aggregate raw task rewards into a purported fair G1-versus-Go1 winner.

The stock default agent is an MLP.  Frozen LIF, rewired LIF, and GRU are not
registered stock default agents for these tasks; any such comparison requires
a later, separate adapter study with explicit capacity and recurrent-state
contracts.  Do not silently include those controllers in this native baseline
queue.

The dry run must verify the pinned Isaac Lab commit and ten installed source
hashes, all 37 frozen custom G1 files, and the frozen 1,429-file `runs/` tree.
It launches no trainer or simulator.  Native RSL-RL resume runs additional
iterations rather than an exact target-total continuation, so an incomplete
job must be retained as an immutable attempt and restarted from scratch in a
new attempt directory; exact resume must not be claimed.

```bash
cd /home/chayanin/Desktop/flyg1
ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

# Pure unit gates for this queue.
"$ISAAC_PYTHON" -m pytest tests/unit/test_default_locomotion_queue_v1.py -q

# Create the two physically separate six-job queues plus the aggregate index.
# This is a dry run only and launches no training process.
"$ISAAC_PYTHON" scripts/default_locomotion_queue_v1.py \
  --config configs/experiments/default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1.json \
  --dry_run

# Read-only status after the verified dry run.
"$ISAAC_PYTHON" scripts/default_locomotion_queue_v1.py \
  --config configs/experiments/default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1.json \
  --status
```

`--execute` is intentionally fail-closed in revision 1.  Do not add or invoke
an executor until the user reviews the dry run and all four one-update memory
smokes pass.  At this handoff no stock locomotion main job has started.
