# Crazyflie task and experiment specification

This document describes the additive implementation beside the paused Unitree
G1 work. It is an execution contract, not a scientific result. The G1 source,
runs, checkpoints, queues, and pause state remain separate and must not be
resumed or changed by these tools.

## Upstream embodiment contract

All three project tasks extend the installed
`Isaac-Quadcopter-Direct-v0` direct environment and retain its
`isaaclab_assets.CRAZYFLIE_CFG` / `cf2x.usd` vehicle, 0.01 s physics step,
decimation 2, 0.02 s control step (50 Hz), and aggregate-wrench action mapping.
The live asset audit is in
[`crazyflie_asset_audit.json`](crazyflie_asset_audit.json) and
[`crazyflie_asset_audit.md`](crazyflie_asset_audit.md). At the recorded audit,
the mass was 0.0282000024 kg, the installed Isaac Lab commit was
`b4c321024792976150ca55fddb26fa34480d974e`, and all pinned native-contract
checks passed. Every executable revalidates rather than assuming those values.

The policy observation has exactly 12 single-step values, with no previous
action or stacked history:

1. body-frame linear velocity, XYZ in m/s;
2. body-frame angular velocity, XYZ in rad/s;
3. projected gravity, XYZ;
4. body-frame displacement to the active goal, XYZ in m.

Every controller uses the same immutable physics-scale normalization. Raw
observations are divided elementwise by
`[2, 2, 2, 5, 5, 5, 1, 1, 1, 2, 2, 2]` and clipped to `[-5, 5]`. The first
three scales apply to body linear velocity, the next three to body angular
velocity, the next three to projected gravity, and the final three to goal
displacement. Its `update` operation is deliberately a no-op: neither training
nor evaluation learns running statistics, and crash-state samples therefore
cannot contaminate normalization. Checkpoints store and validate the fixed
scale/clip contract.

Actions are four tanh-bounded values in `[-1, 1]`. Action 0 maps to
`(a0 + 1) / 2 * 1.9 * vehicle weight` along body +Z. Actions 1--3 map to body
moments of `0.01 * action` N m. These are aggregate force/moment commands, not
individual motor commands.

## Project task IDs and resets

The additive, idempotent registrations are:

- `FlyCrazyflie-WaypointReach-v0`
- `FlyCrazyflie-WaypointSwitch-v0`
- `FlyCrazyflie-GustRecovery-v0`
- `FlyCrazyflie-Mixed-v0` (training-only)

They do not replace or shadow NVIDIA's native task. Each has a 12 s horizon,
600 control decisions, and identical vehicle, observation, action, and failure
conventions. The custom reset explicitly sets `episode_length_buf` to zero;
this removes the native full-reset horizon randomization during fixed
evaluation. Per-environment resets do not modify other environments.

"Default task" has two levels here. The installed
`Isaac-Quadcopter-Direct-v0` is the authoritative native embodiment: it
provides the Crazyflie asset, direct-environment dynamics, 12-value observation
layout, four aggregate-wrench actions, timing, and native bounds. The native
smoke test runs that registration directly. The controller experiment's
active contract profile is `balanced_v3`. `WaypointReach` is its isolated
LIF-first proof task; the four-controller comparison trains on
`FlyCrazyflie-Mixed-v0`, which deterministically rotates Reach, Switch, and
Gust episodes. `WaypointSwitch` and `GustRecovery` still use the identical
body/physics/observation/action contract and remain separate immutable held-out
evaluation scenarios.  Selecting balanced-v3 changes no Isaac physics, asset,
observation, or action hyperparameter.

Training uses the same fixed interaction-gated reset curriculum for every
controller and seed. A stage is selected only when a new episode resets; an
episode in flight is never interrupted or mutated. The exact completed-
interaction boundaries and distributions are:

| Start | Stage | Spawn center Z | Position half-range X/Y, Z | Yaw half-range | Linear/angular velocity half-range | Goal X/Y | Goal Z |
| ---: | --- | ---: | --- | ---: | --- | --- | --- |
| 0 | `near_3d` | 1.00 m | 0.01, 0.005 m | 0.02 rad | 0.00 m/s, 0.00 rad/s | `[-0.35, 0.35]` m | `[0.80, 1.20]` m |
| 200,000 | `local_3d` | 0.85 m | 0.025, 0.01 m | 0.05 rad | 0.02 m/s, 0.03 rad/s | `[-0.75, 0.75]` m | `[0.65, 1.35]` m |
| 500,000 | `mid_3d` | 0.70 m | 0.05, 0.02 m | 0.10 rad | 0.04 m/s, 0.07 rad/s | `[-1.25, 1.25]` m | `[0.60, 1.50]` m |
| 1,000,000 | `full` | 0.50 m | 0.10, 0.05 m | 0.25 rad | 0.10 m/s, 0.20 rad/s | `[-2.00, 2.00]` m | `[0.50, 1.50]` m |

The minimum start-to-target separations are respectively 0.40, 0.50, 0.65,
and 0.75 m. A real vector step advances the exact clock by `num_envs` before
terminal rows auto-reset, so a reset on a boundary immediately uses the new
stage. Checkpoint resume restores the completed-interaction count before the
first reset/rollout and may never decrease it. With four environments and a
100-step rollout, all boundaries are divisible by the 400 interactions per
update. The final stage is byte-fingerprinted and exactly equals the original
full reset/goal distribution. `deterministic_eval=True` or any installed
immutable episode plan always bypasses the training curriculum and selects
that full stage for new episodes.

The immutable held-out manifest uses evaluation seed 101. Its initial states,
targets, switch events, gust directions, scenario seeds, per-plan hashes, and
manifest hash are stored before evaluation. Held-out initial perturbations are
X/Y in `[-0.15, 0.15]` m, Z in `[0.45, 0.65]` m, yaw within `pi/12`, linear
velocity within 0.08 m/s per world axis, and angular velocity within
0.15 rad/s per body axis.

The main protocol contains 16 plans per scenario. The separate `lif_proof`
protocol contains exactly five plans per scenario at the same seed. Its launch
score is `event_episode_count/5`: Reach counts a target dwell, Switch counts
any one of its four target dwells including the initial target, and Gust counts
a post-gust recovery. The stricter all-four-target Switch episode success
remains recorded separately. Reach, Switch, and Gust must each score at least
`1/5` before main execution.

## Scenario timing

`WaypointReach` holds one target for the complete episode and is the isolated
proof task.  The active comparison training task is Mixed; per-environment
scenario code is `(environment_id + episode_index + seed) mod 3`, exactly
balanced over each three episodes.

`WaypointSwitch` has four predeclared targets. It installs the next target at
3, 6, and 9 s (control steps 150, 300, and 450). A switch resets the continuous
success dwell, success latch, previous-distance baseline, and target-local
timer. Progress on the switch transition is forced to zero, so changing the
target cannot create progress reward or success.

`GustRecovery` keeps one waypoint and calls
`Articulation.instantaneous_wrench_composer.set_forces_and_torques` with
`is_global=True`, applying horizontal world-frame force at the Crazyflie body
center of mass with no point offset. Gusts start at steps 150, 300, and 450 and
are active on the five half-open control intervals `[start, start + 5)`, or
0.10 s (ten physics steps). For measured mass `m`:

```text
desired impulse = m * 0.75 m/s
constant force = desired impulse / 0.10 s
```

At the audited 0.0282000024 kg mass this is approximately 0.02115 N s and
0.2115 N. Evidence keeps three quantities separate: the mass-derived expected
impulse, the exact `sum(force * physics_dt)` submitted to the wrench composer,
and the simulator response measured as horizontal `mass * delta-velocity`.
The raw momentum change includes ordinary vehicle dynamics, so Gate C isolates
the gust contribution with two live, identically initialized five-decision
rollouts using identical hover actions: one has the gust disabled and one has
it enabled. Their horizontal momentum-change difference must match the
expected vector within `max(0.0005 N s, 10% * expected_norm)`. The tolerance is
declared here before the acceptance run and is not fitted to results.
Directions are normalized in the horizontal world plane and come from the
episode manifest.

## Success, recovery, and endings

A target succeeds only after both conditions hold continuously for 25 control
intervals (0.50 s):

- 3-D target distance is at most 0.20 m;
- vehicle speed is at most 0.25 m/s.

The success bonus is latched once per target. A switch starts a fresh dwell.
For the switch scenario, an episode-level success requires success on the
initial target and all three switched targets; per-switch outcomes are also
retained separately.

A gust recovery attempt begins after the five gust intervals. Recovery means
completing the same 25-step tube dwell within the following 100 control
intervals (2.0 s). The last permissible completion has latency 2.0 s. Every
scheduled gust remains in the unconditional denominator. The conditional
rate includes only gusts whose vehicle was inside the distance-and-speed tube
immediately before onset. Conditional recovery never replaces the
unconditional result. Termination before recovery is a failed attempt.

Failure termination occurs for height below 0.10 m, height above 2.00 m,
absolute local X or Y beyond 2.75 m, or nonfinite simulator/task state. The
low-height rule is the explicit ground-or-low-height safety proxy; this task
does not claim a dedicated contact-sensor measurement.
Reaching a target does not terminate the episode. A nonfailed 600-step ending
is a time-limit truncation, stored separately from failure termination. The
environment snapshots pre-reset terminal observation, goal, position,
distance, speed, success and event counters, gust records, failure cause, and
work proxy. `drone_terminal_observation` is canonical;
`flyg1_terminal_observation` is only a documented compatibility alias.

Gate-C smoke acceptance includes a separate stabilized live protocol episode.
It advances without a reset through all 600 real simulator decisions while a
diagnostic harness re-pins pose/velocity before each decision; this tests event
and ending semantics, not controller flight skill. It must observe all three
switches or gusts at exact schedule indices, all five-decision gust integrals,
no early ending, and then truncation (not termination) at decision 600 with a
coherent pre-reset terminal snapshot. Separate live teleports exercise the
low-, high-, and horizontal-workspace failure causes. The nonfinite branch is
checked through the same pure classifier used by the environment, without
injecting NaN or infinity into physics. A one-environment smoke tier may pass
with reset isolation explicitly untested; full Gate-C acceptance requires the
two-environment tier to prove that resetting row zero leaves every inspected
row-one state unchanged.

## Active balanced-v3 reward and work proxy

All comparison controllers use one exact balanced interval reward.  With
`h = 2 / 1.9 - 1`, `phi(d) = 2*tanh(d/2)`, and
`q = exp(-0.5*(d/0.50)^2)`:

```text
progress = phi(previous_distance) - phi(current_distance)
proximity = +0.015*q
dwell = +0.020 when distance <= 0.20 m and speed <= 0.25 m/s
braking = -0.010*q*min((speed/0.50)^2, 4)
success = +3.0 once on a new 25-step target success
retention = -0.040*latched*clamp((distance-0.20)/0.50, 0, 1)
survival = +0.010 on a valid nonfailure interval

control_effort = -0.0025*min(
    ((action[0]-h)/0.15)^2 + sum((action[1:4]/0.020)^2), 4)
action_change = -0.0005*min(
    (delta_action[0]/0.10)^2 + sum((delta_action[1:4]/0.010)^2), 4)

boundary = -0.050*min(
    relu((0.30-world_z)/0.20)^2
    + relu((world_z-1.70)/0.20)^2
    + relu((abs(local_x)-2.25)/0.25)^2
    + relu((abs(local_y)-2.25)/0.25)^2, 4)
failure = -25.0 once on failure termination
```

Failure intervals zero proximity, dwell, success, survival, and positive
progress while retaining negative progress and all costs.  Reset/switch
baselines make progress exactly zero.  These are interval/event quantities and
are not multiplied by the timestep again.  The reward SHA-256 is
`bfbba45d2f19ddf8368da6a1318f1dba446d27e49fec9f8b4ec09abf2da611e9`;
the balanced curriculum SHA-256 is
`5aefd508cb3a489db630506c33392f4ad7cf40f39851b0194231a661731f3f0c`.
Any coefficient or distribution change requires a new version and fingerprint.

The retired survival-v2 payload hashes are preserved only in source tests and
the removal audit; its execution artifacts were deleted at the user's request.

The separately reported mechanical-work proxy is
`dt * (|F dot v| + |M dot omega|)` accumulated over control intervals, using
aggregate commanded body-frame force/linear velocity and body-frame
moment/angular velocity. It is a simulator mechanical proxy in joules, not
battery energy.

## Controllers and matching

The comparison conditions are, in fixed matrix order:

1. `frozen_lif_original` -- original MaleCNS-derived topology and frozen core;
2. `frozen_lif_degree_rewired` -- one seed-20260916 directed,
   degree-preserving destination-swap rewire shared by all training seeds;
3. `gru_matched` -- recurrent baseline;
4. `mlp_normal` -- genuine single-step feed-forward baseline.

The rewire preserves source indices, directed in/out degree, weight multiset,
per-source weight/sign multisets, neuron and input/output population counts,
and rejects duplicates or self-loops. Both LIF variants train only their
encoder, decoder, common critic, and action-distribution parameters; the
fixed-core checksum is verified before and after training. Frozen parameters
remain on the differentiable path to the encoder. LIF and GRU keep one state
row per environment and reset only completed rows.

With the checked-in 256-neuron graph and default widths, exact counts are:

| Controller | Trainable actor | Common critic | Total trainable | Frozen synaptic weights | Dynamic state/env |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original frozen LIF | 4,776 | 18,305 | 23,081 | 5,103 | 1,024 |
| Rewired frozen LIF | 4,776 | 18,305 | 23,081 | 5,103 | 1,024 |
| Matched GRU | 4,793 | 18,305 | 23,098 | 0 | 33 |
| Normal MLP | 4,827 | 18,305 | 23,132 | 0 | 0 |

All actor counts are within the predeclared 10% tolerance of the LIF adapter
budget. Every condition uses the same PPO family, critic, transformed Gaussian
action distribution, observation normalizer, task stream, and evaluation
plans.

All four actors also use the same fixed, non-trainable, goal-independent
Crazyflie stabilizer. Let `h = 2 / 1.9 - 1 = 0.05263158...`, the normalized
collective action that maps to vehicle weight. Using physical values
reconstructed from the fixed normalized observation, the prior action is
governed by the fingerprinted contract
`crazyflie_shared_stabilization_bounded_residual_v2`:

```text
collective = h - 0.18*v_z + 0.20*(1 + gravity_z)
roll       = 0.08*gravity_y - 0.02*omega_x + 0.015*v_y
pitch      = -0.08*gravity_x - 0.02*omega_y - 0.015*v_x
yaw        = -0.01*omega_z
```

The prior reads only body velocity, body angular velocity, and projected
gravity (observation indices 0--8). It never reads the active-goal displacement
at indices 9--11, so changing only the waypoint cannot change the fixed
command. The prior is clamped to `[-0.9999, 0.9999]` only to keep its inverse
tanh finite. It supplies basic vertical, tilt, rate, and horizontal-velocity
damping, but cannot perform waypoint navigation by itself.

Each trainable actor sees all 12 normalized values and produces four residual
logits. The shared policy composition is exactly:

```text
latent_mean = atanh(prior_action)
              + [0.35, 0.08, 0.08, 0.05] * tanh(residual_logits)
stochastic_action = tanh(Normal(latent_mean, exp(log_std)))
deterministic_action = tanh(latent_mean)
```

The fixed latent residual scales bound how far learning can move each action
coordinate away from the stabilizer. Every output head starts with zero bias
and deterministic nonzero weights of magnitude `1e-3 / sqrt(fan_in)`. Its four
rows use distinct balanced rows 1, 2, 3, and 4 of a repeated order-eight Walsh
matrix, and the builder requires exact row rank four. The trainable initial
latent standard deviation is `[0.03, 0.003, 0.003, 0.003]`: the collective
coordinate retains the larger exploration scale, while the three sensitive
moment coordinates use one tenth of it. The same values and residual
composition apply to all four controllers.

This fixed stabilizer/residual structure was frozen after a bounded pre-main
LIF attempt showed immediate-survival failure and remains unchanged under
balanced-v3. Earlier
checkpoints have incompatible source and controller-contract fingerprints and
cannot enter the primary comparison. The scientific estimand is therefore the
difference among residual learners on top of one identical fixed flight
stabilizer, not end-to-end unassisted flight. For the two LIF conditions, the
original or rewired LIF core remains frozen and the trainable adapters produce
the residual logits. The stabilizer must not be reported as LIF output,
learned flight skill, or evidence of biological flight control.

## Evaluation records and censoring

The main protocol has 16 plans for each of the three scenarios and always uses
deterministic actions. An episode failure remains a row and denominator member;
missing, duplicate, or noncontiguous episode rows invalidate the whole
checkpoint/scenario cell. Descriptive failure latency uses fixed-horizon
censoring: 12 s for episode success, 3 s for each switched-target segment, and
2 s for gust recovery. Success-only latency is reported alongside the censored
statistic.

Isaac physics, observations, normalization, terminal snapshots, traces, and
metric accumulation remain on CUDA during evaluation. All four controllers
use the same fixed-batch `torch.cuda.CUDAGraph` deterministic action-only
backend on `cuda:0`. It captures after checkpoint loading and runs exactly the
actor mean/state computation followed by `tanh`; unused critic and stochastic
log-probability branches are not evaluated. A mandatory three-step probe
requires the captured action and every recurrent-state tensor to be bit-for-bit
equal to eager CUDA before any episode can run. Capture or parity failure is
fatal and has no silent CPU fallback. Observation, recurrent state, reset mask,
and action stay on `cuda:0`, so policy inference performs zero GPU-to-CPU and
zero CPU-to-GPU tensor transfers per decision. Recurrent rows reset immediately
before their next policy action. The dry-run config exposes this contract,
every evaluation artifact records it, and queue validation requires an exact
match.

`--all_scenarios` remains a GPU-free parent process. It launches one fresh,
sequential Isaac child per scenario and stores each child artifact in a unique,
attempt-scoped directory. Prior attempt evidence is never unlinked or
overwritten; the merged artifact records the exact validated byte hash of each
part. Fingerprint payloads, checkpoint hashes, CUDA device indices, graph
contracts, episode plans, and recomputed summaries must agree before merge.

Episode and aggregate output includes success, final and integrated 3-D error,
speed while geometrically inside the target region, crash/out-of-bounds/
invalid-state rates, command effort, command smoothness, and the explicitly
labeled aggregate-wrench work proxy. Switch reports retain each switch's
success and latency. Gust reports retain unconditional and stable-before-gust
conditional recovery, recovery latency, maximum displacement relative to the
pre-gust position, and the post-gust error integral.

Integrated-error denominators also remain fixed after early termination. The
evaluator uses last-observation-carried-forward censoring: it carries the last
observed goal error through the unobserved remainder of the 600-step episode,
and through the unobserved remainder of a gust's 100-step post-gust window.

## Training, checkpointing, and fingerprints

The main config declares four environments, 100-step rollouts, two PPO epochs,
a full four-environment recurrent vector microbatch, no workers, FP32,
learning rate `3e-5`, gamma 0.99, GAE lambda 0.95, clip ratio 0.2, value
coefficient 0.5, entropy coefficient 0.002, gradient norm 1.0, KL guard 0.05,
and checkpoints every 100 completed updates. The bounded integration config
retains its deliberately tiny 25-step rollout, `1e-4` learning rate, one
environment, one PPO epoch, and microbatch 1. Standalone training defaults to
the main controller-tuning values (100 steps and `3e-5`), so the exact
section-14 MLP smoke command exercises those shared defaults. The implementation
requires the microbatch size to equal the environment count because it updates
one full vector sequence batch. Four environments are usable only after every
controller passes the one-, two-, and four-environment memory gates.

Checkpoint writes are temporary-file-plus-rename atomic operations. A
checkpoint stores policy, critic/adapters/distribution, optimizer, scheduler,
normalizers, completed-update and exact-interaction counters, recurrent state,
Python/NumPy/PyTorch/environment/schedule RNG state, resolved config and
command, manifests, fingerprints, history, and frozen-core checksum. Resume is
at a completed rollout boundary. Isaac state is restarted together with
controller state, recorded as `resume_reset=true`, while counted interactions
remain unchanged and strictly monotone.

The checkpoint-selection rule is fixed and non-optimizing: evaluation always
uses `latest.pt`, written at the exact declared interaction budget. Best-return,
best-success, and other post-hoc checkpoint selection are not permitted.

Per-update metric history is written once into immutable, atomic JSONL segments
at checkpoint boundaries. Checkpoints contain a compact ordered segment list,
per-segment hashes/counter bounds, and a checksum of the complete logical
history; resume resolves and verifies every segment before appending. This
avoids quadratically duplicating 12,500 cumulative metric rows across 125
periodic main checkpoints (plus the update-0 restart checkpoint) while
preserving old checkpoints and exact
no-duplicate resume semantics. The final manifest points to the same history
reference rather than embedding a second full copy.

A fresh run writes an update-0 restart checkpoint immediately after the
environment, controller, optimizer, scheduler, and normalizer initialize. The
job seed is applied once before application startup, then reapplied after the
application and environment exist but immediately before controller
construction; Isaac startup can therefore not shift initial network weights.
Because different controller architectures themselves consume different
numbers of random values while their parameters are built, the entrypoint
reapplies the same seed a third time to Python, NumPy, PyTorch CPU, and every
available CUDA generator. This `post_construction_reseed_v1` boundary occurs
before the update-0 checkpoint, the first environment reset, and the first
exploration sample, so architecture construction cannot shift the
task/exploration random stream. A resumed process still uses the
pre-controller seed for its throw-away construction, then restores the saved
checkpoint RNG and never applies the post-construction fresh-start seed. The
resolved checkpoint configuration, checkpoint metadata, and training manifest
record this exact two-phase contract and whether a fresh reseed or
resume-preservation path was taken.
History validation requires row `i` to be exactly update `i+1` at
`(i+1) * num_envs * horizon` interactions, so skipped or invented samples are
rejected. If an attempt stops before any checkpoint can be committed, a later
fresh retry atomically moves the complete incomplete directory into a sibling
`failed_attempt_archives/` directory and starts clean; no failure evidence is
deleted or overwritten.

The artifact fingerprint set covers all drone scripts/task/controller/
training/evaluation source, every reused connectome/policy/training module,
resolved configuration, evaluation manifest, connectome and exact rewire/core
identity, installed package/runtime/GPU/OS identity, Isaac Lab commit,
upstream task and asset sources, the directly imported installed
`isaaclab/envs/direct_rl_env.py` base environment, the directly imported
`isaaclab/utils/math.py` quaternion helpers, USD identifier, and
observation/action contracts. Hashing those installed files directly makes a
local dirty checkout fail closed even when its Git commit string is unchanged.
Train, resume, and evaluation reject incompatible primary artifacts by default.
The asset audit, smoke, training, evaluation, queue, monitor, and summarizer
CLIs print their applicable resolved configuration, fingerprint scope/value,
and output paths; runtime failure summaries include an explicit reason.

## Memory and queue safety

Memory validation starts with one headless environment, then two, then four,
and includes a real forward/backward update for every controller. The maximum
sampled device-wide `nvidia-smi` usage must remain strictly below 6.8 GiB
(6963.2 MiB);
the gate fails closed if any CUDA sample lacks that telemetry. This is a
sampled maximum, not a continuous hardware peak. System RAM must remain
strictly below 90% under `crazyflie_memory_acceptance_v2`; a sample at 90%
fails. A strictly increasing final four-sample steady-state/optimizer RSS
window remains detected when its net rise exceeds 1.0 MiB, but it is a visible
warning and does not fail acceptance by itself. Sustained paging
uses the same four-sample rule on cumulative swap-out and fails above a 1.0 MiB
net rise. Reports store available RAM, swap counters, PyTorch allocated/
reserved peaks, process RSS, and device-wide samples. Training uses one Isaac
process and one queue job at a time; cameras, video, ROS, notebooks, and
preprocessing are outside the training path.

The 1,000-step smoke labels samples as steady state only after the first full
episode boundary: steps 700/800/900/1000 for the 600-step project tasks and
625/750/875/1000 for the 500-step native task. This excludes the known
one-time Isaac/PhysX/task-logger allocation at the first automatic reset while
still applying the four-sample, 1.0 MiB RSS-warning rule. Short
diagnostics fall back to quartiles and explicitly report that a full
post-boundary steady-state window was unavailable.

Evaluation applies the same fail-closed limits after the environment loads,
after CUDA-Graph capture, after every completed evaluation batch, and at the
scenario end. Each scenario artifact stores every raw snapshot and its
recomputed assessment; missing `nvidia-smi` telemetry or a failed RAM/VRAM/
swap hard gate invalidates the evaluation. RSS-growth warnings are preserved
in each scenario and the merged artifact. Memory values may differ between
fresh scenario processes and therefore remain scenario-specific in a merged
all-scenarios artifact.

The integration config is explicitly non-scientific: four controllers, seed
0, 50 interactions per job, and two episodes per scenario (24 evaluation
episodes total). `execute_drone_integration.sh` uses only that fixed config,
revalidates all immutable queue content before an existing-queue resume, and
summarizes its artifacts. On resume, the runner atomically archives a
persistent pause request and records the resume event; the launcher does not
silently unlink it.

The main config is exactly four controllers by seeds 0--4: 20 jobs at
5,000,000 interactions each, 60 checkpoint/scenario bundles, and 960 held-out
episodes. A dry run must exist for human review before execution. The main
launcher requires a second explicit authorization token and forwards the
runner's own `--authorize_main` gate; it cannot create an unreviewed queue.
An authorized resume uses the same recorded pause-request archival protocol.
The user has authorized main execution in principle, but balanced-v3 adds a
stricter prerequisite: the frozen-LIF proof checkpoint must show at least one
held-out success-event-scored episode in each scenario and pass every hard
memory gate first. Strict full-episode success remains a separate reported
metric.

Each queue embeds and the dry run prints the complete resolved config,
single-process limit, GPU/RAM/RSS/swap limits, every command, output path, seed,
budget, and expected reproduction fingerprint.

```bash
ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

# Required review artifact; does not launch training.
"$ISAAC_PYTHON" scripts/drone_run_matrix.py \
  --config configs/experiments/crazyflie_balanced_v3_main.json \
  --dry_run --output runs/crazyflie_balanced_v3_main_v1.json

# Only after a later explicit user authorization:
bash scripts/execute_drone_matrix.sh --authorize_main
```

`drone_play.py` runs a bounded diagnostic checkpoint rollout and labels it as
non-held-out. `drone_record.py` replays one exact immutable evaluation plan,
uses deterministic actions, records no more than 600 decisions, and writes
tensor traces atomically in bounded chunks. Full per-neuron LIF or GRU state is
opt-in with `--include_neural_state`; default recordings retain bounded state
norm/activity summaries. Neither tool changes a checkpoint.

```bash
"$ISAAC_PYTHON" scripts/drone_play.py \
  --checkpoint RUN/checkpoints/latest.pt \
  --task FlyCrazyflie-WaypointReach-v0 --steps 600 --headless \
  --output RUN/playback.json

"$ISAAC_PYTHON" scripts/drone_record.py \
  --checkpoint RUN/checkpoints/latest.pt \
  --task FlyCrazyflie-GustRecovery-v0 --protocol main --episode_id 0 \
  --chunk_steps 100 --output_dir RUN/trace-gust-episode-0 --headless
```

`drone_visualize_lif.py` is the bounded realtime viewer for an original or
degree-rewired frozen-LIF checkpoint. It always constructs exactly one
environment, uses deterministic inference only, paces the 50 Hz controller at
realtime by default, and stops at the first episode ending, viewer close, or a
hard maximum of 600 requested control steps. The Isaac viewer is enabled by
default; `--headless` leaves the same terminal LIF sidecar available without a
window. The terminal LIF sidecar shows flight telemetry, action, a rolling
relative neural-activity bar, final-neural-substep spike count/rate, membrane
and synapse norms, and the latest process/system/GPU memory sample.

The visualization deliberately labels activity as one sample of the
post-final-neural-substep state per 0.02 s control decision; it does not claim
to count spikes in unobserved internal neural substeps. Its required atomic
JSON report stores bounded telemetry frames, the complete activity summary,
the standard strict `<6.8 GiB` device-wide GPU and `<90%` RAM hard limits,
the warning-only RSS assessment, raw
memory samples, the stopping reason, and before/after checkpoint and frozen
core hashes. It refuses non-LIF checkpoints, output overwrite, stochastic
actions, and any checkpoint mutation. A drone crash is retained as a flight
outcome and is not relabeled as a successful flight merely because the
diagnostic command itself completed.

```bash
# Viewer plus terminal sidecar (default).
"$ISAAC_PYTHON" scripts/drone_visualize_lif.py \
  --checkpoint RUN/checkpoints/latest.pt \
  --task FlyCrazyflie-WaypointReach-v0 --steps 600 \
  --output RUN/lif-visualization.json

# Optional terminal-sidecar-only mode.
"$ISAAC_PYTHON" scripts/drone_visualize_lif.py \
  --checkpoint RUN/checkpoints/latest.pt \
  --task FlyCrazyflie-WaypointReach-v0 --steps 600 --headless \
  --output RUN/lif-visualization-headless.json
```

## Reporting contract

Training manifests retain exact interactions and updates; episodic return and
each reward component; success/failure and terminal-cause counts; goal error
and speed; policy/value/entropy losses and KL; action mean, saturation, and
change; throughput; process/system/PyTorch/device memory; invalid-state counts;
and checkpoint/fingerprint identity. LIF runs additionally retain bounded
activity, dead/saturated fractions, recurrent-state norms, encoder/decoder
gradient norms, and before/after frozen-core checksums.

LIF training activity uses schema version 1 and is rollout-wide, not an
instantaneous final-state snapshot. It is sampled only on the existing bounded
measurement cadence: update 1, every configured checkpoint update, and the
final update. Each LIF history row contains `lif_activity_sampled`; unsampled
rows retain `lif_activity: null`, so absence cannot be mistaken for zero
activity. With the main checkpoint cadence this means updates 1, 100, 200, ...,
and the final update; the two-update integration configuration samples both
updates. No activity hook or activity host synchronization runs on unsampled
updates.

On a sampled update, a device-resident accumulator retains one 256-element
per-neuron spike-count vector and eight scalar values, never a state trace. It
samples the post-final-neural-substep state once per 50 Hz control decision for
exactly `horizon * num_envs` samples; the separate value bootstrap call is
observed and excluded. The sampled spike fraction is the number of observed
spikes divided by neurons and samples, and its explicitly labeled sampled rate
is that fraction divided by the 0.02 s control period. A dead neuron has zero
spikes across the entire sampled rollout. A saturated neuron spikes in every
environment at every counted control decision. Membrane and synapse L2 mean,
RMS, and maximum first take the norm across neurons, then summarize across all
environment/control samples. This sampling does not claim to count spikes at
unobserved internal neural substeps. The summary performs one bounded
device-to-host transfer after collection and fails closed on an unexpected
core-call count, shape/device mismatch, nonfinite state, or nonbinary spike
value.

The report generator emits all episode rows, checkpoint/scenario summaries, a
20-row seed table, a controller comparison containing per-seed values plus
mean, median, sample standard deviation and paired-seed bootstrap 95% intervals,
and a complete execution-status table. Pending, failed, paused, cancelled, or
missing cells remain visible. A partial main report states its exact completed
fraction; seed 0 remains `1/5 seeds`, never a final comparison. Summarization
recomputes the current queue fingerprint from source/config/runtime and rejects
stale artifacts.

## Unitree G1 preservation evidence

The original pre-change manifest is `g1_frozen_sha256_before.json`. Because
that first capture accidentally omitted the explicitly required unchanged
`scripts/memory_watch.py`, a separately labeled before-manifest supplement
records its hash plus pre-existing filesystem birth/modify times; the original
manifest is not rewritten. `g1_frozen_sha256_after.json` covers the complete
37-file frozen set and must match the union of those two before records exactly.
No drone command may edit, resume, rename, or delete a G1 source, run,
checkpoint, queue, log, lock, summary, or pause artifact.

## Scientific limitations

The controller's source graph is leg/VNC-derived and is not a known biological
flight circuit. Results may describe only simulated control performance and
activity. They cannot establish insect flight mechanisms, dopamine
concentration, causal biological function, physical onboard observability,
real-hardware safety, individual-motor performance, or battery consumption.
