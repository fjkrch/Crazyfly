# Prompt for the implementation agent

> Historical request provenance. The 2026-09-17 user-authorized memory and
> LIF-proof scoring changes are normative in `plan.md` and
> `docs/crazyflie_memory_acceptance_v2.md`; the original prompt below is
> intentionally preserved rather than rewritten.

Copy everything inside the block below into a new Codex agent session whose
working directory is `/home/chayanin/Desktop/flyg1`.

```text
Implement the Crazyflie migration described in
/home/chayanin/Desktop/flyg1/plan.md.

Work autonomously through implementation, focused tests, simulator smoke
tests, checkpoint/resume validation, the bounded four-controller integration
matrix, and the full main-matrix dry run. Do not launch the full 20-job,
100-million-interaction main matrix. Stop after its dry-run artifacts are
reviewable; a separate explicit user instruction is required to launch it.

Read plan.md completely before editing. Also read applicable repository
instructions and the installed Crazyflie sources named in the plan. Treat the
installed Isaac Lab checkout and task source as authoritative. Use this
interpreter for Isaac work:

  /home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

The old Unitree G1 work is paused and must remain resumable. Before changing
code, create docs/g1_frozen_sha256_before.json covering the exact frozen set
listed in plan.md. Implement the drone path additively with new
tasks/crazyflie modules, drone_* scripts, Crazyflie configs, focused tests, and
docs. Do not edit the frozen G1 files, old configs, checkpoints, queues, logs,
or runs. At the end create docs/g1_frozen_sha256_after.json and prove that all
frozen hashes match. The archived G1 plan is
docs/g1_plan_paused_20260913.md.

Preserve the scientific comparison. The Crazyflie is the embodiment; it does
not replace the normal MLP controller. Implement all four controller
conditions:

  1. original frozen LIF,
  2. degree-preserving rewired frozen LIF,
  3. matched GRU,
  4. matched normal MLP.

The main configuration must contain five seeds, 0 through 4, for all four
conditions, exactly 20 jobs, with 5,000,000 interactions per job. The full
main matrix must remain unstarted. Implement a smaller seed-0 integration
configuration for validation and keep its artifacts visibly separate from
research results.

Build these tasks on the installed `Isaac-Quadcopter-Direct-v0` contract:

  FlyCrazyflie-WaypointReach-v0
  FlyCrazyflie-WaypointSwitch-v0
  FlyCrazyflie-GustRecovery-v0

Keep the native 12 observations, four aggregate-wrench actions, 0.01 s
physics timestep, decimation 2, and Crazyflie asset. Use a custom 12 s horizon.
Train only on WaypointReach. Evaluate all trained checkpoints on fixed held-out
Reach, Switch, and Gust scenarios. Switches and gusts occur at 3, 6, and 9 s.
A target success requires distance <=0.20 m and speed <=0.25 m/s continuously
for 0.50 s. Gusts last 0.10 s and use mass-normalized desired delta-v
0.75 m/s. Recovery requires returning to the success tube for 0.50 s within
2.0 s after the gust. Freeze 16 held-out plans per scenario at evaluation seed
101 and use the identical plans for every checkpoint.

Implement periodic atomic checkpoints and rollout-boundary pause/resume.
Preserve policy, critic, adapters, optimizer, normalization, counters,
interaction count, RNG states, resolved config, histories, and fingerprints.
If simulator state is not exactly restorable, reset environment and recurrent
state together and label `resume_reset=true`; do not claim bit-exact resume.
Test that resume neither repeats nor skips counted interactions.

Fingerprint all new and reused code, resolved configs, connectome and rewire
manifests, evaluation manifest, installed quadcopter task source, Crazyflie
asset configuration/identifier, package versions, Isaac Lab commit, and
hardware/runtime metadata. Reject incompatible train/evaluate/resume inputs by
default.

Apply the memory gates from plan.md. Start with one environment, headless, no
cameras, `num_workers=0`, FP32 LIF, microbatch 1, and one Isaac process. Measure
native smoke, custom smoke, and a real forward/backward update. Then try two
and four environments sequentially. Use the largest count that passes every
controller. Keep sampled device-wide GPU use strictly below 6.8 GiB
(6963.2 MiB; the boundary fails) on the 8 GiB GPU and system RAM below
80-85% of 24 GiB; stop on OOM, nonfinite values, sustained paging, or a
memory leak. Never run multiple Isaac jobs concurrently.

Implement and validate in the plan's gate order. Use meaningful pure unit
tests for schedules, dwell logic, switches, gust impulses, metrics, complete
matrix enumeration, checkpoint identity, and resume accounting. Inspect the
real direct-environment API instead of guessing from the manager-based G1
path. The built-in task randomizes episode_length_buf on full reset, so make
fixed evaluation start deterministically at step zero and test it.

Required future command interfaces are listed in section 14 of plan.md.
Implement those interfaces, make every script support --help, and update the
docs if an argument must change. After implementation, execute this validation
sequence from the project root:

  cd /home/chayanin/Desktop/flyg1
  ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

  "$ISAAC_PYTHON" -m pytest \
    tests/unit/test_crazyflie_logic.py \
    tests/unit/test_crazyflie_schedules.py \
    tests/unit/test_crazyflie_metrics.py \
    tests/unit/test_crazyflie_checkpoint.py \
    tests/unit/test_crazyflie_matrix.py -q

  "$ISAAC_PYTHON" -m pytest tests/unit -q

  "$ISAAC_PYTHON" scripts/drone_inspect_asset.py --headless \
    --json docs/crazyflie_asset_audit.json \
    --markdown docs/crazyflie_asset_audit.md

  "$ISAAC_PYTHON" scripts/drone_smoke_env.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --num_envs 1 --steps 1000 \
    --hover_centered_random_actions --random_action_scale 0.1 \
    --headless \
    --output_report runs/crazyflie-native-smoke-1x1000.json

Run each custom task with one environment and 1,000 steps as specified in
plan.md. If the memory report passes, repeat with two environments and then
four. Next run the 20,000-interaction MLP checkpoint/evaluation smoke command
from section 14. Then run:

  "$ISAAC_PYTHON" scripts/drone_run_matrix.py \
    --config configs/experiments/crazyflie_integration.json --dry_run

  bash scripts/execute_drone_integration.sh

  "$ISAAC_PYTHON" scripts/drone_run_matrix.py \
    --config configs/experiments/crazyflie_main.json --dry_run

Do not run `scripts/execute_drone_matrix.sh`. That command is reserved for a
later user-authorized full run.

Before finishing, verify that no simulator/training process was left behind,
that no old G1 job resumed, and that the G1 before/after hashes match. Report:

- files added and any deviations from plan.md;
- exact test and smoke commands with pass/fail evidence;
- measured RAM/VRAM/throughput at each environment count;
- parameter counts for all four controllers;
- integration matrix status for every cell;
- checkpoint/resume evidence;
- the main dry-run count, expected 20 jobs and 960 evaluation episodes;
- blockers and scientific limitations;
- the exact command that could later launch the full matrix, clearly marked
  as not executed.

Do not fabricate successful runs, performance, files, or metrics. A failing
gate is a result: preserve its evidence and stop dependent scaling while
continuing any independent implementation or CPU-testable work.
```

## Commands for the later full run

The implementation agent is instructed not to launch the main matrix. After
that agent finishes, review its tests, integration report, memory report,
fingerprints, and 20-cell dry run. If everything passes, a later explicit
instruction may use:

```bash
cd /home/chayanin/Desktop/flyg1
ISAAC_PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python

bash scripts/execute_drone_matrix.sh

"$ISAAC_PYTHON" scripts/drone_monitor_matrix.py \
  --queue runs/crazyflie_main_v1.json
```

Those commands are specifications until the implementation agent creates and
validates the scripts. They have not been run while preparing this prompt.
