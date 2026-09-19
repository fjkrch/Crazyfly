# Crazyflie implementation and acceptance report

> Historical pre-balanced-v3/pre-memory-v2 report. Current acceptance is
> defined by `plan.md` and `docs/crazyflie_memory_acceptance_v2.md`. In
> particular, system RAM is now strictly `<90%`, monotonic RSS is warning-only,
> and the five-episode-per-task LIF event-score proof is not retroactively
> applied to the historical results below.

Generated 2026-09-16 after the additive wing extension. The required Isaac
interpreter was `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python`.
The GPU gate used by every current artifact is strict `< 6963.2 MiB` device
usage, with `< 85%` system RAM, no sustained paging, and one Isaac process at a
time.

## Outcome

- Baseline Crazyflie stack: implemented and accepted through the bounded
  four-controller integration matrix.
- Baseline main queue: dry run only, 4 controllers x 5 seeds = 20 jobs,
  5,000,000 interactions/job, 60 scenario bundles, 960 episodes.
- Wing extension: deterministic 256-neuron wing circuit, wing-only controller,
  and independent-core leg-plus-wing controller implemented.
- Wing extension queue: dry run only, 3 conditions x 5 seeds = 15 jobs,
  5,000,000 interactions/job, 45 scenario bundles, 720 episodes.
- Neither full queue was launched. `scripts/execute_drone_matrix.sh` was not run.
- The bounded 200-interaction leg/wing pilots passed execution, checkpoint,
  resume, checksum, and memory gates, but observed **zero task successes**.
  This is a pipeline pass and is not a claim of learned flight.

## Commands and results

`$PY` below means the required Isaac interpreter.

| Command | Result |
|---|---|
| `$PY -m pytest tests/unit/test_crazyflie_logic.py tests/unit/test_crazyflie_schedules.py tests/unit/test_crazyflie_metrics.py tests/unit/test_crazyflie_checkpoint.py tests/unit/test_crazyflie_matrix.py -q` | PASS (included in the complete suite) |
| `/usr/bin/time -v $PY -m pytest -q tests/unit/test_crazyflie_checkpoint.py tests/unit/test_prepare_malecns_wing.py tests/unit/test_crazyflie_wing_matrix.py` | PASS, 43 tests, 16.68 s, max RSS 763,332 KiB, 0 major faults, 0 swaps |
| `/usr/bin/time -v $PY -m pytest tests/unit -q` | PASS, 334 tests, 22.85 s, max RSS 1,642,800 KiB, 0 major faults, 0 swaps |
| `$PY scripts/drone_inspect_asset.py --headless --json docs/crazyflie_asset_audit.json --markdown docs/crazyflie_asset_audit.md` | PASS; Isaac Lab 0.54.4, Assets 0.2.4, Tasks 0.11.16, Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128 |
| `$PY scripts/drone_smoke_env.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1 --steps 1000 --hover_centered_random_actions --random_action_scale 0.1 --headless --output_report runs/crazyflie-native-smoke-1x1000-v2.json` | PASS |
| Section-14 custom smoke loop for Reach, Switch, and Gust at `--num_envs 1`, then 2, then 4, each `--steps 1000 --headless` | PASS, all 9 final artifacts |
| Same smoke entrypoint for each baseline controller at 1/2/4 envs with `--steps 1000 --ppo_update --curriculum_probe_interactions 1000000 --headless` | PASS, all 12 PPO-update artifacts |
| `$PY scripts/drone_train.py --task FlyCrazyflie-WaypointReach-v0 --policy mlp --seed 0 --num_envs 1 --total_interactions 20000 --headless --run_dir runs/crazyflie-mlp-smoke` | PASS, 20,000/20,000 interactions, 200 updates, checkpoint SHA-256 `c93caaf9c695b373cc41c8147ef908823feb6a4875aef66520290d90998c448d` |
| `$PY scripts/drone_evaluate.py --checkpoint runs/crazyflie-mlp-smoke/checkpoints/latest.pt --protocol integration --all_scenarios --headless --output runs/crazyflie-mlp-smoke/evaluation.json` | PASS pipeline, 6 episodes |
| `$PY scripts/drone_run_matrix.py --config configs/experiments/crazyflie_integration.json --dry_run` | PASS, 4 jobs / 12 bundles / 24 episodes |
| `bash scripts/execute_drone_integration.sh` | PASS, all 4 jobs and all 12 evaluations completed |
| `$PY scripts/drone_run_matrix.py --config configs/experiments/crazyflie_main.json --dry_run --output runs/crazyflie_main_v2_verified.json` | PASS, 20 jobs / 60 bundles / 960 episodes; no training launched |
| Same baseline command with `--resume` | PASS; the existing 20-job queue exactly reproduces current commands and fingerprints |
| `$PY scripts/prepare_malecns_wing.py ...` | PASS; final graph has 256 neurons (32 input, 200 intrinsic, 24 motor) and 8,864 signed edges |
| `$PY scripts/calibrate_wing_lif.py ...` | PASS; selected threshold 0.04 from the predeclared sweep |
| Wing memory loop: `$PY scripts/drone_smoke_env.py --task FlyCrazyflie-WaypointReach-v0 --policy {wing_lif,leg_wing_lif} --num_envs {1,2,4} --steps 1000 --ppo_update --curriculum_probe_interactions 1000000 --headless ...` | PASS, all 6 jobs |
| Wing pilot: `$PY scripts/drone_train.py --task FlyCrazyflie-WaypointReach-v0 --policy wing_lif --seed 0 --num_envs 1 --total_interactions 200 --horizon 100 --checkpoint_every_updates 1 --pause_after_updates 1 --headless --run_dir runs/crazyflie-wing-pilot-wing_lif-resume-v1` | PASS: clean pause at update 1 |
| Same wing pilot command with `--resume` and without `--pause_after_updates` | PASS: fresh-process resume to update 2; `resume_count=1` |
| Combined pilot with the same settings and `--policy leg_wing_lif`, pause then `--resume` | PASS: clean pause and fresh-process resume; `resume_count=1` |
| Matched leg-only pilot with the same settings and `--policy frozen_lif_original` | PASS: 200 interactions / 2 updates |
| `$PY scripts/drone_evaluate.py --checkpoint <each exact pilot checkpoint> --protocol integration --all_scenarios --headless --output <run>/evaluation.json` for all three circuit conditions | PASS pipeline, 18 total episodes; zero successes and zero crashes |
| `$PY scripts/summarize_wing_comparison.py --runs_root runs --output runs/crazyflie-wing-comparison-v1.json --markdown runs/crazyflie-wing-comparison-v1.md` | PASS validation; explicitly reports no task success at 200 interactions |
| `$PY scripts/drone_wing_run_matrix.py --config configs/experiments/crazyflie_wing_main.json --dry_run --output runs/crazyflie_wing_main_v2_verified.json` | PASS, 15 jobs / 45 bundles / 720 episodes; execution disabled |
| Same extension command with `--verify` in place of `--dry_run` | PASS; all 15 commands and fingerprints exactly reproduce current source/config |
| Read-only SHA verifier over the union of both before manifests | PASS, 37/37 frozen G1 files match |
| `ps ...` plus `nvidia-smi --query-compute-apps=...` | PASS, no Crazyflie/Isaac training or evaluation process left running; no compute process reported |

One archived acceptance attempt failed before the final smoke rerun:
`runs/crazyflie_gate_failures_20260916T0637Z/FlyCrazyflie-WaypointReach-v0-1x1000.json`
reported 13.2 MiB monotonic RSS growth. It was preserved, not overwritten. The
final one-, two-, and four-environment artifacts all passed without sustained
paging or monotonic growth. Some Isaac runs emitted upstream Fabric/PhysX
warnings; no final gate treated a warning as a successful result.

The first queue-verification attempt also failed closed after the planner
source changed: `crazyflie_wing_main_v1.json` and
`crazyflie_main_v1_wing_extension_verified.json` were reported stale and were
preserved. Fresh `v2_verified` queues were then generated and both passed an
immediate independent rebuild/verification. No stale queue was relabelled as
successful.

## Memory results

Native/custom environment final smokes:

| Task | Envs | GPU MiB | RSS MiB | RAM % | Result |
|---|---:|---:|---:|---:|---|
| Native | 1 | 2490 | 3231.320 | 31.2 | PASS |
| Reach | 1 / 2 / 4 | 2490 / 2490 / 2490 | 3388.383 / 3404.082 / 3365.672 | 32.1 / 31.9 / 31.4 | PASS |
| Switch | 1 / 2 / 4 | 2490 / 2490 / 2490 | 3360.867 / 3391.629 / 3367.387 | 31.8 / 31.8 / 31.6 | PASS |
| Gust | 1 / 2 / 4 | 2492 / 2492 / 2492 | 3410.031 / 3382.086 / 3414.625 | 32.3 / 31.8 / 31.6 | PASS |

Baseline controller PPO-update smokes:

| Controller | Envs | GPU MiB | RSS MiB | Result |
|---|---:|---:|---:|---|
| Original LIF | 1 / 2 / 4 | 2528 / 2528 / 2528 | 3921.805 / 3969.441 / 3982.141 | PASS |
| Rewired LIF | 1 / 2 / 4 | 2528 / 2528 / 2528 | 3940.914 / 3957.570 / 3953.262 | PASS |
| Matched GRU | 1 / 2 / 4 | 2785 / 2781 / 2784 | 3830.977 / 3864.523 / 3867.422 | PASS |
| Normal MLP | 1 / 2 / 4 | 2782 / 2777 / 2784 | 3852.762 / 3884.652 / 3866.715 | PASS |

Wing controller PPO-update smokes:

| Controller | Envs | GPU MiB | RSS MiB | Result |
|---|---:|---:|---:|---|
| Wing-only LIF | 1 / 2 / 4 | 2546 / 2551 / 2554 | 3972.898 / 4007.941 / 4035.234 | PASS |
| Leg+wing LIF | 1 / 2 / 4 | 2571 / 2571 / 2571 | 3977.391 / 4005.234 / 4022.074 | PASS |

Short MLP: training 2545 MiB GPU / 3971.043 MiB RSS / 35.2% RAM;
evaluation maximum 2653 MiB GPU / 3569.672 MiB RSS / 35.0% RAM. The bounded
four-controller integration jobs used 2543--2545 MiB GPU and
3917.172--3995.965 MiB RSS for training; their 12 evaluations used
2651--2655 MiB GPU and 3547.422--3588.305 MiB RSS.

## Controller accounting

| Controller | Trainable actor | Total trainable | Frozen synapses | Dynamic state/env | Match result |
|---|---:|---:|---:|---:|---|
| Original frozen LIF | 4776 | 23081 | 5103 | 1024 | exact reference |
| Degree-rewired frozen LIF | 4776 | 23081 | 5103 | 1024 | exact reference |
| Matched GRU | 4793 | 23098 | 0 | 33 | +0.356%, within 10% |
| Normal MLP | 4827 | 23132 | 0 | 0 | +1.068%, within 10% |
| Wing-only LIF | 4776 | 23081 | 8864 | 1024 | exact reference; extension |
| Leg+wing LIF | 9224 | 27529 | 13967 | 2048 | +93.132%; intentionally not matched |

Wing threshold 0.04 calibration: motor mean rate 0.01703125, global mean
0.09053223, input mean 0.33507812, dead fraction 0.6640625, saturated fraction
0.0. Frozen core SHA-256 checksums are leg
`cf2c0b8f9fe4319eeebabeefaebd6125084012abd5b8f891422dea2eede4fc54`,
wing `461ff516c0d7e3feb79c8aac86dbc3f5242940564b84de6a18cac9edb27719eb`,
and combined composition
`333dfc4caccb83a6911eeb38b3a0e620f93b070c1605e473cd81e9d00d614443`.

## Checkpoint/resume and bounded flight results

| Condition | Resume | Final checkpoint SHA-256 | Train GPU/RSS MiB | Reach | Switch | Gust |
|---|---:|---|---|---:|---:|---:|
| Leg-only | 0 | `ef8aa0ce1d0bad5a3ca1a46d51cbb5566d5c9dc34ad8367889add15f40a3f02a` | 2579 / 3995.512 | 0/2 | 0/2 | 0/2 |
| Wing-only | 1 | `72e80679fed2f8f1444456e9d3743aea6648ddeef862757046ee47aae2b107fc` | 2581 / 3989.852 | 0/2 | 0/2 | 0/2 |
| Leg+wing | 1 | `5f8e6f0a552f0cef7256477020131d12df42b1dfe034ad2a8c24951a04a6d3a0` | 2581 / 4013.121 | 0/2 | 0/2 | 0/2 |

All three core checksums were identical before and after optimization. All 18
held-out episodes had zero crash, invalid-state, and out-of-bounds events, but
timed out without meeting the success criterion; switch success was 0/18
switches and gust recovery was 0/18 gusts.

## Integration jobs

All were seed 0, 50 interactions, one sequential Isaac process, and completed
three 2-episode evaluation bundles:

| Job | Train GPU/RSS MiB | Checkpoint SHA-256 | Eval result |
|---|---|---|---|
| `frozen_lif_original__seed-0` | 2545 / 3995.965 | `0093b674a73d7af0ddd949c449a83e6a73b0cbf3dba1b78df0cdef47b063539a` | 3/3 bundles complete; 0/6 success, 0 crash |
| `frozen_lif_degree_rewired__seed-0` | 2545 / 3980.957 | `f2b0e5bbd6e7f2add6c74e4cdd5729975d11dfb68ba8e0cde2db36a46709b781` | 3/3 bundles complete; 0/6 success, 0 crash |
| `gru_matched__seed-0` | 2545 / 3917.172 | `212b44e8a2452ebc066932a54b6adc046c8117efae6689ffea9fd672e53b7523` | 3/3 bundles complete; 0/6 success, 0 crash |
| `mlp_normal__seed-0` | 2543 / 3957.598 | `93e1c59054495cb5ad689805d5dc51d74a71ce203c733a32d80f7d54f6cf2dfc` | 3/3 bundles complete; 0/6 success, 0 crash |

## Changed implementation files

This directory is not a Git worktree, so this is the implementation-owned
file set rather than a `git diff`:

- `plan.md`
- `configs/experiments/crazyflie_eval_manifest_v1.json`
- `configs/experiments/crazyflie_integration.json`
- `configs/experiments/crazyflie_main.json`
- `configs/experiments/crazyflie_rewire_seed_20260916.json`
- `configs/experiments/crazyflie_wing_main.json`
- `data/connectome_wing/{manifest,neurons,edges,audit}.json`
- `data/connectome_wing_precalibration/{manifest,neurons,edges,audit}.json`
- `data/connectome_wing_calibration.json`
- `scripts/drone_bootstrap.py`, `drone_inspect_asset.py`, `drone_smoke_env.py`
- `scripts/drone_train.py`, `drone_evaluate.py`, `drone_evaluation_protocol.py`
- `scripts/drone_play.py`, `drone_record.py`, `drone_visualize_lif.py`
- `scripts/drone_run_matrix.py`, `drone_summarize_matrix.py`, `drone_monitor_matrix.py`
- `scripts/drone_wing_run_matrix.py`, `summarize_wing_comparison.py`
- `scripts/prepare_malecns_wing.py`, `calibrate_wing_lif.py`
- `scripts/run_crazyflie_chained_pilot.py`, `plot_drone_lif_checkpoint.py`,
  `plot_lif_pilot_losses.py`, `visualize_crazyflie_tasks.sh`
- `scripts/execute_drone_integration.sh`, `execute_drone_matrix.sh`
- `source/g1_fly_control/g1_fly_control/crazyflie/{__init__,checkpoint,controllers,lif_activity,memory,normalization,stabilization}.py`
- `source/g1_fly_control/g1_fly_control/tasks/crazyflie/{__init__,adapter,env,env_cfg,logic,metrics,registration}.py`
- `tests/unit/test_crazyflie_*.py`, including the new wing-matrix test
- `tests/unit/test_prepare_malecns_wing.py`
- `docs/crazyflie_asset_audit.{json,md}`, `docs/crazyflie_task_spec.md`
- `docs/g1_frozen_sha256_before.json`,
  `docs/g1_frozen_sha256_before_supplement.json`, and
  `docs/g1_frozen_sha256_after.json`
- This report and the separately labelled `runs/crazyflie*` artifacts.

## Preservation and blockers

The current hashes of all 37 frozen G1 inputs exactly match the before-manifest
union. No paused G1 run, checkpoint, queue, or pause file was resumed, renamed,
deleted, or rewritten. There is no implementation blocker for reviewing the
queues. The behavioral blocker is that the deliberately tiny 200-interaction
wing comparison did not learn task success; the full queues are required for a
research comparison, but launching them remains intentionally unauthorized.
