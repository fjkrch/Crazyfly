# G1 free-posture locomotion

An external Isaac Lab project for simulation-only Unitree G1 goal-directed locomotion without an upright, gait, or torso-contact requirement.  The main policy route is:

`G1 observations -> trainable encoder -> frozen LIF circuit -> trainable decoder -> bounded joint-position targets -> Isaac Sim PD actuators`

This repository deliberately contains no Go2, drone, ROS, or hardware-control code. The user's public MaleCNS v1.0 Feather files are copied under `data/connectome/raw/`, and a derived leg VNC circuit is available through `data/connectome/manifest.json`; see the [data provenance and download links](data/connectome/README.md). Synthetic circuits exist only under `tests/fixtures` and are rejected by the research training command.

## Installation

The inspected installation is `/home/chayanin/Downloads/IsaacLab` at commit `b4c321024792976150ca55fddb26fa34480d974e` (Isaac Lab 0.54.4).  Use its dedicated interpreter, not the system Python:

```bash
/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python -m pip install -e '.[data,monitor]'
```

Then use that same Python for every command below.  `scripts/doctor.py` reports any drift instead of guessing compatibility.

```bash
python scripts/doctor.py
python scripts/memory_watch.py --output runs/g1-memory.csv --interval 2 --count 3
python -m pytest tests/unit -q
python scripts/inspect_asset.py --headless
python scripts/smoke_env.py --task FlyG1-GoalReach-FreePosture-v0 --num_envs 16 --steps 1000 --random_actions --random_action_scale 0.1 --headless --output_report runs/smoke-16x1000-verified.json
python scripts/train.py --task FlyG1-GoalReach-FreePosture-v0 --policy mlp --num_envs 16 --max_iterations 10 --seed 0 --headless
python scripts/train.py --task FlyG1-GoalReach-FreePosture-v0 --policy frozen_lif --connectome_manifest data/connectome/manifest.json --num_envs 16 --max_iterations 10 --seed 0 --headless
```

The first simulator launch must be run on a machine with an installed Isaac Sim runtime and a compatible NVIDIA driver.  A successful smoke test only validates loading/stepping; it is not a locomotion result.  See [docs/environment.md](docs/environment.md), [docs/task_spec.md](docs/task_spec.md), and [docs/model_spec.md](docs/model_spec.md).

The [current matrix](runs/main_matrix_malecns_v1_heldout_v3_20260913.json) contains four conditions × five training seeds. The historical [one-update integration v2](runs/main_matrix_integration_v2_20260913.json) completed all four training paths and three held-out evaluations per condition; its [comparison report](runs/main_matrix_integration_v2_20260913_comparison.md) checked schedules and paired initial states but does not establish sustained training. A [60-update original-LIF pilot](runs/train-replay-dense-v5/20260913T145118Z/FlyG1-GoalReach-FreePosture-v0/frozen_lif/seed-0/manifest.json) then completed 30,720 interactions through an episode reset and saved a checkpoint. The v3 queue is **paused**: original-LIF seeds 0 and 1 each passed a 5,000,192-interaction training job and all three 16-episode held-out evaluations; seed 2 stopped after 6,618 logged updates without a checkpoint, and 17 jobs remain untouched and ready. Its saved JSON still labels seed 2 "running," but no matrix runner or trainer is live. The [partial comparison](runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.md) validates 2/20 jobs with no errors, while the complete five-seed comparison remains pending. The [four-condition integration v3](runs/main_matrix_integration_v3_20260913.json) completed four one-update, 512-interaction jobs and all 12 held-out evaluations. Its [comparison report](runs/main_matrix_integration_v3_20260913_comparison.md) validates four jobs with no errors and shared schedule, episode-plan, and initial-state hashes within each scenario across conditions. This is a pipeline check, not evidence of useful learning or comparative efficacy. Resume only when ready: the wrapper revalidates seeds 0 and 1 and restarts seed 2 from update zero. The queue requires a current-code, 16-environment, 1,000-step bounded-random-action smoke report, runs jobs sequentially, and verifies checkpoint and evaluation artifacts on resume. Each full-budget checkpoint is evaluated on GoalReach, GoalSwitch, and PushRecovery with the fixed `heldout_v1` target/push schedule. The comparison summarizer uses independent training seeds and labels incomplete matrices. See [resume instructions](docs/training.md) and [current results and limits](docs/results.md) before interpreting performance.

The user's [RAM/VRAM addendum](docs/memory_plan_source_th.md) is integrated into [plan.md](plan.md) as staged measurement and scale-up gates. Its Windows RTX 4060/24 GB target differs from this Linux RTX 5060 development machine; the existing 16-environment pilot does not prove that the full MaleCNS graph fits either 8 GB card.
