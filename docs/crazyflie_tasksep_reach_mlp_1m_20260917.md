# Task-separated Reach MLP 1M pilot report (2026-09-17)

This report records the user-authorized Reach-only normal-MLP pilot. It is a
diagnostic baseline, not a LIF proof and not a main-matrix job. `PASS` below
means command/artifact integrity unless the row explicitly says empirical
task success.

## Outcome

| Check | Result | Evidence |
|---|---|---|
| 16-environment PPO smoke | PASS | exit 0; GPU 2,543 MiB; RAM 36.8%; no paging |
| 40-environment PPO smoke | PASS | exit 0; GPU 2,542 MiB; RAM 37.3%; no paging |
| 40-environment compute target | PASS | 20-sample training mean 52%, range 49--55% |
| Exact 1M training execution | PASS | 250/250 updates; 1,000,000 interactions; exit 0 |
| Training finite/failure gate | PASS | 250 finite history rows; zero failure terminations; zero rejected PPO steps |
| Training memory gate | PASS | GPU 2,568 MiB; RAM 37.5%; RSS 3,982.52 MiB; no paging |
| Held-out Reach artifact execution | PASS | five fixed seed-101 episodes completed; exit 0 |
| Held-out Reach empirical success | **FAIL** | event score 0/5; strict score 0/5 |
| Held-out safety | PASS | crash 0/5; OOB 0/5; invalid state 0/5 |
| Evaluation memory gate | PASS | GPU 2,648 MiB; RAM 35.8%; RSS 3,589.422 MiB |
| Checkpoint/resume cycle | NOT RUN | this pilot started fresh and completed without interruption |

The vehicle remained stable and moved toward every target, but did not enter
the 0.20 m, <=0.25 m/s success tube for 25 consecutive control intervals.
Mean initial goal error was 1.6127 m and mean final error was 1.1437 m; mean
per-episode distance reduction was 28.44%.

## Exact commands

Interpreter used throughout:

```bash
/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python
```

16-environment smoke:

```bash
/usr/bin/time -v \
  -o runs/crazyflie-tasksep-v1-mlp-reach-16x1000-update.time.txt \
  /home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python \
  scripts/drone_smoke_env.py \
  --task FlyCrazyflie-WaypointReach-v0 \
  --contract_profile balanced_v3 \
  --policy mlp_normal --ppo_update --num_envs 16 --steps 1000 --headless \
  --output_report runs/crazyflie-tasksep-v1-mlp-reach-16x1000-update.json
```

40-environment smoke:

```bash
/usr/bin/time -v \
  -o runs/crazyflie-tasksep-v1-mlp-reach-40x1000-update.time.txt \
  /home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python \
  scripts/drone_smoke_env.py \
  --task FlyCrazyflie-WaypointReach-v0 \
  --contract_profile balanced_v3 \
  --policy mlp_normal --ppo_update --num_envs 40 --steps 1000 --headless \
  --output_report runs/crazyflie-tasksep-v1-mlp-reach-40x1000-update.json
```

Training:

```bash
/usr/bin/time -v \
  -o runs/crazyflie-tasksep-v1-mlp-reach-1m-seed0.time.txt \
  /home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python \
  scripts/drone_train.py \
  --task FlyCrazyflie-WaypointReach-v0 \
  --contract_profile balanced_v3 \
  --policy mlp_normal --seed 0 --num_envs 40 \
  --total_interactions 1000000 --horizon 100 --microbatch_size 40 \
  --ppo_epochs 2 --learning_rate 3e-5 \
  --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.2 \
  --value_coefficient 0.5 --entropy_coefficient 0.002 \
  --max_grad_norm 1.0 --target_kl 0.05 \
  --checkpoint_every_updates 25 \
  --evaluation_protocol lif_proof \
  --run_dir runs/crazyflie-tasksep-v1-mlp-reach-1m-seed0 --headless
```

Held-out Reach evaluation:

```bash
/usr/bin/time -v \
  -o runs/crazyflie-tasksep-v1-mlp-reach-1m-seed0.evaluation.time.txt \
  /home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python \
  scripts/drone_evaluate.py \
  --checkpoint runs/crazyflie-tasksep-v1-mlp-reach-1m-seed0/checkpoints/latest.pt \
  --protocol lif_proof \
  --scenario FlyCrazyflie-WaypointReach-v0 \
  --training_seed 0 --policy mlp_normal --device cuda:0 --headless \
  --output runs/crazyflie-tasksep-v1-mlp-reach-1m-seed0/evaluation-reach-lif-proof.json
```

GPU utilization was sampled with:

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
```

The 20 exact samples and summary are stored in
`runs/crazyflie-tasksep-v1-mlp-reach-1m-seed0/gpu-utilization-sample.json`.

## Training evidence

- Wall time: 3:49.48; exit status 0; CPU 175%; `/usr/bin/time` maximum RSS
  4,095,976 KiB; swaps 0.
- 40 environments x horizon 100 = 4,000 interactions/update; 250 updates =
  exactly 1,000,000 interactions.
- 1,640 completed training episodes; ten target-success events (0.6098%);
  zero failure terminations.
- Loss first/last: 0.050376 / 0.030828. Value loss first/last:
  0.040070 / 0.000630. Maximum attempted KL was 0.0001953 and no step was
  rejected.
- Actor parameters: 4,827 versus the 4,776 original-LIF reference (+1.0678%,
  within the 10% match gate). Total trainable parameters: 23,132; critic:
  18,305; distribution: 4.
- Eleven archived update checkpoints plus `latest.pt` are present. The ten
  immutable history segments cover contiguous updates 1--250 and have
  aggregate history SHA-256
  `fdc10c7359730dfec00dd64e3250c7fc2abbf70a4983a4d67c86b0b83dc47dc5`.

The weighted completed-episode reward was dominated by survival (6.0000) and
proximity (3.0822), while progress was 0.0659 and success bonus only 0.0183.
Loss reduction therefore did not establish target acquisition.

## Held-out episode detail

| Episode | Initial error | Final error | Reduction | Integrated error | Event | Crash/OOB/invalid |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 2.1314 m | 1.6488 m | 22.64% | 22.0286 m*s | 0 | 0/0/0 |
| 1 | 1.8856 m | 1.2842 m | 31.89% | 18.6692 m*s | 0 | 0/0/0 |
| 2 | 1.1211 m | 0.7836 m | 30.10% | 10.4116 m*s | 0 | 0/0/0 |
| 3 | 1.8550 m | 1.1280 m | 39.19% | 17.1157 m*s | 0 | 0/0/0 |
| 4 | 1.0702 m | 0.8738 m | 18.35% | 10.1280 m*s | 0 | 0/0/0 |

All five episodes ran the full 600 steps / 12 seconds. Evaluation wall time
was 29.41 seconds, exit status 0, and swaps 0. Isaac emitted the non-fatal log
line `[Error] [isaacsim.core.cloner.impl.cloner] Failed to clone in Fabric`;
the environment subsequently initialized and produced a complete, finite
artifact. This line is retained as a warning and is not hidden by the command's
zero exit status.

## Critical limitation and provenance

Balanced-v3 starts the full curriculum stage at exactly 1,000,000
interactions. The last completed training cohort reset into `mid_3d`; no
full-stage episode reset or completion occurred before training stopped.
Evaluation deliberately forces the full distribution. The result is therefore
a valid failed generalization test, not evidence that a fully trained MLP
cannot solve Reach.

Primary hashes:

| Artifact | SHA-256 |
|---|---|
| 16-env smoke JSON | `bc44496c27a882637c2913a73841fe251f1130c5e58192333d627128400a4e97` |
| 40-env smoke JSON | `5efdfae13ded3690fea508ab8d87e9a894b1b51da1b9cf1693aef6f48b262d52` |
| Training manifest | `bcc8446134f8c2d4aa84b7a00476d93fcc9d689715643ce63ac5af1b1606843d` |
| Final checkpoint | `e643fe82b714d7f7b586a83429236da94d337bb04f0a2a7e3c9ce420fda1f7d3` |
| Evaluation JSON | `7fbc0f0de9bfea62da0b36b3a8ef757f23bad53f113ce4baebd782725caa176f` |
| GPU sample JSON | `be9a3b4a5a8291da5eddf72a0ced6e0f46c8cb30999f002316697850ce6de7a1` |
| Training plot | `be979bc40aac6435e44854910c51f24b4f90eb1bca093806abe3021b6d2c3731` |

The run fingerprint is
`68b9212b2b4dcfd7bbd1dc5514bd3772f01343714fe3286fa0d8286281332ff9`.
Subsequent additive source work must produce a new fingerprint; it must not
rewrite or relabel this run.

## Preservation/process check

At the post-run audit, the frozen Unitree G1 before-manifest union contained
37 files and all 37 current hashes matched both the before union and stored
after manifest. No frozen G1 file changed. No Isaac, drone trainer, evaluator,
matrix runner, or NVIDIA compute process remained running.
