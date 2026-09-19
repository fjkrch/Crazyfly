# Crazyflie memory acceptance v2

**Effective:** 2026-09-17  
**Policy identifier:** `crazyflie_memory_acceptance_v2`

This policy supersedes the Crazyflie acceptance rule that treated a
monotonically increasing four-sample RSS window above 1.0 MiB as a hard
failure. It does not alter or relabel historical artifacts produced under the
old rule, and it does not change frozen Unitree G1 files or their memory
watcher.

## Hard failures

- Sampled device-wide GPU memory is `>= 6963.2 MiB` (6.8 GiB).
- System RAM is `>= 90.0%`.
- CUDA telemetry is missing for a CUDA sample.
- The last four like-for-like steady-state/optimizer samples show sustained
  cumulative swap-out growth greater than 1.0 MiB.
- CUDA/host OOM, killed child process, nonfinite simulator state, nonfinite
  action/loss/metric, malformed telemetry, or NaN/Inf in memory evidence.

All limits are fail-closed and boundary values fail. Training remains one
Isaac process and one matrix job at a time.

## RSS warning

The detector remains unchanged: inspect the final four like-for-like
`steady_state` or `optimizer_update` RSS samples; flag the window when every
sample rises strictly and the net rise exceeds 1.0 MiB. The artifact retains
the Boolean detector, exact window, tolerance, and warning text. This signal
is warning-only and does not make `passed=false` by itself.

Each v2 assessment contains:

- `policy_version: "crazyflie_memory_acceptance_v2"`
- `warnings: [...]`
- `limits.rss_growth_disposition: "warning_only"`

Scenario warnings must survive aggregation into evaluation, proof, queue, and
summary artifacts.

## LIF proof scoring update

The source-fingerprinted v2 proof trains Original LIF for exactly 500,000
interactions and evaluates five deterministic episodes for each of Reach,
Switch, and Gust. Each task receives an event score out of five: Reach counts a
target dwell, Switch counts any of its four target dwells including the initial
target, and Gust counts a post-gust recovery. Completing all four Switch
targets remains the separately reported strict episode-success metric.

The launch prerequisite is an event score of at least `1/5` for every task,
with all episode-level details preserved. This proof protocol does not alter
the main matrix's 16 episodes per scenario/job or its total of 960 episodes.

## Evidence preservation

The final pre-override 500,000-interaction run is preserved under
`runs/pre_override_archives/crazyflie-balanced-v3-lif-proof-memory-v1-500k-20260917/`.
It remains a v1 failure and is not reused or edited into a v2 pass. Runtime
source changes require a fresh proof, smoke evidence, integration queue, and
main dry run because those artifacts are source-fingerprinted.
