# Crazyflie survival-v2 artifact removal audit

Date: 2026-09-16 (Asia/Bangkok)

The user explicitly stopped the Crazyflie survival-v2 branch and requested
removal of its run artifacts before continuing with the additive balanced-v3
protocol.  This removal does not include Unitree G1 files, G1 runs, shared
connectome inputs, shared source modules, or balanced-v3 artifacts.

The stopped primary attempt had reached 449,200 interactions (1,123 PPO
updates and 756 completed episodes) for `frozen_lif_original`, seed 0.  Its
recorded target-success count remained zero.  It was already cleanly paused;
no Isaac, training, evaluation, or matrix process was running at removal time.

Authenticated identities retained before removal:

- queue `runs/crazyflie_main_v2_verified.json`: SHA-256
  `16cc7528c0cef4bd85de4ed861cc070a8ee68e22e60b305d0ae139b14445098a`
  (750,911 bytes)
- latest checkpoint: SHA-256
  `56cd8f5e05ce34f86fd67a978fafbaafd5f864cf3ee23709910a4aec2fe4efd9`
  (465,788 bytes)
- training manifest: SHA-256
  `37321dd994963427a0b11a73dde2f4cb9e561cf2819fe08be822d42e7e143cd1`
  (41,101 bytes)

Removed artifact roots/files:

- `runs/crazyflie_main_v2_verified/`
- `runs/crazyflie_main_v2_verified.json`
- `runs/crazyflie_main_v2_verified_console.log`
- `runs/crazyflie_main_v2_verified_dry_run.log`
- `runs/crazyflie_main_v2_verified_summary.json`
- `runs/crazyflie_main_v2_verified_verify.log`
- `runs/crazyflie-lif-original-survival-2m-v1/`
- `runs/crazyflie-lif-original-survival-2m-v2/`
- `runs/crazyflie-lif-original-survival-2m-v3/`
- five top-level survival-v2 smoke reports for Reach, Switch, and Gust

The removed data occupied approximately 25 MiB.  At inspection time the root
filesystem was 92% used with 74 GiB available; therefore these artifacts were
not the material cause of overall filesystem usage.

## Supplemental exhaustive run-artifact cleanup

A later content audit found additional historical Crazyflie artifacts whose
JSON provenance contained a non-null `survival_first_contract`.  Before
removing them, the complete 250-file, 33,472,904-byte inventory was recorded
with each file's SHA-256 digest in
`docs/crazyflie_survival_v2_artifacts_before_delete_20260916.json` (file
SHA-256 `92a61215d560fca8bf2d4a4abe156cef0f4393d278c0fb0130573602cc88837b`).

The nine exact directory targets and 68 exact top-level file targets were
moved out of the workspace with the desktop trash service.  They remain
recoverable until that trash is emptied.  The post-removal audit is
`docs/crazyflie_survival_v2_artifacts_after_delete_20260916.json` (file
SHA-256 `ef7cfbcfc3115ac1ca0465880d7d63dde5c6ac8f85790173b0fb010f89a73763`).
It records zero remaining targets, zero JSON files under `runs/` with a
non-null survival contract, and zero mismatches across all 37 files in the
frozen G1 manifest.  Balanced-v3 artifacts and every G1 run, checkpoint, and
queue were excluded from this cleanup.
