# Environment and reproducibility

Last inspected: 2026-09-13.

| Component | Observed value | Status |
| --- | --- | --- |
| OS | Linux 7.0.0-31-generic x86_64 | PASS |
| Isaac Lab checkout | `/home/chayanin/Downloads/IsaacLab` @ `b4c321024792976150ca55fddb26fa34480d974e` | PASS |
| Isaac Lab | 0.54.4 | PASS |
| Isaac Sim Python package | 5.1.0.0 in `env_isaaclab` | PASS |
| Isaac Lab assets/tasks | 0.2.4 / 0.11.16 | PASS |
| Python | 3.11.15, `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python` | PASS |
| PyTorch | 2.7.0+cu128 | PASS |
| Gymnasium | 1.2.1 | PASS |
| rsl_rl | 5.0.1 (`rsl-rl-lib`) | PASS |
| GPU/runtime smoke | 1 and 16 environment 1,000-step runs completed | PASS |
| Real MaleCNS data | Public v1.0 Feather tables in `data/connectome/raw/`; derived 256-neuron manifest in `data/connectome/` | PASS |
| Local 16-environment LIF memory pilot | 2,869/8,151 MiB sampled device-wide GPU; 40.7% system RAM after two updates | PASS for short subset pilot |

The Isaac Lab checkout had an unrelated untracked `synthetic_smolvla/` directory when inspected; this project does not modify it. Use this project's editable installation with the exact interpreter above:

```bash
/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python -m pip install -e /home/chayanin/Desktop/flyg1
/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python /home/chayanin/Desktop/flyg1/scripts/doctor.py
```

`doctor.py` is CPU-safe and reports `PASS`, `BLOCKED`, or `FAIL` capabilities. The simulator asset inspection and 1- and 16-environment 1,000-step smoke tests completed on 2026-09-13. The current-code 16-environment × 1,000-step low-amplitude check passed; a separate full-range random-action check did not complete. The [MaleCNS data README](../data/connectome/README.md) records official source links, local checksums, and the converter command. Final-manifest, five-update original and rewired LIF pilots and their neural traces completed with finite activity; longer training stability remains open. Do not change physics backends between main comparison conditions.
