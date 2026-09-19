# G1 asset audit

Status: **PASS for articulation resolution; physical validation remains pending**.

Run:

```bash
python scripts/inspect_asset.py --headless --output docs/asset_audit.json
```

The generated [asset_audit.json](asset_audit.json) records the resolved G1 articulation: 37 joints, 44 bodies, approximately 32.24 kg total body mass, and a 23-joint action allowlist spanning legs, ankles, torso, shoulders, and elbows. It includes position/velocity/effort limits, PD stiffness/damping, actuator groups, base freedom, self-collision setting, physics timestep (0.005 s), control decimation (4), and control timestep (0.020 s). The checked configuration selects `isaaclab_assets.G1_CFG`. The asset path is a Nucleus identifier; the `asset_identifier_sha256` hashes the identifier string, not the downloaded USD bytes. All 23 named low-amplitude action-to-target probes passed. Their measured joint-position deltas are diagnostics, not proof of isolated joint motion while the untethered robot falls; contact/penetration checks and a visible run remain pending.
