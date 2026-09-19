# Codex implementation plan: Unitree G1 free-posture locomotion

Status: active. This is the single maintained plan for the existing `flyg1` project; engineering and real-circuit pilots have executed, but the full multi-seed comparison has not.
Prepared: 2026-09-13. Updated: 2026-09-13.

## 1. Objective and source boundaries

Implement the Unitree G1 portion of the supplied proposal in an isolated NVIDIA Isaac Lab project using Isaac Sim. Do not implement Go2, a drone, ROS integration, or real-hardware deployment in this project.

The supplied proposal specifies G1 simulation; a MaleCNS-derived circuit with leaky integrate-and-fire (LIF) neurons; frozen circuit wiring and weights; small, trainable embodiment-specific input/output adapters; joint-position targets followed by PD control; reinforcement learning; locomotion without mandatory upright posture or prescribed gait; no termination merely because the torso contacts the floor; synchronized circuit/robot recording; and comparisons with degree-preserving rewiring, a conventional recurrent network, and matched neuronal ablations. [P]

This plan makes those requirements executable. Package names, task IDs, observation definitions, PPO integration, data schemas, numerical starting points, and acceptance tests below are engineering choices, not details supplied by the proposal. The local repository, simulator, GPU, and G1 asset have been inspected under Task 0 and Task 1. Public MaleCNS v1.0 source tables have since been supplied locally and converted into a provenance-tracked research subset; the measured environment and implementation state are recorded below and in `docs/`.

The research question is whether the specified fixed circuit supports useful G1 locomotion and how its simulated activity changes across behaviors. Do not replace this with a requirement to learn humanlike walking. Walking, crawling, shuffling, rolling, and mixed strategies are admissible if physically valid. Do not guarantee that multiple strategies will emerge or that the biological circuit will outperform a baseline.

## 2. Working instructions for Codex

Read the repository's existing AGENTS.md and other applicable instructions first. Inspect before editing. Preserve unrelated changes. Keep this file current under Progress, Decisions, Discoveries, and Outcomes. Work through independently testable milestones; do not launch all experiments before the implementation gates pass. Maintain ordinary repository permission and approval requirements.

Use a separate project and an isolated dependency environment or container for G1. Do not upgrade or modify an existing Go2/drone environment. Prefer an external Isaac Lab project rather than modifications to Isaac Lab itself; the official documentation supports this organization. [S1]

Do not overwrite an existing AGENTS.md. If appropriate, add a short scoped instruction pointing to this plan, preserving existing instructions. OpenAI documents AGENTS.md as project guidance and describes maintained execution plans as one workflow for complex tasks. [S4, S5]

If a required GPU/runtime is unavailable, complete CPU-testable modules and document exact blocked simulator checks. If real circuit data are unavailable, complete the schema, loader, fixture tests, and ordinary-policy baseline; block biological training explicitly. Neither missing dependency permits invented assets, neuron identities, results, or claims of successful execution.

## 3. Intended architecture

    G1 observations + goal
        -> small trainable G1 encoder
        -> fixed MaleCNS-derived LIF recurrent core
        -> small trainable G1 decoder
        -> bounded joint-position targets
        -> PD actuator model with validated limits
        -> Isaac Sim physics

The actor's trainable components are the G1 encoder, decoder, and any explicitly declared action-distribution parameters. A separate value-function network may also be trained. The circuit topology, synaptic weights, and chosen neuron-model parameters stay fixed in the main condition. Membrane potentials, spikes, synaptic traces, and other dynamical state evolve during an episode.

Do not add an observation-to-action bypass around the circuit in the main actor. Put any bypass experiment in a separately named ablation. Do not put the neural core inside a nondifferentiable environment observation wrapper while claiming that policy gradients train the upstream encoder.

For training, implement the hard forward spike rule and an explicitly documented surrogate derivative for backpropagation. This derivative is an optimization device, not a biological measurement. Freeze circuit parameters without enclosing the policy's core forward pass in no_grad or detaching the encoder output: the encoder still needs a gradient path through the fixed transformation. Test that path rather than assuming it exists. No differentiable physics is required by this PPO design.

Maintain one independent neural state per vectorized environment. Environment reset must reset that environment's membrane state, spike history, refractory state, synaptic filters, previous action, and observation histories without changing other environments.

## 4. Project structure

The project is already implemented as an external source-layout package in the current `flyg1` repository. Continue in these existing paths; do not create a second plan or project tree. The logical layout below remains a reference where planned files differ from implemented paths.

    flyg1/
      plan.md
      README.md
      pyproject.toml
      source/g1_fly_control/
        g1_fly_control/
          tasks/g1/{scene,observations,actions,rewards,terminations,events}.py
          policies/{encoder,decoder,lif_core,actor_critic}.py
          connectome/{schema,loader,rewire,manifest}.py
          training/{runner,recurrent_storage,checkpoint}.py
          evaluation/{metrics,perturbations,ablations,recording}.py
      configs/experiments/main.json
      scripts/{doctor,inspect_asset,smoke_env,train,play,evaluate,record,run_matrix}.py
      tests/unit/
      tests/fixtures/synthetic_circuit/
      data/connectome/README.md
      docs/{environment,asset_audit,task_spec,model_spec,training,results}.md
      runs/                         # ignored by Git

Keep pure tensor/circuit modules importable without Isaac Sim. Keep large data, assets, videos, and checkpoints out of Git. Do not commit credentials or download restricted data without authorization.

## 5. Task 0 — Inspect and pin the execution environment

Inspect the working tree and locate any existing reusable circuit implementation. Record Python, Isaac Lab, Isaac Sim, PyTorch, CUDA/runtime, GPU/VRAM, operating system, learning-library version, and relevant commit hashes. Resolve a compatible, pinned combination from the actual installation and its documentation. Do not select an unverified latest version or switch physics backends during the main comparison.

Use the implemented `scripts/doctor.py` to report dependencies and unavailable capabilities without pretending to run simulation. The verified interpreter is `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python`; the Isaac Lab source checkout is `/home/chayanin/Downloads/IsaacLab` at `b4c321024792976150ca55fddb26fa34480d974e`. Keep the dependency export and reproduction instructions in `docs/environment.md`. Do not use `isaaclab.sh -p` here because it selected a different base interpreter during inspection.

The official environment catalog lists Isaac-Velocity-Flat-G1-v0 and Isaac-Velocity-Rough-G1-v0. Use the installed task registry to verify local availability; a task listed online is not proof that it exists in the user's checkout. [S2]

Acceptance: a fresh invocation produces an environment report, a selected compatible configuration, and explicit PASS/BLOCKED/FAIL statuses. No Go2/drone dependencies or files have been changed.

## 6. Task 1 — Create the isolated project and validate the G1 body

Generate an external, manager-based Isaac Lab project using the pinned installation's supported mechanism. Use the supported G1 asset as a starting point. Inspect the loaded articulation rather than assuming a particular number of degrees of freedom.

Export exact joint names/order, actuator groups, masses, root freedom, position limits, velocity limits, effort limits, PD gains, contact bodies, self-collision settings, asset identifier, and asset checksum/version where available. Use an explicit actuated-joint allowlist. The research configuration must make the verified arm, leg, and trunk joints available rather than silently fixing the upper body to enforce walking. Document exclusions such as fingers. Never invent missing joints or stronger actuators.

Implement action normalization and mapping into validated position targets. Keep position, effort, and velocity constraints effective in the actual simulator, not just in a configuration file. Log target and applied quantities. Verify joint order with named, low-amplitude probes. Random-action tests must also obey bounds.

Run a headless scene, then a visible/recorded scene, with zero actions and bounded diagnostic actions. A zero-action robot is allowed to fall; this checks execution and physics, not locomotion skill. A tethered or fixed-base diagnostic is allowed only if separately labeled and excluded from research training.

Acceptance: the asset audit exists; 1-environment and 16-environment smoke tests complete 1,000 control steps without unhandled errors or nonfinite state; resets work; the joint mapping is verified; different environment instances do not collide. Record actual timestep, control decimation, throughput, and memory usage.

## 7. Task 2 — Implement posture-neutral G1 tasks

Create three project-specific task registrations. These names are new project IDs, not existing NVIDIA tasks:

    FlyG1-GoalReach-FreePosture-v0
    FlyG1-GoalSwitch-FreePosture-v0
    FlyG1-PushRecovery-FreePosture-v0

GoalReach is the first training task: approach an XY target on flat terrain with no mandatory gait. GoalSwitch changes target according to a logged schedule and measures redirection. PushRecovery retains the goal objective while applying a logged disturbance schedule. Use shared scene/action/reward code so variants do not silently change the experiment.

Do not copy the stock walking configuration unchanged. The checked G1 implementation contains biped foot-air-time rewards, orientation/posture penalties, and a torso-contact termination inherited by its flat configuration. These differ from the supplied proposal's free-posture objective. Audit all inherited terms, not only the most obvious termination. [S3a, S3b]

### Observations and actions

Start with normalized joint positions and velocities, base angular velocity, projected gravity, base linear velocity, the target relative to the body, and previous action. Mark base velocity and target localization explicitly as simulator-state assumptions, not demonstrated onboard estimators. Define frames, units, normalization, ordering, clipping, and reset handling in docs/task_spec.md. Use the same actor-visible observations across comparison conditions. Do not add vision, gait phase, a motion reference, or posture labels in the first version.

Treat contact signals as diagnostics initially. Adding contact observations later is a versioned design change applied to all matched conditions. Use one declared reference point for world-XY goal distance, such as the verified pelvis/root position.

### Reward and completion

Use target progress, a one-time success bonus, and penalties for mechanical-work proxy, action changes, joint-limit violations, and excessive impacts. Define mechanical work per control interval as the physics-substep sum of abs(applied_torque * joint_velocity) * physics_dt. Call it a mechanical-work proxy, not measured electrical consumption.

Do not reward uprightness, pelvis height, biped foot alternation, human pose imitation, or foot air time. Do not penalize ordinary sustained hand, knee, or torso support contacts merely because they occur. Excessive-impact thresholds must be explicit and identical across compared methods.

A concrete starting goal criterion is XY distance <= 0.30 m for 0.50 s, with a 20 s episode horizon. These are proposed defaults to validate, not proposal requirements or known optimal settings. Freeze the final criterion before comparative runs. Count success once per target and reset distance history on goal changes and environment resets so teleportation cannot create a reward spike.

Define reward units carefully: Isaac Lab reward aggregation may apply timestep scaling in the selected version. Test whether each term is a rate or an interval quantity and prevent double integration. Use a synthetic trajectory to verify reward behavior when control frequency changes.

### Termination, resets, and physics validity

Do not terminate for torso contact, low base height, a non-upright orientation, or ordinary falling. Use a finite time horizon, declared goal completion behavior, workspace escape, and invalid-simulation conditions. Distinguish truncation from termination. Record numerical failures separately rather than treating them as successful recovery or ordinary task completion.

Start debugging with a single valid pose. For research, evaluate collision-valid standing, crouched, and prone/supine reset regimes when the asset supports them. Record the training reset distribution and report performance by evaluation regime; a standing-only start is an experimental bias, not evidence of posture neutrality. No inactivity termination in the initial main task.

Test self-collision, contact response, penetration, actuator saturation, and stability at a smaller physics timestep on representative trajectories. Valid rolling or crawling is not a physics exploit merely because it is not walking. Inspect suspicious progress from excessive penetration, spurious impulses, or unenforced limits.

Acceptance: automated tests prove that ordinary torso contact and non-upright states do not end an episode; all disallowed walking rewards are absent; reward bookkeeping survives goal switches/resets; task IDs are unique; the fully resolved configuration is saved.

## 8. Task 3 — Establish an ordinary-policy engineering baseline

Implement train/play/checkpoint support using a small MLP actor and separate critic with the same custom GoalReach environment. Prefer a PPO integration supported by the installed Isaac Lab stack, after inspecting its actual interface. Do not assume a stock runner accepts a custom recurrent spiking actor without changes.

Use an explicit small smoke budget, for example 16 environments and 10 updates. Increase training scale only after measuring memory and throughput. Save optimizer state, policy state, normalization state, configuration, seeds, and version metadata. Verify inference from a checkpoint in a fresh process.

Keep this engineering baseline distinct from the stock G1 walking task. The stock walking task may separately validate simulator installation, but its rewards and checkpoints are not evidence for the free-posture experiment.

Acceptance: training updates are finite; model parameters change; checkpoints load; evaluation executes and produces real measurements. A smoke run does not have to learn competent locomotion. Before expensive circuit comparisons, establish useful learning in a larger predeclared pilot or document that task learning remains unresolved. Never invent a success rate to satisfy a gate.

## 9. Task 4 — Implement validated circuit ingestion and the frozen LIF actor

### Data boundary

Define an explicit schema with stable neuron IDs, edge pre/post IDs, raw connectivity evidence, model weights, annotation provenance, and a circuit manifest. Store source release, extraction query/rule, neuron subset, edge filtering, sign assumptions, normalization, neuron-model constants, units, and checksums. Keep raw evidence distinct from derived simulation weights. Do not treat synapse counts as physical conductance without a declared model mapping.

The attachment names MaleCNS but did not provide its actual circuit files, chosen subset, or executable neuron parameterization. [P] Public MaleCNS v1.0 source tables were later supplied locally and converted as recorded below. Do not fabricate a biological circuit or label random connectivity as MaleCNS. Put tiny synthetic networks in tests/fixtures only, with synthetic status embedded in their manifests and outputs. A missing-data error must identify what is required; it must not silently fall back to a synthetic research run.

Validate unique IDs, edge endpoints, finite weights, chosen duplicate-edge semantics, orientation of the adjacency matrix, dimensions, metadata coverage, and checksums. Treat missing transmitter signs or anatomical labels as explicit unresolved/model assumptions. Record every transform before freezing. Do not perform continual weight normalization during training.

### Core and adapters

Implement batched LIF dynamics with explicit membrane, spike, synaptic-filter, refractory, reset, and optional delay conventions. Choose and document a numerically tested update rule; do not claim a particular biological parameter set unless supplied and verified. Define the number and duration of neural substeps per policy step and the mapping between neural and simulator time.

Train the encoder to inject currents into a declared input subset. Decode from a declared state summary, such as filtered spikes from the chosen output subset. Keep adapter sizes small and report their parameter counts. Do not silently grow a large encoder/decoder until the fixed core becomes incidental.

Use sparse storage/operations for a nontrivial circuit where the verified autograd path supports them. A tiny dense reference implementation is appropriate for equivalence tests. Profile actual memory cost before increasing neuron count, parallel environments, or sequence length.

Acceptance: known-input LIF tests pass; sparse/reference outputs and gradients agree within declared tolerances; fixed core tensors have unchanged checksums after optimizer steps; encoder and decoder receive nonzero gradients on controlled test inputs; reset isolation passes; checkpoint/reload reproduces an identical action sequence from identical observations and initial neural state. Record dead/saturated neuron fractions on real rollouts without requiring every neuron to fire.

## 10. Task 5 — Integrate recurrent PPO correctly

The circuit state persists across control steps. Implement a recurrent actor/runner interface with initial_state(batch_size), act(observation, state), evaluate_sequence(...), and reset(state, env_ids), adapting names to the chosen library rather than guessing APIs.

Store ordered rollout sequences, sequence-start neural states, episode masks, old log probabilities, values, actions, observations, and any burn-in observations. During updates, replay the differentiable encoder/core/decoder sequence with truncated backpropagation through time. Detach only at declared sequence boundaries, not automatically after every LIF substep. Keep padded samples and reset transitions out of invalid losses.

Declare how state reconstruction and burn-in handle encoder updates across PPO epochs. Do not cache old reservoir features as fixed observations while claiming to train the encoder through them. Document the approximation from truncation and any stale sequence-start states.

Store the policy-space sampled action and its corresponding log probability consistently with the action transformation. Prefer an explicitly modeled bounded distribution or a documented normalized-action/clipping convention. Do not calculate a clipped action's log probability as though no transform occurred.

Handle true terminal transitions versus time limits correctly. When bootstrapping a timeout, use the terminal observation and its aligned recurrent state, not the next episode's reset observation. Check the installed auto-reset/wrapper behavior explicitly.

Acceptance: before any parameter update, re-evaluated rollout log probabilities match the stored values; asynchronous resets affect only the selected environments; masks and padding pass controlled tests; gradients remain finite; core hashes remain unchanged; memory use is bounded across successive iterations; inference uses the same forward dynamics and normalization as training.

## 11. Task 6 — Implement comparisons and a fixed evaluation protocol

Use the original frozen circuit, a frozen degree-preserving rewired circuit, and a conventional trainable recurrent network such as a GRU as the main planned comparisons from the proposal. Retain the MLP engineering baseline. A frozen random recurrent control is an optional additional condition, not a replacement for the specified comparisons. [P]

For the rewired circuit, preserve each neuron's directed in/out degree under declared self-loop and duplicate-edge rules. Define how weights and signs are reassigned and which distributions are preserved. Verify invariants numerically, save the randomized graph/seed/checksum, and freeze it throughout training. Keep the same input/output interface, neuron count, neuron dynamics, adapters, and optimization settings as the original-circuit condition. Do not describe a simple neuron relabeling as meaningful rewiring.

A fully trained GRU has a different trainable-parameter budget from a frozen circuit with adapters. Report total and trainable parameters separately. Use declared size-matching targets, equal environment-interaction budgets, comparable tuning budgets, and consistent observations, actions, rewards, resets, curricula, and evaluation scenarios. Record wall-clock and memory costs rather than claiming identical compute from equal environment steps.

Prepare a configurable initial matrix of five training seeds per main condition. This is a starting experimental design, not a claim of statistical sufficiency. Pair evaluation scenarios where possible. Keep held-out targets, reset conditions, and disturbance schedules separate from training/tuning. Treat independent training seeds, not individual vectorized episodes, as the main independent units for uncertainty summaries.

Report success, XY progress, time to target, work proxy, excessive impacts, limit/saturation frequency, recovery success/time, and observable posture/contact regimes. Define recovery as a preregistered sustained resumption of goal-directed progress or subsequent goal completion within a fixed window; do not require standing up. Measure applied disturbance impulse from force and duration in a declared frame/body, and save the exact schedule.

Acceptance: a dry-run emits a complete experiment manifest without starting a sweep; a small actual evaluation produces auditable per-episode records and per-seed summaries; task and model configuration differences are visible; no performance claim is made from unexecuted runs.

## 12. Task 7 — Record robot/circuit activity and test ablations

Record synchronized video, simulation/control timestamps, environment and episode IDs, root pose/velocity, joints, actions, applied torques, contact summaries, goal state, each reward component, neural voltages, spikes/filtered rates, and disturbances. Tie every record to a configuration hash, circuit manifest, model checkpoint, and seed. Preserve the mapping from array index to source neuron ID.

Keep high-volume neural recording out of default all-environment training. Record selected episodes or neuron subsets, with full-resolution short windows on demand. Retain spike timestamps at neural-substep resolution when needed; a video frame is not a neuron update. Define downsampling and synchronization explicitly.

Use published or supplied metadata for annotated neuronal groups. Unknown groups stay unknown. Do not infer that a group controls G1 behavior solely because it activates during that behavior. Do not label simulated activity from dopamine-associated neurons as dopamine concentration. These interpretation limits are explicit in the proposal. [P]

Implement an acute ablation with a fixed checkpoint: clamp the specified group's outgoing spikes to zero, state exactly which quantities are affected, and compare against multiple random groups matched on declared properties such as size and degree. Keep the environment scenarios paired and weights unchanged. Retraining after ablation is a separate experiment, not the acute test. Include a sham intervention.

Acceptance: a recorded perturbation aligns across video, robot traces, and neural traces; ablation targets resolve to valid source IDs; output suppression affects only intended neurons; sham and matched random controls execute; resulting claims stay confined to the simulated model.

## 13. Task 8 — Package reproducibility and report actual status

Expose doctor, inspect_asset, smoke_env, train, play, evaluate, record, and run_matrix entry points with documented --help output. The names below are interfaces for Codex to implement, not commands claimed to exist already. Run them from the project root with the validated Isaac-compatible interpreter; substitute the pinned installation's launcher where required.

    python scripts/doctor.py
    python scripts/inspect_asset.py --headless
    python scripts/smoke_env.py --task FlyG1-GoalReach-FreePosture-v0 --num_envs 16 --steps 1000 --headless
    python -m pytest tests/unit -q
    python scripts/train.py --task FlyG1-GoalReach-FreePosture-v0 --policy mlp --num_envs 16 --max_iterations 10 --seed 0 --headless
    python scripts/train.py --task FlyG1-GoalReach-FreePosture-v0 --policy frozen_lif --connectome_manifest data/connectome/manifest.json --num_envs 16 --max_iterations 10 --seed 0 --headless
    python scripts/run_matrix.py --config configs/experiments/main.json --dry_run

Document the exact checkpoint path emitted by training and the exact play/evaluate/record commands that use it; do not invent an output file that the runner did not create. The frozen_lif command must fail clearly until a valid real-data manifest is supplied. Unit tests may use the explicitly synthetic fixture without confusing the two modes.

Distinguish functional checkpoint loading from exact mid-episode simulator continuation. Do not claim exact continuation unless simulator, policy, optimizer, RNG, normalization, and recurrent state are all captured and restored under a tested mechanism. Document nondeterminism and tolerances.

Acceptance: README gives a fresh-user setup and run path; source/model/task data provenance is complete; tests distinguish CPU and simulator requirements; results are labeled executed versus pending; there is no hardware-control code or hidden coupling to Go2/drone packages.

## 14. Progress

Check an item only when its full acceptance criteria have corresponding evidence. Partial implementation does not count as completed research validation.

- [x] Task 0: environment audit and version pinning.
- [ ] Task 1: isolated project and G1 asset/control validation.
- [ ] Task 2: free-posture task family and reward/reset tests.
- [ ] Task 3: ordinary-policy training/checkpoint baseline.
- [ ] Task 4: circuit data validation and frozen LIF actor.
- [ ] Task 5: recurrent PPO integration and state/gradient tests.
- [ ] Task 6: matched experiments and held-out evaluation.
- [ ] Task 7: synchronized recording and controlled ablations.
- [ ] Task 8: documentation, reproduction commands, and status report.

## 15. Decisions, discoveries, and outcomes

Initial decisions: G1 only; simulation only; external isolated project; manager-based task implementation; flat terrain first; no imposed gait; frozen main circuit; separate critic permitted; synthetic circuits confined to explicit software tests. These are a mixture of proposal requirements and engineering choices distinguished above.

Discoveries to record during implementation: actual asset variant and joint availability; installed API differences; reward timestep behavior; sparse-gradient compatibility; compute constraints; missing source data; reset-wrapper semantics; physical or numerical artifacts.

Outcome report at each stopping point: what changed, exact commands executed, pass/fail/blocked results, artifacts created, unresolved dependencies, and the next uncompleted milestone. A working pipeline is not proof of the research hypothesis. A negative result is not an implementation failure when the code and experiment are valid.

2026-09-13 implementation update: an external `g1_fly_control` source-layout package now exists in this repository. The inspected Isaac Lab checkout is `b4c321024792976150ca55fddb26fa34480d974e`, Isaac Lab 0.54.4, Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128, on an RTX 5060 Laptop GPU with 8151 MiB VRAM. `scripts/doctor.py` reports that environment. The simulator resolved `G1_CFG` to 37 joints, 44 bodies, approximately 32.24 kg total mass, and 23 selected controllable joints including legs, ankles, torso, shoulders, and elbows. `docs/asset_audit.json` contains resolved names, limits, masses, gains, and 23 named target-mapping probes. Earlier one- and sixteen-environment 1,000-step headless bounded-action checks passed with finite state. A 10-update 16-environment MLP pilot and a one-update GRU pilot completed, and fresh-process playback of an MLP checkpoint passed. A four-episode MLP evaluation returned 0/4 sustained successes for one training seed; this is an executed smoke measurement, not evidence about general locomotion. GoalSwitch and PushRecovery fired their respective target-change and timed-force events in short simulator checks. A 10-step trace and 10-frame MP4 aligned in count. Twelve CPU unit tests pass. PPO rollouts now carry the environment and recurrent state across update boundaries; LIF and GRU CPU continuation/replay tests pass, while simulator retraining under that fix remains pending. The real MaleCNS circuit is still absent, so biological-circuit training, rewired comparison, and ablations are blocked by data provenance. Tasks 1–8 remain unchecked because GUI viewing, full contact/penetration and timestep validation, useful-learning pilot, real-circuit and multi-seed runs, full neural/video alignment, and full reproduction gates remain outstanding. See `docs/results.md` for precise executed artifacts and pending checks.

### Paused handoff — 2026-09-13

The user asked to stop all work and write the current state here. The pasted attachment and this file have the same substantive plan; the current file has been updated in place rather than replaced by a new plan. This turn also changed recurrent PPO collection to continue observations and neural/GRU state across update boundaries, save a sequence-start state for replay, and cover both policies with CPU tests. The exact latest unit command, `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/unit -q`, passed: 12 tests. The new 16-environment, 1,000-step simulator diagnostic (`scripts/smoke_env.py --task FlyG1-GoalReach-FreePosture-v0 --num_envs 16 --steps 1000 --random_actions --headless`) was stopped on request; no PASS result or throughput from that invocation is claimed. No simulator or training process remains running. No new training/evaluation/ablation run was started after the PPO change. If resumed, first rerun the simulator diagnostic and a small MLP/GRU training check with the current code, then continue the pending acceptance gates. Real MaleCNS source files and provenance remain a hard prerequisite for the biological and rewired comparisons.

### Resumed outcome — 2026-09-13

CPU validation was rerun with `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/unit -q` and passed: 12 tests. `scripts/run_matrix.py --config configs/experiments/main.json --dry_run` emitted `runs/matrix_manifest.json` with 20 planned jobs and started none. This was the valid queue artifact at that historical stage; the real MaleCNS manifest was still absent, so the two frozen-LIF conditions could not then be launched without inventing data. Two attempts to rerun the requested 16-environment, 1,000-step random-action smoke command initialized the G1 scene but did not complete a measured rollout after five minutes; both were interrupted cleanly. They produce no PASS result and no throughput figure. `scripts/smoke_env.py` was revised so its target-limit, finite-state, and root-separation validation accumulates device-side across every step and synchronizes only once at completion, avoiding diagnostic-induced per-step CUDA stalls. The edit compiles; the simulator runtime check is still pending and must complete before the small MLP/GRU retraining checks. The configured machine reports a CPU powersave profile, mismatched PCIe link width, and enabled IOMMU during simulator launch; these are environment observations, not a proven cause of the non-completion. No simulator or training process remains running.

### Main-matrix queue continuation — 2026-09-13

At this historical stage, the main comparison queue became executable and resumable under that code version. `scripts/run_matrix.py` maps the four planned conditions to 20 independent training jobs, checks a real connectome with the research loader, requires a matching task/diagnostic source fingerprint in the smoke PASS report, enforces at least 16 environments × 1,000 steps with bounded random actions, verifies requested versus actual training interactions and checkpoint metadata, and evaluates completed checkpoints on GoalReach, GoalSwitch, and PushRecovery. `--dry_run` and `--queue_only` launch no training. `configs/experiments/main.json` declares 5,000,000 requested interactions per seed, five seeds, 16 environments, horizon 32, and a common evaluation seed. Full rollouts round each run to 5,000,192 interactions. The [queue](runs/main_matrix_20260913.json) and [summary](runs/main_matrix_20260913_summary.json) were created with `--dry_run`, then prepared with `--queue_only --resume --smoke_report runs/smoke-16x1000-low-amplitude.json`: **10 GRU/MLP jobs ready, 10 original/rewired LIF jobs blocked by missing authorized MaleCNS data, zero main jobs executed**. This queue is historical and must not be resumed.

The current-code [smoke report](runs/smoke-16x1000-low-amplitude.json) passed all 1,000 steps in 16 environments using normalized random actions in [-0.1, 0.1] after ten zero-action steps, with 16 resets and 553.17 environment-control-steps/s. A separate full-range [-1, 1] random-action diagnostic reached step 650 and then stopped advancing during validation; it was interrupted and is not a PASS. Two-update current-code MLP and GRU PPO retraining checks completed with finite updates and checkpoints. The updated evaluator produced four-episode GoalReach/GoalSwitch/PushRecovery records from the MLP checkpoint and a four-episode GoalReach record from the GRU checkpoint, including target-relative progress and switch/push counts. These are pipeline checks, not matched scientific results or proof of useful learning. The exact current unit command, `/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/unit -q`, passed 28 tests. `docs/results.md` lists artifacts and numeric outputs. Still open: authorized real circuit files/provenance, full-range random-action stability, a useful-learning pilot, verified held-out targets/reset regimes/disturbance pairing, multi-seed main execution, recording/ablation evidence, and physical-validity/GUI checks.

### MaleCNS data continuation — 2026-09-13

After the earlier snapshots above, three public MaleCNS v1.0 Feather tables were copied from the user's Downloads folder to `data/connectome/raw/`: body annotations, neurotransmitter predictions, and the minconf-0.5 flat connectome. Their exact filenames, official download URLs, row counts, and local SHA-256 values are in [data/connectome/README.md](data/connectome/README.md). `scripts/prepare_malecns.py` streams the source connectivity and creates `data/connectome/{manifest,neurons,edges,audit}.json`. The selected leg VNC circuit has 256 neurons: 24 proprioceptive sensory inputs, eight descending inputs, 200 interneurons, and 24 motor readouts. Its 5,103 directed edges include 3,549 assigned positive and 1,554 assigned negative model weights; all 32 inputs reach an output and all 24 outputs are reachable. The source synapse counts, predicted transmitter signs, selection rules, scaling assumption, and derived-file checksums are explicit in the manifest. This is a selected and modeled circuit from real connectome data, not a measured fly neuron parameter set or the full MaleCNS graph.

One-update simulator pilots for the original and degree-rewired frozen LIF conditions completed and saved checkpoints in `runs/pilot-real-lif/`, establishing that both real-data pathways launch. A diagnostic trace of the first original-circuit pilot found zero spikes at its initial dimensionless threshold of 1.0; later calibration and pilot results are recorded below. The old 20-job queue in `runs/main_matrix_20260913.json` belongs to earlier circuit settings and has a stale checksum, regardless of its subsequently updated ready count. **At that historical point, no five-seed main-matrix job or comparative evaluation had been executed.** Full-range simulator stability, useful-learning, held-out schedules, physical-validity/GUI, and episode-level neural/video and ablation gates remain open.

### Final MaleCNS intake and queued matrix — 2026-09-13

The reproducible [derived circuit](data/connectome/manifest.json) is now fixed at 256 neurons, 5,103 edges, a dimensionless LIF threshold of 0.047, and surrogate beta 1.0. The selected real MaleCNS data load with checksum `69362d71d0ae016c78bad67cbbd6673522adad43e2983796ff25862590c7169f`. These constants were chosen from short encoder-current and simulator pilots before the main comparison; they are not measured fly physiology. The first threshold-1.0 circuit was silent. A threshold-0.05 post-update pilot became almost motor-silent; reducing beta from 10 to 1 resolved its extreme raw gradient, and the final 0.047 circuit retained motor activity after short training.

With a 1e-6 LIF learning rate and a 0.05 post-step PPO KL guard, five-update, 16-environment pilots accepted 7/10 original-circuit PPO epochs and 10/10 rewired epochs. Their 100-control-step traces recorded 13/24 original and 6/24 rewired motor cells active, with no motor cell at or above a 0.45 per-control-step spike rate. This verifies a functioning signal and a short numerical gate, not learning or long-run stability. The first [20-job queue](runs/main_matrix_malecns_v1_20260913.json) was later superseded by the held-out evaluation protocol; do not resume that stale configuration. Each job requests 5,000,000 environment interactions. The separate full-range random-action smoke remains unresolved, as do useful-learning, GUI/physical validation, synchronized episode video, and full ablation evidence.

The earlier held-out queue was superseded when the evaluator and resume checks changed. The later [heldout_v2 20-job queue](runs/main_matrix_malecns_v1_heldout_v2_20260913.json) and [summary](runs/main_matrix_malecns_v1_heldout_v2_20260913_summary.json) were bound to the circuit and execution-code fingerprints at creation. All 20 were software-ready then, and the first full-budget original-circuit seed was launched. It stopped after roughly 16 PPO updates, before a checkpoint, when stored and replayed policy log probabilities disagreed before an update. The queue JSON/summary may still show one job running and 19 ready, but that is a stale serialized state, not an active or safely resumable matrix. Subsequent execution-source changes invalidate its fingerprint. A separate [four-condition integration matrix](runs/main_matrix_integration_v2_20260913.json) completed one 512-interaction update and all three held-out evaluations per condition. Its [comparison report](runs/main_matrix_integration_v2_20260913_comparison.json) validated four jobs, 12 result files, shared per-scenario schedules, and paired initial-state hashes with no errors. GoalSwitch recorded three target changes; PushRecovery recorded three 20 N·s pushes per episode. This is an executed historical check of the train/checkpoint/evaluate/report path, not evidence that recurrent PPO stays correct over sustained training, useful learning, or stability over 5 million interactions. The first GoalSwitch attempt exposed a missing callback default at its five-second event; that was fixed and the full 1,000-step scenario then passed. Only the default standing reset has been exercised. A one-episode final-schedule [acute ablation smoke](runs/ablation-smoke-real-lif-final-schedule.json) passed sham, a two-neuron intervention, and two size/degree-matched random controls with verified initial-state/schedule pairing; its single episode gives no effect estimate. The CPU suite passed 50 tests at that point. At this historical point, long-run PPO replay validation and a fresh fingerprint-bound queue were still pending. Full multi-seed training, useful-learning, posture-reset validation, neural-substep recording, episode video, and scientific ablation results remain open.

### Deterministic 256-neuron replay and v3 queue — 2026-09-13

The LIF replay mismatch was traced to nondeterministic CUDA edge accumulation near hard spike thresholds. For the present 256-neuron subset, the recurrent multiply now uses a deterministic derived dense weight matrix; its 256 × 256 FP32 storage is modest, and tests compare its outputs/input gradients with the ordered-edge reference. Frozen-LIF PPO rejects graphs above 1,024 neurons until a deterministic sparse backend is validated, rather than silently applying a dense full-MaleCNS matrix. The full CPU unit suite now passes **104/104 tests**, including a full-shape report test with 20 jobs and 960 paired episode rows, read-only live-monitor checks, and guarded post-main handoff checks.

A fresh-process 16-environment [60-update training run](runs/train-replay-dense-v5/20260913T145118Z/FlyG1-GoalReach-FreePosture-v0/frozen_lif/seed-0/manifest.json) completed **30,720 actual interactions** and saved a [checkpoint](runs/train-replay-dense-v5/20260913T145118Z/FlyG1-GoalReach-FreePosture-v0/frozen_lif/seed-0/checkpoint.pt). It crossed the 1,000-control-step episode timeout/reset boundary and continued PPO replay without the earlier mismatch. Across 60 updates, 104 PPO epochs were accepted and 12 update attempts were rejected by the 0.05 post-step KL guard; the largest accepted KL was 0.04869. Three fresh-process, one-episode `heldout_v1` [GoalReach](runs/train-replay-dense-v5/evaluation_heldout_v1/GoalReach.json), [GoalSwitch](runs/train-replay-dense-v5/evaluation_heldout_v1/GoalSwitch.json), and [PushRecovery](runs/train-replay-dense-v5/evaluation_heldout_v1/PushRecovery.json) evaluations completed from that checkpoint. They recorded no sustained successes; GoalSwitch logged three target changes, and PushRecovery logged three 20 N·s pushes (60 N·s summed magnitude) with no recovery successes. These are execution and reset-boundary proofs, not useful-learning evidence or a five-seed comparison.

The [heldout_v3 main queue](runs/main_matrix_malecns_v1_heldout_v3_20260913.json) and [summary](runs/main_matrix_malecns_v1_heldout_v3_20260913_summary.json) are **paused: original-LIF seeds 0 and 1 passed, seed 2 stopped without a checkpoint, and 17 jobs remain ready**, four conditions × five seeds, each requesting 5,000,000 interactions. The earlier heldout_v2 queue remains stale and non-resumable. The four-condition [integration v3 matrix](runs/main_matrix_integration_v3_20260913.json) completed all four one-update, 512-interaction jobs and all 12 `heldout_v1` evaluation JSONs. Its [comparison report](runs/main_matrix_integration_v3_20260913_comparison.json) is complete with four validated jobs, no errors, and shared schedule, episode-plan, and initial-state hashes within each scenario across conditions. This is a current-code train/checkpoint/evaluate/report integration check, not evidence of useful learning or multi-seed efficacy. The `scripts/execute_main_matrix_v3.sh` wrapper launched the queue; on resume, it will revalidate completed artifacts and write a comparison report when execution exits. The first job saved its full-budget checkpoint after 5,000,192 actual interactions (9,766 updates) and passed all three fresh-process, 16-episode held-out evaluations. It recorded 0/16 target successes in each scenario and 0/48 successful push recoveries; this is execution evidence, not useful-learning evidence. Seed 1 also saved a full-budget checkpoint and passed all 48 held-out episodes as result files, with 0/16 target successes per scenario and 0/48 strict push recoveries. The [main comparison report](runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.json) and [Markdown table](runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.md) are correctly **incomplete: 2/20 jobs validated, with no validation errors** while the queue is paused. Between-condition pairing remains untested while only the original-LIF condition has completed jobs. The regenerated integration-v3 report remains complete at 4/4 with zero errors under the latest analysis source. `scripts/summarize_matrix.py` rejects missing, nonfinite, or episode-inconsistent metrics and incomplete, out-of-order, or nonfinite per-update training histories while allowing documented conditional nulls, such as time-to-target without a success or recovery metrics without a completed recovery. It now checks on-disk training metadata and reports the scalar Task 6 metrics currently captured by the evaluator, trainable/frozen/total parameter counts, actual interactions, elapsed training time, per-seed training-manifest SHA-256 provenance, and sampled GPU/RAM usage. The integration-v3 comparison contains these resource tables; they remain one-update samples, not full-budget measurements. Observable posture/contact regimes are absent from the evaluation JSONs and require paired trace recording. A separate `scripts/record_heldout_regimes.py` sidecar is prepared to check a completed 16-episode held-out result against its checkpoint, schedule, paired plans, initial states, outcomes, event timing, and total steps while saving bounded, named-body pre-action traces. `scripts/regime_metrics.py` computes CPU-only root orientation/height and net-force proxies by episode and training seed; `scripts/execute_regime_matrix.sh` will run 60 sequential companion replays and write source-checked comparisons after the main queue completes. Their CPU checks pass, but companion simulator replay and regime results remain untested until the main matrix completes; net force does not identify ground support or a gait. The other 18 full-budget results, complete multi-seed comparison, useful learning, physical-validity and posture-reset checks, episode-level neural/video recording, and research ablations remain outstanding.

The present matrix fixes equal environment interactions and paired held-out schedules, but does not meet Task 6's requested model-size matching or comparable tuning-budget target. The integration-v3 manifests report 42,767 trainable parameters for each frozen-LIF condition, 124,975 for the GRU, and 193,583 for the MLP. Report these as distinct-capacity engineering comparisons, and do not attribute a later performance difference solely to circuit topology. A size-matched control and tuning-budget sensitivity study remain separate follow-up experiments.

### User-requested main-matrix pause — 2026-09-14

The main wrapper, queue runner, seed-2 trainer, and guarded regime handoff were stopped at the user's request. A read-only check at 2026-09-13 23:08 UTC found no live runner or trainer PID. The saved queue still serializes seed 2 as `running`, but its log contains only 6,618 of 9,766 updates (3,388,416 observed interactions) and no checkpoint or held-out evaluation. Seeds 0 and 1 retain verified full-budget checkpoints and all three evaluations each; the other 17 jobs have not started. The source-backed main comparison remains incomplete at 2/20. The [pause snapshot](runs/main_matrix_pause_20260913T230933Z/pause.json) preserves the queue, partial comparison, and interrupted seed-2 log as read-only reference files. Restarting the original fingerprint-bound queue with `bash scripts/execute_main_matrix_v3.sh` revalidates the completed jobs and retrains seed 2 from update zero, since a partial training log is not a resumable simulator/optimizer checkpoint. The old guarded handoff is not armed; after the new main wrapper starts, it must be rearmed with that wrapper's live PID or the 60 regime replays run separately only after the main report validates all 20 jobs. See [resume instructions](docs/training.md). No training or evaluation workload should start until the user chooses to continue.

## 16. Memory and VRAM operating plan — user addendum, 2026-09-13

The user's [Thai memory plan](docs/memory_plan_source_th.md) is incorporated here as a resource gate for this project. Its target machine is **Windows 11, 24 GB RAM, RTX 4060 Laptop with 8 GB VRAM**. The inspected development machine for this G1 repository is **Linux, about 24 GB RAM, RTX 5060 Laptop with about 8 GB VRAM**; measurements on one do not establish performance on the other. The installed Isaac Sim 5.1.0/Isaac Lab 0.54.4 are pinned for the present tests. NVIDIA's [latest requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) list 32 GB RAM and 16 GB VRAM as minimum and say Isaac Lab training needs more. Small local G1 pilots ran below those published specifications, but that is not a guarantee for full MaleCNS, other embodiments, or Windows. The reported approximately 3 GB peak GPU allocation in the [Flyhard rate-core pilot](https://github.com/MarkUnthank/flyhard/blob/main/docs/pilot-2026-09-09.md) is not an Isaac/LIF RL memory budget.

**Scope and experimental identity.** This repository remains G1-only; Go2 and drone are separate future embodiments, each in its own process, seed, and declared task. The current 256-neuron leg VNC circuit is an anatomically selected **change in model scope**, not a lossless memory compression of full MaleCNS. Do not silently prune edges or relabel it the full graph. Preserve original neuron-ID to compact-index mapping, raw counts, transmitter provenance, and graph checksum. The current small circuit retains frozen edge records and derives a deterministic dense 256 × 256 recurrent multiply for PPO replay; a full 165k-neuron/25M-edge circuit would require a separately validated CSR implementation with float32 values and tested int32 or int64 indices. Keep one graph shared across environments, use sparse `W @ H` and `W.T @ dY` for a frozen graph, and compare forward and input gradients with a tiny dense reference before using a new backend. Avoid retaining `environments × edges × timesteps` messages. Prepare original and rewired graphs one at a time in a separate process, persist to SSD, then release source DataFrames and temporary representations. [PyTorch sparse matrix multiplication](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html) and the Flyhard pilot are implementation references, not proof that this project's full-graph LIF path fits 8 GB.

For the cited Flyhard pilot dimensions (165,122 neurons, 25,563,197 directed pairs), a **single dense FP32 adjacency** would require about 101.57 GiB. Numeric CSR arrays alone would be about 196 MiB with FP32 values/int32 indices or 294 MiB with int64 indices; a second CSR transpose for backward would bring those arrays to about 391–588 MiB. These are calculated storage sizes, excluding simulator, states, activations, temporary sparse kernels, optimizer, allocator, and host copies. The Flyhard pilot's 3 GB device allocation measured a different rate-based system, not this LIF training workload.

**Resource targets and staged gates.** Treat system RAM use under roughly 80–85%, several GB available RAM, and total device memory use under roughly 85% (about 6.8 GB on an 8 GB card) as initial *measurement targets*, not guaranteed allocations. Run headless with one flat scene, one robot type, no image observation, camera, livestream, or training video. Keep contact geometry, actuator limits, and physics timing required for the locomotion question. For a new target machine or full core, validate 1 environment before 2 and 4; choose a larger count only after stage measurements. The current 16-environment small-subset G1 smoke is evidence only for that particular local workload. Keep DataLoader workers at zero and do not run a notebook, graph preprocessing, and Isaac simultaneously on 24 GB RAM. Keep rollout data in bounded numeric arrays; use chunked SSD reads or memory mapping when appropriate rather than retaining duplicate full tables in RAM. Use the installed Isaac release's supported viewport controls; [latest optimization advice](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html) must not be copied blindly into version 5.1.

Measure system used/available RAM, process RSS, device-wide GPU used/total, and `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` after synchronization at scene load, rollout, backward/update, and checkpoint save, plus warm-up throughput. PyTorch's counters omit Isaac/driver allocations. A separate two-second CSV watcher may warn at 85% but must not silently change settings or terminate a run. Stop scale-up on OOM, non-finite gradients/states, sustained paging, growing memory across updates, unacceptable throughput, or physics changes. Passing memory alone is insufficient: encoder and decoder gradients must be finite and nonzero where expected, frozen core checksums unchanged, and locomotion physics valid. A Windows page file can absorb commit spikes but is not a substitute for RAM.

**Gradient-preserving training.** Freeze core weights/parameters and optimize only the encoder, decoder, and small critic; do not put the core forward pass under `torch.no_grad()` during PPO replay because the encoder needs its input gradient. The collection pass remains gradient-free. Retain observations, actions, rewards, done masks, old log probabilities, and sequence-start neural state in a bounded buffer, then recompute short encoder→core→decoder sequences for backward. The attachment proposes neural microbatch 1 and eight **control decisions** as the starting truncated-BPTT test. Try microbatches 2 and 4, then sequence lengths 16 and 32, only when measured memory and gradient sensitivity permit. These are control decisions, not LIF integration substeps. Detach gradient history at sequence boundaries while preserving numerical neural state and episode reset masks; if burn-in is used, recompute it with current parameters and test sensitivity to sequence and burn-in length. Our current pilot uses horizon 32 and full rollout replay; it has **not** yet implemented microbatch-1/eight-decision PPO or a capped CPU rollout buffer. Those changes are a required design and validation step before scaling to a full graph or claiming the attachment's 8 GB workflow. Activation checkpointing (`use_reentrant=False`) is an optional measured response to activation pressure: pass neural state as explicit tensor arguments, avoid in-place global-state changes, and verify stochastic recomputation. Gradient accumulation can reduce simultaneous update graphs, but neither technique reduces simulator memory or accelerates sample collection. Changing LIF dt/time constants or switching to a rate core changes the model. Start core dynamics/sparse operations in FP32; test mixed precision on adapters separately rather than assuming thresholded LIF is unaffected.

**Checkpointing and recording.** A full-core workflow must store immutable graph data once and reference its checksum and input/output population IDs from checkpoints, instead of duplicating a huge graph per checkpoint. Save adapter/critic/optimizer, normalization, configuration, and RNG state needed for replay. The current small-circuit checkpoints include the core buffers, which is acceptable for their size but does not satisfy the full-core storage design. Keep high-volume per-neuron traces out of default training. Record compact group spike statistics during training; write selected episode windows in bounded chunks to SSD for detailed analysis, including neural-substep spike events where claims require them. For stochastic dynamics, retain the seeds and state needed to verify replay. A control-rate sample is not a complete spike train.

**Execution order.** A: test the pinned Isaac installation, empty headless app, and one simple robot alone with measured memory. B: with Isaac closed, load the intended sparse neural graph and test real encoder/decoder forward/backward at microbatch 1 for 8, 16, and 32 control decisions. C: combine one robot and the neural model, then increase environments only after the memory and numerical gates hold. D: run original/rewired controls, seeds, and other embodiments sequentially with documented hyperparameters, interaction budgets, and truncated-gradient limits. If A fails, changing the neural graph cannot fix Isaac's base cost. If A and B pass separately but C fails, a process-separated rollout/update workflow is a possible engineering experiment, with fresh on-policy rollouts and correct recurrent boundaries. If B fails, a declared locomotor subnetwork may be studied as a different model scope. Do not interpret any of these memory adaptations as evidence of learned locomotion.

**Measured addendum status on the local Linux machine.** `scripts/memory_watch.py` now logs two-second CSV RAM/device-GPU samples with advisory warnings and tolerates missing `nvidia-smi`; `scripts/train.py` records process RSS, system RAM, device-wide GPU snapshots, and synchronized PyTorch CUDA peaks at launch/load, selected rollout/update stages, and checkpoint. An earlier 16-environment, two-update frozen-LIF pilot with the 256-neuron circuit recorded 1,024 environment interactions, at most 2,869/8,151 MiB (35.2%) device-wide GPU in sampled stages, 40.7% system RAM, and 77.12 MiB peak PyTorch allocation during an optimizer stage; [run manifest](runs/memory-lif-smoke/20260913T135457Z/FlyG1-GoalReach-FreePosture-v0/frozen_lif/seed-0/manifest.json). The later 60-update dense-replay run sampled 2,822/8,151 MiB device-wide GPU at its first and last measured optimizer stages, system RAM 41.2% then 40.8%, and PyTorch optimizer-stage allocation peaks 31.13 then 35.56 MiB. The first two completed 5,000,192-interaction main-matrix seeds sampled at most 2,812/8,151 MiB device-wide GPU, 4,072.52 MiB process RSS, and 44.8% system RAM across 201 recorded memory samples per seed; the first seed's [training manifest](runs/main_matrix_malecns_v1_heldout_v3_20260913/training/20260913T150442Z/FlyG1-GoalReach-FreePosture-v0/frozen_lif/seed-0/manifest.json) and [partial comparison](runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.json) retain the source values. These are sampled readings, not guaranteed whole-run peaks or a Windows/full-graph benchmark. The proposed full-core CSR, microbatch-1/eight-decision PPO, CPU buffer cap, Go2/drone embodiments, and full-budget memory growth checks remain unimplemented here.

## 17. Sources

[P] User-supplied fly_robot_proposal_th_go2_drone(4).docx, one-page proposal. Relevant sections: selected robots; encoder -> MaleCNS + LIF -> decoder -> motor control; training without imposed posture; synchronized neural/robot observations; rewiring and RNN comparisons; matched group ablation; simulation-only G1 scope. The document contains no delivered circuit data or runnable repository.

[S1] NVIDIA Isaac Lab, Create new project or task. Checked 2026-09-13. `https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html`

[S2] NVIDIA Isaac Lab, Available Environments. Checked 2026-09-13. `https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html`

[S3a] Isaac Lab upstream G1 flat configuration, main branch, inspected 2026-09-13. `https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py`

[S3b] Isaac Lab upstream G1 rough configuration, main branch, inspected 2026-09-13. `https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`

[S4] OpenAI, Custom instructions with AGENTS.md. Checked 2026-09-13. `https://developers.openai.com/codex/guides/agents-md`

[S5] OpenAI, Using PLANS.md for multi-hour problem solving. Archived cookbook article, dated 2025-10-07; used only for the maintained-plan workflow, not current model or API recommendations. `https://developers.openai.com/cookbook/articles/codex_exec_plans`

The online main branch is a reference, not a reproducible version pin. Task 0 must record the actual versions and commits used locally.

## 18. Full user-supplied RAM/VRAM plan (Thai)

The original user-supplied text is reproduced below. Section 16 states how it applies to this G1-only repository and distinguishes the proposed Windows target from measured local Linux results.

แผนลดการใช้ RAM / VRAM สำหรับโปรเจกต์ MaleCNS → Go2, G1 และโดรน

เครื่องเป้าหมาย: Windows 11, RAM 24 GB, RTX 4060 Laptop 8 GB VRAM
ขอบเขต: ฝึก encoder + decoder โดยคง topology, weights และพารามิเตอร์ของ MaleCNS core ไว้ หากใช้ PPO จะมี value network / critic ขนาดเล็กที่ฝึกเพิ่มด้วย แต่ไม่ใช่การฝึก core
ตรวจเอกสาร: 13 กันยายน 2026
สถานะ: แผนทดลองและเกณฑ์วัด ไม่ใช่ผล benchmark บนเครื่องของผู้ใช้

ข้อจำกัดที่ต้องแยกก่อน

NVIDIA ระบุขั้นต่ำของ Isaac Sim รุ่นปัจจุบันไว้ที่ RAM 32 GB และ VRAM 16 GB และระบุว่า Isaac Lab training ใช้เพิ่มเติม [1] เครื่องนี้จึงต่ำกว่าสเปกที่รองรับ การ optimize เพิ่มโอกาสให้ workload ขนาดเล็กทำงานได้ แต่ไม่รับประกันว่าจะรัน Isaac + full MaleCNS + RL พร้อมกันได้

Flyhard รายงาน peak allocated GPU memory ประมาณ 3 GB ใน steering pilot ที่มี 165,122 neurons และ 25,563,197 neuron-pair edges แต่เป็น rate-based core ที่อัปเดตสี่ครั้งแล้ว reset state ทุก decision ไม่ใช่ long-horizon LIF locomotion RL และไม่ใช่ตัวเลข RAM ของทั้งเครื่อง [2] ห้ามนำ 3 GB ไปใช้เป็นงบที่รับประกันสำหรับงานเรา

1. ค่าเริ่มต้นและเกณฑ์หยุดเพิ่มขนาด

รายการ

ค่าเริ่มต้นที่เสนอ

หุ่นในแต่ละรัน

หนึ่งประเภท / หนึ่ง seed

หุ่นแรก

Go2 บนพื้นเรียบ

Parallel environments

1 ก่อน แล้วค่อยทดสอบ 2 และ 4

ข้อมูลเข้า

joint states, orientation/angular velocity, contacts, task command; ยังไม่ใช้ภาพ

Rendering

Headless, ไม่มี camera/livestream/video ระหว่างฝึก และปิด viewport ที่ไม่ใช้

Neural training microbatch

1 sequence ก่อน แล้วค่อย 2 และ 4

Gradient sequence length

8 control decisions เป็นจุดทดสอบเริ่มต้น ไม่ใช่ 8 LIF integration steps

Precision เริ่มต้น

FP32 สำหรับ dynamics และ sparse core

DataLoader workers

0

เป้าหมาย RAM ทั้งเครื่อง

ไม่เกินประมาณ 80–85% และเหลือ available memory หลาย GB

เป้าหมาย GPU memory ทั้งอุปกรณ์

ไม่เกินประมาณ 85% หรือราว 6.8 GB ของการ์ด 8 GB

ตัวเลขทั้งหมดเป็น ค่าเริ่มทดลอง / เป้าหมายเผื่อพื้นที่ ไม่ใช่การคาดการณ์ว่า Isaac จะกินเท่าไรจริง ต้องตรวจ peak ตอนโหลดฉาก, rollout, backward, optimizer step และ save checkpoint รวมทั้ง throughput หลัง warm-up

2. จัดเก็บกราฟให้เล็กโดยไม่ตัดวงจร

เริ่มจากข้อมูลการเชื่อมต่อระดับคู่เซลล์ ไม่โหลดภาพ EM, skeleton ทุกเซลล์ หรือ mesh สมองเข้ากระบวนการฝึก

แปลงข้อมูลเป็น CSR sparse matrix เพียงครั้งเดียว เก็บเฉพาะค่าการเชื่อมต่อ, column indices และ row pointers พร้อมเก็บตาราง original neuron ID → compact index ไว้แยกกัน ใช้ float32 สำหรับน้ำหนัก และ integer indices ตามที่ backend รองรับ: int32 เมื่อผ่านการทดสอบจริง หรือ int64 เป็นทางเลือก [3]

จากขนาดกราฟ pilot ของ Flyhard คำนวณเองได้ว่า:

Dense FP32 ขนาด 165,122 × 165,122 ต้องใช้ประมาณ 101.57 GiB เฉพาะเมทริกซ์เดียว

CSR แบบ FP32 + int32 indices ใช้ประมาณ 196 MiB; ถ้า indices เป็น int64 ประมาณ 294 MiB

หากเก็บ transpose ใน CSR อีกชุดสำหรับ backward จะรวมประมาณ 391–588 MiB

ตัวเลข CSR นี้เป็นเฉพาะ numeric arrays ไม่รวม temporary buffers, states, autograd, optimizer, framework หรือ simulator

ใช้กราฟชุดเดียวร่วมกับสถานะหลาย environment: W @ H โดย H มีหนึ่งคอลัมน์ต่อ environment ไม่คัดลอก W ตามจำนวน environment และไม่ทำ tensor messages ขนาดทุก edge × ทุก environment × ทุก timestep ค้างไว้

สำหรับ frozen W, backward ของ Y = W @ H ต้องคำนวณ gradient ของ H เป็น W.T @ dY แต่ไม่ต้องคำนวณ gradient ของ W ทดสอบ sparse path กับ dense reference บนกราฟจิ๋วก่อนเสมอ ถ้า backend ยังสร้าง intermediate ใหญ่ ค่อยเขียน custom backward เฉพาะ input gradient ห้ามแอบตัด edges เพื่อให้พอดีโดยไม่บันทึกการเปลี่ยนโมเดล [2,3]

เตรียมกราฟและ randomized controls ทีละตัวใน process แยกจาก Isaac เก็บกราฟสำเร็จไว้บน SSD แล้วปิด process เตรียมข้อมูล หลีกเลี่ยงเก็บทั้ง CSV DataFrame, Python list, COO และ CSR หลายสำเนาพร้อมกัน

3. ฝึก encoder + decoder โดยไม่เผลอตัด gradient

ตั้ง core parameters เป็น requires_grad=False และให้ optimizer รับเฉพาะ encoder, decoder และ critic ที่ใช้จริง ไม่ใส่ graph values เป็น trainable Parameters [4]

ห้ามครอบ core ด้วย torch.no_grad() ในช่วงคำนวณ loss ถ้าต้องการฝึก encoder ผ่านมัน เพราะ gradient ต้องวิ่งกลับผ่าน core ไปถึง encoder การ freeze parameters กับการหยุด autograd เป็นคนละเรื่อง [4]

แบ่งวงจรการฝึกเป็นสองช่วง:

เก็บ rollout: ไม่เก็บ autograd graph; บันทึก observations, actions, rewards, done masks, old log-probabilities และข้อมูลเริ่ม sequence ที่จำเป็นลง CPU buffer ขนาดจำกัด

อัปเดตโมเดล: นำ sequence สั้นกลับมาคำนวณ encoder → core → decoder ใหม่ แล้ว backward; ไม่เก็บ graph ของทั้ง episode

ใช้ truncated BPTT เป็นวิธีประมาณเริ่มต้น: detach gradient history เมื่อข้ามขอบ sequence แต่ คงค่า neural state ไว้ในการจำลอง ไม่ reset state ทุก control step เพียงเพื่อประหยัดหน่วยความจำ ใช้ recurrent-aware PPO batching และ episode masks ให้ถูกต้อง หากใช้ burn-in ให้คำนวณด้วย parameters ปัจจุบัน และทดสอบ sensitivity ของผลต่อความยาว sequence/burn-in

เริ่ม microbatch = 1 และ gradient sequence = 8 control decisions แล้วลอง 16/32 เมื่อพื้นที่พอ ตัวเลขนี้ต้องแยกจากจำนวน integration substeps ภายใน LIF การลด simulation timestep resolution หรือเปลี่ยน time constant เพื่อให้เร็วขึ้นคือการเปลี่ยน dynamics ไม่ใช่ optimization ที่เทียบเท่าเดิม

ใช้ activation checkpointing เมื่อ memory ของ activations เป็นปัญหาจริง: use_reentrant=False, ส่ง state เป็น tensor arguments ชัดเจน, หลีกเลี่ยงแก้ global state แบบ in-place และตรวจ randomness ให้การคำนวณซ้ำตรงกัน วิธีนี้แลก compute เพิ่มกับ memory ลด [5]

Gradient accumulation ช่วยสะสม gradients จาก microbatch โดยไม่เก็บ computation graphs ไว้พร้อมกัน แต่ไม่ได้ลด memory ของ simulator หรือ rollout buffer และไม่ได้ทำให้เก็บ environment samples ได้เร็วขึ้น

เริ่ม core/dynamics เป็น FP32 ก่อน ทดลอง mixed precision เฉพาะ encoder/decoder ภายหลังเมื่อ gradient และพฤติกรรมผ่านการตรวจ ไม่ใช้ INT8/FP16 ทั้ง core เป็นค่าเริ่มต้น เพราะผลต่อ firing thresholds และ numerical stability ต้องวัดแยก

หากใช้ LIF แบบ hard spike ต้องใช้วิธีฝึกผ่าน spike ที่กำหนดไว้ เช่น surrogate gradient; การใส่ sparse matrix ลง PyTorch อย่างเดียวไม่ได้ทำให้ threshold กลายเป็น differentiable อย่าสลับไป rate-based core แล้วเรียกว่า LIF โดยไม่แจ้ง

4. ลดภาระ Isaac โดยไม่บิดโจทย์ locomotion

ใช้หนึ่ง scene เรียบและหุ่นหนึ่งประเภทต่อ process ปิด GUI/cameras/video ระหว่าง training ปิด unused viewport ด้วยวิธีที่ตรงกับ Isaac Sim/Isaac Lab release ที่ติดตั้ง การตั้ง headless อย่างเดียวอาจยังมี default viewport work ใน standalone workflow [6]

ไม่โหลดห้องสมจริง แสง/texture จำนวนมาก ROS bridge, lidar หรือ asset ที่ไม่ใช้ รักษา collision geometry, contacts, actuator limits และ physics settings ที่มีผลกับ gait ไว้ อย่าลดความละเอียด physics จนเกิดการลื่นหรือกระเด้งผิดจริงเพียงเพื่อให้เร็ว

เริ่มจาก environment ที่มีใน Isaac Lab [7]:

Go2: Isaac-Velocity-Flat-Unitree-Go2-v0

G1: Isaac-Velocity-Flat-G1-v0

โดรน Crazyflie: Isaac-Quadcopter-Direct-v0

ใช้ environment เดิมเพื่อตรวจ installation/memory/control interface ก่อน ไม่ใช่ถือว่า reward เดิมเป็น emergence experiment การทดลองเดินเทียบคลานของ G1 ต้องตรวจและแก้ posture/contact rewards, termination, action space และ reset distribution โดยเฉพาะ ไม่ใช้การล็อกแขนหรือ controller เดินสำเร็จรูปแล้วสรุปว่า agent เลือกเดินเอง

Pin เวอร์ชัน Isaac Sim, Isaac Lab, Python, PyTorch และ CUDA ที่เข้ากันและผ่านการทดสอบ ไม่อัปเดตแพ็กเกจทั้งหมดแยกกันเพื่อหวังให้เร็วขึ้น ไม่ใช้ flag จากเอกสาร latest โดยไม่ตรวจว่า release ที่ติดตั้งรองรับ

5. RAM 24 GB: อย่าใช้หมดกับข้อมูลประกอบ

ใช้ num_workers=0 ก่อน เพราะ worker processes อาจเพิ่มสำเนาข้อมูลฝั่ง CPU [8] ไม่เปิด notebook training, viewer, graph preprocessing และ Isaac หลายตัวพร้อมกัน เก็บ dataset/rollout เป็น numeric arrays และอ่านเป็นชิ้นจาก SSD ด้วย memory mapping เมื่อเหมาะสม [9]

ไม่ย้าย activations ทั้งหมดจาก GPU ไป CPU โดยอัตโนมัติ เพราะแก้ VRAM เต็มแต่ไปทำให้ RAM 24 GB เต็มแทน ต้องกำหนด buffer cap ชัดเจน เก็บเฉพาะ sequence ที่จำเป็นและทิ้ง tensors หลังจบ update

Checkpoint ให้แยกกราฟคงที่หนึ่งไฟล์กับ encoder/decoder/critic/optimizer/normalizers/config/RNG state เก็บ graph checksum และ input/output population IDs ไว้ด้วย เพื่อให้โหลดกลับมาแล้วเป็นวงจรเดิม ไม่บันทึกกราฟเต็มซ้ำทุก checkpoint

เปิด Windows page file แบบ system-managed บน SSD ที่มีพื้นที่ว่างพอเพื่อรองรับ commit spikes แต่ไม่ถือว่า page file แทน RAM จริงได้ ถ้ามี paging ต่อเนื่องและ throughput ตก ให้ลด workload [10]

6. บันทึก neural activity แบบที่ยังทำวิจัยได้

อย่าเก็บ activation ทุกเซลล์ทุก timestep ลง Python list ระหว่าง train

ตัวอย่างคำนวณ: 165,122 เซลล์ × 50 samples/s × 600 s × 4 bytes ≈ 18.45 GiB สำหรับตัวแปร FP32 เพียงตัวเดียวและ environment เดียว ถ้าเก็บทั้ง voltage, synaptic state และตัวแปรอื่น ปริมาณยิ่งเพิ่ม

ระหว่าง training เก็บ reward, body state และสถิติกลุ่มเซลล์ เช่น mean/variance/activity counts โดยคำนวณสรุปเป็นช่วง ๆ ส่วนการวิเคราะห์ละเอียดให้โหลด checkpoint แล้วรัน evaluation episode แยกต่างหาก เก็บทุกเซลล์เฉพาะช่วงที่วางแผนไว้ เขียนเป็น chunks ลง SSD ด้วย buffer จำกัดและ detach ก่อนเก็บ

สำหรับโมเดล spiking ให้บันทึก spike counts/events ที่ temporal resolution เหมาะสม ไม่สุ่มอ่าน voltage ที่ 20 Hz แล้วอ้างว่าเห็น spikes ครบ หากมี noise ต้องเก็บ seed/state ที่จำเป็นและตรวจการ replay แทนการสมมุติว่าเหมือนเดิมเสมอ

7. ลำดับทดสอบที่แยกสาเหตุได้

A — Isaac อย่างเดียว: Compatibility Checker → empty headless app → Go2 หนึ่งตัวกับ policy เล็กหรือคำสั่งทดสอบ วัด RAM/VRAM หลัง warm-up และขณะ save/record

B — Neural model อย่างเดียว: ปิด Isaac; โหลด full sparse core, encoder/decoder, microbatch 1 และทดสอบ forward/backward จริง วัดว่า encoder/decoder gradients finite และไม่เป็นศูนย์ทั้งหมด ขณะที่ core ไม่เปลี่ยน ทดสอบหลาย sequence lengths

C — รวมระบบ: หนึ่ง Go2, ไม่มีภาพ, หนึ่ง environment, gradient sequence สั้น เมื่อใช้ memory ต่ำกว่าเกณฑ์และไม่เพิ่มต่อเนื่อง จึงลอง 2 และ 4 environments ทีละระดับ ไม่เปิดทั้งสามร่างพร้อมกัน

D — ขยายงาน: ทำ randomized controls, seeds และ embodiments ทีละรัน ใช้สเปกการฝึกเหมือนกันในคู่เปรียบเทียบ และรายงานข้อจำกัดของ truncated gradients

หาก A ไม่ผ่าน แม้ scene เล็กแล้ว การลดขนาด neural core ไม่ช่วยแก้ base cost ของ Isaac ต้องใช้เครื่องอื่นสำหรับ simulator หรือประเมิน simulator ที่เบากว่าเป็นแผนสำรองอย่างเปิดเผย

หาก A กับ B ผ่านแยกกันแต่ C ไม่ผ่าน ลอง workflow แบบสลับ process: เก็บ rollout ภายใต้ current policy โดยไม่ทำ backward → ปิด process Isaac → อัปเดต policy จาก rollout นั้น → เปิด simulator เก็บ rollout ใหม่ การ restart มี overhead และ PPO ห้ามใช้ rollout เก่าวนไม่จำกัดเมื่อ policy เปลี่ยน หาก core เป็น recurrent ต้องรักษา episode boundaries และ reconstruction ของ state ให้ถูกต้อง

หาก B ไม่ผ่านเต็มกราฟ ให้เลือก locomotor subnetwork ด้วยเกณฑ์ทางกายวิภาคล่วงหน้า แล้วทำ matched controls บน subnetwork เดียวกัน ชัดเจนว่านี่คือ เปลี่ยนขอบเขตโมเดล ไม่ใช่การบีบอัดที่คง full MaleCNS ทุกอย่าง

8. วัดอะไรและใช้อะไรตัดสิน

สคริปต์ memory_watch.py ที่แนบเป็นตัวอ่าน RAM ทั้งเครื่องและ GPU memory ผ่าน nvidia-smi ทุกสองวินาที บันทึก CSV และเตือนที่ 85% โดยไม่แก้ settings และไม่หยุด process ให้อัตโนมัติ [11]

python -m pip install psutil
python memory_watch.py --output go2_memory.csv

ดู GPU Dedicated memory และ RAM/Available ใน Windows Task Manager ควบคู่กัน บน Windows/driver บางชุด GPU query อาจรายงาน N/A ให้ใช้ Task Manager แทน

ใน process training ให้เก็บ torch.cuda.max_memory_allocated() และ torch.cuda.max_memory_reserved() หลัง synchronize ในแต่ละขั้น ทั้งสองค่าไม่รวม memory ที่ Isaac/driver จัดสรรนอก PyTorch และ empty_cache() ไม่ได้ลบ live tensors หรือทำให้โมเดลที่ใหญ่เกินไปพอดีทันที [12]

เกณฑ์ผ่านต้องมีทั้ง memory เหลือ, ไม่เกิด NaN/OOM, encoder ได้ gradient, core weights ไม่เปลี่ยน, throughput ใช้งานได้ และการจำลองยังรักษา physics/dynamics ตามที่ประกาศ การรันได้อย่างเดียวไม่เท่ากับฝึกให้ locomotion สำเร็จ

การทดสอบไฟล์แนบ: ตรวจ syntax และทดสอบ RAM logging/กรณีไม่มี nvidia-smi ในสภาพแวดล้อม Linux แบบ CPU เท่านั้น ยังไม่ได้ทดสอบบน Windows 11, RTX 4060, MaleCNS หรือ Isaac ของผู้ใช้

แหล่งอ้างอิง

[1] NVIDIA Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

[2] Flyhard pilot 2026-09-09, รายงานของผู้พัฒนา: https://github.com/MarkUnthank/flyhard/blob/main/docs/pilot-2026-09-09.md

[3] PyTorch sparse matrix multiplication: https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html

[4] PyTorch autograd / freezing parameters: https://docs.pytorch.org/docs/2.14/notes/autograd.html

[5] PyTorch activation checkpointing: https://docs.pytorch.org/docs/2.14/checkpoint.html

[6] NVIDIA performance optimization handbook: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html

[7] Isaac Lab available environments: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html

[8] PyTorch DataLoader: https://docs.pytorch.org/docs/2.14/data.html

[9] NumPy memory-mapped arrays: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

[10] Microsoft page file documentation: https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file

[11] NVIDIA SMI: https://docs.nvidia.com/deploy/nvidia-smi/index.html

[12] PyTorch CUDA memory management: https://docs.pytorch.org/docs/2.14/notes/cuda.html
