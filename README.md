# ReadTeam — Replay vs Grounding Detector for VLAs

Measures whether vision-language-action (VLA) policies **replay memorized motion** or **ground on object position** when the target is perturbed. Two black-box probes:

1. **Position probe** — perturb object position, regress grasp-error vs displacement
2. **Occlusion probe (PatchMask)** — mask visual regions, measure action shift

Tested on **OpenVLA** (`openvla-7b-finetuned-libero-spatial`, ~7B) and **Pi0.5** (`pi05_libero`, ~3.3B) in LIBERO-Spatial with controlled object-position sweeps on Modal.

## Final verdict

**OUTCOME 3 — Structural limitation.** The position probe (grasp-error-vs-displacement) resolves replay only at ±20 cm (replayer R²=0.43). At that range:
- **Pi0.5** is flat/grounded (slope=−0.14, n=56 successes)
- **OpenVLA** success collapses to 9% (n=9) — unmeasurable

No range simultaneously resolves replay AND yields enough OpenVLA successes for comparison. See [FINAL_VERDICT.md](FINAL_VERDICT.md).

### Earlier narrow-range results (replicated across 2 tasks)

| Task | Success gap (Pi0.5 − OpenVLA) | Grounding Δ (matched band) |
|------|--------------------------------|------------------------------|
| task_id=0 (bowl between plate & ramekin) | +40.3% | +0.183 |
| task_id=2 (bowl from table center) | +72.2% | +0.301 |

See [COMPARISON.md](COMPARISON.md) and [COMPARISON_task2.md](COMPARISON_task2.md).

## Repository layout

```
ReadTeam/
├── modal_perturbation_sweep.py       # OpenVLA rollout sweeps (Modal, ±7–20 cm grid)
├── modal_pi05_sweep.py               # Pi0.5 rollout sweeps (Modal)
├── run_wide_position_sweep.py        # Orchestrate wide-range ±10/15/20 cm sweeps
├── libero_task_config.py             # Task presets + wide sweep path helpers
├── analyze_position_grasperror.py    # Grasp-error vs displacement (best metric)
├── analyze_control_grasperror.py     # Control diagnostic (replayer/oracle validation)
├── analyze_wide_position.py          # Wide-range analysis → FINAL_VERDICT.md
├── analyze_*.py                      # Other grasp regression / trajectory metrics
├── make_controls.py                  # Synthetic gain-sweep controls
├── validate_detector.py              # Calibrate detector (replayer→0, oracle→1)
├── compare_models.py                 # Score + compare models; replication table
├── recompute_action_sensitivity.py   # Velocity-L2 action sensitivity (weak metric)
├── modal_patchmask_probe.py          # PatchMask occlusion probe (Modal)
├── run_live_validation.py            # Cross-model PatchMask validation
├── *_rollouts.csv                    # Rollout summaries per model/task/range
├── *_trajectories/                   # EE trajectory JSON per rollout
├── controls/ controls_task2/         # Synthetic validation data
├── *.png                             # Analysis plots
├── FINAL_VERDICT.md                  # Wide-range position probe conclusion
├── POSITION_PROBE_VERDICT.md         # Narrow-range grasp-error re-analysis
├── CONTROL_DIAGNOSTIC.md             # Replayer/oracle metric validation
├── PREREGISTRATION.md                # Pre-registered PatchMask success criteria
├── FINDINGS.md                       # OpenVLA task 0 writeup
├── VALIDATION.md                     # Detector calibration (task 0)
├── COMPARISON.md                     # OpenVLA vs Pi0.5 (task 0)
└── COMPARISON_task2.md               # Replication on task 2
```

**Not in git:** `.venv/`, vendored clones (`LIBERO/`, `openvla/`, `VLA-Trace/`, `RoboSemanticBench/`, `vla-evaluation-harness/`), and PatchMask frame images.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install modal numpy matplotlib scipy
```

Modal account required for rollout collection. Weights are cached on Modal Volumes at first run.

## Collect rollouts

```bash
# Narrow sweeps (±7 cm, 5×5 × 3 repeats = 75 rollouts)
./.venv/bin/modal run modal_perturbation_sweep.py          # OpenVLA task 0
./.venv/bin/modal run modal_pi05_sweep.py                  # Pi0.5 task 0
./.venv/bin/modal run modal_perturbation_sweep.py --task-id 2
./.venv/bin/modal run modal_pi05_sweep.py --task-id 2

# Wide sweeps (±10/15/20 cm, 5×5 × 5 repeats = 125 rollouts each)
./.venv/bin/python run_wide_position_sweep.py              # all 6 runs
./.venv/bin/python run_wide_position_sweep.py --models pi05 --half-widths 0.20
```

## Analyze

```bash
# Position probe (narrow + wide)
./.venv/bin/python analyze_position_grasperror.py          # narrow ±7 cm
./.venv/bin/python analyze_control_grasperror.py           # metric validation
./.venv/bin/python analyze_wide_position.py                # wide → FINAL_VERDICT.md

# Detector calibration
./.venv/bin/python make_controls.py
./.venv/bin/python validate_detector.py

# Compare models (narrow range)
./.venv/bin/python compare_models.py
./.venv/bin/python compare_models.py --task-id 2
```

## Metrics

### Position probe: grasp-error vs displacement (primary)
- **perturbation_distance** = ‖object_xy − grid_center‖
- **grasp_error** = ‖grasp_xy − object_xy‖
- Regress grasp_error on perturbation_distance (successes only); **lower slope = more grounded**
- Validated: synthetic replayer shows rising error (slope≈0.81), oracle stays flat

### Narrow-range grounding score
- `clip(mean(|slope_x|, |slope_y|), 0, 1)` anchored to synthetic controls (g=0 replayer, g=1 oracle)

## Experiment design

- **Suite:** LIBERO-Spatial, single target object (`akita_black_bowl_1`) per task
- **Perturbation:** override object xy before policy runs; reject if override error > 1.5 cm
- **Grasp point:** minimum-z end-effector position in recorded trajectory
- **Ranges:** narrow ±7 cm (3 repeats) and wide ±10/15/20 cm (5 repeats)
- **Replication:** only `task_id` and grid center move; same models, grid size, seeds, and detector
