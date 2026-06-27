# ReadTeam — Replay vs Grounding Detector for VLAs

Measures whether vision-language-action (VLA) policies **replay memorized motion** or **ground on object position** when the target is perturbed. Scores policies on a controls-validated **0–1 grounding scale** (0 = canned replayer, 1 = oracle reacher).

Tested on **OpenVLA** (`openvla-7b-finetuned-libero-spatial`, ~7B) and **Pi0.5** (`pi05_libero`, ~3.3B) in LIBERO-Spatial with controlled object-position sweeps on Modal.

## Headline results (replicated across 2 tasks)

| Task | Success gap (Pi0.5 − OpenVLA) | Grounding Δ (matched band) |
|------|--------------------------------|------------------------------|
| task_id=0 (bowl between plate & ramekin) | +40.3% | +0.183 |
| task_id=2 (bowl from table center) | +72.2% | +0.301 |

Pi0.5 grounds more and succeeds more on both tasks. See [COMPARISON.md](COMPARISON.md) and [COMPARISON_task2.md](COMPARISON_task2.md).

## Repository layout

```
ReadTeam/
├── modal_perturbation_sweep.py   # OpenVLA rollout sweeps (Modal)
├── modal_pi05_sweep.py           # Pi0.5 rollout sweeps (Modal)
├── libero_task_config.py         # Task presets (grid center, output paths)
├── analyze_*.py                  # Grasp regression, trajectory metrics
├── make_controls.py              # Synthetic gain-sweep controls
├── validate_detector.py          # Calibrate detector (replayer→0, oracle→1)
├── compare_models.py             # Score + compare models; replication table
├── *_rollouts.csv                # Rollout summaries per model/task
├── *_trajectories/               # EE trajectory JSON per rollout (tracked)
├── controls/                     # Task 0 synthetic validation data
├── controls_task2/               # Task 2 synthetic validation data
├── *.png                         # Analysis plots (tracked)
├── FINDINGS.md                   # OpenVLA task 0 writeup
├── VALIDATION.md                 # Detector calibration (task 0)
├── COMPARISON.md                 # OpenVLA vs Pi0.5 (task 0)
└── COMPARISON_task2.md           # Replication on task 2
```

**Not in git:** `.venv/`, and vendored clones `LIBERO/`, `openvla/`, `vla-evaluation-harness/` (used by Modal images, not versioned here).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install modal numpy matplotlib scipy
```

Modal account required for rollout collection. Weights are cached on Modal Volumes at first run.

## Collect rollouts

```bash
# Task 0 (default)
./.venv/bin/modal run modal_perturbation_sweep.py --test
./.venv/bin/modal run modal_perturbation_sweep.py

./.venv/bin/modal run modal_pi05_sweep.py --test
./.venv/bin/modal run modal_pi05_sweep.py

# Task 2 replication
./.venv/bin/modal run modal_perturbation_sweep.py --task-id 2
./.venv/bin/modal run modal_pi05_sweep.py --task-id 2
```

Each full sweep: 5×5 grid ±7 cm, 3 repeats → 75 commanded rollouts.

Outputs: `spike3_rollouts.csv` / `pi05_rollouts.csv` (task 0) or `openvla_task2_rollouts.csv` / `pi05_task2_rollouts.csv` (task 2), plus matching `*_trajectories/` dirs.

## Analyze

```bash
# Validate detector on synthetic controls
./.venv/bin/python make_controls.py
./.venv/bin/python validate_detector.py

# Task 2 controls + validation
./.venv/bin/python make_controls.py --task-id 2
./.venv/bin/python validate_detector.py --task-id 2

# Compare models
./.venv/bin/python compare_models.py              # task 0
./.venv/bin/python compare_models.py --task-id 2  # task 2 + replication verdict
```

## Metric

- **Grounding score** = `clip(mean(|slope_x|, |slope_y|), 0, 1)` from grasp-point vs object-position regression on successes
- **Replay score** = `1 − grounding_score`
- **Controlled lift** and **tube width** corroborate; grasp slope is the calibrated primary metric
- **Matched-band recompute** restricts comparison to grid cells where both models have ≥1 success

Anchors: synthetic controls where known gain g ∈ {0, 0.25, …, 1.0} must recover g (see [VALIDATION.md](VALIDATION.md)).

## Experiment design

- **Suite:** LIBERO-Spatial, single target object (`akita_black_bowl_1`) per task
- **Perturbation:** override object xy before policy runs; reject if override error > 1.5 cm
- **Grasp point:** minimum-z end-effector position in recorded trajectory
- **Change one thing for replication:** only `task_id` and grid center move; same models, grid size, seeds, and detector
