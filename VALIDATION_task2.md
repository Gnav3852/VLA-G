# Detector Validation (synthetic gain sweep)

## Goal
Validate that grasp-vs-object regression (mean |slope| on x,y) recovers known tracking gain
from synthetic controls before scoring real models.

## Control design
- **Positive control (g=0):** recorded successful spike3 trajectory (rollout 6) replayed verbatim; only object position varies.
- **Oracle (g=1):** same backbone with grasp displaced 1:1 toward each object (ramped xy shift to grasp step).
- **Sweep:** g ∈ {0, 0.25, 0.5, 0.75, 1.0} × {clean, noisy (σ=0.5 cm/step)} over the 62 valid spike3 object positions.

## Pass/fail criteria
- g=0: recovered |gain| < 0.1, lift < 0.5 cm
- g=1: recovered gain ∈ [0.85, 1.15]
- Sweep (clean): linear fit recovered vs true gain has R² > 0.95 and slope ∈ [0.85, 1.15]

## Result: **PASS** — Detector recovers synthetic gain across the sweep (calibration R²=1.0000, slope=1.000). Replayer g=0 recovered 0.000; oracle g=1 recovered 1.000.

## Calibration table (clean)

| true g | recovered | slope_x | slope_y | lift (cm) | traj r | n_success |
|--------|-----------|---------|---------|-----------|--------|-----------|
| 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.00 | nan | 2 |
| 0.25 | 0.2500 | 0.2500 | 0.2500 | 1.43 | 1.0000 | 4 |
| 0.50 | 0.5000 | 0.5000 | 0.5000 | 2.86 | 1.0000 | 4 |
| 0.75 | 0.7500 | 0.7500 | 0.7500 | 4.29 | 1.0000 | 5 |
| 1.00 | 1.0000 | 1.0000 | 1.0000 | 5.73 | 1.0000 | 0 |

## Calibration table (noisy)

| true g | recovered | lift (cm) | traj r |
|--------|-----------|-----------|--------|
| 0.00 | 0.0082 | -0.00 | 0.0636 |
| 0.25 | 0.2498 | 0.93 | 0.9942 |
| 0.50 | 0.5035 | 2.29 | 0.9987 |
| 0.75 | 0.7503 | 3.68 | 0.9996 |
| 1.00 | 0.9948 | 5.10 | 0.9997 |

## Replay-score definition
- `grounding_score = clip(mean(|slope_x|, |slope_y|), 0, 1)` — 0 = replay, 1 = oracle
- `replay_score = 1 - grounding_score`

## OpenVLA placement (spike3)
**Primary (successes only, n=16):**
- slope_x = 0.6835, slope_y = 0.4705
- tracking_gain = 0.5770
- OpenVLA spike3 (successes only, n=16): grounding score = **0.577** (replay score = 0.423) — partial/compressed grounding on a validated 0–1 scale.
- All-valid (n=62): grounding = 0.390, replay = 0.610 (diluted by non-tracking failures).

**All valid (n=62):** slope_x = -0.5125, slope_y = 0.2673, tracking_gain = 0.3899

## Honesty notes
- Successes-only controlled traj-r is partly tautological; **grasp slope is the primary calibration anchor**.
- Synthetic controls validate metric math, not all real-world failure modes.
- Noisy controls inject per-step xy noise only; they do not simulate physics or vision.

Plot: `control_calibration_task2.png`
