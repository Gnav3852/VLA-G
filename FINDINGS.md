# Controlled-Perturbation Pilot Findings

## Setup

- Input CSV: `spike2_rollouts.csv`
- Trajectories: `spike2_trajectories/rollout_*.json`
- Grasp definition: EE `[x, y, z]` at the global minimum-z timestep (earliest if tied)
- Valid rollouts used in regression: 27
- Successful rollouts among valid rows: 2/27
- Skipped rollouts: 9

## Position Spread

| Axis | Wide pilot object span | Narrow spike object span |
|------|------------------------|--------------------------|
| x | 30.17 cm | 5.29 cm |
| y | 30.02 cm | 2.96 cm |

## Regression: grasp_point ~ actual_object_init

| Axis | Slope | R2 | Pearson r | Tracking ratio |
|------|-------|----|-----------|----------------|
| x | -0.1782 | 0.1179 | -0.3433 | 0.6260 |
| y | 0.0676 | 0.0900 | 0.3000 | 0.4199 |

## Read

x: grasp barely tracks object (slope=-0.18, R2=0.12); replay/shortcut signal present.
y: grasp barely tracks object (slope=0.07, R2=0.09); replay/shortcut signal present.

Compared to the narrow spike (`x`: slope=0.54, R2=0.53; `y`: slope=0.24, R2=0.09), this wide perturbation run is the first real test of whether the detector separates grounding from replay outside the natural 3-5 cm variation.

## Honesty Notes

- This is still a pilot, not proof.
- Regression uses all valid completed rollouts because only 2 succeeded; a successes-only fit would be too underpowered for this pilot.
- If many extreme perturbations fail or clip into scene objects, interpret the usable range rather than the commanded range.

---

---

---

---

---

---

# Tight-Grid Phase 1 Findings (spike3)

## Setup

- Input CSV: `spike3_rollouts.csv`
- Trajectories: `spike3_trajectories/rollout_*.json`
- Grid: 5×5 over ±7 cm of center `(-0.06, 0.20)`, 3 repeats per cell → 75 commanded
- Valid / rejected / successes: 62 / 13 / 16
- Skipped in analysis: 13

## Success vs perturbation distance

Usable range (max distance with any success): 9.89 cm

| bin | n | successes | rate |
|-----|---|-----------|------|
| 0-3cm | 3 | 2 | 0.667 |
| 3-6cm | 19 | 8 | 0.421 |
| 6-10cm | 39 | 6 | 0.154 |
| 10cm+ | 1 | 0 | 0.000 |

## Trajectory similarity (success-independent)

- Metric: mean per-step Euclidean distance after linear resampling to 100 points
- Pairwise r (object dist vs traj dist): 0.0637
- Mean / median traj distance: 9.00 / 9.17 cm
- Failed-pair mean traj dist: 7.24 cm (r=0.089)

NOTE: the naive metric is confounded — repeats vary `base_idx` (robot/scene start), so
pairs differ for reasons unrelated to object position. The controlled metric below fixes this.

## Trajectory similarity — base_idx-controlled

Splits pairs to isolate the object's effect from robot-start noise:

| split | meaning | n pairs | r (obj dist vs traj dist) | mean traj dist |
|-------|---------|---------|---------------------------|----------------|
| same robot-start, diff object | object-driven (clean grounding test) | 611 | 0.0844 | 8.57 cm |
| same position, diff robot-start | non-object baseline (noise floor) | 55 | 0.0345 | 8.48 cm |

- Object-driven mean MINUS baseline mean (lift) = +0.09 cm (positive => object position adds trajectory variation beyond robot-start noise)
- Object-driven successes-only: n=40, r=0.5774

Plots: `spike3_success_vs_distance.png`, `spike3_traj_similarity_vs_object_distance.png`, `spike3_traj_controlled.png`

## Failed-trajectory overlay (robot start held fixed)

Direct visual + dispersion test of the "replay-like failure" claim. Tube width = mean
per-timestep spread of EE paths around their centroid, pooled within fixed `base_idx`
(so robot-start variance is excluded).

| group | within-start tube width |
|-------|-------------------------|
| failed | 4.83 cm |
| succeeded | 3.95 cm |

Read: failed tube WIDER than succeeded => failures scatter more than successes (mild breakage), NOT a tight canned replay.
This supersedes the confounded both-failed pairwise comparison above.
In the plot: star = object init, X = grasp point, color = object perturbation distance.
Visual: centroid paths are near-identical across positions/starts (strong canned reaching backbone);
object stars span y~0.13-0.35 while grasp X's stay compressed (~0.10-0.20) => partial, under-reaching tracking.

Plot: `spike3_failed_overlay.png`

## Grasp regression: grasp_point ~ actual_object_init

| subset | n | x slope/R²/r | y slope/R²/r | x tracking ratio | y tracking ratio |
|--------|---|--------------|--------------|------------------|------------------|
| all valid | 62 | 0.2931 / 0.0596 / 0.2441 | 0.1656 / 0.1588 / 0.3985 | 1.7393 | 0.7359 |
| successes only | 16 | 0.5520 / 0.2659 / 0.5157 | 0.4604 / 0.5627 / 0.7501 | 1.1672 | 0.7277 |

Object span (all valid): x=14.47 cm, y=14.90 cm

Plot: `spike3_grasp_vs_object.png`

## Interpretation matrix

| Signal | Pattern in spike3 | Read |
|--------|-------------------|------|
| Controlled traj (object-driven) | r>0, lift>0 over baseline | grounding (object changes path) |
| Controlled traj (object-driven) | r≈0, lift≈0 | replay (path fixed vs object) |
| Controlled traj (object-driven) | high mean, r≈0, lift large | residual breakage |
| Grasp regression | slope≈1, high R² on successes | grounding |
| Grasp regression | 0<slope<1, moderate R² | partial/compressed grounding |
| Grasp regression | low slope/R² | replay or no tracking |
| Tube width | failed < succeeded | replay-like failure |
| Tube width | failed > succeeded | failures scatter (breakage) |
| Naive traj (all pairs) | confounded by base_idx | use controlled / tube instead |

## Combined verdict

VERDICT: partial/local grounding + failures scatter (mild breakage) — Controlled trajectory test (robot start fixed): object-driven r=0.084, object-driven mean=8.6cm vs non-object baseline=8.5cm (lift=+0.1cm). On SUCCESSES, grasp PARTIALLY tracks object (y slope=0.46 <1, R²=0.56, r=0.75) — it shifts toward the object but under-reaches, on top of a strong shared canned reaching backbone (centroid paths near-identical across positions and starts). The controlled successes-only traj r=0.58 is partly tautological (a grasp ends at the object). Across ALL valid rollouts object adds ~nothing beyond robot-start noise (lift=+0.1cm). Within-start tube width is WIDER for failures than successes (4.8cm > 3.9cm) — failures scatter more, leaning mild breakage rather than tight replay. (The earlier both-failed pairwise mean 7.2cm < baseline 8.5cm was confounded by mixing positions; the within-start tube width supersedes it.) Net: a dominant canned reach with partial object tracking; out-of-band cases fail by scatter, not by a tighter replay. Power note: 16/62 successes (13 rejected); controlled object-driven pairs n=611, baseline pairs n=55. Success-only regression usable but modest.

## Comparison to wide pilot (spike2)

Wide pilot: 2/27 successes, usable range ~8 cm, verdict was OOD-breakage-dominated (mass failure outside envelope).
Tight grid moves into the empirically measured envelope; 16 successes enable both metrics side by side.

## Honesty notes

- Still a pilot on one task (`libero_spatial` task_id=0) and one model (OpenVLA).
- LIBERO-PRO reports OpenVLA position-perturbation success ≈0 on libero-spatial; 16 successes here reflects the narrower band, not contradiction.
- The successes-only controlled trajectory correlation (r=0.58) is partly tautological: a successful grasp trajectory must end at the object, so far-apart objects force far-apart trajectories. Weight the grasp regression and the tube-width test more than this single number.
- CORRECTION: an earlier reading called failures "replay-like" because both-failed pairwise mean (7.2 cm) fell below the same-position baseline (8.5 cm). That comparison was confounded (it mixed different object positions). The clean within-start tube-width test (robot start fixed) shows failures are WIDER than successes (4.8 vs 3.9 cm) — so failures scatter, they do not collapse into a tight canned replay.
- Grasp tracking on successes is PARTIAL/compressed (slope ~0.46 <1): the gripper shifts toward the object but under-reaches, riding a strong shared canned reaching backbone (see overlay centroid paths).
- Across all valid rollouts the object adds ~0 trajectory variation beyond robot-start noise (lift=+0.1 cm); the (partial) grounding signal lives in the success subset.
- Phase 2 (OpenVLA vs Pi0.5) deferred until this read is reviewed.
