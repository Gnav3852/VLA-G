# OpenVLA vs Pi0.5 — Replay vs Grounding Comparison

Same perturbation grid as spike3: 5×5 over ±7 cm, 3 repeats per cell (75 commanded).
The 13 rejected positions are model-independent (override check fires before the policy
runs), so both models share an identical 62-position valid set.

## Summary table (raw successes-only grounding score)

| model | n_valid | n_success | success_rate | grounding (succ) | replay (succ) | slope_x | slope_y | lift (cm) | failed tube | succ tube |
|-------|---------|-----------|--------------|------------------|---------------|---------|---------|-----------|-------------|-----------|
| OpenVLA (spike3) | 62 | 16 | 25.8% | 0.506 | 0.494 | 0.552 | 0.460 | 0.09 | 4.83 | 3.95 |
| Pi0.5 | 62 | 41 | 66.1% | 0.709 | 0.291 | 0.841 | 0.577 | 4.41 | 7.77 | 4.00 |

## Matched object-position band (removes successes-on-different-subsets confound)

Raw successes-only grounding is computed over *each model's own* successes, which occupy
different object-position bands (OpenVLA's successes are sparser and miss the far-y row).
The matched-band recompute fixes this:

- **Shared-cell intersection:** the 11 grid cells where BOTH models have
  ≥1 success. Each model's grasp-slope is recomputed over its own successes within those
  identical cells — like-for-like.
- **Bounding box (OpenVLA (spike3) successes):** x∈[-0.134, 0.009],
  y∈[0.130, 0.235]; robustness check against high-y leverage points.

| model | n (shared cells) | grounding (matched) | slope_x | slope_y | n (bbox) | grounding (bbox) |
|-------|------------------|---------------------|---------|---------|----------|------------------|
| OpenVLA (spike3) | 16 | 0.506 | 0.552 | 0.460 | 16 | 0.506 |
| Pi0.5 | 28 | 0.689 | 0.826 | 0.552 | 28 | 0.732 |

## Read
- **Success-rate gap (confound-free):** Pi0.5 66.1% vs OpenVLA 25.8% = +40.3% on the identical 62-position valid set.
- **Grounding delta (raw successes-only, different subsets):** +0.203 (Pi0.5 0.709 vs OpenVLA 0.506).
- **Grounding delta (matched shared-cell band):** +0.183 (Pi0.5 0.689 vs OpenVLA 0.506) over 11 identical cells — **this is the defensible number**.
- **Grounding delta (matched bbox, robustness):** +0.226 (Pi0.5 0.732 vs OpenVLA 0.506).

## Scale definition
- `grounding_score = clip(mean(|slope_x|, |slope_y|), 0, 1)` on successes-only grasp regression
- `replay_score = 1 - grounding_score`
- Anchored by synthetic controls (see [VALIDATION.md](VALIDATION.md)): 0 = replayer, 1 = oracle

## Notes
- Lead with the **success-rate gap** — it's pure counting on the identical 62-position set, fully confound-free.
- The **matched shared-cell grounding delta** is the defensible grounding headline; the raw delta over-credits
  Pi0.5 by measuring it over a wider object band than OpenVLA.
- **Controlled lift** (all-valid, already matched-band) and **tube width** corroborate; grasp slope is the calibrated primary metric.
- All-valid grounding scores are lower when failures dominate (non-tracking grasps dilute the slope).
