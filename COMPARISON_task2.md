# OpenVLA vs Pi0.5 — Replay vs Grounding Comparison (task_id=2)

**Task:** `Pick the akita black bowl from table center and place it on the plate`

Same perturbation grid: 5×5 over ±7 cm centered at (-0.075, 0.000), 3 repeats per cell (75 commanded).
Override rejections are model-independent (check fires before policy runs).

## Summary table (raw successes-only grounding score)

| model | n_valid | n_success | success_rate | grounding (succ) | replay (succ) | slope_x | slope_y | lift (cm) | failed tube | succ tube |
|-------|---------|-----------|--------------|------------------|---------------|---------|---------|-----------|-------------|-----------|
| OpenVLA (task 2) | 72 | 20 | 27.8% | 0.577 | 0.423 | 0.684 | 0.471 | -3.51 | 7.26 | 3.30 |
| Pi0.5 | 72 | 72 | 100.0% | 0.829 | 0.171 | 0.872 | 0.786 | 3.81 | nan | 4.34 |

## Matched object-position band (removes successes-on-different-subsets confound)

- **Shared-cell intersection:** the 13 grid cells where BOTH models have ≥1 success.
- **Bounding box (OpenVLA (task 2) successes):** x∈[-0.144, -0.004],
  y∈[-0.070, 0.070].

| model | n (shared cells) | grounding (matched) | slope_x | slope_y | n (bbox) | grounding (bbox) |
|-------|------------------|---------------------|---------|---------|----------|------------------|
| OpenVLA (task 2) | 20 | 0.577 | 0.684 | 0.471 | 20 | 0.577 |
| Pi0.5 | 36 | 0.878 | 0.830 | 0.926 | 66 | 0.855 |

## Read
- **Success-rate gap (confound-free):** Pi0.5 100.0% vs OpenVLA 27.8% = +72.2% on the identical valid set.
- **Grounding delta (raw successes-only, different subsets):** +0.252 (Pi0.5 0.829 vs OpenVLA 0.577).
- **Grounding delta (matched shared-cell band):** +0.301 (Pi0.5 0.878 vs OpenVLA 0.577) over 13 identical cells — **this is the defensible number**.
- **Grounding delta (matched bbox, robustness):** +0.278 (Pi0.5 0.855 vs OpenVLA 0.577).

## Replication verdict (vs task_id=0)

| metric | task_0 | task_2 | replicates? |
|--------|--------|--------|-------------|
| success gap | +40.3% | +72.2% | yes |
| grounding delta (matched) | +0.183 | +0.301 | yes |
| lift Pi0.5 / OpenVLA | 4.41 / 0.09 | 3.81 / -3.51 | direction |
| x-slope Pi0.5 / OpenVLA | 0.826 / 0.552 | 0.830 / 0.684 | direction |

**Verdict: REPLICATES** — Same direction on both success rate and matched-band grounding (Pi0.5 > OpenVLA).

OpenVLA n_success=20, Pi0.5 n_success=72 (task_0: 16 / 41).

## Scale definition
- `grounding_score = clip(mean(|slope_x|, |slope_y|), 0, 1)` on successes-only grasp regression
- `replay_score = 1 - grounding_score`
- Anchored by synthetic controls (see VALIDATION.md / controls_task2/)

## Notes
- Lead with the **success-rate gap** — pure counting on the identical valid set.
- **Matched shared-cell grounding delta** is the defensible grounding headline.
- Two tasks = replicated finding, not generalization claim.

## Plain-English summary

On task_id=2 (Pick the akita black bowl from table center and place it on the plate), the Pi0.5 vs OpenVLA gap **replicates** task_0: success gap +72.2% (task_0 +40.3%), matched grounding delta +0.301 (task_0 +0.183). This is replication across two LIBERO-Spatial tasks, not a claim of generalization.
