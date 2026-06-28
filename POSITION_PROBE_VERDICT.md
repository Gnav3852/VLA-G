# Position Probe Verdict: Grasp-Error vs Displacement

Re-analysis of existing position perturbation sweeps (OpenVLA + Pi0.5, tasks 0 and 2).
No new rollouts. Grasp point = min-z EE position from logged CSV (multi-grasp not audited).

## Metric

- **perturbation_distance** = distance from grid center to actual object XY
- **grasp_error** = XY distance from grasp point to true object position
- **Regression (successes):** grasp_error ~ perturbation_distance; **lower slope = more grounded**

## Pooled results (successes, both tasks)

| Model | n | slope | 95% CI | R² | calibrated grounding (0=replay, 1=grounded) |
|-------|---|-------|--------|-----|------------------------------------------|
| openvla | 36 | 0.106 | [-0.18, 0.52] | 0.015 | 0.808 |
| pi05 | 113 | -0.049 | [-0.21, 0.08] | 0.003 | 1.000 |

## Separation comparison (OpenVLA vs Pi0.5)

| Metric | Δ (OpenVLA − Pi0.5 magnitude) | Separates better than velocity-L2? | vs visual_dep (0.303)? | vs attn_mass (0.045)? |
|--------|------------------------------|-------------------------------------|-------------------------|----------------------|
| **Grasp-error slope** | **0.155** | — | — | — |
| Velocity-L2 (success pairs) | 0.076 | baseline | — | — |
| VLA-Trace visual_dep | 0.303 | — | white-box ref | — |
| VLA-Trace attn_mass | 0.045 | — | — | white-box ref |

- Grasp-error slope Δ **>** velocity-L2 success Δ (0.155 vs 0.076)
- Grasp-error slope Δ **≤** VLA-Trace visual_dep Δ (0.155 vs 0.303)
- Grasp-error slope Δ **>** VLA-Trace attn_mass Δ (0.155 vs 0.045)

## Per-axis (pooled successes)

- **openvla**: x slope=0.332 (R²=0.129), y slope=0.003 (R²=0.000)
- **pi05**: x slope=0.049 (R²=0.008), y slope=-0.064 (R²=0.007)
- Separation is driven mainly by **x-axis** on task 2 (OpenVLA x-slope ~0.10 vs Pi0.5 ~0.04); **y-axis** remains noisy (small y perturbation span on task 0).

## Catch-radius context

### Task 0
**openvla** (success rate & grasp error vs perturbation):
  - 0-3cm: n=3, success=67%, mean grasp error (success)=4.8cm, frac error>3cm=100%
  - 3-6cm: n=19, success=42%, mean grasp error (success)=4.9cm, frac error>3cm=100%
  - 6-10cm: n=35, success=17%, mean grasp error (success)=5.8cm, frac error>3cm=83%
  - 10-15cm: n=5, success=0%, mean grasp error (success)=nancm, frac error>3cm=nan%
**pi05** (success rate & grasp error vs perturbation):
  - 0-3cm: n=3, success=100%, mean grasp error (success)=4.7cm, frac error>3cm=100%
  - 3-6cm: n=19, success=79%, mean grasp error (success)=4.4cm, frac error>3cm=87%
  - 6-10cm: n=35, success=54%, mean grasp error (success)=5.2cm, frac error>3cm=63%
  - 10-15cm: n=5, success=80%, mean grasp error (success)=3.8cm, frac error>3cm=50%

### Task 2
**openvla** (success rate & grasp error vs perturbation):
  - 0-3cm: n=3, success=67%, mean grasp error (success)=4.6cm, frac error>3cm=100%
  - 3-6cm: n=24, success=46%, mean grasp error (success)=4.6cm, frac error>3cm=100%
  - 6-10cm: n=45, success=16%, mean grasp error (success)=5.2cm, frac error>3cm=86%
**pi05** (success rate & grasp error vs perturbation):
  - 0-3cm: n=3, success=100%, mean grasp error (success)=3.9cm, frac error>3cm=100%
  - 3-6cm: n=24, success=100%, mean grasp error (success)=4.4cm, frac error>3cm=88%
  - 6-10cm: n=45, success=100%, mean grasp error (success)=3.8cm, frac error>3cm=76%


## Per-task regression

### Task 0
- **openvla**: n_success=16, slope=0.123, R²=0.013, x-axis slope=0.605, y-axis slope=-0.259
- **pi05**: n_success=41, slope=0.038, R²=0.001, x-axis slope=0.069, y-axis slope=0.071
  - task slope Δ=0.085, velocity-L2 success Δ=0.043

### Task 2
- **openvla**: n_success=20, slope=0.084, R²=0.019, x-axis slope=0.096, y-axis slope=0.214
- **pi05**: n_success=72, slope=-0.086, R²=0.013, x-axis slope=0.043, y-axis slope=-0.144
  - task slope Δ=0.170, velocity-L2 success Δ=0.108


## Honest verdict

**INCONCLUSIVE: grasp-error metric beats velocity-L2 on Δ but regressions are not significant**

Pooled point-estimate slope Δ=0.155 vs velocity-L2 success Δ=0.076 (2× better).
OpenVLA slope=0.106 (R²=0.015, n=36), Pi0.5 slope=-0.049 (R²=0.003, n=113); lower slope = more grounded.
95% bootstrap CIs overlap=True (OpenVLA [-0.18, 0.52], Pi0.5 [-0.21, 0.08]).
Neither model shows a reliable grasp_error ~ displacement trend (R²<0.05; CIs include 0 and overlap).
The cross-model slope gap may reflect different success subsets / baseline error, not verified grounding.
→ Before scaling: (1) audit grasp point (multi-grasp), (2) try x-axis-only metric (task 2 shows weak OpenVLA x-tracking), (3) bootstrap slope-difference test on pooled data.

## Caveats
- Grasp point uses min-z heuristic; multi-grasp contamination not audited.
- Success-only regressions have uneven n (OpenVLA task 0 has few successes).
- Control calibration uses task-0 controls only for pooled grounding score.
- Catch radius reference: 3 cm.