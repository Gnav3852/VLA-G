# Final Verdict: Wide-Range Position Probe

Last run: OpenVLA + Pi0.5 position sweeps at ±10/15/20 cm (task 0). Metric validated on synthetic replayer (R²≥0.30 required at resolving range).

## **OUTCOME 3 — STRUCTURAL LIMITATION: discriminating range and success range do not overlap**

- ±10cm: OpenVLA success=20% (n_succ=21), Pi0.5 success=76% (n_succ=79), replayer R²=0.10
- ±15cm: OpenVLA success=11% (n_succ=11), Pi0.5 success=59% (n_succ=61), replayer R²=0.27
- ±20cm: OpenVLA success=9% (n_succ=9), Pi0.5 success=54% (n_succ=56), replayer R²=0.43
- Metric-resolving range ±20cm: replayer slope=0.81, R²=0.43.
- OpenVLA at ±20cm: only 9 successes (<10) — not measurable where metric resolves.
- Pi0.5 at ±20cm: slope=-0.140, R²=0.027, n=56 (grounded if |slope|<0.40).
- Pi0.5 is flat/grounded at ±20cm (the only metric-resolving range).
- OpenVLA success collapses to 9% (n=9) at ±20cm — cannot test it where the metric works.
- Cross-model position-grounding comparison is blocked by a catch-radius/OOD bind.

## Success rates and slopes (report successes first)

### ±10 cm

| Policy | n_valid | success rate | n_success | slope | R² | 95% CI |
|--------|---------|--------------|-----------|-------|-----|--------|
| Replayer | 25 | 100% | 25 | 0.514 | 0.095 | [-0.04, 1.12] |
| Oracle | 25 | 100% | 25 | 0.000 | nan | [0.00, 0.00] |
| openvla | 104 | 20% | 21 | 1.196 | 0.199 | [-0.09, 2.64] |
| pi05 | 104 | 76% | 79 | 0.225 | 0.020 | [0.03, 0.46] |

- 0-5cm: n=16, success=50%
- 5-10cm: n=29, success=28%
- 10-15cm: n=59, success=8%

### ±15 cm

| Policy | n_valid | success rate | n_success | slope | R² | 95% CI |
|--------|---------|--------------|-----------|-------|-----|--------|
| Replayer | 25 | 100% | 25 | 0.714 | 0.269 | [0.31, 1.21] |
| Oracle | 25 | 100% | 25 | 0.000 | nan | [0.00, 0.00] |
| openvla | 103 | 11% | 11 | 1.329 | 0.443 | [0.22, 2.47] |
| pi05 | 103 | 59% | 61 | 0.106 | 0.011 | [-0.02, 0.26] |

- 0-5cm: n=5, success=80%
- 5-10cm: n=20, success=20%
- 10-15cm: n=25, success=8%
- 15-20cm: n=37, success=3%
- 20-21cm: n=16, success=0%

### ±20 cm

| Policy | n_valid | success rate | n_success | slope | R² | 95% CI |
|--------|---------|--------------|-----------|-------|-----|--------|
| Replayer | 25 | 100% | 25 | 0.809 | 0.432 | [0.48, 1.22] |
| Oracle | 25 | 100% | 25 | 0.000 | nan | [0.00, 0.00] |
| openvla | 103 | 9% | 9 | 0.604 | 0.119 | [-0.32, 2.67] |
| pi05 | 103 | 54% | 56 | -0.140 | 0.027 | [-0.36, -0.01] |

- 0-5cm: n=5, success=80%
- 5-10cm: n=7, success=14%
- 10-15cm: n=22, success=14%
- 15-20cm: n=8, success=0%
- 20-29cm: n=61, success=2%

## Overlap diagnostic

A range **resolves** if replayer R²≥0.3 AND model has ≥10 successes.

- ±20cm: pi05 n=56, slope=-0.140

## Pre-registration note

PREREGISTRATION.md targets PatchMask cross-model validation (≥3 models, pooled r>0.70). This position probe uses **two models only** — interpret as exploratory, not a registered pass/fail.

## Conclusion

This is the project's closing experiment on position grounding.

**What we learned:** The grasp-error-vs-displacement metric resolves replay only at ±20cm (replayer R²≥0.3). At ±7–15cm the metric is in its dead zone.

**Pi0.5:** At ±20cm (n=56 successes), grasp-error slope is flat (≈−0.14, R²=0.03) while replayer rises (slope≈0.81) — Pi0.5 grounds position on this task.

**OpenVLA:** Success rate falls to 9% at ±20cm (n=9), below the minimum for slope inference. OpenVLA cannot be tested at the only range where the metric works. Conditional-on-success slopes at ±10–15cm are confounded by survivorship bias (successes concentrate near center).

**Cross-model:** No range simultaneously (a) resolves replay and (b) yields ≥10 OpenVLA successes. A position-grounding gap between OpenVLA and Pi0.5 cannot be established with this probe.