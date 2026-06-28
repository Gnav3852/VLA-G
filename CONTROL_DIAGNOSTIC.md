# Control Diagnostic: Grasp-Error vs Displacement

Synthetic replayer (g=0, fixed trajectory) and oracle (g=1) from `make_controls.py`, compared to real OpenVLA / Pi0.5 successes.

## Detection thresholds

- Replayer **works**: slope ≥ 0.2, 95% CI excludes 0, R² ≥ 0.3
- Oracle **flat**: |slope| ≤ 0.15

## Primary verdict (task 0)

**METRIC WORKS ONLY AT WIDE RANGE: replayer detection fails at ±7cm (R² below threshold) but succeeds at ±20cm. → the real-model range was too narrow; real models must be re-tested at wider displacement before any conclusion. Detector is fine, experiment range was off.**

Narrow replayer: slope=0.550, R²=0.090, CI=[0.214, 0.939]
Wide replayer: slope=0.809, R²=0.376, CI=[0.703, 0.922]
Narrow oracle (make_controls g=1): slope=-0.000, mean error=3.9cm (flat vs displacement)
Wide ideal oracle: slope=0.000, mean error=0.0cm
At ±7cm: positive slope (0.55) but R²=0.09 (below 0.3 bar) — not a reliable detection.

## Task 0 (source rollout 6)

### Narrow ±7 cm

| Policy | n | slope | R² | 95% CI | mean error |
|--------|---|-------|-----|--------|------------|
| Replayer g=0 | 62 | 0.550 | 0.090 | [0.214, 0.939] | 10.8 cm |
| Oracle g=1 | 62 | -0.000 | -5.640 | [-0.000, 0.000] | 3.9 cm |
| OpenVLA | 16 | 0.123 | 0.013 | [-0.351, 0.946] | 5.2 cm |
| Pi0.5 | 41 | 0.038 | 0.001 | [-0.195, 0.359] | 4.8 cm |

### Wide ±20 cm (synthetic)

| Policy | n | slope | R² | 95% CI | mean error |
|--------|---|-------|-----|--------|------------|
| Replayer | 289 | 0.809 | 0.376 | [0.703, 0.922] | 18.1 cm |
| Oracle (ideal) | 289 | 0.000 | nan | [0.000, 0.000] | 0.0 cm |

## Task 2 (source rollout 66)

### Narrow ±7 cm

| Policy | n | slope | R² | 95% CI | mean error |
|--------|---|-------|-----|--------|------------|
| Replayer g=0 | 72 | 0.412 | 0.054 | [0.072, 0.758] | 10.8 cm |
| Oracle g=1 | 72 | -0.000 | -8.047 | [-0.000, 0.000] | 3.8 cm |
| OpenVLA | 20 | 0.084 | 0.019 | [-0.180, 0.488] | 4.8 cm |
| Pi0.5 | 72 | -0.086 | 0.013 | [-0.326, 0.059] | 4.0 cm |

### Wide ±20 cm (synthetic)

| Policy | n | slope | R² | 95% CI | mean error |
|--------|---|-------|-----|--------|------------|
| Replayer | 289 | 0.826 | 0.406 | [0.720, 0.930] | 18.0 cm |
| Oracle (ideal) | 289 | 0.000 | nan | [0.000, 0.000] | 0.0 cm |

Task 2 verdict: **METRIC WORKS ONLY AT WIDE RANGE: replayer detection fails at ±7cm (R² below threshold) but succeeds at ±20cm. → the real-model range was too narrow; real models must be re-tested at wider displacement before any conclusion. Detector is fine, experiment range was off.**

## Implications

Real models at ±7 cm show flat grasp-error vs displacement (R²<0.02). This diagnostic determines whether that flatness is a metric artifact or a real finding.

- **If metric works (or works at wide range):** flat real models may genuinely ground position.
- **If metric broken:** real-model flatness tells us nothing; probe needs redesign.

## Caveats
- Narrow oracle = make_controls trajectory shift (constant offset, flat slope).
- Wide oracle = idealized grasp-at-object (not min-z EE).
- Wide replayer uses fixed grasp from source rollout min-z point.