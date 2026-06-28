# Pre-Registered Validation Criteria

**Date**: 2026-06-27 (before multi-model data is collected)
**Status**: Pre-registered. No multi-model behavioral data exists yet. Only OpenVLA
has produced clean live rollouts (preliminary within-model r=0.93, n=5).

## Context

We are validating a non-circular behavioral PatchMask action-shift probe against
VLA-Trace's white-box delta_sr. The validation claim is cross-model: "our cheap
behavioral probe ranks models the way the expensive white-box method does."

This document defines what counts as success *before* the multi-model data comes in,
so we cannot rationalize a weak result into a win or a confounded result into a finding.

## Models

| Model | Architecture | VLA-Trace Ground Truth |
|-------|-------------|----------------------|
| OpenVLA | Autoregressive VLM | Table 4 (14 conditions) |
| Pi0.5 | Flow-matching | Table 4 (14 conditions) |
| OpenVLA-OFT | Autoregressive + action head | Table 4 (14 conditions) |
| X-VLA | LeRobot / transformer | Table 4 (14 conditions) |

## Success Criteria

### Primary: Pooled Cross-Model Correlation

- **Pass**: Pooled Pearson r > 0.70 (action_shift vs delta_sr, all models x conditions)
- **Strong pass**: Pooled Pearson r > 0.80

Rationale: r > 0.70 means ~50% shared variance between our behavioral probe
and the white-box metric. Below 0.70 the probe does not meaningfully track the
white-box signal across models.

### Secondary: Within-Model Consistency

- **Pass**: Pearson r > 0.50 for each individual model (within-model, across conditions)
- **Acceptable exception**: One model may fail this if the other three pass

Rationale: within-model correlation tests whether our probe correctly ranks mask
conditions for a given model, not just "general model goodness."

### Required: Model Spread

- **Pass**: At least 3 models with distinct delta_sr profiles
- **Distinct** means: for any pair of models, at least 2 of their 5 condition-level
  delta_sr values differ by more than 5 percentage points
- If models cluster on the white-box metric, even a high correlation is uninformative

### Required: Sanity Gate

- All models included in the correlation must pass the sanity gate:
  - Visual: mask overlap with primary target >= 80% of target pixels
  - Action: normalized target shift > normalized background shift

### Interpretation Rules

- **Validated probe (floor)**: pooled r > 0.70, within-model holds, models spread,
  all sanity gates pass. This is a methods contribution.
- **Dissociation finding (ceiling)**: a model has benchmark SR > 80% but action_shift
  in the bottom quartile across conditions. Flag for deeper investigation as potential
  evidence of replay/shortcut behavior.
- **Honest negative**: pooled r < 0.70 cross-model, even if single-model r is high.
  Reportable as "probe works within-model but does not generalize cross-model."
- **Confounded**: models do not spread on white-box metric. Result is uninterpretable
  regardless of correlation value.

## Dissociation Hunt

Once multiple models produce clean rollouts, actively check:

- Does any model show high benchmark success (SR > 80%) but low sensitivity to
  target masking (action_shift for target_black in bottom quartile)?
- This would suggest the model succeeds without actually using the target object's
  visual features -- potential replay or shortcut behavior.

This is the higher-ceiling parallel goal. The validation (probe agrees with
white-box) is the floor; a dissociation finding is what makes this more than
a workshop note.

## What This Document Protects Against

1. Post-hoc rationalization of a weak cross-model correlation
2. Treating a single-model result as validation (it is not)
3. Ignoring model spread (high r on clustered models is meaningless)
4. Confusing within-model ranking with cross-model validation
