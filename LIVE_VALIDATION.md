# Live PatchMask Validation

**Proxy correlation (Pearson ~0.74 / Spearman ~0.88) is RETRACTED** — see [CIRCULARITY_AUDIT.md](CIRCULARITY_AUDIT.md).

This report uses **live** paired masked/unmasked action-shift scores only.

## Data
- Source: `patchmask_behavioral_scores_live.json`
- Models: openvla, pi05
- Aligned conditions: 5

## Pooled correlation (action shift vs VLA-Trace ΔSR)

| Metric | n | Pearson r | 95% CI | Spearman ρ | small-n |
|--------|---|-----------|--------|------------|---------|
| raw L2 shift | 5 | 0.932 | [0.747, 1.000] | 0.900 | yes |
| normalized L2 shift | 5 | 0.944 | [0.501, 1.000] | 0.900 | yes |

## Within-model correlation (stronger claim)

- **openvla** (n=5): Pearson r=0.932, Spearman ρ=0.900

## Target vs background mask contrast (black)

- **openvla**: target=0.2704, background=0.3039, ratio=0.89, target>background=False
- **pi05**: missing target or background condition

## Magnitude sensitivity

- Raw vs normalized shift Pearson: 0.992
- High raw-vs-normalized agreement suggests metric tracks relative shift, not just action magnitude. If raw and normalized correlations with ΔSR diverge, prefer normalized for grounding claims.

## Honest read

Only one model has passing live scores; pooled correlation is exploratory (n=5). Models blocked: pi05.
