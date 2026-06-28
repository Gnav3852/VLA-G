# Validation: behavioral action-sensitivity vs VLA-Trace white-box

> **RETRACTED (Probe C per-condition numbers):** The PatchMask correlation in §Per-condition below used **circular proxy** scores (`position_sensitivity × ΔSR`). Pearson 0.74 / Spearman 0.88 are **invalid** — see [CIRCULARITY_AUDIT.md](CIRCULARITY_AUDIT.md). Real numbers: [LIVE_VALIDATION.md](LIVE_VALIDATION.md).

## Per-model position sensitivity (Probe A)

| model | raw sensitivity | calibrated 0-1 | VLA-Trace visual_dep | attn mass | target_bg drop |
|-------|-----------------|----------------|----------------------|-----------|----------------|
| openvla | 0.1878 | 1.000 | 0.687 | 0.588 | 47.8 |
| pi05 | 0.1321 | 0.832 | 0.990 | 0.633 | 76.5 |

**Separation test:** raw Δ=0.0558 cal Δ=0.168 vs VLA-Trace visual_dep Δ=0.303 (attn_mass Δ=0.045).

- Action sensitivity separates more than visual_dep: **False**
- Action sensitivity separates more than attn mass: **True**

## Per-condition PatchMask correlation (Probe C)

- N aligned conditions: 56
- Pearson r: 0.7434
- R²: 0.5526
- Spearman ρ: 0.8785
- Partial r (control baseline SR): 0.8853

## Per-model rank sanity check

- Models: openvla, pi05
- Spearman ρ (position vs target_bg drop): nan

## Interpretation

Action-sensitivity did **not** clearly outperform white-box separation on pi0.5 vs OpenVLA; interpret per-condition results cautiously.
Per-condition PatchMask correlation is **positive** (r=0.74): larger VLA-Trace SR drops align with larger behavioral action shifts. **This claim is retracted** — the proxy was circular; see CIRCULARITY_AUDIT.md.
