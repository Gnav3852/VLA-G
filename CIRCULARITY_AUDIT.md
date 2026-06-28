# Circularity Audit: Proxy PatchMask Behavioral Scores

## Verdict

**CIRCULAR: the proxy behavioral score is derived from VLA-Trace ΔSR. The reported Pearson 0.74 / Spearman 0.88 are artifacts. The real correlation is unknown until live measurement.**

Those numbers are **RETRACTED** and must not be cited as validation evidence.

---

## 1. How the proxy score is computed

Source: [`patchmask_behavioral.py`](patchmask_behavioral.py), called by [`collect_patchmask_behavioral.py`](collect_patchmask_behavioral.py).

```python
wb = vtg.patchmask_conditions()          # loads delta_sr from vla_trace_numbers.json
spatial = [r for r in wb if r["suite"] == "libero_spatial"]
...
rel = row["delta_sr"] / denom              # denom = max delta_sr for that model
score = model_cals[model] * rel            # model_cals from position sensitivity JSON
```

Where:
- `row["delta_sr"]` comes from `vla_trace_ground_truth.patchmask_conditions()`, which computes `baseline_sr - masked_sr` from **VLA-Trace Table 4** in [`vla_trace_numbers.json`](vla_trace_numbers.json).
- `model_cals[model]` is a **single scalar per model** from position-sensitivity recompute (OpenVLA / Pi0.5), independent of mask type.
- `denom` is the maximum ΔSR for that model across libero_spatial conditions.

**Closed form per condition:**

\[
\text{proxy\_score}(m, c) = S_m \cdot \frac{\Delta\text{SR}_{m,c}}{\max_{c'} \Delta\text{SR}_{m,c'}}
\]

where \(S_m\) = position sensitivity for model \(m\), \(\Delta\text{SR}_{m,c}\) = VLA-Trace white-box success drop for condition \(c\).

---

## 2. Does the proxy use white-box ΔSR as input?

**Yes, directly.** `delta_sr` from `vla_trace_numbers.json` is a multiplicative factor in every proxy score. Position sensitivity only sets the per-model scale \(S_m\); it does **not** vary by mask type.

---

## 3. Correlation axes

Source: [`per_condition_correlation.py`](per_condition_correlation.py) + [`run_validation_correlation.py`](run_validation_correlation.py).

| Axis | Quantity | Source |
|------|----------|--------|
| **X (ground truth)** | `delta_sr` = baseline_sr − masked_sr | VLA-Trace Table 4 via `vla_trace_ground_truth.py` |
| **Y (behavioral)** | `action_sensitivity` from `patchmask_behavioral_scores.json` | Proxy = \(S_m \times \Delta\text{SR} / \max\Delta\text{SR}\) |

**Shared term:** Y contains X (ΔSR) as a direct multiplicative factor.

For a fixed model \(m\):

\[
Y_c = k_m \cdot X_c \quad\text{with}\quad k_m = S_m / \max_{c'} X_{m,c'}
\]

That is a **linear pass-through** (zero intercept). Pearson and Spearman correlation between \(X\) and \(Y\) within a model are **1.0 by construction**.

Pooling across models with different \(k_m\) still yields very high correlation because Y is proportional to X with only modest slope differences. The observed ρ ≈ 0.88 is exactly what a constructed \(Y \propto X\) relationship produces — not independent confirmation.

---

## 4. What the live measurement must do differently

The live PatchMask behavioral score must:

1. **Run the model** on LIBERO with unmasked vs masked images (VLA-Trace PatchMask protocol: instance seg → mask region → black / bg-fill / mosaic).
2. **Record actions** from our rollouts only: at each timestep (same physical state), compute \(\|a_\text{unmasked} - a_\text{masked}\|\).
3. **Never read** `vla_trace_numbers.json`, `delta_sr`, or any VLA-Trace white-box number when computing Y.
4. Correlate **independently measured** action shift (Y) against VLA-Trace ΔSR (X).

Both measure “how much does masking matter,” but at different levels (action vs success). Agreement would be non-trivial evidence; the proxy agreement was tautological.

---

## 5. Position-only component

Position sensitivity \(S_m\) itself is **not** circular with ΔSR — it comes from object-position perturbation trajectories. But it enters the proxy only as a per-model constant multiplier, so it cannot rescue the per-condition correlation: mask-type variation in Y comes entirely from ΔSR.

---

## Summary

| Question | Answer |
|----------|--------|
| Proxy uses ΔSR? | **Yes** |
| Y shares term with X? | **Yes** — \(Y \propto X\) per model |
| 0.88 stands? | **No — retracted** |
| Next step | Live masked/unmasked action-shift measurement |
