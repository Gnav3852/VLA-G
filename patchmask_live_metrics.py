"""Live PatchMask action-shift metrics (independent of VLA-Trace white-box)."""

from __future__ import annotations

from typing import Any

import numpy as np


def per_step_shifts(
    actions_clean: list[np.ndarray],
    actions_masked: list[np.ndarray],
) -> tuple[list[float], list[float]]:
    """Raw and magnitude-normalized L2 shifts per aligned timestep."""
    n = min(len(actions_clean), len(actions_masked))
    raw: list[float] = []
    norm: list[float] = []
    for i in range(n):
        a = np.asarray(actions_clean[i], dtype=np.float64).ravel()
        b = np.asarray(actions_masked[i], dtype=np.float64).ravel()
        d = float(np.linalg.norm(a - b))
        raw.append(d)
        norm.append(d / (float(np.linalg.norm(a)) + 1e-6))
    return raw, norm


def aggregate_shifts(raw: list[float], norm: list[float]) -> dict[str, Any]:
    if not raw:
        return {
            "raw_l2_shift": float("nan"),
            "normalized_l2_shift": float("nan"),
            "n_steps": 0,
        }
    return {
        "raw_l2_shift": float(np.mean(raw)),
        "normalized_l2_shift": float(np.mean(norm)),
        "raw_l2_std": float(np.std(raw)),
        "n_steps": len(raw),
    }


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 4:
        return {"lo": float("nan"), "hi": float("nan")}
    stats = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        stats.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    if not stats:
        return {"lo": float("nan"), "hi": float("nan")}
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return {"lo": lo, "hi": hi}
