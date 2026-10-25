"""Normalize fixed-panel paired IMP, preserving seed/board dependence in bootstrap."""

import numpy as np


def rate(candidate, random_anchor, oracle_anchor, *, draws=2000, seed=20261026, anchor=None):
    """Inputs are panel-weighted IMP arrays, shape (replicate seeds, shared boards).

    All three arrays must use the same board/seed coordinates. Resample the same
    board columns and seed rows jointly, including both estimated anchors.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in (candidate, random_anchor, oracle_anchor)]
    if (arrays[0].ndim != 2 or min(arrays[0].shape) < 1 or
            any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays)):
        raise ValueError("Aligned finite nonempty seed-by-board arrays required")
    if type(draws) is not int or draws < 200:
        raise ValueError("At least 200 bootstrap draws required")
    if anchor not in (None, "random", "oracle"):
        raise ValueError("Unknown anchor")
    means = [float(a.mean()) for a in arrays]
    gap = means[2] - means[1]
    if gap <= 0:
        raise ValueError("DDS/random calibration gap must be positive; expand or revise the panel")
    rng = np.random.default_rng(seed)
    samples, gaps, improvements = [], [], []
    rows, columns = arrays[0].shape
    for _ in range(draws):
        ri = rng.integers(rows, size=rows)
        ci = rng.integers(columns, size=columns)
        values = [float(a[np.ix_(ri, ci)].mean()) for a in arrays]
        denominator = values[2] - values[1]
        gaps.append(denominator)
        improvements.append(values[0] - values[1])
        if denominator > 0:
            samples.append(100. * (values[0]-values[1])/denominator)
    gap_ci = np.quantile(gaps, [.025, .975]).tolist()
    reliable = gap_ci[0] > 0 and len(samples) == draws
    point = 0. if anchor == "random" else 100. if anchor == "oracle" else 100.*(means[0]-means[1])/gap
    interval = ([point, point] if anchor else np.quantile(samples, [.025, .975]).tolist()) if reliable else None
    return {"rating": point, "rating_95_ci": interval,
            "status": "calibrated" if reliable else "insufficient_anchor_separation",
            "panel_mean_imps": means[0], "random_panel_mean_imps": means[1],
            "oracle_panel_mean_imps": means[2], "anchor_gap_imps": gap, "anchor_gap_95_ci": gap_ci,
            "gain_over_random_95_ci": np.quantile(improvements, [.025, .975]).tolist(),
            "boards": columns, "random_replicates": rows, "bootstrap_draws": draws,
            "bootstrap": "joint shared board columns and replicate seed rows; percentile 95%; estimated anchors included",
            "interval_scope": "fixed policies and benchmark distribution, not training-seed or human-level uncertainty",
            "clipped": False}
