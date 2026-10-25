"""Unbatched decision timing and validation-only search configuration selection."""

import math
import time

import numpy as np


class TimedPlayPolicy:
    """Measure observation-to-action time, including device/RPC synchronization."""
    def __init__(self, policy):
        self.policy = policy
        self.seconds, self.forced = [], []

    def predict(self, observation):
        start = time.perf_counter()
        result = self.policy.predict(observation)
        elapsed = time.perf_counter()-start
        if result.get("state_id") != observation["state_id"] or result["card_id"] not in observation["legal_card_ids"]:
            raise ValueError("Benchmark policy returned a stale or illegal decision")
        self.seconds.append(elapsed)
        self.forced.append(len(observation["legal_card_ids"]) == 1)
        return result

    def predict_batch(self, observations):
        return [self.predict(observation) for observation in observations]

    def metrics(self):
        values = np.asarray(self.seconds, np.float64)*1000
        nonforced = values[~np.asarray(self.forced, dtype=bool)]

        def summary(samples):
            if not len(samples):
                return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
            return {"count": len(samples), "mean_ms": float(samples.mean()),
                    "p50_ms": float(np.quantile(samples, .5)), "p95_ms": float(np.quantile(samples, .95)),
                    "max_ms": float(samples.max())}

        return {"all": summary(values), "nonforced": summary(nonforced),
                "forced": summary(values[np.asarray(self.forced, dtype=bool)]),
                "timing": "batch=1 observation-to-decision; includes sampling/RPC and synchronous CPU action output"}


def choose_search_configuration(trials, *, mean_budget_ms=100., p95_budget_ms=250., tie_band_imps=.25):
    """Select using validation outcomes only, with a latency budget and tie band.

    Among eligible search variants, take the highest mean-IMP group within the
    tie band, then prefer the lowest nonforced latency. Nonpositive validation
    gains or no eligible search variant fall back to the frozen neural policy.
    This is a selection rule, not a significance test on validation results.
    """
    if any(not math.isfinite(v) or v < 0 for v in (mean_budget_ms, p95_budget_ms, tie_band_imps)):
        raise ValueError("Finite nonnegative selection budgets required")
    if not trials:
        raise ValueError("No validation trials")
    ids = [t["config"]["id"] for t in trials]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate validation configuration ID")
    neural = [t for t in trials if t["config"]["kind"] == "neural"]
    if len(neural) != 1:
        raise ValueError("Exactly one frozen neural baseline required")
    eligible = []
    for trial in trials:
        score = trial["mean_imps"]
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation score")
        if trial["config"]["kind"] == "neural":
            continue
        latency = trial["latency"]["nonforced"]
        mean, tail = latency["mean_ms"], latency["p95_ms"]
        if mean is None or tail is None:
            continue
        if not all(math.isfinite(v) and v >= 0 for v in (mean, tail)):
            raise ValueError("Invalid validation latency")
        if mean <= mean_budget_ms and tail <= p95_budget_ms:
            eligible.append(trial)
    selected, reason = neural[0], "No latency-eligible search variant with positive validation mean"
    if eligible:
        best = max(t["mean_imps"] for t in eligible)
        if best > 0:
            near = [t for t in eligible if t["mean_imps"] > 0 and t["mean_imps"] >= best-tie_band_imps]
            selected = min(near, key=lambda t: (t["latency"]["nonforced"]["mean_ms"],
                                               t["config"]["samples"], t["config"]["id"]))
            reason = "Fastest eligible search variant within the predeclared mean-IMP tie band"
    return {"config": selected["config"], "validation_mean_imps": selected["mean_imps"],
            "validation_nonforced_latency": selected["latency"]["nonforced"],
            "eligible_config_ids": [t["config"]["id"] for t in eligible], "reason": reason,
            "rule": {"mean_budget_ms": mean_budget_ms, "p95_budget_ms": p95_budget_ms,
                     "tie_band_imps": tie_band_imps}}
