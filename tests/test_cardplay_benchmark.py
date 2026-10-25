import copy

import pytest

pytest.importorskip("numpy")

from bridgemind.cardplay.benchmark import TimedPlayPolicy, choose_search_configuration


def trial(name, kind, samples, score, mean, tail):
    return {"config": {"id": name, "kind": kind, "samples": samples}, "mean_imps": score,
            "latency": {"nonforced": {"count": 20, "mean_ms": mean, "p95_ms": tail}}}


def test_selection_applies_latency_budgets_before_choosing_a_validation_winner():
    rows = [trial("neural", "neural", 0, 0, 2, 3), trial("too-slow", "uniform", 32, 12, 110, 120),
            trial("tail-too-slow", "hybrid", 32, 11, 50, 300), trial("usable", "uniform", 8, 5, 10, 20)]
    result = choose_search_configuration(rows)
    assert result["config"]["id"] == "usable" and result["eligible_config_ids"] == ["usable"]


def test_selection_prefers_latency_within_the_predeclared_tie_band():
    rows = [trial("neural", "neural", 0, 0, 2, 3), trial("uniform-8", "uniform", 8, 7, 10, 20),
            trial("hybrid-32", "hybrid", 32, 7.2, 40, 60)]
    assert choose_search_configuration(rows)["config"]["id"] == "uniform-8"
    assert choose_search_configuration(rows, tie_band_imps=.1)["config"]["id"] == "hybrid-32"


@pytest.mark.parametrize("score,mean,tail", [(-1, 10, 20), (0, 10, 20), (8, 101, 200), (8, 40, 251)])
def test_selection_falls_back_to_neural_when_search_does_not_meet_the_rule(score, mean, tail):
    rows = [trial("neural", "neural", 0, 0, 2, 3), trial("search", "uniform", 8, score, mean, tail)]
    assert choose_search_configuration(rows)["config"]["id"] == "neural"


def test_selection_rejects_duplicate_or_nonfinite_inputs():
    rows = [trial("neural", "neural", 0, 0, 2, 3), trial("search", "uniform", 8, 1, 10, 20)]
    with pytest.raises(ValueError, match="Duplicate"):
        choose_search_configuration(rows+[copy.deepcopy(rows[1])])
    rows[1]["mean_imps"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        choose_search_configuration(rows)


def test_timing_rejects_stale_results_and_handles_all_forced_actions():
    class Policy:
        def predict(self, obs):
            return {"state_id": "a", "card_id": 3}
    wrapper = TimedPlayPolicy(Policy())
    wrapper.predict({"state_id": "a", "legal_card_ids": [3]})
    assert wrapper.metrics()["all"]["count"] == 1
    assert wrapper.metrics()["nonforced"]["count"] == 0
    assert wrapper.metrics()["nonforced"]["mean_ms"] is None
    with pytest.raises(ValueError, match="stale or illegal"):
        wrapper.predict({"state_id": "b", "legal_card_ids": [3]})
