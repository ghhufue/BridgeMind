"""Prevent experiment bookkeeping from accidentally reusing a reported deal."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "long_sl_evaluation", Path(__file__).resolve().parents[1]/"examples/cardplay_long_sl_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_historical_deal_exclusion_reads_nested_reports_without_confusing_model_hashes():
    a, b, model = "a"*64, "b"*64, "c"*64
    report = {"model_sha256": model, "board_ids": [a],
              "rooms": [{"validation": {"board_ids": [b, a]}}]}
    assert evaluation.collect_report_ids(report) == {a.encode(), b.encode()}


@pytest.mark.parametrize("ids", [None, "a"*64, ["short"], ["Z"*64]])
def test_malformed_historical_ids_are_rejected_instead_of_silently_skipped(ids):
    with pytest.raises(ValueError, match="historical evaluation board IDs"):
        evaluation.collect_report_ids({"nested": {"board_ids": ids}})
