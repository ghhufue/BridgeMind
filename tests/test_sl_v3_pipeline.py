import importlib.util
from pathlib import Path
import sys
import pytest

from bridgemind.rating.system import write, read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"examples"))
spec = importlib.util.spec_from_file_location("sl_v3_pipeline", ROOT/"examples/cardplay_sl_v3_run.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_pilot_selection_uses_declared_tradeoff_without_test_results():
    rows = [{"fraction": .25, "mean_imps": 1., "novel_accuracy": .8, "augmented_nll": .5},
            {"fraction": .125, "mean_imps": .9, "novel_accuracy": .95, "augmented_nll": .6}]
    assert pipeline.select_pilot(rows)["fraction"] == .125
    rows[1]["mean_imps"] = .7
    assert pipeline.select_pilot(rows)["fraction"] == .25
    with pytest.raises(ValueError):
        pipeline.select_pilot([])


def test_run_clone_is_immutable_and_keeps_original_initialization(tmp_path):
    source, target = tmp_path/"source", tmp_path/"pilot"
    source.mkdir()
    (source/"init.pt").write_bytes(b"frozen-v2-initialization")
    (source/"curriculum.json").write_text('{"train": [], "valid": []}')
    from bridgemind.training.runner import sha256
    config = {"batch_size": 64, "curriculum_fraction": .25, "real_samples_per_batch": 48,
              "initial_sha256": sha256(source/"init.pt")}
    write(source/"protocol.json", config)
    pipeline.clone_run(source, target, .125)
    assert read(target/"protocol.json")["real_samples_per_batch"] == 56
    assert (target/"init.pt").read_bytes() == (source/"init.pt").read_bytes()
    assert read(source/"protocol.json") == config
    pipeline.clone_run(source, target, .125)
    with pytest.raises(ValueError, match="differs"):
        pipeline.clone_run(source, target, .25)
    (target/"init.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        pipeline.clone_run(source, target, .125)


def test_stage_stop_does_not_start_more_training(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    pipeline.check_stop()
    (tmp_path/"STOP").write_text("")
    with pytest.raises(pipeline.StopRequested):
        pipeline.child("unused", ["not-a-real-command"])
