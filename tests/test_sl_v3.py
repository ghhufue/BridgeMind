import copy
import numpy as np
import pytest
import torch

from bridgemind.dds import Deal
from bridgemind.dds.models import SEATS
from bridgemind.cardplay.data import prepare_dataset
from bridgemind.cardplay.model import ModelConfig, PlayModel, save_checkpoint, read_checkpoint
from bridgemind.foundation.cases import observe, transform_case, transform_observation, PERMUTATIONS
from bridgemind.sl_v3.courses import TEMPLATES, build_case, certify, witness_value
from bridgemind.sl_v3.data import OLD_FAMILIES
from bridgemind.sl_v3.training import prepare, train, passes_guard
from bridgemind.diagnostics.cardplay import hidden_worlds
from bridgemind.engine.play import card_id
from bridgemind.training.data import ownership, board_ids
from test_sl_v2 import raw_trajectory


@pytest.mark.parametrize("family", TEMPLATES)
@pytest.mark.parametrize("role", ["declarer", "dummy", "defender"])
def test_new_tactics_have_legal_histories_and_public_certificates(family, role, backend):
    case = build_case(family, 390101, role)
    obs = observe(case)
    actual_role = "declarer" if obs["to_play"] == obs["contract"]["declarer"] else "dummy" if obs["player"] == obs["contract"]["declarer"] else "defender"
    assert role == actual_role
    assert len(case["board"]["owners"]) == 52 and len(case["prefix"]) >= 36
    result = certify(case, backend)
    if role == "defender" and family != "draw_trumps_before_cash":
        assert result["status"] == "ungraded"
        return
    assert result["status"] == "certified" and result["worlds"] >= 20
    assert len(result["accepted_cards"]) < len(obs["legal_card_ids"])
    if family.startswith("unblock"):
        assert result["accepted_cards"] == ["CA"]
    if family == "draw_trumps_before_cash":
        assert result["accepted_cards"] == ["SA"]
    if family == "preserve_friendly_trump_entry":
        assert "SA" not in result["accepted_cards"] and len(result["accepted_cards"]) == 3
    for world in hidden_worlds(obs):
        upper = {m.card: m.tricks_for_evaluated_side for m in backend.solve_position(world).moves}
        for card in result["accepted_cards"]:
            recipe = result["witnesses"][card]["continuation"]
            assert witness_value(world, card_id(card), obs["contract"]["declarer"], recipe) == upper[card] == max(upper.values())


@pytest.mark.parametrize("family", TEMPLATES)
def test_new_full_trajectory_transforms_preserve_every_public_field(family):
    case = build_case(family, 390103, "declarer")
    original = observe(case)
    for permutation in PERMUTATIONS:
        for rotation in (0, 1):
            actual = observe(transform_case(case, permutation, rotation))
            expected = transform_observation(original, permutation, rotation)
            for key in expected:
                if key != "state_id":
                    assert actual[key] == expected[key], key


def test_improved_new_skills_cannot_hide_old_family_regression():
    baseline = {"novel_macro_accuracy": .5, "by_family": {f: 1. for f in OLD_FAMILIES}}
    assert passes_guard(baseline, baseline)
    improved = copy.deepcopy(baseline); improved["novel_macro_accuracy"] = 1.
    assert passes_guard(improved, baseline)
    improved["by_family"][OLD_FAMILIES[0]] = .99
    assert not passes_guard(improved, baseline)
    worse = copy.deepcopy(baseline); worse["novel_macro_accuracy"] = .49
    assert not passes_guard(worse, baseline)


def test_v3_cpu_resume_and_calibration_metrics(tmp_path, monkeypatch, deal, backend):
    import bridgemind.sl_v3.training as training
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "load_suite", lambda path: {"bases": []})
    hands = {s: list(h) for s, h in deal.hands.items()}
    hands["N"][0], hands["E"][0] = hands["E"][0], hands["N"][0]
    paths = []
    for label, d in (("train", deal), ("valid", Deal(hands))):
        source = tmp_path/(label+".txt"); source.write_text(raw_trajectory(d))
        path = tmp_path/label
        prepare_dataset(source, path, allowed_ids=board_ids(np.asarray([ownership(d)])), max_boards=1)
        paths.append(path)
    initial = tmp_path/"init.pt"
    torch.manual_seed(12); save_checkpoint(initial, PlayModel(ModelConfig(32, 1, 4)))
    class Solver:
        def solve_position(self, p):
            return backend.solve_position(p)
        def get_info(self):
            return {"solver_identity": backend.info["identity"], "solver_version": backend.info["version"]}
    def setup(name):
        root = tmp_path/name
        prepare(root, paths[0], paths[1], initial, Solver(), train_seeds=(8301,), valid_seeds=(9303,),
                device="cpu", batch_size=8, eval_states=8, eval_boards=0, eval_interval=10000,
                save_interval=10000, cache_bytes=1024**2)
        return root
    a, b = setup("a"), setup("b")
    first = train(a, seconds=1000, max_steps=4)
    train(b, seconds=1000, max_steps=2)
    resumed = train(b, seconds=1000, max_steps=4, resume=True)
    state_a, state_b = read_checkpoint(a/"last.pt"), read_checkpoint(b/"last.pt")
    for name, value in state_a["state_dict"].items():
        torch.testing.assert_close(value, state_b["state_dict"][name], rtol=0, atol=0)
    assert state_a["numpy_rng"] == state_b["numpy_rng"]
    assert first["validation"] == resumed["validation"]
    for view in ("real_canonical", "real_augmented"):
        metrics = first["validation"][view]
        assert metrics["belief_nll"] >= 0 and 0 <= metrics["belief_accuracy"] <= 1
        assert 0 <= metrics["belief_brier"] <= 2 and metrics["value_mse"] >= 0
