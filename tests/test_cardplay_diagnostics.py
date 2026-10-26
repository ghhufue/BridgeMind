"""Legal targeted positions, complete hidden-world audits and actual single-move replay."""

import copy

import pytest

pytest.importorskip("torch")

from bridgemind.dds.models import CARDS
from bridgemind.engine.play import card_id
from bridgemind.rating.policies import RulePolicy, decision, oracle_position
from bridgemind.diagnostics.cardplay import (TEMPLATES, construct_case, make_environment,
                                             evaluate_case, hidden_worlds, play_room)


@pytest.mark.parametrize("family", list(TEMPLATES))
@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("history", (1, 2))
def test_constructed_full_deals_have_legal_prefixes_and_expected_ending(family, rotation, history):
    case = construct_case(family, rotation, history)
    env = make_environment(case["board"])
    for card in case["prefix"]:
        env.apply_card(card)
    obs = env.observation()
    assert obs["auction_source"] == "synthetic_contract"
    assert obs["player"] == "S" and obs["to_play"] == "S"
    assert card_id(case["suggested_good"]) in obs["legal_card_ids"]
    assert card_id(case["suggested_bad"]) in obs["legal_card_ids"]
    assert len(case["board"]["owners"]) == 52
    assert len(case["prefix"]) == len(set(case["prefix"]))
    assert set(case["prefix"]) | {c for cards in oracle_position(case["board"], obs).remaining_hands.values() for c in cards} == set(CARDS)


class NativeFacade:
    def __init__(self, backend):
        self.backend = backend

    def solve_position(self, position, options=None):
        return self.backend.solve_position(position)


@pytest.mark.parametrize("family", list(TEMPLATES))
def test_suit_variants_are_exact_permutations_of_the_complete_deal_and_history(family):
    from bridgemind.engine.play import card_name
    for history in (1, 2):
        base = construct_case(family, 0, history)
        for rotation in (1, 2, 3):
            variant = construct_case(family, rotation, history)
            def translate(card):
                return "CDHS"[("CDHS".index(card[0])+rotation) % 4]+card[1]
            assert variant["prefix"] == [translate(c) for c in base["prefix"]]
            assert variant["suggested_good"] == translate(base["suggested_good"])
            for card in range(52):
                assert variant["board"]["owners"][card_id(translate(card_name(card)))] == base["board"]["owners"][card]


@pytest.mark.parametrize("family", list(TEMPLATES))
def test_native_dds_confirms_labels_and_exhaustive_public_worlds(family, backend):
    case = construct_case(family)

    class BadPolicy:
        def predict_batch(self, observations, return_beliefs=False):
            outputs = []
            for obs in observations:
                assert "owners" not in obs and "remaining_hands" not in obs
                action = card_id(case["suggested_bad"])
                outputs.append({**decision(obs, action), "probabilities": [float(c == action) for c in range(52)]})
            return outputs

    result = evaluate_case(case, BadPolicy(), NativeFacade(backend))
    assert result["model_dds_regret"] > 0
    assert result["hidden_world_audit"]["good_never_worse"]
    assert result["hidden_world_audit"]["worlds"] > 1
    env = make_environment(case["board"])
    for card in case["prefix"]:
        env.apply_card(card)
    obs = env.observation()
    actual = oracle_position(case["board"], obs).to_dict()
    assert actual in [p.to_dict() for p in hidden_worlds(obs)]


def test_single_move_patch_changes_actual_outcome_without_mutating_or_freezing_future_moves(backend):
    case = construct_case("win_now_keep_lead")
    target_ply = len(case["prefix"])

    class PrefixThenRule:
        def __init__(self):
            self.future_calls = 0

        def predict(self, obs):
            ply = len(obs["play_history"])
            if ply < target_ply:
                return decision(obs, card_id(case["prefix"][ply]))
            if ply == target_ply:
                return decision(obs, card_id(case["suggested_bad"]))
            self.future_calls += 1
            return RulePolicy().predict(obs)

        def predict_batch(self, observations, return_beliefs=False):
            result = []
            for obs in observations:
                d = self.predict(obs)
                d["probabilities"] = [float(c == d["card_id"]) for c in range(52)]
                result.append(d)
            return result

    neural, opponent = PrefixThenRule(), PrefixThenRule()
    original = play_room(case["board"], 0, neural, opponent, NativeFacade(backend))
    saved = copy.deepcopy(original)
    before = neural.future_calls
    patched = play_room(case["board"], 0, neural, opponent, original=original["trace"],
                        patch={"ply": target_ply, "card_id": card_id(case["suggested_good"])})
    assert original == saved
    assert neural.future_calls > before
    assert patched["result"]["declarer_tricks"] == original["result"]["declarer_tricks"]+2
