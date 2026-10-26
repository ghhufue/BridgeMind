"""Proof bounds, public information, legal isomorphisms and independent score axes."""

import copy
from collections import defaultdict

import numpy as np
import pytest

pytest.importorskip("torch")

from bridgemind.dds import Position
from bridgemind.dds.models import SEATS
from bridgemind.engine.play import card_id, card_name
from bridgemind.rating.policies import oracle_position
from bridgemind.foundation.cases import (TEMPLATES, PERMUTATIONS, ROLES, build_base, observe,
                                          transform_case, transform_observation, card_map)
from bridgemind.foundation.certificates import certify, witness_value, public_view, public_action
from bridgemind.foundation.system import evaluate_rows, summarize, build_suite, load_suite, PublicPolicy, evaluate, export, install_suite


class NativeFacade:
    def __init__(self, backend):
        self.backend = backend

    def solve_position(self, position):
        return self.backend.solve_position(position)

    def get_info(self):
        return {"solver_version": self.backend.info["version"], "solver_identity": self.backend.info["identity"]}


@pytest.mark.parametrize("family", list(TEMPLATES))
@pytest.mark.parametrize("role", list(ROLES))
def test_every_template_has_legal_full_history_and_the_requested_controller(family, role):
    base = build_base(family, 1701, role)
    obs = observe(base)
    assert obs["to_play"] == "S"
    assert obs["player"] == ("N" if role == "dummy" else "S")
    assert len(base["prefix"]) == len(set(base["prefix"]))
    assert len(base["board"]["owners"]) == 52
    assert sorted(base["board"]["owners"]) == [0]*13+[1]*13+[2]*13+[3]*13
    if base["mechanical"]:
        assert len(obs["legal_card_ids"]) == 1


@pytest.mark.parametrize("family,role", [("fourth_hand_take_trick", "dummy"), ("small_ruff_preserve_master", "defender")])
def test_all_24_suit_bijections_and_four_seats_replay_to_the_exact_transformed_observation(family, role):
    base = build_base(family, 1709, role)
    original = observe(base)
    for permutation in PERMUTATIONS:
        assert sorted(card_map(permutation)) == list(range(52))
        for rotation in range(4):
            transformed = transform_case(base, permutation, rotation)
            actual = observe(transformed)
            wanted = transform_observation(original, permutation, rotation)
            actual.pop("state_id"); wanted.pop("state_id")
            assert actual == wanted


@pytest.mark.parametrize("family", ["fourth_hand_take_trick", "small_ruff_preserve_master", "overruff_not_undertrump", "discard_loser_keep_master"])
def test_certified_action_has_one_public_witness_in_all_hidden_worlds(family, backend):
    from bridgemind.diagnostics.cardplay import hidden_worlds
    base = build_base(family, 1701, "declarer")
    obs = observe(base)
    certificate = certify(obs, NativeFacade(backend))
    assert certificate["status"] == "certified"
    for position in hidden_worlds(obs):
        result = backend.solve_position(position)
        witness = certificate["witness"]
        lower = witness_value(position, card_id(witness["first_card"]), obs["contract"]["declarer"], witness["continuation"])
        assert lower == result.best_tricks
    assert certificate["witness"]["verified_worlds"] == certificate["worlds"]


def test_ambiguous_information_and_unproven_oracle_agreement_are_not_graded(backend):
    solver = NativeFacade(backend)
    ambiguous = certify(observe(build_base("cheapest_winning_honor", 1701, "defender")), solver)
    assert ambiguous["status"] == "ungraded" and ambiguous["reason"].startswith("no_action")
    no_witness = certify(observe(build_base("preserve_partner_entry", 1701, "defender")), solver)
    assert no_witness["status"] == "ungraded" and "without_a_public" in no_witness["reason"]


def test_equivalent_correct_moves_are_all_accepted(backend):
    cert = certify(observe(build_base("equivalent_winning_honors", 1701, "declarer")), NativeFacade(backend))
    assert cert["status"] == "certified"
    assert set(cert["accepted_cards"]) == {"HA", "HK"}
    assert set(cert["witnesses"]) == {"HA", "HK"}


def test_witness_value_is_a_lower_bound_not_an_oracle_policy(backend):
    from bridgemind.diagnostics.cardplay import hidden_worlds
    base = build_base("preserve_partner_entry", 1701, "defender")
    obs = observe(base)
    strict_gap = False
    for position in hidden_worlds(obs):
        dds = backend.solve_position(position)
        values = {card_id(m.card): m.tricks_for_evaluated_side for m in dds.moves}
        for action in obs["legal_card_ids"]:
            lower = witness_value(position, action, obs["contract"]["declarer"], "rule-basic")
            assert lower <= values[action]
            strict_gap |= lower < values[action]
    assert strict_gap


def test_public_witness_does_not_change_when_hidden_cards_swap():
    base = build_base("fourth_hand_take_trick", 1701, "declarer")
    obs = observe(base)
    pos = oracle_position(base["board"], obs)
    hands = tuple(tuple(sorted(card_id(c) for c in pos.remaining_hands[s])) for s in SEATS)
    altered = list(hands)
    altered[1], altered[3] = hands[3], hands[1]
    # Both hidden hands have equal remaining sizes in this position.
    trick = tuple(card_id(c) for c in pos.current_trick)
    a = public_view(hands, trick, SEATS.index(pos.trick_leader), 2, "NT")
    b = public_view(tuple(altered), trick, SEATS.index(pos.trick_leader), 2, "NT")
    assert a == b
    assert public_action(a, "cash-public-master") == public_action(b, "cash-public-master")


def toy_suite():
    base = build_base("fourth_hand_take_trick", 1701, "declarer")
    base["certificate"] = {"status": "certified", "accepted_cards": ["HA"], "worlds": 6,
                           "worst_world_trick_regret": {"HA": 0, "HQ": 2}}
    return {"bases": [base], "suit_permutations": list(PERMUTATIONS), "seat_rotations": [0, 1, 2, 3]}


class ConsistentWrong:
    def predict_batch(self, observations, **kwargs):
        result = []
        for obs in observations:
            assert "owners" not in obs and "remaining_hands" not in obs and "certificate" not in obs
            assert len(obs["state_id"]) == 64 and all(c in "0123456789abcdef" for c in obs["state_id"])
            # Pick the queen, an isomorphic and consistently wrong action.
            action = next(c for c in obs["legal_card_ids"] if card_name(c)[1] == "Q")
            probs = [float(c == action) for c in range(52)]
            result.append({"state_id": obs["state_id"], "card_id": action, "probabilities": probs})
        return result


def test_consistently_wrong_policy_has_zero_skill_even_with_perfect_symmetry():
    rows = evaluate_rows(toy_suite(), ConsistentWrong(), batch_size=7)
    scores = summarize(rows)
    assert scores["all_suits_basic_percent"] == 0
    assert scores["suit_distribution_equivariance_percent"] == 100
    assert scores["exact_suit_choice_agreement_percent"] == 100
    assert scores["all_suits_pass_position_percent"] == 0
    assert len(rows) == 96


def test_correct_public_policy_passes_all_suit_and_seat_variants():
    rows = evaluate_rows(toy_suite(), PublicPolicy("cash-public-master"), batch_size=11)
    scores = summarize(rows)
    assert scores["all_suits_basic_percent"] == 100
    assert scores["suit_distribution_equivariance_percent"] == 100
    assert scores["seat_rotation_choice_consistency_percent"] == 100


def test_controls_do_not_inflate_the_basic_macro_average():
    rows = evaluate_rows(toy_suite(), ConsistentWrong())
    control = copy.deepcopy(rows)
    for row in control:
        row.update(family="easy-control", base_id="control", mechanical=True, correct=True)
    assert summarize(rows+control)["all_suits_basic_percent"] == 0


def test_probability_drift_is_reported_even_when_the_argmax_stays_correct():
    class Biased:
        def predict_batch(self, observations, **kwargs):
            result = []
            for obs in observations:
                good = next(c for c in obs["legal_card_ids"] if card_name(c)[1] == "A")
                bad = next(c for c in obs["legal_card_ids"] if card_name(c)[1] == "Q")
                p = .51 if card_name(good)[0] == "C" else .99
                probs = [0.]*52; probs[good] = p; probs[bad] = 1-p
                result.append({"state_id": obs["state_id"], "card_id": good, "probabilities": probs})
            return result
    scores = summarize(evaluate_rows(toy_suite(), Biased()))
    assert scores["all_suits_basic_percent"] == 100 and scores["exact_suit_choice_agreement_percent"] == 100
    assert scores["suit_distribution_equivariance_percent"] < 100


def test_stale_or_incomplete_outputs_are_rejected():
    class Incomplete:
        def predict_batch(self, obs, **kwargs):
            return []
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_rows(toy_suite(), Incomplete())
    with pytest.raises(ValueError, match="bijection"):
        card_map("CCCC")


def test_complete_suite_run_is_versioned_cached_exportable_and_rejects_tampering(tmp_path, backend):
    from bridgemind.rating.system import read, write
    root = tmp_path/"suite"
    suite = build_suite(root, NativeFacade(backend), seeds=(1701,))
    assert len(suite["bases"]) == 30
    assert sum(b["certificate"]["status"] == "ungraded" for b in suite["bases"]) == 2
    report = evaluate(root, name="reference", recipe="cash-public-master", device="cpu")
    assert report["summary"]["all_suits_basic_percent"] == 100
    assert evaluate(root, name="reference", recipe="cash-public-master", device="cpu") == report
    with pytest.raises(ValueError, match="differs"):
        evaluate(root, name="reference", recipe="rule-basic", device="cpu")
    output = tmp_path/"exported"
    assert export(root, output)["reports"] == 1
    assert (output/"report.md").is_file()
    restored = tmp_path/"restored"
    install_suite(output, restored)
    assert load_suite(restored) == suite
    assert not (restored/"reports").exists()
    records = root/"predictions/reference.jsonl"
    records.write_text(records.read_text()+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache changed"):
        evaluate(root, name="reference", recipe="cash-public-master", device="cpu")
    changed = read(root/"suite.json")
    changed["seeds"] = [999]
    write(root/"suite.json", changed)
    with pytest.raises(ValueError, match="suite changed"):
        load_suite(root)


def test_accepted_sets_are_transported_with_the_game_not_left_in_original_suits(backend):
    base = build_base("small_ruff_preserve_master", 1701, "declarer")
    base["certificate"] = certify(observe(base), NativeFacade(backend))
    for permutation in PERMUTATIONS:
        variant = transform_case(base, permutation, 1)
        pos = oracle_position(variant["board"], observe(variant))
        result = backend.solve_position(pos)
        assert set(variant["certificate"]["accepted_cards"]) == set(result.best_cards)
        assert variant["certificate"]["transport"]["seat_rotation"] == 1
