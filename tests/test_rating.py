"""Rating correctness, privileged access boundary, reproducible matches and recovery."""

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from bridgemind.dds import Deal, Position, PositionResult, Move
from bridgemind.dds.models import SEATS
from bridgemind.engine import BridgePlayEnv, Contract
from bridgemind.engine.play import card_id
from bridgemind.training.data import ownership, board_ids
from bridgemind.rating.policies import RulePolicy, OraclePolicy, oracle_position
from bridgemind.rating.scoring import rate
from bridgemind.rating import system as rating


def board_for(deal, contract=None, **kwargs):
    env = BridgePlayEnv(deal, contract=contract or Contract(3, "NT", "N"), **kwargs)
    owners = ownership(deal)
    return {"id": board_ids(owners[None])[0].decode(), "owners": owners.tolist(), "auction": env.auction,
            "dealer": env.dealer, "vulnerable": env.vulnerable, "system_id": 0}


def test_normalization_endpoints_and_joint_bootstrap_cancel_shared_noise():
    noise = np.arange(18).reshape(3, 6)
    baseline, oracle = noise, noise+10
    mid = noise+5
    result = rate(mid, baseline, oracle, draws=300)
    assert result["rating"] == 50 and result["rating_95_ci"] == pytest.approx([50, 50])
    assert rate(baseline, baseline, oracle, draws=200, anchor="random")["rating"] == 0
    assert rate(oracle, baseline, oracle, draws=200, anchor="oracle")["rating"] == 100
    assert rate(noise+12, baseline, oracle, draws=200)["rating"] == 120
    assert rate(noise-1, baseline, oracle, draws=200)["rating"] == -10


def test_weak_calibration_refuses_a_confident_interval():
    result = rate([[0, 1]], [[0, 0]], [[-1, 3]], draws=300)
    assert result["status"] == "insufficient_anchor_separation" and result["rating_95_ci"] is None
    with pytest.raises(ValueError, match="positive"):
        rate([[0, 1]], [[1, 1]], [[0, 0]], draws=200)
    with pytest.raises(ValueError, match="Aligned"):
        rate([[1, np.nan]], [[0, 0]], [[1, 2]], draws=200)


def test_rules_are_legal_and_ignore_hidden_hands(deal):
    env = BridgePlayEnv(deal, contract=Contract(3, "NT", "N"))
    for _ in range(52):
        obs = env.observation()
        assert "hands" not in obs and "owners" not in obs
        result = RulePolicy().predict(obs)
        assert result["card_id"] in obs["legal_card_ids"]
        env.apply_card(result["card_id"])
    assert env.is_terminal()
    # Actual acting seat determines partnership even when declarer controls dummy.
    obs = {"state_id": "x", "to_play": "S", "player": "N", "legal_card_ids": [card_id("H2"), card_id("HA")],
           "current_trick": [{"seat": "N", "card": "HK"}, {"seat": "E", "card": "H3"}],
           "contract": {"strain": "NT"}}
    assert RulePolicy().predict(obs)["card_id"] == card_id("H2")


def test_oracle_position_matches_true_hands_during_an_incomplete_trick(deal):
    board = board_for(deal)
    env = rating.environment(board)
    for _ in range(7):
        env.apply_card(env.legal_card_ids()[0])
    obs = env.observation()
    position = oracle_position(board, obs)
    assert position.to_play == obs["to_play"]
    assert set(map(card_id, position.legal_cards())) == set(obs["legal_card_ids"])
    played = {e["card"] for e in obs["play_history"]}
    for seat in SEATS:
        assert set(position.remaining_hands[seat]) == set(deal.hands[seat])-played
    assert len(position.current_trick) == 3
    assert not hasattr(OraclePolicy(None), "predict")


def test_oracle_uses_acting_side_and_deterministic_lowest_card_tie(deal):
    env = rating.environment(board_for(deal))
    obs = env.observation()
    legal = obs["legal_card_ids"]

    class Solver:
        def solve_position(self, position, options):
            assert options.state_id == obs["state_id"]
            return PositionResult(position.to_play, "EW", 13, 9, tuple(),
                                  tuple(Move(rating.RulePolicy("lowest").predict({**obs, "legal_card_ids": [c]})["card"], 9) for c in legal))

    action = OraclePolicy(Solver()).predict_position(obs, oracle_position(board_for(deal), obs))
    assert action["card_id"] == min(legal)


def test_native_oracle_agrees_with_solver_on_real_late_positions(deal, backend):
    board = board_for(deal, Contract(4, "S", "S"), vulnerable=(True, False))
    env = rating.environment(board)
    rng = np.random.default_rng(12)
    for _ in range(44):
        env.apply_card(int(rng.choice(env.legal_card_ids())))

    class NativeFacade:
        def solve_position(self, position, options):
            return backend.solve_position(position)

    oracle = OraclePolicy(NativeFacade())
    while not env.is_terminal():
        obs = env.observation()
        position = oracle_position(board, obs)
        result = backend.solve_position(position)
        chosen = oracle.predict_position(obs, position)
        assert chosen["card_id"] in {card_id(c) for c in result.best_cards}
        env.apply_card(chosen["card_id"])


def test_duel_self_symmetry_reversal_and_no_privileged_input_to_ordinary_policy(deal):
    board = board_for(deal)
    random = rating.specification("random", "random")
    low = rating.specification("lowest", "lowest")
    factory = rating.Factory("cpu")
    assert rating.duel(board, random, random, 42, factory)["imps"] == 0
    a = rating.duel(board, random, low, 42, factory)
    b = rating.duel(board, low, random, 42, factory)
    assert a["imps"] == -b["imps"] and a["raw_difference"] == -b["raw_difference"]

    class Spy:
        def predict(self, observation):
            assert "owners" not in observation and "remaining_hands" not in observation
            return RulePolicy("lowest").predict(observation)

    class SpyFactory:
        def create(self, spec, seed):
            return Spy()

    assert rating.duel(board, random, low, 42, SpyFactory())["imps"] == 0


@pytest.fixture
def stored_system(tmp_path, monkeypatch, deal):
    board = board_for(deal)
    hands = {s: list(h) for s, h in deal.hands.items()}
    hands["N"][0], hands["E"][0] = hands["E"][0], hands["N"][0]
    boards = [board, board_for(Deal(hands))]
    protocol = {"name": "test-v1", "schema_version": 1, "anchors": {"random": 0, "oracle": 100},
                "implementation_sha256": rating.implementation(), "runtime": rating.runtime("cpu"),
                "boards_sha256": rating.digest(boards), "boards": 2, "seeds": [42, 43], "bootstrap_draws": 200,
                "panel": [{"id": k, "weight": .25} for k in rating.DEFAULT_PANEL], "purpose": "tracking", "limits": []}
    rating.write(tmp_path/"protocol.json", protocol)
    rating.write(tmp_path/"boards.json", boards)
    for key in rating.DEFAULT_PANEL:
        rating.write(tmp_path/"opponents"/(key+".json"), rating.specification(key, key))
    return rating.RatingSystem(tmp_path)


def test_registry_is_append_only_and_panel_does_not_expand(stored_system):
    system = stored_system
    before = copy.deepcopy(system.protocol)
    spec = rating.specification("search-4", "sampling-dds", samples=4)
    system.register(spec)
    system.register(spec)
    assert system.protocol == before
    with pytest.raises(ValueError, match="immutable"):
        system.register(rating.specification("search-4", "sampling-dds", samples=8))
    with pytest.raises(ValueError, match="start with"):
        rating.specification("../escape", "lowest")
    with pytest.raises(ValueError, match="training deals"):
        rating.check_training_overlap({"id": "leak", "training_ids": [system.boards[0]["id"]]}, system.boards)


def test_match_resume_validates_coordinates_and_reuses_reverse_results(stored_system, monkeypatch):
    system = stored_system
    original = rating.duel
    calls = []

    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(*args)

    monkeypatch.setattr(rating, "duel", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        system.match("random", "lowest")
    assert len(rating.read(system.root/"matches/lowest--random.json")["records"]) == 1
    monkeypatch.setattr(rating, "duel", original)
    forward, _ = system.match("random", "lowest")
    reverse, _ = system.match("lowest", "random")
    np.testing.assert_array_equal(forward, -reverse)
    path = system.root/"matches/lowest--random.json"
    data = rating.read(path)
    data["records"][0]["seed"] += 1
    data["records_sha256"] = rating.digest(data["records"])
    rating.write(path, data)
    with pytest.raises(ValueError, match="coordinates"):
        system.match("random", "lowest")


def test_modified_code_boards_and_weights_are_rejected(stored_system, monkeypatch, tmp_path):
    system = stored_system
    from bridgemind.cardplay.model import PlayModel, ModelConfig, save_checkpoint
    path = tmp_path/"candidate.pt"
    save_checkpoint(path, PlayModel(ModelConfig(32, 1, 4)))
    system.register(rating.specification("candidate", "neural", checkpoint=path))
    path.write_bytes(path.read_bytes()+b"modified")
    with pytest.raises(ValueError, match="checkpoint changed"):
        system.verify_entry("candidate")
    monkeypatch.setattr(rating, "implementation", lambda: "different")
    with pytest.raises(ValueError, match="implementation changed"):
        rating.RatingSystem(system.root)


def test_writer_lock_prevents_concurrent_mutation(tmp_path):
    with rating.writer_lock(tmp_path):
        with pytest.raises((OSError, ValueError)):
            with rating.writer_lock(tmp_path):
                pytest.fail("Second writer acquired lock")


def test_reports_keep_fixed_ratings_when_pool_grows(stored_system, monkeypatch):
    system = stored_system
    strengths = {"random": 0, "lowest": 2, "rule-basic": 5, "oracle": 10}

    def matches(a, b):
        return np.full((2, 2), strengths[a]-strengths[b], dtype=float), {"signature": rating.digest([a, b])}

    monkeypatch.setattr(system, "match", matches)
    assert system.evaluate("random")["score"]["rating"] == 0
    assert system.evaluate("oracle")["score"]["rating"] == 100
    before = system.evaluate("rule-basic")["score"]
    assert before["rating"] == 50
    system.register(rating.specification("future-search", "sampling-dds", samples=64))
    assert system.evaluate("rule-basic")["score"] == before
    board = system.leaderboard()
    assert [r["id"] for r in board["ratings"]] == ["oracle", "rule-basic", "random"]
    assert "future-search" in board["unrated"]


def test_initialize_freezes_board_copy_and_rejects_exclusions(tmp_path, monkeypatch, deal):
    boards = [board_for(deal), board_for(deal, Contract(4, "S", "S"))]
    # A source must have distinct complete-deal IDs; use another actual deal.
    hands = {s: list(h) for s, h in deal.hands.items()}
    hands["N"][0], hands["E"][0] = hands["E"][0], hands["N"][0]
    boards[1] = board_for(Deal(hands))

    class Source:
        manifest = {"boards": 2}
        ids = np.array([b["id"].encode() for b in boards])

        def __init__(self, path):
            pass

        def load_games(self):
            return boards

    class Solver:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_info(self):
            return {"solver_version": "3.1.0", "solver_identity": "test"}

    monkeypatch.setattr(rating, "PlayData", Source)
    monkeypatch.setattr(rating, "fingerprint", lambda path: "source-hash")
    monkeypatch.setattr(rating, "SolverClient", Solver)
    with pytest.raises(ValueError, match="overlaps excluded"):
        rating.initialize(tmp_path/"bad", "data", name="bad", count=2, exclude_data=["train"])
    root = tmp_path/"good"
    rating.initialize(root, "data", name="good", count=2, draws=200)
    system = rating.RatingSystem(root)
    assert len(system.boards) == 2 and system.protocol["anchors"] == {"random": 0, "oracle": 100}
    changed = rating.read(root/"boards.json")
    changed[0]["vulnerable"] = [True, True]
    rating.write(root/"boards.json", changed)
    with pytest.raises(ValueError, match="boards changed"):
        rating.RatingSystem(root)
