import random

import numpy as np
import pytest
import torch

from bridgemind.cardplay.encoding import collate, encode_observation
from bridgemind.cardplay.model import ModelConfig, PlayModel
from bridgemind.cardplay.policy import NeuralPlayPolicy
from bridgemind.dds import Deal
from bridgemind.dds.models import CARDS, SEATS
from bridgemind.engine import BridgeBiddingEnv
from bridgemind.engine.play import card_id
from bridgemind.web.analysis import AnalysisService, PROTOCOL
from bridgemind.web.bidding import validate_bidding
from test_web_analysis import fixture_request


@pytest.mark.parametrize("declarer", SEATS)
@pytest.mark.parametrize("steps", [0, 1, 9, 48, 51])
def test_debug_matches_raw_model_heads_and_absolute_ownership(declarer, steps):
    torch.set_num_threads(1)
    model = PlayModel(ModelConfig(d_model=32, heads=4, layers=1)).eval()
    with torch.no_grad():
        model.value_head[-1].weight.zero_()
        model.value_head[-1].bias.fill_(.25)
    policy = NeuralPlayPolicy(model=model)
    request, obs = fixture_request(steps=steps, declarer=declarer)
    result = AnalysisService(policy, model_name="tiny", model_sha256="a"*64).analyze(request, "ai")
    d = result["diagnostics"]
    with torch.no_grad():
        distribution, value, beliefs = model.distribution(collate([encode_observation(obs)]))
    assert d["modelSha256"] == "a"*64
    assert d["value"]["normalized"] == float(value[0]) == .25
    assert d["value"]["predictedScore"] == 1900
    assert d["value"]["perspective"] == ("NS" if obs["player"] in ("N", "S") else "EW")
    assert d["winProbability"] is None
    assert len(d["actionProbabilities"]) == len(obs["visible_hands"][obs["to_play"]])
    assert sum(a["probability"] for a in d["actionProbabilities"]) == pytest.approx(1)
    for action in d["actionProbabilities"]:
        assert action["legal"] == (card_id(action["card"]) in obs["legal_card_ids"])
        assert action["probability"] == pytest.approx(float(distribution.probs[0, card_id(action["card"])]))
        if not action["legal"]:
            assert action["probability"] == 0
    assert len(d["ownership"]) == 52
    known = {c: s for s, hand in obs["visible_hands"].items() for c in hand}
    played = {e["card"]: e["seat"] for e in obs["play_history"]}
    raw = beliefs[0].softmax(-1).numpy()
    for row in d["ownership"]:
        assert sum(row["probabilities"].values()) == pytest.approx(1)
        if row["card"] in played or row["card"] in known:
            seat = played.get(row["card"], known.get(row["card"]))
            assert row["probabilities"][seat] == 1
            assert row["status"] == ("played" if row["card"] in played else "known")
        else:
            assert row["status"] == "unknown"
            for i, seat in enumerate(SEATS):
                assert row["probabilities"][seat] == pytest.approx(float(raw[card_id(row["card"]), (i-SEATS.index(declarer)) % 4]))


def bidding_request(env):
    obs = env.observation()
    return {"protocol": PROTOCOL, "stateId": obs["state_id"], "board": 1, "dealer": env.dealer,
            "vulnerable": list(env.vulnerable), "turn": obs["player"], "auction": obs["auction"],
            "hand": {"seat": obs["player"], "cards": obs["hand"]}, "legalCalls": env.legal_calls()}


def test_bidding_validator_agrees_with_openspiel_on_full_random_auctions():
    rng = random.Random(260627)
    count = 0
    for game in range(32):
        deck = list(CARDS); rng.shuffle(deck)
        env = BridgeBiddingEnv(None, Deal({s: deck[i*13:(i+1)*13] for i, s in enumerate(SEATS)}),
                               dealer=SEATS[game % 4], vulnerable=(bool(game & 1), bool(game & 2)), prefetch=False)
        while not env.is_terminal():
            actual = validate_bidding(bidding_request(env))
            expected = env.observation()
            assert set(actual) == {"state_id", "player", "dealer", "hand", "vulnerable", "auction", "legal_call_ids"}
            for key, value in actual.items():
                assert value == expected[key]
            count += 1
            # Includes ascending contracts, doubles/redoubles and passed-out deals.
            calls = env.legal_calls()
            env.apply_call(rng.choice(calls) if rng.random() < .6 else "P")
    assert count > 150


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(hands={s: [] for s in SEATS}),
    lambda r: r.update(turn="E"),
    lambda r: r["hand"].update(seat="S"),
    lambda r: r["legalCalls"].pop(),
    lambda r: r["hand"]["cards"].pop(),
    lambda r: r.update(auction=[{"seat": "E", "call": "P"}]),
])
def test_bidding_rejects_hidden_hands_and_forged_state(mutation):
    env = BridgeBiddingEnv(None, Deal({s: list(CARDS)[i*13:(i+1)*13] for i, s in enumerate(SEATS)}), prefetch=False)
    request = bidding_request(env)
    mutation(request)
    with pytest.raises(ValueError):
        validate_bidding(request)


def test_bidding_service_receives_only_private_actor_hand():
    class Bidder:
        def predict(self, observation, stochastic=False):
            assert set(observation) == {"state_id", "player", "dealer", "hand", "vulnerable", "auction", "legal_call_ids"}
            assert len(observation["hand"]) == 13 and not stochastic
            return {"call": "P", "probabilities": {"P": 1.}}
    env = BridgeBiddingEnv(None, Deal({s: list(CARDS)[i*13:(i+1)*13] for i, s in enumerate(SEATS)}), prefetch=False)
    request = bidding_request(env)
    result = AnalysisService(bidder=Bidder(), bidding_name="fixture").analyze(request, "bidding")
    assert result["call"] == "P" and result["source"] == "bidding" and result["stateId"] == request["stateId"]
