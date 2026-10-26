import json
import random
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from bridgemind.dds import Deal
from bridgemind.dds.models import CARDS, SEATS
from bridgemind.engine import BridgePlayEnv, Contract
from bridgemind.engine.play import card_name
from bridgemind.web.analysis import AnalysisService, PROTOCOL, validate_request
from bridgemind.web.server import LanServer


def fixture_request(source="ai", steps=0, declarer="S", dealer="S"):
    deck = list(CARDS)
    random.Random(251).shuffle(deck)
    deal = Deal({seat: deck[i * 13:(i + 1) * 13] for i, seat in enumerate(SEATS)})
    synthetic = BridgePlayEnv(deal, contract=Contract(3, "NT", declarer), dealer=dealer)
    env = BridgePlayEnv(deal, auction=synthetic.auction, dealer=dealer)
    for _ in range(steps):
        env.apply_card(env.legal_card_ids()[0])
    obs = env.observation()
    n = len(env._plays) // 4
    tricks = [{"cards": env._plays[i * 4:i * 4 + 4],
               "winner": env._plays[(i + 1) * 4]["seat"] if (i + 1) * 4 < len(env._plays) else env._leader} for i in range(n)]
    request = {"protocol": PROTOCOL, "stateId": obs["state_id"], "board": 1, "dealer": dealer,
               "vulnerable": [False, False], "contract": {"level": 3, "strain": "NT", "declarer": declarer, "doubled": 0},
               "toPlay": obs["to_play"], "controller": obs["player"], "auction": obs["auction"],
               "currentTrick": obs["current_trick"], "tricks": tricks,
               "legalCards": [card_name(i) for i in obs["legal_card_ids"]], "information": "visible" if source == "ai" else "full"}
    if source == "ai":
        request.update(hand={"seat": obs["player"], "cards": obs["visible_hands"][obs["player"]]},
                       dummy={"seat": obs["dummy"], "cards": obs["visible_hands"][obs["dummy"]]} if steps else None,
                       remainingCounts=obs["remaining_counts"])
    else:
        request["hands"] = {s: list(env._hands[s]) for s in SEATS}
    return request, obs


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4, 9, 20, 39, 48, 51])
@pytest.mark.parametrize("declarer", SEATS)
def test_visible_observation_matches_existing_backend(steps, declarer):
    request, expected = fixture_request(steps=steps, declarer=declarer)
    actual = validate_request(request, "ai")
    assert actual.observation == expected
    assert actual.dds_position is None
    assert set(actual.observation["visible_hands"]) == set(expected["visible_hands"])
    assert "hands" not in actual.observation


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 12, 39, 48, 51])
def test_dds_remaining_position_and_legal_cards(steps):
    request, expected = fixture_request("dds", steps)
    parsed = validate_request(request, "dds")
    assert parsed.dds_position.to_play == expected["to_play"]
    assert set(parsed.dds_position.legal_cards()) == set(parsed.legal)
    assert parsed.observation["visible_hands"] == {}


def test_rejects_hidden_information_at_ai_boundary():
    request, _ = fixture_request()
    request["hands"] = {s: [] for s in SEATS}
    with pytest.raises(ValueError, match="fields"):
        validate_request(request, "ai")


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(toPlay="N"),
    lambda r: r.update(controller=next(s for s in SEATS if s != r["controller"])),
    lambda r: r["legalCards"].pop(),
    lambda r: r["contract"].update(declarer="E"),
    lambda r: r["hand"]["cards"].append(r["hand"]["cards"][0]),
    lambda r: r.update(dummy=None),
    lambda r: r["remainingCounts"].update(N=13),
    lambda r: r["auction"].append({"seat": "N", "call": "P"}),
])
def test_rejects_inconsistent_client_state(mutation):
    request, _ = fixture_request(steps=5)
    mutation(request)
    with pytest.raises(ValueError):
        validate_request(request, "ai")


def test_cannot_reveal_dummy_before_lead_or_forge_winner():
    request, _ = fixture_request()
    request["dummy"] = {"seat": "N", "cards": []}
    with pytest.raises(ValueError, match="hidden"):
        validate_request(request, "ai")
    request, _ = fixture_request(steps=4)
    request["tricks"][0]["winner"] = next(s for s in SEATS if s != request["tricks"][0]["winner"])
    with pytest.raises(ValueError, match="winner"):
        validate_request(request, "ai")


class FakePolicy:
    def predict_batch(self, observations, return_beliefs=False):
        obs = observations[0]
        assert return_beliefs and "hands" not in obs
        card = card_name(obs["legal_card_ids"][0])
        return [{"card": card, "probabilities": [1 / len(obs["legal_card_ids"]) if i in obs["legal_card_ids"] else 0 for i in range(52)]}]


def test_service_calls_model_only_with_visible_observation():
    request, _ = fixture_request()
    result = AnalysisService(FakePolicy(), model_name="fixture").analyze(request, "ai")
    assert result["stateId"] == request["stateId"] and result["source"] == "ai"
    assert result["bestCards"][0] in request["legalCards"]
    assert sum(c["probability"] for c in result["candidates"]) == pytest.approx(1)


@pytest.fixture
def server(tmp_path):
    (tmp_path / "index.html").write_text("bridge table", encoding="utf-8")
    service = AnalysisService(FakePolicy(), model_name="fixture", solver_port=1)
    http = LanServer(("127.0.0.1", 0), service, tmp_path)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{http.server_port}"
    http.shutdown()
    thread.join(3)
    http.server_close()


def test_real_http_routes_cors_and_errors(server):
    request, _ = fixture_request(steps=1)
    payload = json.dumps(request).encode()
    with urlopen(Request(server + "/api/v1/analysis/ai", data=payload,
                         headers={"Content-Type": "application/json", "Origin": "http://localhost"}), timeout=5) as response:
        assert response.status == 200
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost"
        assert json.load(response)["bestCards"][0] in request["legalCards"]
    with urlopen(Request(server + "/api/v1/analysis/ai", method="OPTIONS", headers={"Origin": "http://localhost"}), timeout=5) as response:
        assert response.status == 204
    with urlopen(server + "/api/v1/health", timeout=5) as response:
        assert json.load(response)["services"]["ai"]["ready"] is True
    with urlopen(server, timeout=5) as response:
        assert response.read() == b"bridge table"
    for path, data, headers, expected in [
        ("/api/v1/analysis/ai", payload, {"Content-Type": "text/plain"}, 415),
        ("/api/v1/analysis/ai", b"{}", {"Content-Type": "application/json"}, 400),
        ("/api/v1/analysis/ai", payload, {"Content-Type": "application/json", "Origin": "http://evil.invalid"}, 403),
        ("/api/unknown", payload, {"Content-Type": "application/json"}, 404),
    ]:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(server + path, data=data, headers=headers), timeout=5)
        assert error.value.code == expected
