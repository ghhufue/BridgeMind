import json
import threading
from urllib.request import urlopen

import pytest

from bridgemind.engine.bidding import CALLS
from bridgemind.web.analysis import AnalysisService, PROTOCOL
from bridgemind.web.models import ModelRegistry
from bridgemind.web.server import LanServer
from test_web_analysis import fixture_request


class Policy:
    def __init__(self, last=False):
        self.last = last
        self.observations = []

    def predict_batch(self, observations, return_beliefs=False):
        self.observations.extend(observations)
        legal = observations[0]["legal_card_ids"]
        action = legal[-1 if self.last else 0]
        from bridgemind.engine.play import card_name
        return [{"card": card_name(action), "probabilities": [float(i == action) for i in range(52)]}]


def registry(tmp_path, monkeypatch):
    default, alternate = Policy(), Policy(last=True)
    service = AnalysisService(default, model_name="default")
    service.models = ModelRegistry(service, tmp_path)
    (tmp_path / "trusted.pt").write_bytes(b"fixture")
    service.models.paths["ai"] = {"alternate": ("alternate", "trusted.pt")}
    loads = []
    def load(path, **kwargs):
        loads.append(path)
        return alternate
    monkeypatch.setattr("bridgemind.cardplay.policy.NeuralPlayPolicy", load)
    return service, default, alternate, loads


def test_selection_routes_to_distinct_cached_models_without_changing_default(tmp_path, monkeypatch):
    service, default, alternate, loads = registry(tmp_path, monkeypatch)
    request, obs = fixture_request(steps=1)
    baseline = service.analyze({**request, "modelId": "default"}, "ai")
    selected = service.analyze({**request, "modelId": "alternate"}, "ai")
    assert baseline["bestCards"] != selected["bestCards"]
    assert selected["modelId"] == selected["model"] == "alternate"
    assert alternate.observations == [obs]  # No hidden hands introduced by model selection.
    assert service.policy is default
    service.analyze({**request, "modelId": "alternate"}, "ai")
    assert len(loads) == 1
    assert service.analyze(request, "ai")["model"] == "default"


@pytest.mark.parametrize("identity", ["../../private.pt", "/tmp/model.pt", "missing", "dds", 2, None, ""])
def test_client_cannot_select_files_or_unknown_models(tmp_path, monkeypatch, identity):
    service, _, _, loads = registry(tmp_path, monkeypatch)
    request, _ = fixture_request(steps=1)
    with pytest.raises(ValueError):
        service.analyze({**request, "modelId": identity}, "ai")
    assert not loads


def test_invalid_information_is_rejected_before_alternate_model_load(tmp_path, monkeypatch):
    service, _, _, loads = registry(tmp_path, monkeypatch)
    request, _ = fixture_request()
    with pytest.raises(ValueError):
        service.analyze({**request, "modelId": "alternate", "hands": {}}, "ai")
    assert not loads
    full, _ = fixture_request("dds")
    with pytest.raises(ValueError):
        service.analyze({**full, "modelId": "alternate"}, "dds")


def test_bidding_selection_uses_private_observation_and_separate_cache(tmp_path, monkeypatch):
    class Bidder:
        def predict(self, obs, stochastic=False):
            assert "hands" not in obs and len(obs["hand"]) == 13
            return {"call": "P", "probabilities": {"P": 1.}}
    from bridgemind.dds.models import CARDS
    service, _, _, _ = registry(tmp_path, monkeypatch)
    service.models.paths["bidding"] = {"alternate": ("bidder", "trusted.pt")}
    monkeypatch.setattr("bridgemind.training.policy.BiddingPolicy", lambda *args, **kwargs: Bidder())
    request = {"protocol": PROTOCOL, "stateId": "b", "board": 1, "dealer": "N", "turn": "N", "vulnerable": [False, False],
               "auction": [], "hand": {"seat": "N", "cards": list(CARDS[:13])}, "legalCalls": ["P", *CALLS[3:]], "modelId": "alternate"}
    result = service.analyze(request, "bidding")
    assert result["call"] == "P" and result["modelId"] == "alternate" and result["model"] == "bidder"
    assert ("bidding", "alternate") in service.models.cache


def test_catalog_http_does_not_load_weights_or_expose_paths(tmp_path, monkeypatch):
    service, _, _, loads = registry(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "health", lambda: {"services": {"dds": {"ready": True}}})
    server = LanServer(("127.0.0.1", 0), service, tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/v1/models") as response:
            data = json.load(response)
        assert data["protocol"] == "bridgemind.models.v1"
        assert {row["id"] for row in data["play"]} == {"default", "alternate", "dds"}
        assert all(set(row) == {"id", "name", "available"} for row in data["play"])
        assert not loads
    finally:
        server.shutdown(); server.server_close(); worker.join()
