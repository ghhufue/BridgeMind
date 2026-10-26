"""Server-owned model allowlist for per-seat spectating; no client file paths."""

from pathlib import Path


class ModelRegistry:
    def __init__(self, service, root):
        self.service, self.root = service, Path(root)
        self.cache = {}
        self.paths = {
            "ai": {
                "sl-long": ("长训 SL", "runs/play-sl-long-20261025/best.pt"),
                "sl-pilot": ("SL 早期模型", "runs/play-sl-pilot/best.pt"),
                "ppo-pilot": ("PPO 试验模型", "runs/play-ppo-pilot/best.pt"),
            },
            "bidding": {"ppo-pilot": ("PPO 叫牌试验模型", "runs/ppo-pilot/last.pt")},
        }

    def catalog(self):
        result = {}
        for source, key in (("ai", "play"), ("bidding", "bidding")):
            default = self.service.policy if source == "ai" else self.service.bidder
            name = self.service.model_name if source == "ai" else self.service.bidding_name
            label = ("SL v2（当前）" if source == "ai" and name and "play-sl-v2" in name else
                     "SL 叫牌（当前）" if source == "bidding" and name and "sl-pilot" in name else name or "默认模型")
            result[key] = [{"id": "default", "name": label, "available": default is not None}]
            result[key] += [{"id": identity, "name": label, "available": True}
                            for identity, (label, path) in self.paths[source].items() if (self.root / path).is_file()]
        result["play"].append({"id": "dds", "name": "DDS 全知求解", "available": self.service.health()["services"]["dds"]["ready"]})
        return {"protocol": "bridgemind.models.v1", **result}

    def resolve(self, source, identity):
        if identity not in self.paths[source]:
            raise ValueError("Unknown modelId")
        label, relative = self.paths[source][identity]
        path = self.root / relative
        if not path.is_file():
            raise ValueError("Model is not available")
        if not self.service.lock.acquire(timeout=1):
            raise TimeoutError("Model is busy")
        try:
            key = (source, identity)
            if key not in self.cache:
                from .analysis import AnalysisService
                if source == "ai":
                    from bridgemind.cardplay.policy import NeuralPlayPolicy
                    from bridgemind.training.runner import sha256
                    child = AnalysisService(NeuralPlayPolicy(path, device="cpu", stochastic=False),
                                            model_name=label, model_sha256=sha256(path))
                else:
                    from bridgemind.training.policy import BiddingPolicy
                    child = AnalysisService(bidder=BiddingPolicy(path, device="cpu"), bidding_name=label)
                child.lock = self.service.lock
                self.cache[key] = child
            return self.cache[key]
        finally:
            self.service.lock.release()
