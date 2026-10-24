"""Small inference facade for an environment or a future game server."""

import numpy as np
import torch

from bridgemind.engine.bidding import CALLS
from .encoding import encode_observation
from .model import load_model


class BiddingPolicy:
    def __init__(self, checkpoint, *, device="cpu", seed=None):
        self.model, self.metadata = load_model(checkpoint, device)
        self.model.eval().requires_grad_(False)
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, *, stochastic=False):
        features, mask = encode_observation(observation)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            distribution, _ = self.model.distribution(torch.as_tensor(features[None], device=device),
                                                     torch.as_tensor(mask[None], device=device))
            probabilities = distribution.probs[0].cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        action = int(self.rng.choice(38, p=probabilities) if stochastic else probabilities.argmax())
        return {"state_id": observation["state_id"], "call_id": action, "call": CALLS[action],
                "probabilities": {CALLS[i]: float(probabilities[i]) for i in np.flatnonzero(mask)}}
