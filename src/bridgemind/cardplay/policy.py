"""Neural and search-enhanced policies consume public/private observations only."""

import numpy as np
import torch

from bridgemind.dds.models import SEATS
from bridgemind.engine.play import card_name, card_id
from .encoding import encode_observation, collate
from .model import load_model


def act_batch(model, encoded, rng, *, stochastic=True, return_beliefs=False):
    device = next(model.parameters()).device
    with torch.no_grad():
        distribution, values, beliefs = model.distribution(collate(encoded, device))
        probabilities = distribution.probs.cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum(-1, keepdims=True)
        actions = np.array([rng.choice(52, p=p) for p in probabilities]) if stochastic else probabilities.argmax(-1)
        logprobs = distribution.log_prob(torch.as_tensor(actions, device=device)).cpu().numpy()
        result = (actions, logprobs, values.cpu().numpy())
        if return_beliefs:
            return (*result, probabilities, beliefs.softmax(-1).cpu().numpy())
        return result


class NeuralPlayPolicy:
    def __init__(self, checkpoint=None, *, model=None, device="cpu", seed=None, stochastic=False):
        if (checkpoint is None) == (model is None):
            raise ValueError("Supply checkpoint or model")
        self.model = load_model(checkpoint, device)[0] if checkpoint else model
        self.model.eval()
        self.rng = np.random.default_rng(seed)
        self.stochastic = stochastic

    def predict_batch(self, observations, *, return_beliefs=False):
        encoded = [encode_observation(obs) for obs in observations]
        result = act_batch(self.model, encoded, self.rng, stochastic=self.stochastic, return_beliefs=return_beliefs)
        actions, _, _ = result[:3]
        decisions = []
        for i, obs in enumerate(observations):
            decision = {"state_id": obs["state_id"], "card_id": int(actions[i]), "card": card_name(int(actions[i]))}
            if return_beliefs:
                _, _, _, priors, relative = result
                declarer = SEATS.index(obs["contract"]["declarer"])
                absolute = relative[i][:, [(s-declarer) % 4 for s in range(4)]].copy()
                for seat, cards in obs["visible_hands"].items():
                    for card in cards:
                        absolute[card_id(card)] = np.eye(4)[SEATS.index(seat)]
                for event in obs["play_history"]:
                    absolute[card_id(event["card"])] = np.eye(4)[SEATS.index(event["seat"])]
                decision["ownership_probabilities"] = absolute.tolist()
                decision["probabilities"] = priors[i].tolist()
            decisions.append(decision)
        return decisions

    def predict(self, observation):
        return self.predict_batch([observation])[0]


class HybridPlayPolicy:
    def __init__(self, neural, search):
        self.neural, self.search = neural, search

    def predict(self, observation):
        estimate = self.neural.predict_batch([observation], return_beliefs=True)[0]
        return self.search.predict(observation, probabilities=estimate["ownership_probabilities"],
                                   prior=estimate["probabilities"])
