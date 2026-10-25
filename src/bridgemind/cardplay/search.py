"""PIMC baseline using the independent DDS facade; no access to the real deal."""

import time

import numpy as np

from bridgemind.dds import Options, PositionResult, SolverError
from bridgemind.dds.models import SEATS
from bridgemind.engine.play import card_name
from bridgemind.engine.scoring import Contract, score_contract
from .sampling import DealSampler


class SamplingDdsPolicy:
    def __init__(self, solver, *, samples=32, batch_size=32, seed=None, wait_timeout_ms=30000):
        if type(samples) is not int or samples < 1 or type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("Invalid sample count or DDS batch size")
        self.solver, self.samples, self.batch_size = solver, samples, batch_size
        self.rng = np.random.default_rng(seed)
        self.wait_timeout_ms = wait_timeout_ms

    def predict(self, observation, *, probabilities=None, prior=None):
        started = time.perf_counter()
        legal = observation["legal_card_ids"]
        if not legal:
            raise ValueError("No legal card")
        if len(legal) == 1:
            return {"state_id": observation["state_id"], "card_id": legal[0], "card": card_name(legal[0]),
                    "samples": 0, "seconds": time.perf_counter() - started, "evaluation_model": "forced_card"}
        positions = DealSampler(observation, probabilities).sample(self.samples, self.rng)
        contract = Contract(**observation["contract"])
        declarer_side = SEATS.index(contract.declarer) % 2
        acting_side = SEATS.index(observation["to_play"]) % 2
        remaining = 13 - sum(observation["tricks"])
        scores = {card_name(i): [] for i in legal}
        for start in range(0, self.samples, self.batch_size):
            options = Options(wait_timeout_ms=self.wait_timeout_ms, state_id=observation["state_id"], priority="interactive")
            batch = positions[start:start+self.batch_size]
            items = self.solver.solve_positions(batch, options)
            if len(items) != len(batch):
                raise ValueError("Incomplete DDS batch")
            for item in items:
                if item["status"] != "succeeded":
                    raise SolverError.from_dict(item["error"])
                if item.get("state_id") != observation["state_id"]:
                    raise ValueError("Stale DDS result")
                result = PositionResult.from_dict(item["result"])
                if result.to_play != observation["to_play"] or set(m.card for m in result.moves) != set(scores):
                    raise ValueError("DDS action set does not match observation")
                for move in result.moves:
                    future = move.tricks_for_evaluated_side
                    tricks = observation["tricks"][declarer_side] + (future if acting_side == declarer_side else remaining-future)
                    score = score_contract(contract, tricks, observation["vulnerable"][declarer_side])
                    scores[move.card].append(score if acting_side == declarer_side else -score)
        means = {card: float(np.mean(values)) for card, values in scores.items()}
        maximum = max(means.values())
        tied = [i for i in legal if means[card_name(i)] == maximum]
        action = max(tied, key=lambda i: float(prior[i])) if prior is not None else tied[0]
        return {"state_id": observation["state_id"], "card_id": action, "card": card_name(action),
                "mean_team_scores": means, "samples": self.samples, "seconds": time.perf_counter() - started,
                "evaluation_model": "sampled_double_dummy", "utility": "expected_raw_duplicate_score"}


class RandomPlayPolicy:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def predict(self, observation):
        action = int(self.rng.choice(observation["legal_card_ids"]))
        return {"state_id": observation["state_id"], "card_id": action, "card": card_name(action)}
