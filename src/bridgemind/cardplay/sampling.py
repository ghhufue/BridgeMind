"""Exact constrained sampling of a factorized card-ownership proposal.

The proposal is uniform by default. Learned per-card weights are conditioned on
known hands, public voids and exact remaining hand capacities using log-space DP.
This is not a claim that a factorized proposal is the true bridge posterior.
"""

from functools import lru_cache

import numpy as np

from bridgemind.dds import Position
from bridgemind.dds.models import SEATS
from bridgemind.engine.play import card_id, card_name


class DealSampler:
    def __init__(self, observation, probabilities=None):
        self.observation = observation
        counts = observation["remaining_counts"]
        visible = observation["visible_hands"]
        if set(counts) != set(SEATS) or any(type(v) is not int or not 0 <= v <= 13 for v in counts.values()):
            raise ValueError("Invalid remaining capacities")
        if observation["player"] not in visible or observation["to_play"] not in visible:
            raise ValueError("Controller and acting hand must be visible")
        self.known = {s: list(cards) for s, cards in visible.items()}
        played = [card_id(r["card"]) for r in observation["play_history"]]
        known_cards = [card_id(c) for cards in visible.values() for c in cards]
        if len(set(played + known_cards)) != len(played + known_cards):
            raise ValueError("Duplicated visible or played card")
        if sum(counts.values()) + len(played) != 52:
            raise ValueError("Inconsistent public card counts")
        if any(sum(r["seat"] == s for r in observation["play_history"]) != 13-counts[s] for s in SEATS):
            raise ValueError("Public play history disagrees with hand capacities")
        if any(s not in SEATS or len(cards) != counts[s] for s, cards in visible.items()):
            raise ValueError("Invalid visible hand size")
        if any(c[0] in observation["voids"][s] for s, cards in visible.items() for c in cards):
            raise ValueError("Visible hand contradicts public voids")
        self.seats = [s for s in SEATS if s not in visible]
        self.capacities = tuple(counts[s] for s in self.seats)
        unavailable = set(played + known_cards)
        self.unknown = [i for i in range(52) if i not in unavailable]
        if len(self.unknown) != sum(self.capacities):
            raise ValueError("Unknown cards disagree with capacities")
        weights = np.ones((52, 4), np.float64) if probabilities is None else np.asarray(probabilities, dtype=np.float64)
        if weights.shape != (52, 4) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Invalid ownership weights")
        self.logs = np.full((len(self.unknown), len(self.seats)), -np.inf)
        for i, card in enumerate(self.unknown):
            for j, seat in enumerate(self.seats):
                weight = weights[card, SEATS.index(seat)]
                if card_name(card)[0] not in observation["voids"][seat] and weight > 0:
                    self.logs[i, j] = np.log(weight)

        @lru_cache(maxsize=None)
        def log_mass(capacities):
            if not any(capacities):
                return 0.
            i = len(self.unknown) - sum(capacities)
            total = -np.inf
            for j, remaining in enumerate(capacities):
                if remaining and np.isfinite(self.logs[i, j]):
                    after = list(capacities)
                    after[j] -= 1
                    total = np.logaddexp(total, self.logs[i, j] + log_mass(tuple(after)))
            return float(total)

        self._log_mass = log_mass
        if not np.isfinite(log_mass(self.capacities)):
            raise ValueError("No hidden deal satisfies public constraints and proposal weights")

    def sample(self, count, rng):
        if type(count) is not int or count < 1:
            raise ValueError("Positive sample count required")
        positions = []
        for _ in range(count):
            hands = {s: list(self.known.get(s, [])) for s in SEATS}
            capacities = list(self.capacities)
            for i, card in enumerate(self.unknown):
                logs = np.full(len(self.seats), -np.inf)
                for j, remaining in enumerate(capacities):
                    if remaining and np.isfinite(self.logs[i, j]):
                        after = list(capacities)
                        after[j] -= 1
                        logs[j] = self.logs[i, j] + self._log_mass(tuple(after))
                probabilities = np.exp(logs - self._log_mass(tuple(capacities)))
                probabilities /= probabilities.sum()
                selected = int(rng.choice(len(self.seats), p=probabilities))
                hands[self.seats[selected]].append(card_name(card))
                capacities[selected] -= 1
            obs = self.observation
            positions.append(Position(hands, obs["contract"]["strain"], obs["trick_leader"],
                                      [r["card"] for r in obs["current_trick"]]))
        return positions
