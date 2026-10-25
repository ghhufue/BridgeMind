"""Public-information rules and an explicitly privileged evaluation-only oracle."""

from bridgemind.dds import Position, Options
from bridgemind.dds.models import SEATS
from bridgemind.engine.play import card_id, card_name
from bridgemind.training.data import deal_from_ownership


def decision(observation, action):
    return {"state_id": observation["state_id"], "card_id": int(action), "card": card_name(int(action))}


def strength(card, led, trump):
    name = card_name(card)
    return (2 if name[0] == trump else 1 if name[0] == led else 0, card // 4)


class RulePolicy:
    """Deterministic v1 rules; consumes only the ordinary private observation."""

    def __init__(self, kind="rule-basic"):
        if kind not in ("lowest", "rule-basic"):
            raise ValueError("Unknown rule policy")
        self.kind = kind

    def predict(self, observation):
        legal = sorted(observation["legal_card_ids"])
        if not legal:
            raise ValueError("No legal card")
        if self.kind == "lowest":
            return decision(observation, legal[0])
        trick, trump = observation["current_trick"], observation["contract"]["strain"]
        if not trick:
            # Lead low from the longest nontrump suit, ties C/D/H/S.
            suits = [s for s in "CDHS" if s != trump and any(card_name(c)[0] == s for c in legal)]
            suit = min(suits, key=lambda s: (-sum(card_name(c)[0] == s for c in legal), "CDHS".index(s))) if suits else trump
            return decision(observation, next(c for c in legal if card_name(c)[0] == suit))
        led = trick[0]["card"][0]
        winner = max(trick, key=lambda item: strength(card_id(item["card"]), led, trump))
        partner_winning = SEATS.index(winner["seat"]) % 2 == SEATS.index(observation["to_play"]) % 2
        if partner_winning:
            return decision(observation, legal[0])
        winning = [c for c in legal if strength(c, led, trump) > strength(card_id(winner["card"]), led, trump)]
        action = min(winning, key=lambda c: (strength(c, led, trump), c)) if winning else legal[0]
        return decision(observation, action)


def oracle_position(board, observation):
    """Build true remaining hands in the evaluator, never in policy observations."""
    hands = {s: set(cards) for s, cards in deal_from_ownership(board["owners"]).hands.items()}
    for event in observation["play_history"]:
        hands[event["seat"]].remove(event["card"])
    return Position({s: sorted(cards) for s, cards in hands.items()}, observation["contract"]["strain"],
                    observation["trick_leader"], [e["card"] for e in observation["current_trick"]])


class OraclePolicy:
    """Only the evaluation runner calls this method with a true Position.

    Maximize acting side's remaining tricks; ties use ascending engine card ID.
    There is deliberately no predict(observation) interface for this policy.
    """

    def __init__(self, solver):
        self.solver = solver

    def predict_position(self, observation, position):
        result = self.solver.solve_position(position, Options(state_id=observation["state_id"]))
        if result.to_play != observation["to_play"] or {card_id(m.card) for m in result.moves} != set(observation["legal_card_ids"]):
            raise ValueError("DDS returned a mismatched position")
        action = min(card_id(m.card) for m in result.moves if m.tricks_for_evaluated_side == result.best_tricks)
        return decision(observation, action)
