"""A conservative public-information certificate, beyond oracle agreement alone.

A label is accepted only if DDS agrees in every feasible hidden world AND a
single public continuation policy attains that value against an omniscient
adversary in all those worlds. This supplies an implementable witness instead
of choosing a different omniscient continuation for each hidden deal.
"""

from functools import lru_cache

from bridgemind.dds.models import SEATS, CARDS
from bridgemind.engine.play import card_id, card_name
from bridgemind.rating.policies import RulePolicy, strength
from bridgemind.diagnostics.cardplay import hidden_worlds


RECIPES = ("rule-basic", "cash-public-master")


def public_action(obs, recipe):
    """Receives ordinary observations only, never a complete hidden allocation."""
    if recipe not in RECIPES:
        raise ValueError("Unknown public continuation recipe")
    if recipe == "cash-public-master" and not obs["current_trick"]:
        acting_side = SEATS.index(obs["to_play"]) % 2
        friendly = {c for s, cards in obs["visible_hands"].items() if SEATS.index(s) % 2 == acting_side for c in cards}
        played = {e["card"] for e in obs["play_history"]}
        possible_enemy = set(CARDS)-friendly-played
        trump = obs["contract"]["strain"]
        masters = []
        for action in obs["legal_card_ids"]:
            suit, rank = card_name(action)
            if trump != "NT" and suit != trump and any(c[0] == trump for c in possible_enemy):
                continue
            if not any(c[0] == suit and card_id(c)//4 > action//4 for c in possible_enemy):
                masters.append(action)
        if masters:
            return max(masters, key=lambda c: (c//4, -c % 4))
    return RulePolicy().predict(obs)["card_id"]


def public_view(hands, trick, leader, declarer, trump):
    """Produce exactly the public/actor-visible fields used by witness recipes."""
    actor = (leader+len(trick)) % 4
    controller = declarer if actor % 2 == declarer % 2 else actor
    dummy = (declarer+2) % 4
    visible = {controller, dummy}
    remaining = {c for h in hands for c in h} | set(trick)
    played = set(range(52))-remaining
    # Recipes inspect the identities of played cards, not their order or owners.
    history = [{"card": card_name(c)} for c in sorted(played | set(trick))]
    legal = list(hands[actor])
    if trick:
        following = [c for c in legal if c % 4 == trick[0] % 4]
        legal = following or legal
    return {"state_id": "certificate", "to_play": SEATS[actor], "player": SEATS[controller],
            "contract": {"strain": trump}, "current_trick": [{"seat": SEATS[(leader+i) % 4], "card": card_name(c)} for i, c in enumerate(trick)],
            "legal_card_ids": legal, "visible_hands": {SEATS[s]: [card_name(c) for c in hands[s]] for s in visible},
            "play_history": history}


def witness_value(position, first_action, declarer, recipe):
    """Exact remaining tricks: fixed public team policy vs all enemy legal moves."""
    root_side = SEATS.index(position.to_play) % 2
    d = SEATS.index(declarer)
    trump = position.trump
    initial = tuple(tuple(sorted(card_id(c) for c in position.remaining_hands[s])) for s in SEATS)

    @lru_cache(maxsize=None)
    def search(hands, trick, leader):
        if not trick and not any(hands):
            return 0
        actor = (leader+len(trick)) % 4
        view = public_view(hands, trick, leader, d, trump)
        legal = view["legal_card_ids"]
        if actor % 2 == root_side:
            return step(hands, trick, leader, public_action(view, recipe))
        return min(step(hands, trick, leader, card) for card in legal)

    def step(hands, trick, leader, card):
        actor = (leader+len(trick)) % 4
        after = list(hands)
        after[actor] = tuple(c for c in hands[actor] if c != card)
        current = (*trick, card)
        if len(current) < 4:
            return search(tuple(after), current, leader)
        led = card_name(current[0])[0]
        winner = (leader+max(range(4), key=lambda i: strength(current[i], led, trump))) % 4
        return int(winner % 2 == root_side)+search(tuple(after), (), winner)

    if card_name(first_action) not in position.legal_cards():
        raise ValueError("Certificate first move is illegal")
    return step(initial, tuple(card_id(c) for c in position.current_trick), SEATS.index(position.trick_leader), first_action)


def certify(observation, solver):
    worlds = list(hidden_worlds(observation))
    if not worlds:
        raise ValueError("No feasible hidden world")
    tables, common = [], set(observation["legal_card_ids"])
    for position in worlds:
        result = solver.solve_position(position)
        values = {card_id(m.card): m.tricks_for_evaluated_side for m in result.moves}
        if set(values) != set(observation["legal_card_ids"]):
            raise ValueError("Certificate DDS legal set mismatch")
        tables.append(values)
        common &= {card for card, value in values.items() if value == max(values.values())}
    if not common:
        return {"status": "ungraded", "reason": "no_action_optimal_in_every_publicly_feasible_world", "worlds": len(worlds)}
    witnesses = {}
    for card in sorted(common):
        witness = None
        for recipe in RECIPES:
            proven = 0
            for position, values in zip(worlds, tables):
                lower = witness_value(position, card, observation["contract"]["declarer"], recipe)
                if lower > values[card]:
                    raise ValueError("Witness exceeds DDS minimax bound")
                if lower != values[card]:
                    break
                proven += 1
            if proven == len(worlds):
                witness = {"first_card": card_name(card), "continuation": recipe, "verified_worlds": proven,
                           "adversary": "omniscient, all legal continuations enumerated"}
                break
        if witness:
            witnesses[card_name(card)] = witness
        else:
            return {"status": "ungraded", "reason": "oracle_agreement_without_a_public_continuation_witness", "worlds": len(worlds)}
    worst_regret = {card_name(c): max(max(row.values())-row[c] for row in tables) for c in observation["legal_card_ids"]}
    return {"status": "certified", "accepted_cards": [card_name(c) for c in sorted(common)], "worlds": len(worlds),
            "worst_world_trick_regret": worst_regret, "witness": next(iter(witnesses.values())), "witnesses": witnesses,
            "scope": "each accepted move has universal hidden-world DDS optimality and one implementable public team continuation; ignores auction priors, conservative not exhaustive"}
