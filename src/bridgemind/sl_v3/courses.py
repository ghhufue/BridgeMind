"""Four new public-information tactics with exhaustive DDS/witness certificates."""

import copy
from functools import lru_cache
import numpy as np

from bridgemind.dds import Deal
from bridgemind.dds.models import CARDS, RANKS, SEATS
from bridgemind.engine import BridgePlayEnv, Contract
from bridgemind.engine.play import card_id, card_name
from bridgemind.foundation.certificates import public_action as old_action, public_view
from bridgemind.foundation.cases import observe
from bridgemind.diagnostics.cardplay import hidden_worlds
from bridgemind.rating.policies import strength
from bridgemind.training.data import ownership, board_ids


TEMPLATES = {
    "unblock_discard_short_ace": {"hands": {"N": ["HK", "CK", "CQ", "CJ"], "E": ["H2", "C2", "C3", "C4"],
        "S": ["CA", "D2", "D3", "D4"], "W": ["H3", "H4", "C5", "D5"]}, "leader": "W", "current": ["H3", "HK", "H2"], "strain": "NT"},
    "draw_trumps_before_cash": {"hands": {"N": ["S3", "C2", "H2"], "E": ["S2", "H3", "H4"],
        "S": ["SA", "CA", "CK"], "W": ["S4", "C3", "H5"]}, "leader": "S", "current": [], "strain": "S"},
    "unblock_before_crossing": {"hands": {"N": ["CA", "H2", "D2", "D3"], "E": ["C2", "H3", "D4", "D5"],
        "S": ["CK", "CQ", "CJ", "HA"], "W": ["C3", "C4", "C5", "H4"]}, "leader": "N", "current": [], "strain": "NT"},
    "preserve_friendly_trump_entry": {"hands": {"N": ["SK", "CA", "CK", "C2"], "E": ["H3", "H4", "C3", "C5"],
        "S": ["SA", "S2", "S3", "H2"], "W": ["H5", "C4", "C6", "C7"]}, "leader": "S", "current": [], "strain": "S"},
}


def build_case(family, seed, role):
    template = TEMPLATES[family]
    rng = np.random.default_rng(seed)
    low = sorted({c[1] for hand in template["hands"].values() for c in hand if c[1] in "23456789"}, key=RANKS.index)
    replacements = sorted(rng.choice(list("23456789"), len(low), replace=False), key=RANKS.index)
    ranks = dict(zip(low, replacements))
    def rename(card):
        return card[0]+ranks.get(card[1], card[1])
    reserved = {s: [rename(c) for c in hand] for s, hand in template["hands"].items()}
    used = {c for hand in reserved.values() for c in hand}
    absent = next(s for s in "CDHS" if not any(c[0] == s for c in used))
    # A legal mixed-suit bridge trick adjusts residue counts, then disappears
    # into the public prefix. Nobody has the led (absent) suit in the ending.
    extra_cards = {s: next(c for c in CARDS if c[0] == s and c not in used) for s in "CDHS"}
    winner = template["leader"]
    trump = template["strain"]
    extras = {winner: extra_cards[absent if trump == "NT" else trump]}
    remaining_seats = [s for s in SEATS if s != winner]
    remaining_suits = [s for s in "CDHS" if extra_cards[s] not in extras.values()]
    if trump != "NT":
        mixed_leader = remaining_seats.pop(0)
        extras[mixed_leader] = extra_cards[absent]
        remaining_suits.remove(absent)
    else:
        mixed_leader = winner
    extras.update({s: extra_cards[suit] for s, suit in zip(remaining_seats, remaining_suits)})
    used.update(extras.values())
    groups = []
    for suit in "CDHS":
        available = [c for c in CARDS if c[0] == suit and c not in used]
        if len(available) % 4:
            raise ValueError("Template cannot use whole-suit filler tricks")
        groups.extend(available[i:i+4] for i in range(0, len(available), 4))
    rng.shuffle(groups)
    hands, planned = copy.deepcopy(reserved), []
    for s, card in extras.items():
        hands[s].append(card)
    for i, group in enumerate(groups):
        winner = mixed_leader if i == len(groups)-1 else SEATS[i % 4]
        seats = [s for s in SEATS if s != winner]+[winner]
        cards = dict(zip(seats, sorted(group, key=card_id)))
        for s, card in cards.items():
            hands[s].append(card)
        planned.append(cards)
    actor = (SEATS.index(template["leader"])+len(template["current"])) % 4
    declarer = actor if role == "declarer" else (actor+2) % 4 if role == "dummy" else (actor+1) % 4
    deal = Deal(hands)
    contract = Contract(1, template["strain"], SEATS[declarer])
    env = BridgePlayEnv(deal, contract=contract)
    prefix = []
    for group in planned:
        for _ in range(4):
            card = group[env.to_play]
            env.apply_card(card); prefix.append(card)
    for _ in range(4):
        card = extras[env.to_play]
        env.apply_card(card); prefix.append(card)
    if env.to_play != template["leader"]:
        raise ValueError("Wrong generated leader")
    for card in template["current"]:
        card = rename(card); env.apply_card(card); prefix.append(card)
    own = ownership(deal)
    return {"id": f"{family}-{role}-{seed}", "family": family, "role": role, "seed": seed,
            "mechanical": False, "control": False, "prefix": prefix,
            "board": {"id": board_ids(own[None])[0].decode(), "owners": own.tolist(), "contract": contract.to_dict(),
                      "auction": env.auction, "auction_source": "synthetic_contract", "system_id": 0,
                      "dealer": "N", "vulnerable": [False, False]}}


def public_action(obs, recipe):
    if recipe != "cash-then-cross":
        return old_action(obs, recipe)
    ordinary = old_action(obs, "cash-public-master")
    actor = obs["to_play"]
    side = SEATS.index(actor) % 2
    friendly = {c for s, hand in obs["visible_hands"].items() if SEATS.index(s) % 2 == side for c in hand}
    enemies = set(CARDS)-friendly-{e["card"] for e in obs["play_history"]}
    def master(card):
        return not any(c[0] == card[0] and card_id(c)//4 > card_id(card)//4 for c in enemies)
    trick, trump = obs["current_trick"], obs["contract"]["strain"]
    if trick:
        # A partner's low card need not survive a later opponent. Ruff it when
        # void and a following enemy could still beat it; this is a fixed public
        # continuation, subsequently checked against every adversarial response.
        winner = max(trick, key=lambda e: strength(card_id(e["card"]), trick[0]["card"][0], trump))
        ruffs = [c for c in obs["legal_card_ids"] if card_name(c)[0] == trump]
        if len(trick) < 3 and trump != "NT" and winner["card"][0] != trump and ruffs and not master(winner["card"]) and not any(card_name(c)[0] == trick[0]["card"][0] for c in obs["legal_card_ids"]):
            return min(ruffs)
        return ordinary
    if trump != "NT":
        return ordinary
    # Prefer a current-hand master. If none exists, a public partner master can
    # provide an entry. Every decision uses only an ordinary private observation.
    own_masters = [c for c in obs["legal_card_ids"] if master(card_name(c))]
    if own_masters:
        return max(own_masters, key=lambda c: (c//4, -c % 4))
    partner_cards = [c for s, hand in obs["visible_hands"].items() if s != actor and SEATS.index(s) % 2 == side for c in hand]
    crossing = [c for c in obs["legal_card_ids"] if any(p[0] == card_name(c)[0] and master(p) for p in partner_cards)]
    return min(crossing) if crossing else ordinary


def witness_value(position, first, declarer, recipe):
    root_side, d = SEATS.index(position.to_play) % 2, SEATS.index(declarer)
    trump = position.trump
    initial = tuple(tuple(sorted(card_id(c) for c in position.remaining_hands[s])) for s in SEATS)
    @lru_cache(maxsize=None)
    def search(hands, trick, leader):
        if not any(hands) and not trick:
            return 0
        actor = (leader+len(trick)) % 4
        obs = public_view(hands, trick, leader, d, trump)
        if actor % 2 == root_side:
            return step(hands, trick, leader, public_action(obs, recipe))
        return min(step(hands, trick, leader, c) for c in obs["legal_card_ids"])
    def step(hands, trick, leader, card):
        actor = (leader+len(trick)) % 4
        after = list(hands); after[actor] = tuple(c for c in hands[actor] if c != card)
        current = (*trick, card)
        if len(current) < 4:
            return search(tuple(after), current, leader)
        winner = (leader+max(range(4), key=lambda i: strength(current[i], card_name(current[0])[0], trump))) % 4
        return int(winner % 2 == root_side)+search(tuple(after), (), winner)
    if card_name(first) not in position.legal_cards():
        raise ValueError("Illegal certificate root action")
    return step(initial, tuple(card_id(c) for c in position.current_trick), SEATS.index(position.trick_leader), first)


def certify(case, solver):
    obs = observe(case)
    worlds = list(hidden_worlds(obs))
    if not worlds:
        raise ValueError("No feasible worlds")
    tables, common = [], set(obs["legal_card_ids"])
    for world in worlds:
        result = solver.solve_position(world)
        values = {card_id(m.card): m.tricks_for_evaluated_side for m in result.moves}
        if set(values) != set(obs["legal_card_ids"]):
            raise ValueError("DDS legal set mismatch")
        tables.append(values); common &= {c for c, v in values.items() if v == max(values.values())}
    if not common:
        return {"status": "ungraded", "reason": "no_universal_optimal_action", "worlds": len(worlds)}
    witnesses = {}
    for card in sorted(common):
        for recipe in ("rule-basic", "cash-public-master", "cash-then-cross"):
            lower = [witness_value(w, card, obs["contract"]["declarer"], recipe) for w in worlds]
            if any(v > table[card] for v, table in zip(lower, tables)):
                raise ValueError("Public witness exceeded DDS bound")
            if all(v == table[card] for v, table in zip(lower, tables)):
                witnesses[card_name(card)] = {"first_card": card_name(card), "continuation": recipe, "verified_worlds": len(worlds),
                                             "adversary": "omniscient, all legal continuations"}
                break
        else:
            return {"status": "ungraded", "reason": "no_public_continuation_witness", "worlds": len(worlds)}
    # All-equivalent cases do not teach a preference and are excluded from training.
    if len(common) == len(obs["legal_card_ids"]):
        return {"status": "ungraded", "reason": "all_moves_equivalent_control", "worlds": len(worlds)}
    return {"status": "certified", "accepted_cards": sorted(witnesses, key=card_id), "worlds": len(worlds),
            "witnesses": witnesses, "worst_world_trick_regret": {card_name(c): max(max(t.values())-t[c] for t in tables) for c in obs["legal_card_ids"]},
            "scope": "universal hidden-world optimality plus one executable public continuation per accepted card"}
