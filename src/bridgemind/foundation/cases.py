"""Legal full-game fixtures and exact transformations of board and observation."""

import copy
from itertools import permutations

import numpy as np

from bridgemind.dds import Deal
from bridgemind.dds.models import CARDS, RANKS, SEATS
from bridgemind.engine import BridgePlayEnv, Contract
from bridgemind.engine.play import card_id, card_name
from bridgemind.training.data import ownership, board_ids
from bridgemind.diagnostics.cardplay import TEMPLATES as PREVIOUS, make_environment
from bridgemind.rating.system import digest


SUITS = "CDHS"
PERMUTATIONS = tuple("".join(p) for p in permutations(SUITS))
ROLES = {"declarer": "S", "dummy": "N", "defender": "E"}
TEMPLATES = {k: {**copy.deepcopy(v), "strain": "NT", "mechanical": False} for k, v in PREVIOUS.items()}
TEMPLATES.update({
    "cash_master_before_loser": {"hands": {"N": ["H2", "CA"], "E": ["HK", "D3"], "S": ["HA", "HQ"], "W": ["H3", "S4"]},
                                  "leader": "S", "current": [], "level": 1, "strain": "NT"},
    "discard_loser_keep_master": {"hands": {"N": ["H3", "CA"], "E": ["H4", "D3"], "S": ["HA", "H2"], "W": ["HK", "S2"]},
                                    "leader": "W", "current": ["S2", "CA", "D3"], "level": 1, "strain": "NT"},
    "small_ruff_preserve_master": {"hands": {"N": ["C8", "S3", "S4"], "E": ["CK", "CQ", "D2"],
                                                 "S": ["S2", "SA", "H3"], "W": ["C3", "CJ", "S5"]},
                                     "leader": "W", "current": ["C3", "C8", "CK"], "level": 1, "strain": "S"},
    "overruff_not_undertrump": {"hands": {"N": ["C8", "CA", "C2"], "E": ["SK", "S3", "D2"],
                                              "S": ["SA", "SQ", "H3"], "W": ["C3", "CK", "S2"]},
                                  "leader": "W", "current": ["C3", "C8", "SK"], "level": 1, "strain": "S"},
    "equivalent_winning_honors": {"hands": {"N": ["H2", "CA"], "E": ["HQ", "D3"],
                                                "S": ["HA", "HK"], "W": ["H3", "S4"]},
                                    "leader": "W", "current": ["H3", "H2", "HQ"], "level": 1, "strain": "NT", "control": True},
    "forced_follow_suit": {"hands": {"N": ["HK", "C2", "C4"], "E": ["HA", "HQ", "C5"],
                                         "S": ["H2", "C3", "D2"], "W": ["H3", "CA", "S2"]},
                             "leader": "W", "current": ["H3", "HK", "HA"], "level": 1, "strain": "NT", "mechanical": True},
})


def build_base(family, seed, role):
    """Vary low ranks, hidden allocation and legal history before model evaluation."""
    t = TEMPLATES[family]
    rng = np.random.default_rng(seed)
    used_low = sorted({c[1] for h in t["hands"].values() for c in h if c[1] in "23456789"}, key=RANKS.index)
    replacements = sorted(rng.choice(list("23456789"), len(used_low), replace=False), key=RANKS.index)
    ranks = dict(zip(used_low, replacements))
    def renamed(c):
        return c[0]+ranks.get(c[1], c[1])
    reserved = {s: [renamed(c) for c in cards] for s, cards in t["hands"].items()}
    used = {c for h in reserved.values() for c in h}
    groups = []
    for suit in SUITS:
        available = [c for c in CARDS if c[0] == suit and c not in used]
        if len(available) % 4:
            raise ValueError("Ending suit counts must permit whole-suit filler tricks")
        groups.extend(available[i:i+4] for i in range(0, len(available), 4))
    rng.shuffle(groups)
    hands, planned = copy.deepcopy(reserved), []
    for i, group in enumerate(groups):
        winner = t["leader"] if i == len(groups)-1 else SEATS[i % 4]
        seats = [s for s in SEATS if s != winner]+[winner]
        cards = dict(zip(seats, sorted(group, key=card_id)))
        for s, card in cards.items():
            hands[s].append(card)
        planned.append(cards)
    deal = Deal(hands)
    contract = Contract(t["level"], t["strain"], ROLES[role])
    env = BridgePlayEnv(deal, contract=contract)
    prefix = []
    for group in planned:
        for _ in range(4):
            card = group[env.to_play]
            env.apply_card(card); prefix.append(card)
    if env.to_play != t["leader"]:
        raise ValueError("Filler prefix did not produce requested leader")
    for card in t["current"]:
        card = renamed(card)
        env.apply_card(card); prefix.append(card)
    own = ownership(deal)
    board = {"id": board_ids(own[None])[0].decode(), "owners": own.tolist(), "contract": contract.to_dict(),
             "auction": env.auction, "auction_source": "synthetic_contract", "system_id": 0,
             "dealer": "N", "vulnerable": [False, False]}
    return {"id": f"{family}-{role}-{seed}", "family": family, "seed": seed, "role": role,
            "mechanical": t.get("mechanical", False), "control": t.get("control", False), "board": board, "prefix": prefix}


def observe(case):
    env = make_environment(case["board"])
    for card in case["prefix"]:
        env.apply_card(card)
    return env.observation()


def card_map(permutation):
    if permutation not in PERMUTATIONS:
        raise ValueError("Expected a bijection of CDHS")
    return np.array([r*4+SUITS.index(permutation[s]) for r in range(13) for s in range(4)], dtype=np.int64)


def transform_observation(observation, permutation, rotation=0):
    """Exact suit/seat renaming, including trump, auction, voids and scores."""
    mapping = card_map(permutation)
    if rotation not in range(4):
        raise ValueError("Seat rotation must be 0..3")
    def seat(s):
        return SEATS[(SEATS.index(s)+rotation) % 4]
    def card(c):
        return card_name(int(mapping[card_id(c)]))
    def suit(s):
        return "NT" if s == "NT" else permutation[SUITS.index(s)]
    obs = copy.deepcopy(observation)
    obs["state_id"] = digest([observation["state_id"], permutation, rotation])
    for key in ("player", "to_play", "dealer", "dummy", "trick_leader"):
        obs[key] = seat(observation[key])
    obs["contract"]["declarer"] = seat(observation["contract"]["declarer"])
    obs["contract"]["strain"] = suit(observation["contract"]["strain"])
    obs["visible_hands"] = {seat(s): sorted((card(c) for c in h), key=card_id) for s, h in observation["visible_hands"].items()}
    obs["remaining_counts"] = {seat(s): n for s, n in observation["remaining_counts"].items()}
    obs["voids"] = {seat(s): sorted(suit(v) for v in values) for s, values in observation["voids"].items()}
    for key in ("play_history", "current_trick"):
        obs[key] = [{"seat": seat(e["seat"]), "card": card(e["card"])} for e in observation[key]]
    obs["auction"] = [{"seat": seat(e["seat"]), "call": e["call"] if e["call"] in ("P", "X", "XX") else e["call"][0]+suit(e["call"][1:])}
                      for e in observation["auction"]]
    obs["legal_card_ids"] = sorted(int(mapping[c]) for c in observation["legal_card_ids"])
    if rotation % 2:
        obs["tricks"] = list(reversed(observation["tricks"]))
        obs["vulnerable"] = list(reversed(observation["vulnerable"]))
    return obs


def transform_case(case, permutation, rotation=0):
    obs = transform_observation(observe(case), permutation, rotation)
    mapping = card_map(permutation)
    owners = np.empty(52, dtype=np.uint8)
    for c, owner in enumerate(case["board"]["owners"]):
        owners[mapping[c]] = (owner+rotation) % 4
    result = copy.deepcopy(case)
    result["board"].update(owners=owners.tolist(), id=board_ids(owners[None])[0].decode(), contract=obs["contract"],
                            dealer=obs["dealer"], vulnerable=obs["vulnerable"], auction=obs["auction"])
    result["prefix"] = [card_name(int(mapping[card_id(c)])) for c in case["prefix"]]
    if "certificate" in result:
        cert = result["certificate"]
        cert["transport"] = {"suit_permutation": permutation, "seat_rotation": rotation,
                             "rule": "inverse-transform the public observation, run canonical witness, transform its action back"}
        if "accepted_cards" in cert:
            cert["accepted_cards"] = [card_name(int(mapping[card_id(c)])) for c in cert["accepted_cards"]]
        if "worst_world_trick_regret" in cert:
            cert["worst_world_trick_regret"] = {card_name(int(mapping[card_id(c)])): v for c, v in cert["worst_world_trick_regret"].items()}
        if "witness" in cert:
            cert["witness"]["first_card"] = card_name(int(mapping[card_id(cert["witness"]["first_card"])]))
        if "witnesses" in cert:
            moved = {}
            for card, witness in cert["witnesses"].items():
                # Deep copy also avoids aliasing the legacy first-witness field.
                entry = copy.deepcopy(witness)
                renamed = card_name(int(mapping[card_id(card)]))
                entry["first_card"] = renamed
                moved[renamed] = entry
            cert["witnesses"] = moved
    return result
