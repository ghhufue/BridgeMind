"""brl/Pgx 480-bit input layout; independent of OpenSpiel's tensor padding.

Reference: harukaki/brl and sotetsuk/pgx, Apache-2.0. See docs/training.md.
"""

import numpy as np

from bridgemind.dds.models import CARD_ORDER, RANKS, SEATS
from bridgemind.engine.bidding import CALLS

ENCODING_VERSION = "brl480_v1"


def encode480(hand, auction, player, vulnerable):
    if player not in SEATS or len(hand) != 13 or len(set(hand)) != 13 or any(c not in CARD_ORDER for c in hand):
        raise ValueError("Invalid player hand")
    if len(vulnerable) != 2 or any(type(v) not in (bool, np.bool_) for v in vulnerable):
        raise ValueError("Invalid vulnerability")
    position = SEATS.index(player)
    own = position % 2
    result = np.zeros(480, dtype=np.bool_)
    result[int(vulnerable[own])] = True
    result[2 + int(vulnerable[1 - own])] = True
    last_bid = None
    for record in auction:
        actor, call = SEATS.index(record["seat"]), CALLS.index(record["call"])
        relative = (actor - position) % 4
        if call == 0 and last_bid is None:
            result[4 + relative] = True
        elif call >= 3:
            last_bid = call - 3
            result[8 + last_bid * 12 + relative] = True
        elif call in (1, 2):
            if last_bid is None:
                raise ValueError("Double before contract")
            result[8 + last_bid * 12 + call * 4 + relative] = True
    for card in hand:
        result[428 + RANKS.index(card[1]) * 4 + "CDHS".index(card[0])] = True
    return result


def encode_env(env):
    return encode_observation(env.observation())


def encode_observation(obs):
    # OpenSpiel 2.0.2's wheel omits the dealer offset in some history features.
    # Rebuild from public records, never use its padded history block blindly.
    value = encode480(obs["hand"], obs["auction"], obs["player"], obs["vulnerable"])
    mask = np.zeros(38, dtype=np.bool_)
    mask[obs["legal_call_ids"]] = True
    return value, mask
