"""Visible information only: 52 card tokens plus complete auction/play prefixes."""

import numpy as np
import torch

from bridgemind.dds.models import SEATS, STRAINS
from bridgemind.engine.bidding import CALLS
from bridgemind.engine.play import card_id

ENCODING_VERSION = "play_tokens_v1"
META_SIZE = 52
MAX_AUCTION = 319


def encode_observation(obs):
    declarer = SEATS.index(obs["contract"]["declarer"])
    def relative(seat):
        return (SEATS.index(seat) - declarer) % 4
    auction, plays = obs["auction"], obs["play_history"]
    if not 1 <= len(auction) <= MAX_AUCTION or len(plays) > 51:
        raise ValueError("Invalid auction or nonterminal play prefix length")
    # Columns: kind, value, declarer-relative seat, within-phase order, card status.
    tokens = [[0, 0, relative(obs["player"]), 0, 0]]
    card_states = np.zeros((52, 2), dtype=np.int16)
    card_states[:, 0] = 4  # Unknown seat; never infer it from a private full state.
    for seat, hand in obs["visible_hands"].items():
        for card in hand:
            card_states[card_id(card)] = (relative(seat), 1)
    for event in plays:
        card_states[card_id(event["card"])] = (relative(event["seat"]), 2)
    tokens.extend([1, i, int(seat), i, int(status)] for i, (seat, status) in enumerate(card_states))
    tokens.extend([2, 52 + CALLS.index(r["call"]), relative(r["seat"]), i, 0] for i, r in enumerate(auction))
    tokens.extend([3, card_id(r["card"]), relative(r["seat"]), i, 0] for i, r in enumerate(plays))
    meta = [obs["contract"]["level"] / 7]
    meta += [float(obs["contract"]["strain"] == s) for s in STRAINS]
    meta += [float(obs["contract"]["double_status"] == s) for s in range(3)]
    meta += [float(obs["vulnerable"][declarer % 2]), float(obs["vulnerable"][1-declarer % 2])]
    for seat in (obs["player"], obs["to_play"], obs["trick_leader"], obs["dealer"]):
        meta += [float(relative(seat) == s) for s in range(4)]
    meta += [float(obs["dummy_visible"]), obs["tricks"][declarer % 2] / 13,
             obs["tricks"][1-declarer % 2] / 13, float(obs["auction_source"] == "recorded")]
    relative_seats = [SEATS[(declarer+i) % 4] for i in range(4)]
    meta += [obs["remaining_counts"][s] / 13 for s in relative_seats]
    meta += [float(not plays)]
    meta += [float(suit in obs["voids"][s]) for s in relative_seats for suit in "CDHS"]
    if len(meta) != META_SIZE:
        raise RuntimeError("Metadata layout mismatch")
    legal = np.zeros(52, dtype=np.bool_)
    legal[obs["legal_card_ids"]] = True
    return {"tokens": np.asarray(tokens, np.int16), "meta": np.asarray(meta, np.float32), "legal": legal,
            "unknown": card_states[:, 1] == 0, "declarer": relative(obs["player"]) == 0,
            "system": obs.get("system_id", 0)}


def collate(encoded, device="cpu"):
    if not encoded:
        raise ValueError("Empty batch")
    length = max(len(x["tokens"]) for x in encoded)
    tokens = np.zeros((len(encoded), length, 5), dtype=np.int64)
    padding = np.ones((len(encoded), length), dtype=np.bool_)
    for i, x in enumerate(encoded):
        size = len(x["tokens"])
        tokens[i, :size] = x["tokens"]
        padding[i, :size] = False
    result = {"tokens": tokens, "padding": padding}
    for name in ("meta", "legal", "unknown", "declarer", "system"):
        result[name] = np.asarray([x[name] for x in encoded])
    return {name: torch.as_tensor(value, device=device) for name, value in result.items()}
