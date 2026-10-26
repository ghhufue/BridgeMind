"""Transform every suit-indexed field and score target together."""

import copy
from functools import lru_cache
from pathlib import Path

import numpy as np

from bridgemind.cardplay.encoding import encode_observation
from bridgemind.dds.models import SEATS, STRAINS
from bridgemind.engine.bidding import CALLS
from bridgemind.engine.play import card_id, contract_auction
from bridgemind.engine.scoring import Contract, score_contract
from bridgemind.foundation.cases import PERMUTATIONS, card_map, observe, build_base, ROLES, TEMPLATES
from bridgemind.foundation.certificates import certify
from bridgemind.rating.system import read, write, digest


@lru_cache(maxsize=24)
def bid_map(permutation):
    mapped = []
    for call in CALLS:
        if call in ("P", "X", "XX") or call.endswith("NT"):
            mapped.append(CALLS.index(call))
        else:
            mapped.append(CALLS.index(call[0]+permutation["CDHS".index(call[1])]))
    return np.asarray(mapped)


@lru_cache(maxsize=4096)
def recover_tricks(level, strain, doubled, vulnerable, old_score):
    matches = [n for n in range(14) if score_contract(Contract(level, strain, "N", doubled), n, vulnerable) == old_score]
    if len(matches) != 1:
        raise ValueError(f"Cannot recover terminal tricks from supervised score: {old_score}")
    return matches[0]


def score_target(encoded, old_return, new_strain):
    """A minor/major suit swap changes duplicate score, even with identical tricks."""
    meta = encoded["meta"]
    level = int(round(float(meta[0])*7))
    strain = STRAINS[int(np.argmax(meta[1:6]))]
    doubled = int(np.argmax(meta[6:9]))
    vulnerable = bool(meta[9])
    sign = 1 if bool(encoded["declarer"]) else -1
    old_score = int(round(float(old_return)*7600*sign))
    tricks = recover_tricks(level, strain, doubled, vulnerable, old_score)
    return sign*score_contract(Contract(level, new_strain, "N", doubled), tricks, vulnerable)/7600.


def augment(row, permutation, *, contract_only=False):
    if permutation not in PERMUTATIONS:
        raise ValueError("Invalid suit permutation")
    encoded = row["encoded"]
    result = dict(row)
    out = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in encoded.items()}
    mapping = card_map(permutation)
    tokens = out["tokens"]
    # Card tokens are rank-major and must remain indexed by their new card ID.
    cards = encoded["tokens"][1:53].copy()
    tokens[1+mapping] = cards
    tokens[1:53, 1] = np.arange(52)
    tokens[1:53, 3] = np.arange(52)
    bids = tokens[:, 0] == 2
    tokens[bids, 1] = 52+bid_map(permutation)[tokens[bids, 1]-52]
    plays = tokens[:, 0] == 3
    tokens[plays, 1] = mapping[tokens[plays, 1]]
    for name in ("legal", "unknown"):
        out[name][mapping] = encoded[name]
    suits = mapping[:4]
    out["meta"][1+suits] = encoded["meta"][1:5]
    voids = encoded["meta"][-16:].reshape(4, 4)
    out["meta"][-16:].reshape(4, 4)[:, suits] = voids
    if contract_only:
        # Arbitrary suit bijections do NOT preserve the bid ordering or SAYC.
        # Use an explicitly synthetic legal contract summary for augmented real
        # rows; retain original recorded auctions on the unaugmented half.
        level = int(round(float(out["meta"][0])*7))
        strain = STRAINS[int(np.argmax(out["meta"][1:6]))]
        doubled = int(np.argmax(out["meta"][6:9]))
        dealer = int(np.argmax(out["meta"][23:27]))
        calls = contract_auction(Contract(level, strain, "N", doubled), dealer=SEATS[dealer])
        auction = np.asarray([[2, 52+CALLS.index(call), (dealer+i) % 4, i, 0] for i, call in enumerate(calls)], dtype=tokens.dtype)
        out["tokens"] = np.concatenate((tokens[:53], auction, tokens[tokens[:, 0] == 3]))
        out["meta"][30] = 0.  # synthetic_contract, never falsely marked recorded
        out["system"] = 0     # no assumed conventions after renaming
    result["encoded"] = out
    result["action"] = int(mapping[int(row["action"])])
    if "owners" in row:
        result["owners"] = np.empty_like(row["owners"])
        result["owners"][mapping] = row["owners"]
    result["accepted"] = np.zeros(52, dtype=bool)
    original = row.get("accepted")
    if original is None:
        original = np.zeros(52, dtype=bool); original[int(row["action"])] = True
    result["accepted"][mapping] = original
    if not row.get("curriculum", False):
        result["return"] = score_target(encoded, row["return"], STRAINS[int(np.argmax(out["meta"][1:6]))])
    return result


def orbit_key(owners):
    """Group complete deals across all suit bijections and compass rotations."""
    own = np.asarray(owners, dtype=np.uint8)
    variants = []
    for permutation in PERMUTATIONS:
        mapped = np.empty(52, dtype=np.uint8)
        mapped[card_map(permutation)] = own
        variants.extend(((mapped+r) % 4).tobytes() for r in range(4))
    return min(variants).hex()


def prepare_courses(solver, *, train_seeds, valid_seeds, excluded_orbits=()):
    if set(train_seeds) & set(valid_seeds):
        raise ValueError("Curriculum seed overlap")
    families = [k for k, v in TEMPLATES.items() if not v.get("mechanical") and not v.get("control")]
    banned = set(excluded_orbits)
    splits, audits = {}, {}
    for split, seeds in (("valid", valid_seeds), ("train", train_seeds)):
        cases, seen, rejected = [], set(), {}
        for seed in seeds:
            for family in families:
                for role in ROLES:
                    base = build_base(family, seed, role)
                    orbit = orbit_key(base["board"]["owners"])
                    if orbit in banned:
                        rejected["excluded_deal_orbit"] = rejected.get("excluded_deal_orbit", 0)+1
                        continue
                    certificate = certify(observe(base), solver)
                    if certificate["status"] != "certified":
                        reason = certificate["reason"]
                        rejected[reason] = rejected.get(reason, 0)+1
                        continue
                    base["certificate"] = certificate
                    base["orbit"] = orbit
                    cases.append(base); seen.add(orbit)
            print(f"Curriculum {split}: seed {seed}, certified cases {len(cases)}", flush=True)
        if {c["family"] for c in cases} != set(families):
            raise ValueError("Curriculum lost a required tactical family")
        splits[split] = cases
        audits[split] = {"cases": len(cases), "distinct_deal_orbits": len(seen), "rejected": rejected, "seeds": list(seeds)}
        banned.update(seen)
    if {c["orbit"] for c in splits["train"]} & {c["orbit"] for c in splits["valid"]}:
        raise ValueError("Augmentation-equivalent curriculum overlap")
    return splits, audits


class Curriculum:
    def __init__(self, cases):
        self.cases, self.rows, self.groups = cases, [], {}
        for case in cases:
            obs = observe(case)
            accepted = np.zeros(52, dtype=bool)
            accepted[[card_id(c) for c in case["certificate"]["accepted_cards"]]] = True
            row = {"encoded": encode_observation(obs), "accepted": accepted, "action": int(np.flatnonzero(accepted)[0]),
                   "curriculum": True, "owners": np.zeros(52, np.uint8), "return": 0., "family": case["family"]}
            self.groups.setdefault(case["family"], []).append(len(self.rows))
            self.rows.append(row)
        self.families = sorted(self.groups)

    def sample(self, count, rng):
        rows = []
        for _ in range(count):
            family = self.families[int(rng.integers(len(self.families)))]
            indices = self.groups[family]
            rows.append(self.rows[indices[int(rng.integers(len(indices)))]])
        return rows
