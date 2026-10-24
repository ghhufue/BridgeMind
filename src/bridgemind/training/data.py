"""Offline datasets. Independent reader for Pgx's packed public DDS data format."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from bridgemind.dds import Deal, TableResult
from bridgemind.dds.models import RANKS, SEATS
from bridgemind.engine import BridgeBiddingEnv
from .encoding import ENCODING_VERSION, encode_env


def ownership(deal):
    deal = Deal.from_dict(deal)
    result = np.empty(52, dtype=np.uint8)
    for seat, hand in deal.hands.items():
        for card in hand:
            result[RANKS.index(card[1]) * 4 + "CDHS".index(card[0])] = SEATS.index(seat)
    return result


def deal_from_ownership(owners):
    owners = np.asarray(owners)
    if owners.shape != (52,) or np.any((owners < 0) | (owners > 3)):
        raise ValueError("Invalid ownership vector")
    return Deal({seat: ["CDHS"[i % 4] + RANKS[i // 4] for i in range(52) if owners[i] == s] for s, seat in enumerate(SEATS)})


def board_ids(owners):
    return np.array([hashlib.sha256(row.tobytes()).hexdigest().encode() for row in np.asarray(owners, dtype=np.uint8)], dtype="S64")


@dataclass
class BoardSet:
    owners: np.ndarray
    tables: np.ndarray
    dealers: np.ndarray
    vulnerable: np.ndarray

    def __post_init__(self):
        self.owners, self.tables, self.dealers, self.vulnerable = map(np.asarray, (self.owners, self.tables, self.dealers, self.vulnerable))
        n = len(self.owners)
        if not n or self.owners.shape != (n, 52) or self.tables.shape != (n, 5, 4) or self.dealers.shape != (n,) or self.vulnerable.shape != (n, 2):
            raise ValueError("Invalid dataset shape")
        for array, lo, hi in ((self.owners, 0, 3), (self.tables, 0, 13), (self.dealers, 0, 3), (self.vulnerable, 0, 1)):
            if array.dtype.kind not in "biu" or np.any((array < lo) | (array > hi)):
                raise ValueError("Invalid dataset values")
        if any(np.any((self.owners == seat).sum(axis=1) != 13) for seat in range(4)):
            raise ValueError("Invalid hand sizes")
        self.owners, self.tables, self.dealers = [a.astype(np.uint8) for a in (self.owners, self.tables, self.dealers)]
        self.vulnerable = self.vulnerable.astype(np.bool_)
        self.ids = board_ids(self.owners)
        if len(np.unique(self.ids)) != n:
            raise ValueError("Duplicate boards")

    def __len__(self):
        return len(self.owners)

    def board(self, index):
        return (deal_from_ownership(self.owners[index]), TableResult(tuple(tuple(int(v) for v in row) for row in self.tables[index])),
                SEATS[int(self.dealers[index])], tuple(bool(v) for v in self.vulnerable[index]))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, owners=self.owners, tables=self.tables, dealers=self.dealers,
                            vulnerable=self.vulnerable, ids=self.ids, schema_version=np.array(1))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if data["schema_version"].item() != 1:
                raise ValueError("Unsupported board dataset version")
            result = cls(*(data[key] for key in ("owners", "tables", "dealers", "vulnerable")))
            if not np.array_equal(result.ids, data["ids"]):
                raise ValueError("Dataset ID mismatch")
            return result


def decode_pgx(keys, values):
    keys, values = np.asarray(keys), np.asarray(values)
    if keys.ndim != 2 or keys.shape[1] != 4 or values.shape != keys.shape:
        raise ValueError("Invalid Pgx key/value arrays")
    owners = np.zeros((len(keys), 52), dtype=np.uint8)
    for suit, symbol in enumerate("SHDC"):
        for rank, symbol_rank in enumerate("A23456789TJQK"):
            owners[:, RANKS.index(symbol_rank) * 4 + "CDHS".index(symbol)] = (keys[:, suit] >> (2 * (12 - rank))) & 3
    tables = np.stack([(values >> (4 * (4 - s))) & 15 for s in range(5)], axis=1).astype(np.uint8)
    return owners, tables


def make_splits(owners, tables, sizes, seed=42):
    if len(sizes) != 3 or any(type(n) is not int or n <= 0 for n in sizes) or sum(sizes) > len(owners):
        raise ValueError("Invalid split sizes")
    if len(np.unique(board_ids(owners))) != len(owners):
        raise ValueError("Duplicate boards before split")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(owners))[:sum(sizes)]
    output, start = {}, 0
    for name, count in zip(("train", "valid", "test"), sizes):
        ix = indices[start:start + count]
        output[name] = BoardSet(owners[ix], tables[ix], rng.integers(0, 4, count, dtype=np.uint8), rng.integers(0, 2, (count, 2), dtype=np.uint8).astype(bool))
        start += count
    return output


def prepare_supervised(source, destination, *, max_boards=None, excluded_ids=(), shard_states=50000):
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        raise ValueError("Supervised destination already has a manifest")
    destination.mkdir(parents=True, exist_ok=True)
    seen, excluded = set(), set(excluded_ids)
    features, masks, labels, ids = [], [], [], []
    shards, total, boards, excluded_count = [], 0, 0, 0

    def flush():
        nonlocal total
        if not labels:
            return
        name = f"shard-{len(shards):05d}.npz"
        np.savez_compressed(destination / name, observations=np.packbits(np.array(features), axis=1),
                            masks=np.packbits(np.array(masks), axis=1), labels=np.array(labels, dtype=np.uint8), ids=np.array(ids, dtype="S64"))
        shards.append({"path": name, "states": len(labels)})
        total += len(labels)
        features.clear(); masks.clear(); labels.clear(); ids.clear()

    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            if max_boards is not None and boards >= max_boards:
                break
            actions = [int(v) for v in line.split()]
            if len(actions) < 56 or sorted(actions[:52]) != list(range(52)):
                raise ValueError("Invalid OpenSpiel trajectory")
            owner = np.zeros(52, dtype=np.uint8)
            owner[actions[:52]] = np.arange(52) % 4
            board_id = board_ids(owner[None])[0]
            if board_id in seen or board_id in excluded:
                excluded_count += 1
                continue
            seen.add(board_id)
            env = BridgeBiddingEnv(None, deal_from_ownership(owner), prefetch=False)
            for raw in actions[52:]:
                if env.is_terminal():
                    break
                action = raw - 52
                obs, mask = encode_env(env)
                if not 0 <= action < 38 or not mask[action]:
                    raise ValueError("Invalid supervised call")
                features.append(obs); masks.append(mask); labels.append(action); ids.append(board_id)
                env.apply_call(action)
            if not env.is_terminal():
                raise ValueError("Incomplete auction")
            boards += 1
            if len(labels) >= shard_states:
                flush()
    flush()
    if not boards:
        raise ValueError("No supervised boards")
    np.save(destination / "board-ids.npy", np.array(sorted(seen), dtype="S64"), allow_pickle=False)
    manifest = {"schema_version": 1, "encoding": ENCODING_VERSION, "boards": boards, "states": total,
                "source": str(Path(source).resolve()), "excluded_boards": excluded_count, "shards": shards}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class SupervisedData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["encoding"] != ENCODING_VERSION:
            raise ValueError("Wrong supervised encoding")
        self.ends = np.cumsum([s["states"] for s in self.manifest["shards"]])
        self.cache = OrderedDict()

    def sample(self, count, rng):
        selected = rng.integers(0, int(self.ends[-1]), count)
        shard_ids = np.searchsorted(self.ends, selected, side="right")
        features, masks, labels = np.empty((count, 480), bool), np.empty((count, 38), bool), np.empty(count, np.int64)
        for shard_id in np.unique(shard_ids):
            key = int(shard_id)
            if key not in self.cache:
                with np.load(self.directory / self.manifest["shards"][key]["path"], allow_pickle=False) as data:
                    self.cache[key] = {n: data[n] for n in ("observations", "masks", "labels")}
                while len(self.cache) > 4:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            data = self.cache[key]
            positions = np.where(shard_ids == key)[0]
            local = selected[positions] - (self.ends[key - 1] if key else 0)
            features[positions] = np.unpackbits(data["observations"][local], axis=1, count=480)
            masks[positions] = np.unpackbits(data["masks"][local], axis=1, count=38)
            labels[positions] = data["labels"][local]
        return features, masks, labels
