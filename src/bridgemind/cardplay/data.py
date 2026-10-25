"""Validated full-play replay, ragged token shards and privileged training labels."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from bridgemind.engine import BridgePlayEnv
from bridgemind.dds.models import SEATS
from bridgemind.training.data import board_ids, deal_from_ownership
from bridgemind.training.runner import sha256
from .encoding import ENCODING_VERSION, encode_observation

SCORE_SCALE = 7600.


def prepare_dataset(source, destination, *, allowed_ids, max_boards, shard_states=4096, system_id=0,
                    system_description="unknown"):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty cardplay dataset destination")
    destination.mkdir(parents=True, exist_ok=True)
    allowed, seen, records, games, shards = set(allowed_ids), set(), [], [], []
    states = 0

    def flush():
        nonlocal states
        if not records:
            return
        name = f"shard-{len(shards):05d}.npz"
        encoded = [r["encoded"] for r in records]
        lengths = [len(r["tokens"]) for r in encoded]
        arrays = {"tokens": np.concatenate([r["tokens"] for r in encoded]),
                  "offsets": np.cumsum([0, *lengths], dtype=np.int32)}
        for key in ("meta", "declarer", "system"):
            arrays[key] = np.asarray([r[key] for r in encoded])
        for key in ("legal", "unknown"):
            arrays[key] = np.packbits(np.asarray([r[key] for r in encoded]), axis=1)
        for key, dtype in (("action", np.uint8), ("owners", np.uint8), ("return", np.float32),
                           ("game", np.int32), ("ply", np.uint8), ("id", "S64")):
            arrays[key] = np.asarray([r[key] for r in records], dtype=dtype)
        np.savez_compressed(destination / name, **arrays)
        shards.append({"path": name, "states": len(records)})
        states += len(records)
        records.clear()

    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            if len(games) >= max_boards:
                break
            actions = [int(x) for x in line.split()]
            if len(actions) < 56 or sorted(actions[:52]) != list(range(52)):
                raise ValueError("Invalid raw deal")
            owner = np.zeros(52, np.uint8)
            owner[actions[:52]] = np.arange(52) % 4
            identity = board_ids(owner[None])[0]
            if identity not in allowed or identity in seen:
                continue
            end = 52
            while end < len(actions) and actions[end] >= 52:
                end += 1
            auction = [x-52 for x in actions[52:end]]
            env = BridgePlayEnv(deal_from_ownership(owner), auction=auction, system_id=system_id)
            if env.is_terminal():
                if end != len(actions):
                    raise ValueError("Passed-out trajectory has card actions")
                continue
            if len(actions[end:]) != 52 or sorted(actions[end:]) != list(range(52)):
                raise ValueError("Expected all 52 card-play actions")
            game_index = len(games)
            game = {"owners": owner.tolist(), "auction": auction, "cards": actions[end:],
                    "dealer": "N", "vulnerable": [False, False], "system_id": system_id, "id": identity.decode()}
            buffer = []
            declarer = SEATS.index(env.contract.declarer)
            relative_owners = ((owner.astype(np.int16)-declarer) % 4).astype(np.uint8)
            for ply, action in enumerate(game["cards"]):
                obs = env.observation()
                encoded = encode_observation(obs)
                if not encoded["legal"][action]:
                    raise ValueError("Recorded play is illegal")
                buffer.append({"encoded": encoded, "action": action, "owners": relative_owners,
                               "game": game_index, "ply": ply, "id": identity, "controller": SEATS.index(obs["player"])})
                env.apply_card(action)
            result = env.result()  # Cross-checks every imported terminal score against OpenSpiel.
            for record in buffer:
                record["return"] = result["returns"][record.pop("controller")] / SCORE_SCALE
            records.extend(buffer)
            seen.add(identity)
            games.append(game)
            if len(records) >= shard_states:
                flush()
    if len(games) != max_boards:
        raise ValueError(f"Source exhausted: imported {len(games)} of {max_boards} requested boards")
    flush()
    with (destination / "games.jsonl").open("w", encoding="utf-8") as stream:
        for game in games:
            stream.write(json.dumps(game) + "\n")
    np.save(destination / "board-ids.npy", np.array(sorted(seen), dtype="S64"), allow_pickle=False)
    manifest = {"schema_version": 1, "encoding": ENCODING_VERSION, "boards": len(games), "states": states,
                "source_sha256": sha256(source), "system_id": system_id, "system_description": system_description,
                "value_target": "actual_played_team_score/7600", "shards": shards}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class PlayData:
    def __init__(self, directory, *, cache_bytes=1024**3):
        if type(cache_bytes) is not int or cache_bytes < 1:
            raise ValueError("Positive shard cache byte budget required")
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["schema_version"] != 1 or self.manifest["encoding"] != ENCODING_VERSION:
            raise ValueError("Unsupported cardplay dataset")
        self.ends = np.cumsum([s["states"] for s in self.manifest["shards"]])
        if self.ends[-1] != self.manifest["states"]:
            raise ValueError("State count mismatch")
        self.ids = np.load(self.directory / "board-ids.npy", allow_pickle=False)
        if len(np.unique(self.ids)) != len(self.ids) or len(self.ids) != self.manifest["boards"]:
            raise ValueError("Invalid or duplicate board IDs")
        self.cache = OrderedDict()
        self.cache_bytes, self.cached_bytes = cache_bytes, 0
        self.games = None

    def row(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.manifest["states"]:
            raise ValueError("State index outside dataset")
        shard = int(np.searchsorted(self.ends, index, side="right"))
        if shard not in self.cache:
            with np.load(self.directory / self.manifest["shards"][shard]["path"], allow_pickle=False) as arrays:
                self.cache[shard] = {key: arrays[key] for key in arrays.files}
            self.cached_bytes += sum(a.nbytes for a in self.cache[shard].values())
            # Keep the requested shard even when it alone exceeds the budget.
            while len(self.cache) > 1 and self.cached_bytes > self.cache_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= sum(a.nbytes for a in evicted.values())
        self.cache.move_to_end(shard)
        data = self.cache[shard]
        local = int(index - (self.ends[shard-1] if shard else 0))
        tokens = data["tokens"][data["offsets"][local]:data["offsets"][local+1]]
        encoded = {"tokens": tokens, **{key: data[key][local] for key in ("meta", "declarer", "system")},
                   **{key: np.unpackbits(data[key][local], count=52).astype(bool) for key in ("legal", "unknown")}}
        return {"encoded": encoded, **{key: data[key][local] for key in ("action", "owners", "return", "game", "ply", "id")}}

    def sample(self, count, rng, *, min_ply=0):
        if not 0 <= min_ply <= 51:
            raise ValueError("min_ply must be in 0..51")
        result = []
        while len(result) < count:
            for index in rng.integers(0, int(self.ends[-1]), max(count-len(result), 32)):
                row = self.row(int(index))
                if row["ply"] >= min_ply:
                    result.append(row)
                    if len(result) == count:
                        break
        return result

    def load_games(self):
        if self.games is None:
            games = [json.loads(line) for line in (self.directory / "games.jsonl").read_text().splitlines()]
            for game in games:
                owners = np.asarray(game["owners"])
                if owners.shape != (52,) or owners.dtype.kind not in "iu" or (owners < 0).any() or (owners > 3).any():
                    raise ValueError("Invalid replay ownership")
                deal_from_ownership(owners)  # Also validates thirteen cards per seat.
                if board_ids(owners[None])[0].decode() != game["id"] or sorted(game["cards"]) != list(range(52)):
                    raise ValueError("Replay identity or card list mismatch")
            if len(games) != len(self.ids) or {g["id"].encode() for g in games} != set(self.ids):
                raise ValueError("Replay games disagree with board IDs")
            self.games = games
        return self.games

    def environment(self, game_index, ply=0):
        if not isinstance(game_index, (int, np.integer)) or not 0 <= game_index < self.manifest["boards"]:
            raise ValueError("Game index outside dataset")
        if not isinstance(ply, (int, np.integer)) or not 0 <= ply <= 52:
            raise ValueError("Play prefix must be in 0..52")
        game = self.load_games()[int(game_index)]
        env = BridgePlayEnv(deal_from_ownership(game["owners"]), auction=game["auction"], dealer=game["dealer"],
                            vulnerable=tuple(game["vulnerable"]), system_id=game["system_id"])
        for card in game["cards"][:int(ply)]:
            env.apply_card(int(card))
        return env
