"""Immutable benchmark protocols, append-only roster, resumable paired matches."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
import torch

from bridgemind.cardplay.data import PlayData
from bridgemind.cardplay.model import load_model, read_checkpoint
from bridgemind.cardplay.policy import NeuralPlayPolicy, HybridPlayPolicy
from bridgemind.cardplay.runner import fingerprint
from bridgemind.cardplay.search import SamplingDdsPolicy, RandomPlayPolicy
from bridgemind.dds import SolverClient
from bridgemind.dds.models import SEATS
from bridgemind.engine import BridgePlayEnv
from bridgemind.training.data import deal_from_ownership
from bridgemind.training.rollout import imps
from bridgemind.training.runner import sha256, resolve_device
from .policies import RulePolicy, OraclePolicy, oracle_position
from .scoring import rate


KINDS = ("random", "oracle", "lowest", "rule-basic", "sampling-dds", "neural", "hybrid")
DEFAULT_PANEL = ("random", "lowest", "rule-basic", "oracle")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise ValueError("IDs must start with a-z and contain only a-z, 0-9, hyphens (max 64)")
    return value


@contextmanager
def writer_lock(root):
    """OS-released single-writer lock, including after a crashed worker."""
    with (Path(root)/".writer.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Another rating writer is running") from exc
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def implementation():
    package = Path(__file__).resolve().parents[1]
    # Freeze the engine, encoders, policies, sampler, solver client and scorer.
    files = {p.relative_to(package).as_posix(): sha256(p)
             for folder in ("rating", "engine", "cardplay", "training", "dds")
             for p in sorted((package/folder).glob("*.py"))}
    return digest(files)


def runtime(device):
    return {"device": resolve_device(device), "torch": str(torch.__version__), "numpy": str(np.__version__),
            "open_spiel": importlib.metadata.version("open-spiel"), "python": platform.python_version(),
            "cpu_threads": 1}


def solver_identity(solver):
    info = solver.get_info()
    return {key: info[key] for key in ("solver_version", "solver_identity")}


def specification(name, kind, *, checkpoint=None, samples=None, training_data=()):
    identifier(name)
    if kind not in KINDS:
        raise ValueError("Unknown opponent kind")
    if (kind in ("neural", "hybrid")) != (checkpoint is not None):
        raise ValueError("Only neural/hybrid policies require a cardplay checkpoint")
    if kind in ("sampling-dds", "hybrid"):
        if type(samples) is not int or samples < 1:
            raise ValueError("Positive fixed sample count required")
    elif samples is not None:
        raise ValueError("This policy does not use a sample count")
    result = {"id": name, "kind": kind, "policy_version": 1}
    if checkpoint:
        path = Path(checkpoint).resolve()
        read_checkpoint(path)
        result.update(checkpoint=str(path), checkpoint_sha256=sha256(path))
        if training_data:
            datasets = [PlayData(p) for p in training_data]
            result["training_ids"] = sorted({i.decode() for data in datasets for i in data.ids})
            result["declared_training_data"] = [{"path": str(Path(p).resolve()), "fingerprint": fingerprint(p)} for p in training_data]
        result["training_audit"] = "known_local_training_ids" if "training_ids" in result else "training_history_not_fully_audited"
    elif training_data:
        raise ValueError("Training datasets only apply to neural/hybrid policies")
    if samples is not None:
        result["samples"] = samples
    return result


def check_training_overlap(spec, boards):
    overlap = set(spec.get("training_ids", [])).intersection(b["id"] for b in boards)
    if overlap:
        raise ValueError(f"{spec['id']}: benchmark overlaps {len(overlap)} known training deals")


def initialize(root, data, *, name, count=128, seeds=(20261026, 20261027, 20261028),
               device="cpu", port=18765, draws=2000, purpose="tracking", exclude_data=(), source_audit=None):
    root = Path(root).resolve()
    identifier(name)
    if type(count) is not int or count < 2 or type(draws) is not int or draws < 200:
        raise ValueError("At least two boards and 200 bootstrap draws required")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Distinct nonnegative replicate seeds required")
    if purpose not in ("tracking", "confirmation"):
        raise ValueError("Unknown benchmark purpose")
    if purpose == "confirmation" and not exclude_data:
        raise ValueError("Confirmation requires --exclude-data training/validation/prior benchmark datasets")
    source = PlayData(data)
    games = source.load_games()
    if count > len(games):
        raise ValueError("Requested board count exceeds source dataset")
    indices = np.random.default_rng(seeds[0]).permutation(len(games))[:count]
    boards = [{key: games[int(i)][key] for key in ("id", "owners", "auction", "dealer", "vulnerable", "system_id")}
              for i in indices]
    audits = []
    for path in exclude_data:
        excluded = set(i.decode() for i in PlayData(path).ids)
        if excluded.intersection(b["id"] for b in boards):
            raise ValueError(f"Benchmark overlaps excluded dataset: {path}")
        audits.append({"path": str(Path(path).resolve()), "fingerprint": fingerprint(path), "intersection": 0})
    # Validate each frozen auction and retain no original hidden-card labels in observations.
    for board in boards:
        if environment(board).is_terminal():
            raise ValueError("Cardplay rating excludes passed-out boards")
    with SolverClient(port=port) as solver:
        identity = solver_identity(solver)
    protocol = {"schema_version": 1, "name": name, "created_at_utc": now(), "purpose": purpose,
                "implementation_sha256": implementation(), "runtime": runtime(device), "solver": identity,
                "source": {"path": str(Path(data).resolve()), "fingerprint": fingerprint(data),
                           "manifest": source.manifest, "exclusion_audits": audits, "preparation_audit": source_audit},
                "boards_sha256": digest(boards), "boards": count, "seeds": list(seeds), "bootstrap_draws": draws,
                "panel": [{"id": key, "weight": 1./len(DEFAULT_PANEL)} for key in DEFAULT_PANEL],
                "anchors": {"random": 0, "oracle": 100}, "oracle_objective": "max_acting_side_tricks_then_lowest_card_id",
                "scope": "cardplay only; fixed recorded auction; same policy controls both teammates",
                "score": "100*(panel_IMP(candidate)-panel_IMP(random))/(panel_IMP(oracle)-panel_IMP(random))",
                "limits": ["Repeated tracking/model selection consumes test independence; reserve a separate confirmation suite",
                           "Oracle has privileged information; 100 is a reference, not a hard ceiling",
                           "Public pretraining history cannot be fully audited"]}
    entries = [specification(key, key) for key in DEFAULT_PANEL]
    entries += [specification(f"sampling-dds-{n}", "sampling-dds", samples=n) for n in (8, 32)]
    root.mkdir(parents=True, exist_ok=False)
    write(root/"boards.json", boards)
    for entry in entries:
        write(root/"opponents"/(entry["id"]+".json"), entry)
    # Commit the protocol last; a partial initialization never looks complete.
    write(root/"protocol.json", protocol)
    return protocol


def environment(board):
    return BridgePlayEnv(deal_from_ownership(board["owners"]), auction=board["auction"], dealer=board["dealer"],
                         vulnerable=tuple(board["vulnerable"]), system_id=board["system_id"])


def seeded(seed, board_id, policy_id):
    return int(digest([seed, board_id, policy_id])[:16], 16)


class Factory:
    def __init__(self, device, solver=None):
        self.device, self.solver, self.models = device, solver, {}

    def create(self, spec, seed):
        kind = spec["kind"]
        if kind == "random":
            return RandomPlayPolicy(seed)
        if kind in ("lowest", "rule-basic"):
            return RulePolicy(kind)
        if kind == "oracle":
            return OraclePolicy(self.solver)
        if kind == "sampling-dds":
            return SamplingDdsPolicy(self.solver, samples=spec["samples"], seed=seed)
        if spec["checkpoint_sha256"] not in self.models:
            self.models[spec["checkpoint_sha256"]] = load_model(spec["checkpoint"], self.device)[0]
        neural = NeuralPlayPolicy(model=self.models[spec["checkpoint_sha256"]], seed=seed)
        return neural if kind == "neural" else HybridPlayPolicy(neural, SamplingDdsPolicy(self.solver, samples=spec["samples"], seed=seed))


def duel(board, left, right, seed, factory):
    """Two actual rooms; ordinary actors never receive a true Position or the board."""
    rooms, latency = [], {left["id"]: [], right["id"]: []}
    for room in (0, 1):
        env = environment(board)
        policies = {s["id"]: factory.create(s, seeded(seed, board["id"], s["id"])) for s in (left, right)}
        for _ in range(52):
            obs = env.observation()
            spec = left if SEATS.index(obs["player"]) % 2 == room else right
            policy = policies[spec["id"]]
            started = time.perf_counter()
            if spec["kind"] == "oracle":
                result = policy.predict_position(obs, oracle_position(board, obs))
            else:
                result = policy.predict(obs)
            latency[spec["id"]].append((time.perf_counter()-started)*1000.)
            if result.get("state_id") != obs["state_id"] or result.get("card_id") not in obs["legal_card_ids"]:
                raise ValueError("Stale or illegal rating policy action")
            env.apply_card(result["card_id"])
        if not env.is_terminal():
            raise ValueError("Incomplete rated game")
        rooms.append(env.result())
    difference = rooms[0]["returns"][0]-rooms[1]["returns"][0]
    side = SEATS.index(rooms[0]["contract"]["declarer"]) % 2
    return {"board_id": board["id"], "seed": seed, "imps": imps(difference), "raw_difference": difference,
            "left_declarer_tricks": rooms[side]["declarer_tricks"],
            "right_declarer_tricks": rooms[1-side]["declarer_tricks"],
            "latency": {key: {"count": len(values), "mean_ms": float(np.mean(values)),
                              "p95_ms": float(np.quantile(values, .95)), "max_ms": max(values)}
                        for key, values in latency.items()}}


class RatingSystem:
    def __init__(self, root, *, port=18765):
        self.root, self.port = Path(root).resolve(), port
        self.protocol = read(self.root/"protocol.json")
        self.boards = read(self.root/"boards.json")
        p = self.protocol
        if (p.get("schema_version") != 1 or p.get("anchors") != {"random": 0, "oracle": 100} or
                p["panel"] != [{"id": k, "weight": 1./len(DEFAULT_PANEL)} for k in DEFAULT_PANEL]):
            raise ValueError("Unsupported rating protocol or changed fixed panel/anchors")
        if p["implementation_sha256"] != implementation():
            raise ValueError("Rating implementation changed; create a new benchmark version")
        if p["runtime"] != runtime(p["runtime"]["device"]):
            raise ValueError("Rating runtime changed; create a new benchmark version")
        if p["boards_sha256"] != digest(self.boards) or len(self.boards) != p["boards"]:
            raise ValueError("Frozen benchmark boards changed")
        self.protocol_hash = digest(p)
        self.roster = {path.stem: read(path) for path in sorted((self.root/"opponents").glob("*.json"))}
        for key, entry in self.roster.items():
            if key != entry["id"]:
                raise ValueError("Opponent registry ID mismatch")
        for item in p["panel"]:
            if self.roster[item["id"]] != specification(item["id"], item["id"]):
                raise ValueError("Frozen reference panel changed")

    def register(self, spec):
        check_training_overlap(spec, self.boards)
        path = self.root/"opponents"/(identifier(spec["id"])+".json")
        if path.exists():
            if read(path) != spec:
                raise ValueError("Opponent ID already exists with a different immutable specification")
            return
        write(path, spec)
        self.roster[spec["id"]] = spec

    def verify_entry(self, name):
        identifier(name)
        if name not in self.roster:
            raise ValueError(f"Unknown opponent: {name}")
        spec = self.roster[name]
        check_training_overlap(spec, self.boards)
        if "checkpoint" in spec and sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {name}")
        return spec

    def match(self, a, b):
        names = sorted((a, b))
        left, right = [self.verify_entry(key) for key in names]
        signature = digest({"protocol": self.protocol_hash, "left": left, "right": right})
        path = self.root/"matches"/("--".join(names)+".json")
        result = read(path) if path.exists() else {"signature": signature, "left": names[0], "right": names[1], "records": []}
        if result["signature"] != signature:
            raise ValueError("Cached match belongs to different frozen policies/protocol")
        if path.exists() and result.get("records_sha256") != digest(result["records"]):
            raise ValueError("Cached match records checksum mismatch")
        coordinates = [(seed, board) for seed in self.protocol["seeds"] for board in self.boards]
        if len(result["records"]) > len(coordinates):
            raise ValueError("Too many cached match records")
        for record, (seed, board) in zip(result["records"], coordinates):
            if record["seed"] != seed or record["board_id"] != board["id"] or record["imps"] != imps(record["raw_difference"]):
                raise ValueError("Cached match coordinates or IMP mismatch")
        if len(result["records"]) < len(coordinates):
            needs_solver = a != b and any(s["kind"] in ("oracle", "sampling-dds", "hybrid") for s in (left, right))
            with self.solver(needs_solver) as solver:
                factory = Factory(self.protocol["runtime"]["device"], solver)
                for seed, board in coordinates[len(result["records"]):]:
                    # An identical policy uses identical per-board RNG in both rooms.
                    # Its paired result is exactly zero; no oracle computation needed.
                    record = ({"board_id": board["id"], "seed": seed, "imps": 0, "raw_difference": 0,
                               "self_comparison": True} if a == b else duel(board, left, right, seed, factory))
                    result["records"].append(record)
                    result["records_sha256"] = digest(result["records"])
                    write(path, result)
                    if len(result["records"]) % 8 == 0 or len(result["records"]) == len(coordinates):
                        print(json.dumps({"match": names, "completed": len(result["records"]), "total": len(coordinates)}), flush=True)
        sign = 1 if a == names[0] else -1
        values = np.asarray([r["imps"]*sign for r in result["records"]], dtype=np.float64)
        return values.reshape(len(self.protocol["seeds"]), len(self.boards)), result

    @contextmanager
    def solver(self, needed):
        if not needed:
            yield None
            return
        with SolverClient(port=self.port) as solver:
            if solver_identity(solver) != self.protocol["solver"]:
                raise ValueError("DDS binary/version differs from frozen benchmark")
            yield solver

    def panel(self, name):
        matrices, details = [], {}
        for item in self.protocol["panel"]:
            values, result = self.match(name, item["id"])
            matrices.append(values*item["weight"])
            means = values.mean(axis=0)
            rng = np.random.default_rng(12345)
            samples = []
            for _ in range(self.protocol["bootstrap_draws"]):
                rows = rng.integers(values.shape[0], size=values.shape[0])
                cols = rng.integers(values.shape[1], size=values.shape[1])
                samples.append(values[np.ix_(rows, cols)].mean())
            details[item["id"]] = {"mean_imps": float(values.mean()), "bootstrap_95_ci": np.quantile(samples, [.025, .975]).tolist(),
                                   "boards": len(means), "replicates": values.shape[0],
                                   "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
                                   "counts_scope": "seed-board outcomes, not distinct independent deals",
                                   "match_signature": result["signature"]}
        return np.sum(matrices, axis=0), details

    def evaluate(self, name):
        candidate, details = self.panel(name)
        random_values, _ = self.panel("random")
        oracle_values, _ = self.panel("oracle")
        score = rate(candidate, random_values, oracle_values, draws=self.protocol["bootstrap_draws"],
                     anchor=name if name in ("random", "oracle") else None)
        report = {"schema_version": 1, "id": name, "benchmark": self.protocol["name"], "protocol_sha256": self.protocol_hash,
                  "opponent_sha256": digest(self.roster[name]), "created_at_utc": now(), "purpose": self.protocol["purpose"],
                  "score": score, "versus": details, "policy": {k: v for k, v in self.roster[name].items() if k != "training_ids"},
                  "limits": self.protocol["limits"]}
        write(self.root/"reports"/(name+".json"), report)
        return report

    def leaderboard(self):
        rows = []
        for path in sorted((self.root/"reports").glob("*.json")):
            report = read(path)
            if report["protocol_sha256"] != self.protocol_hash or report["opponent_sha256"] != digest(self.verify_entry(report["id"])):
                raise ValueError("Stale leaderboard report")
            rows.append({"id": report["id"], **report["score"]})
        rows.sort(key=lambda row: row["rating"], reverse=True)
        board = {"benchmark": self.protocol["name"], "protocol_sha256": self.protocol_hash,
                 "purpose": self.protocol["purpose"], "ratings": rows,
                 "unrated": sorted(set(self.roster)-{row["id"] for row in rows})}
        write(self.root/"leaderboard.json", board)
        lines = [f"# {self.protocol['name']}", "", f"Purpose: {self.protocol['purpose']}; fixed auctions; random=0, full-information DDS=100.",
                 "", "| Policy | Rating | 95% CI | Status |", "|---|---:|---|---|"]
        for row in rows:
            ci = row["rating_95_ci"]
            interval = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "unavailable"
            lines.append(f"| {row['id']} | {row['rating']:.2f} | {interval} | {row['status']} |")
        (self.root/"leaderboard.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        return board

    def export(self, output):
        """Export reviewable protocol/results; full board records and weights stay local."""
        output = Path(output).resolve()
        board = self.leaderboard()
        write(output/"protocol.json", self.protocol)
        write(output/"leaderboard.json", board)
        output.mkdir(parents=True, exist_ok=True)
        (output/"leaderboard.md").write_text((self.root/"leaderboard.md").read_text(encoding="utf-8"), encoding="utf-8")
        write(output/"roster.json", [{k: v for k, v in spec.items() if k != "training_ids"} for spec in self.roster.values()])
        write(output/"board-ids.json", {"board_ids": [b["id"] for b in self.boards], "boards_sha256": digest(self.boards)})
        for row in board["ratings"]:
            write(output/"reports"/(row["id"]+".json"), read(self.root/"reports"/(row["id"]+".json")))
        write(output/"matches-index.json", [{"path": str(p.relative_to(self.root)), "file_sha256": sha256(p),
                                            "signature": read(p)["signature"], "records": len(read(p)["records"])}
                                           for p in sorted((self.root/"matches").glob("*.json"))])
        return {"output": str(output), "rated": len(board["ratings"]), "unrated": len(board["unrated"])}
