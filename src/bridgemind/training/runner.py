"""Reproducible SL / PPO runs, independent from the DDS server process."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .data import BoardSet, SupervisedData
from .evaluation import evaluate_duplicate, supervised_metrics
from .league import OpponentPool, frozen_cpu
from .model import BiddingModel, load_model, restore_training, save_model
from .optim import ppo_update, supervised_update
from .rollout import collect_duplicate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_fingerprint(path):
    path = Path(path)
    if path.is_file():
        return sha256(path)
    manifest = json.loads((path / "manifest.json").read_text())
    files = ["manifest.json", "board-ids.npy", *[s["path"] for s in manifest["shards"]]]
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update(sha256(path / name).encode())
    return digest.hexdigest()


def assert_disjoint(*id_groups):
    seen = set()
    for ids in id_groups:
        current = set(ids)
        if seen.intersection(current):
            raise ValueError("Training/validation/test board overlap")
        seen.update(current)


def provenance():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
            "open_spiel": importlib.metadata.version("open-spiel")}


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return device


def _emit(directory, record):
    record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    console = dict(record)
    for key in ("validation", "rollout"):
        if key in console:
            console[key] = {k: v for k, v in console[key].items() if k not in ("imps", "board_ids", "raw_differences")}
    print(json.dumps(console, ensure_ascii=False, allow_nan=False), flush=True)


def _setup(args, kind, config):
    directory = Path(args.run).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    if args.resume:
        checkpoint = directory / "last.pt"
        model, metadata = load_model(checkpoint, device)
        if metadata["kind"] != kind or metadata["config"] != config:
            raise ValueError("Resume configuration/data mismatch; reuse the original training options")
        step = metadata["step"]
    else:
        if (directory / "last.pt").exists() or (directory / "metrics.jsonl").exists():
            raise ValueError("Run already exists; use --resume or a new --run directory")
        model = load_model(args.init, device)[0] if args.init else BiddingModel(args.hidden_size).to(device)
        metadata, step = {}, 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return directory, model, optimizer, rng, metadata, step, device


def _save(directory, model, optimizer, rng, kind, config, step, **extra):
    metadata = {"kind": kind, "step": step, "config": config, "provenance": provenance(), **extra}
    save_model(directory / "last.pt", model, optimizer=optimizer, metadata=metadata, rng=rng)


def train_sl(args):
    train, valid = SupervisedData(args.train), SupervisedData(args.valid)
    assert_disjoint(*(np.load(Path(p) / "board-ids.npy", allow_pickle=False) for p in (args.train, args.valid)))
    config = {"train": data_fingerprint(args.train), "valid": data_fingerprint(args.valid),
              "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size, "cpu_threads": args.cpu_threads,
              "eval_states": args.eval_states, "eval_every": args.eval_every}
    directory, model, optimizer, rng, metadata, step, device = _setup(args, "sl", config)
    if args.resume:
        restore_training(directory / "last.pt", optimizer, rng)
    else:
        _emit(directory, {"event": "start", "kind": "sl", "step": 0, "device": device,
                          "validation": supervised_metrics(model, valid, count=args.eval_states), "config": config})
        _save(directory, model, optimizer, rng, "sl", config, 0)
    if args.steps < step:
        raise ValueError("--steps is a total target and must not be less than saved step")
    for step in range(step + 1, args.steps + 1):
        metrics = supervised_update(model, optimizer, *train.sample(args.batch_size, rng))
        if step % args.eval_every == 0 or step == args.steps:
            validation = supervised_metrics(model, valid, count=args.eval_states)
            _emit(directory, {"event": "update", "kind": "sl", "step": step, "train": metrics, "validation": validation})
            _save(directory, model, optimizer, rng, "sl", config, step)
    return directory / "last.pt"


def train_ppo(args):
    boards, valid = BoardSet.load(args.train), BoardSet.load(args.valid)
    assert_disjoint(boards.ids, valid.ids)
    config = {"train": data_fingerprint(args.train), "valid": data_fingerprint(args.valid), "seed": args.seed,
              "lr": args.lr, "boards_per_update": args.boards_per_update, "epochs": args.epochs,
              "minibatch_size": args.minibatch_size, "entropy_coef": args.entropy_coef,
              "snapshot_every": args.snapshot_every, "pool_capacity": args.pool_capacity,
              "eval_every": args.eval_every, "eval_boards": args.eval_boards, "cpu_threads": args.cpu_threads}
    directory, model, optimizer, rng, metadata, step, device = _setup(args, "ppo", config)
    if args.resume:
        pool = OpponentPool(directory / "league", capacity=args.pool_capacity, history=metadata["history"])
        restore_training(directory / "last.pt", optimizer, rng)
    else:
        # The imported SL value head has not learned this normalized IMP target.
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.zero_()
        pool = OpponentPool(directory / "league", model, capacity=args.pool_capacity)
        baseline = evaluate_duplicate(frozen_cpu(model), pool.get("anchor.pt"), valid, count=args.eval_boards)
        _emit(directory, {"event": "start", "kind": "ppo", "step": 0, "device": device,
                          "validation": baseline, "config": config})
        _save(directory, model, optimizer, rng, "ppo", config, 0, history=pool.history)
    if args.updates < step:
        raise ValueError("--updates is a total target and must not be less than saved update")
    for step in range(step + 1, args.updates + 1):
        started = time.perf_counter()
        policy = frozen_cpu(model)
        opponents, names = pool.roster(rng)
        indices = rng.choice(len(boards), size=min(args.boards_per_update, len(boards)), replace=False)
        batch, rollout = collect_duplicate(policy, opponents, boards, indices, rng)
        rollout_seconds = time.perf_counter() - started
        metrics = ppo_update(model, optimizer, batch, epochs=args.epochs, minibatch_size=args.minibatch_size,
                             rng=rng, entropy_coef=args.entropy_coef)
        if step % args.snapshot_every == 0:
            pool.snapshot(model, step)
        record = {"event": "update", "kind": "ppo", "step": step, "train": metrics,
                  "rollout": rollout, "opponent_slots": names, "rollout_seconds": rollout_seconds,
                  "update_seconds": time.perf_counter() - started - rollout_seconds}
        if step % args.eval_every == 0 or step == args.updates:
            record["validation"] = evaluate_duplicate(frozen_cpu(model), pool.get("anchor.pt"), valid, count=args.eval_boards)
        _emit(directory, record)
        _save(directory, model, optimizer, rng, "ppo", config, step, history=pool.history)
    return directory / "last.pt"
