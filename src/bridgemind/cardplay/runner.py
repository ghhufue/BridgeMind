"""Separate play training runs and checkpoints; existing bidding runs are unchanged."""

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from bridgemind.training.runner import assert_disjoint, provenance, resolve_device, sha256
from bridgemind.engine.play import card_name
from .data import PlayData
from .evaluation import evaluate_pairs, supervised_metrics, play_pairs
from .model import ModelConfig, PlayModel, load_model, restore_training, save_checkpoint
from .optim import supervised_update, ppo_update
from .search import SamplingDdsPolicy


def fingerprint(path):
    import hashlib
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text())
    digest = hashlib.sha256()
    for name in ["manifest.json", "board-ids.npy", "games.jsonl", *[s["path"] for s in manifest["shards"]]]:
        digest.update(name.encode()); digest.update(sha256(path/name).encode())
    return digest.hexdigest()


def emit(directory, record):
    record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with (directory/"metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False)+"\n")
    console = copy.deepcopy(record)
    for name in ("validation", "rollout"):
        if name in console:
            console[name] = {k: v for k, v in console[name].items() if k not in (
                "imps", "raw_differences", "board_ids", "current_as_declarer_tricks", "opponent_as_declarer_tricks")}
    print(json.dumps(console, allow_nan=False), flush=True)


def frozen(model):
    result = copy.deepcopy(model).eval()
    result.requires_grad_(False)
    return result


class PlayLeague:
    def __init__(self, directory, device, *, model=None, history=None, capacity=8):
        if capacity < 2:
            raise ValueError("Pool capacity must be >=2")
        self.directory, self.device, self.capacity = Path(directory), device, capacity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history, self.cache = list(history or []), {}
        if model is not None:
            if (self.directory/"anchor.pt").exists():
                raise ValueError("Opponent pool already exists")
            save_checkpoint(self.directory/"anchor.pt", model, metadata={"kind": "play_anchor"})
            self.snapshot(model, 0)
        if not self.history or any(not (self.directory/p).is_file() for p in ["anchor.pt", *self.history]):
            raise ValueError("Incomplete play opponent pool")

    def snapshot(self, model, step):
        name = f"snapshot-{step:08d}.pt"
        if (self.directory/name).exists():
            raise ValueError("Refusing to overwrite immutable play opponent snapshot")
        save_checkpoint(self.directory/name, model, metadata={"kind": "play_snapshot", "step": step})
        self.history = (self.history+[name])[-self.capacity:]
        self.cache = {k: v for k, v in self.cache.items() if k in self.history or k == "anchor.pt"}

    def get(self, name):
        if name not in self.cache:
            model, _ = load_model(self.directory/name, self.device)
            self.cache[name] = model.eval().requires_grad_(False)
        return self.cache[name]

    def roster(self, rng):
        previous = str(rng.choice(self.history[:-1] or self.history))
        names = [self.history[-1]]*4+[previous]*2+["anchor.pt"]*2
        return [self.get(name) for name in names], names


def setup(args, kind, extra):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, valid = PlayData(args.train), PlayData(args.valid)
    assert_disjoint(train.ids, valid.ids)
    config = {"train": fingerprint(args.train), "valid": fingerprint(args.valid), "seed": args.seed, "lr": args.lr,
              "cpu_threads": args.cpu_threads, "eval_every": args.eval_every, **extra}
    directory = Path(args.run).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.resume:
        model, metadata = load_model(directory/"last.pt", device)
        if metadata["kind"] != kind or metadata["config"] != config:
            raise ValueError("Resume configuration/data mismatch")
        step = metadata["step"]
    else:
        if (directory/"last.pt").exists() or (directory/"metrics.jsonl").exists():
            raise ValueError("Run exists; use --resume or a new directory")
        model = load_model(args.init, device)[0] if args.init else PlayModel(ModelConfig(args.d_model, args.layers, args.heads)).to(device)
        metadata, step = {}, 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        restore_training(directory/"last.pt", optimizer, rng)
    return directory, model, optimizer, rng, train, valid, config, metadata, step


def save(directory, model, optimizer, rng, kind, config, step, **extra):
    metadata = {"kind": kind, "config": config, "step": step, "provenance": provenance(), **extra}
    save_checkpoint(directory/"last.pt", model, metadata=metadata, optimizer=optimizer, rng=rng)


def train_supervised(args):
    kind = "play_distill" if args.command == "distill" else "play_sl"
    extra = {"batch_size": args.batch_size, "eval_states": args.eval_states,
             "belief_coef": args.belief_coef, "value_coef": args.value_coef}
    if kind == "play_distill":
        extra.update(samples=args.samples, min_ply=args.min_ply, temperature=args.temperature)
    directory, model, optimizer, rng, train, valid, config, metadata, step = setup(args, kind, extra)
    solver, teacher = None, None
    if kind == "play_distill":
        from bridgemind.dds import SolverClient
        solver = SolverClient(port=args.port)
        teacher = SamplingDdsPolicy(solver, samples=args.samples, seed=args.seed+129)
        if args.resume:
            teacher.rng.bit_generator.state = metadata["search_rng"]
    best = metadata.get("best_nll", float("inf"))
    try:
        if not args.resume:
            validation = supervised_metrics(model, valid, count=args.eval_states)
            best = validation["nll"]
            save_checkpoint(directory/"best.pt", model, metadata={"kind": kind, "step": 0, "validation": validation})
            emit(directory, {"event": "start", "kind": kind, "step": 0, "config": config, "validation": validation})
            save(directory, model, optimizer, rng, kind, config, 0, best_nll=best,
                 **({"search_rng": teacher.rng.bit_generator.state} if teacher else {}))
        if args.steps < step:
            raise ValueError("--steps is a total target below saved step")
        for step in range(step+1, args.steps+1):
            started = time.perf_counter()
            rows = train.sample(args.batch_size, rng, min_ply=args.min_ply if teacher else 0)
            targets = None
            if teacher:
                targets = []
                for row in rows:
                    obs = train.environment(row["game"], row["ply"]).observation()
                    result = teacher.predict(obs)  # Receives the visible observation, never true owners.
                    target = np.zeros(52, np.float32)
                    legal = obs["legal_card_ids"]
                    if len(legal) == 1:
                        target[legal[0]] = 1
                    else:
                        scores = np.array([result["mean_team_scores"][card_name(i)] for i in legal])
                        weights = np.exp((scores-scores.max())/args.temperature)
                        target[legal] = weights/weights.sum()
                    targets.append(target)
            model.train()
            metrics = supervised_update(model, optimizer, rows, belief_coef=args.belief_coef, value_coef=args.value_coef, targets=targets)
            if step % args.eval_every == 0 or step == args.steps:
                validation = supervised_metrics(model, valid, count=args.eval_states)
                if validation["nll"] < best:
                    best = validation["nll"]
                    save_checkpoint(directory/"best.pt", model, metadata={"kind": kind, "step": step, "validation": validation})
                emit(directory, {"event": "update", "kind": kind, "step": step, "train": metrics,
                                 "validation": validation, "last_step_seconds": time.perf_counter()-started})
                save(directory, model, optimizer, rng, kind, config, step, best_nll=best,
                     **({"search_rng": teacher.rng.bit_generator.state, "teacher_solver": solver.get_info()} if teacher else {}))
        return directory/"last.pt"
    finally:
        if solver:
            solver.close()


def train_ppo(args):
    extra = {key: getattr(args, key) for key in ("boards_per_update", "epochs", "minibatch_size", "entropy_coef",
             "belief_coef", "snapshot_every", "pool_capacity", "eval_boards")}
    directory, model, optimizer, rng, train, valid, config, metadata, step = setup(args, "play_ppo", extra)
    device = next(model.parameters()).device
    if args.resume:
        pool = PlayLeague(directory/"league", device, history=metadata["history"], capacity=args.pool_capacity)
    else:
        with torch.no_grad():
            model.value_head[-1].weight.zero_(); model.value_head[-1].bias.zero_()
        pool = PlayLeague(directory/"league", device, model=model, capacity=args.pool_capacity)
    best = metadata.get("best_mean_imps", 0.)
    if not args.resume:
        validation = evaluate_pairs(model, pool.get("anchor.pt"), valid, count=args.eval_boards)
        save_checkpoint(directory/"best.pt", model, metadata={"kind": "play_ppo", "step": 0, "validation": validation})
        emit(directory, {"event": "start", "kind": "play_ppo", "step": 0, "config": config, "validation": validation})
        save(directory, model, optimizer, rng, "play_ppo", config, 0, history=pool.history, best_mean_imps=best)
    if args.updates < step:
        raise ValueError("--updates is a total target below saved step")
    for step in range(step+1, args.updates+1):
        started = time.perf_counter()
        opponents, names = pool.roster(rng)
        indices = rng.choice(len(train.load_games()), min(args.boards_per_update, len(train.load_games())), replace=False)
        model.eval()
        records, rollout = play_pairs(model, opponents, train, indices, rng, collect=True, stochastic=True)
        rollout_seconds = time.perf_counter()-started
        model.train()
        metrics = ppo_update(model, optimizer, records, rng, epochs=args.epochs, minibatch_size=args.minibatch_size,
                             entropy_coef=args.entropy_coef, belief_coef=args.belief_coef)
        if step % args.snapshot_every == 0:
            pool.snapshot(model, step)
        record = {"event": "update", "kind": "play_ppo", "step": step, "train": metrics, "rollout": rollout,
                  "opponent_slots": names, "rollout_seconds": rollout_seconds, "update_seconds": time.perf_counter()-started-rollout_seconds}
        if step % args.eval_every == 0 or step == args.updates:
            validation = evaluate_pairs(model, pool.get("anchor.pt"), valid, count=args.eval_boards)
            record["validation"] = validation
            if validation["mean_imps"] > best:
                best = validation["mean_imps"]
                save_checkpoint(directory/"best.pt", model, metadata={"kind": "play_ppo", "step": step, "validation": validation})
        emit(directory, record)
        save(directory, model, optimizer, rng, "play_ppo", config, step, history=pool.history, best_mean_imps=best)
    return directory/"last.pt"
