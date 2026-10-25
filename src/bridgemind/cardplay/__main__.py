"""python -m bridgemind.cardplay --help"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from bridgemind.training.__main__ import ASSETS, positive
from bridgemind.training.runner import assert_disjoint, resolve_device, sha256
from .data import PlayData, prepare_dataset
from .evaluation import evaluate_pairs, supervised_metrics
from .model import load_model
from .policy import HybridPlayPolicy, NeuralPlayPolicy
from .runner import fingerprint, train_supervised, train_ppo
from .search import SamplingDdsPolicy, RandomPlayPolicy


def prepare(args):
    groups = []
    for split in ("valid", "test", "train"):
        source = Path(args.source)/f"openspiel-{split}.txt"
        if sha256(source) != ASSETS[source.name][1]:
            raise ValueError(f"Source SHA256 mismatch: {source.name}")
        allowed = np.load(Path(args.bidding_data)/split/"board-ids.npy", allow_pickle=False)
        assert_disjoint(*groups, allowed)
        groups.append(allowed)
        path = Path(args.output)/split
        count = getattr(args, f"{split}_boards")
        if (path/"manifest.json").exists():
            data = PlayData(path)
            if data.manifest["boards"] != count or not set(data.ids).issubset(set(allowed)) or data.manifest["source_sha256"] != sha256(source):
                raise ValueError("Existing cardplay dataset differs; use a new output")
            result = data.manifest
        else:
            result = prepare_dataset(source, path, allowed_ids=allowed, max_boards=count, system_id=1,
                                     system_description="WBridge5 SAYC")
        print(json.dumps({"split": split, "boards": result["boards"], "states": result["states"]}), flush=True)


def audit(args):
    groups, report = [], {}
    for split in ("train", "valid", "test"):
        path = Path(args.data)/split
        data = PlayData(path)
        ids = data.ids
        games = data.load_games()
        if len(games) != len(ids) or {g["id"].encode() for g in games} != set(ids):
            raise ValueError("Games disagree with board ID file")
        if args.bidding_data:
            allowed = set(np.load(Path(args.bidding_data)/split/"board-ids.npy", allow_pickle=False))
            if not set(ids).issubset(allowed):
                raise ValueError("Play split is not a subset of the corresponding bidding split")
        groups.append(ids)
        report[split] = {"boards": len(ids), "states": data.manifest["states"], "fingerprint": fingerprint(path)}
    assert_disjoint(*groups)
    print(json.dumps({"disjoint": True, "datasets": report}, indent=2))


def make_policy(kind, model_path, args, solver, seed):
    if kind == "random":
        return RandomPlayPolicy(seed)
    if kind == "dds":
        return SamplingDdsPolicy(solver, samples=args.samples, seed=seed)
    if not model_path:
        raise ValueError(f"{kind} needs a model checkpoint")
    model, _ = load_model(model_path, resolve_device(args.device))
    model.eval()
    if kind == "neural":
        return model
    return HybridPlayPolicy(NeuralPlayPolicy(model=model), SamplingDdsPolicy(solver, samples=args.samples, seed=seed))


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    solver = None
    if args.policy in ("dds", "hybrid") or args.opponent in ("dds", "hybrid"):
        from bridgemind.dds import SolverClient
        solver = SolverClient(port=args.port)
    try:
        data = PlayData(args.data)
        current = make_policy(args.policy, args.model, args, solver, args.seed+1)
        if args.supervised:
            if args.policy != "neural":
                raise ValueError("Supervised metrics require --policy neural")
            report = supervised_metrics(current, data, count=args.states, seed=args.seed)
        else:
            opponent = make_policy(args.opponent, args.opponent_model, args, solver, args.seed+2)
            started = time.perf_counter()
            report = evaluate_pairs(current, opponent, data, count=args.count, seed=args.seed, batch_size=args.batch_size)
            report["seconds"] = time.perf_counter()-started
            report["policy"], report["opponent"] = args.policy, args.opponent
            report["batch_size"] = args.batch_size
        report["dataset_fingerprint"] = fingerprint(args.data)
        if args.model:
            report["model_sha256"] = sha256(args.model)
        if args.opponent_model:
            report["opponent_model_sha256"] = sha256(args.opponent_model)
        if solver:
            report["solver"] = solver.get_info()
            report["samples_per_search"] = args.samples
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("imps", "raw_differences", "board_ids",
              "current_as_declarer_tricks", "opponent_as_declarer_tricks")}, indent=2))
    finally:
        if solver:
            solver.close()


def demo(args):
    from bridgemind.engine import BridgeBiddingEnv
    from bridgemind.training.data import BoardSet
    from bridgemind.training.policy import BiddingPolicy
    torch.set_num_threads(args.cpu_threads)
    boards = BoardSet.load(args.boards)
    deal, _, dealer, vulnerable = boards.board(args.index)
    bidding = BridgeBiddingEnv(None, deal, dealer=dealer, vulnerable=vulnerable, prefetch=False)
    bidder = BiddingPolicy(args.bidding_model)
    while not bidding.is_terminal():
        bidding.apply_call(bidder.predict(bidding.observation())["call_id"])
    env = bidding.start_play()
    policy = NeuralPlayPolicy(args.model, device=resolve_device(args.device))
    auction = env.auction
    history = []
    while not env.is_terminal():
        obs = env.observation()
        decision = policy.predict(obs)
        history.append({"controller": obs["player"], "seat": obs["to_play"], "card": decision["card"]})
        env.apply_card(decision["card_id"])
    result = {"board_id": boards.ids[args.index].decode(), "auction": auction, "play": history, "result": env.result(),
              "bidding_model_sha256": sha256(args.bidding_model), "play_model_sha256": sha256(args.model)}
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parser():
    root = argparse.ArgumentParser(description="BridgeMind full cardplay: constrained DDS, Transformer, SL/distillation/PPO")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Import full WBridge5 plays, preserving existing bidding board splits")
    p.add_argument("--source", default="data/downloads")
    p.add_argument("--bidding-data", default="data/supervised")
    p.add_argument("--output", default="data/cardplay")
    for split, count in (("train", 1000), ("valid", 128), ("test", 128)):
        p.add_argument(f"--{split}-boards", type=positive, default=count)
    p.set_defaults(handler=prepare)
    p = commands.add_parser("audit")
    p.add_argument("--data", default="data/cardplay")
    p.add_argument("--bidding-data", default="data/supervised")
    p.set_defaults(handler=audit)
    for command, handler in (("train-sl", train_supervised), ("distill", train_supervised), ("train-ppo", train_ppo)):
        p = commands.add_parser(command)
        p.add_argument("--run", required=True)
        p.add_argument("--train", default="data/cardplay/train")
        p.add_argument("--valid", default="data/cardplay/valid")
        p.add_argument("--init")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--cpu-threads", type=positive, default=1)
        p.add_argument("--d-model", type=positive, default=256)
        p.add_argument("--layers", type=positive, default=4)
        p.add_argument("--heads", type=positive, default=8)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--lr", type=float, default=1e-5 if command == "train-ppo" else 1e-4)
        p.add_argument("--eval-every", type=positive, default=5 if command == "train-ppo" else 50)
        p.add_argument("--belief-coef", type=float, default=.05 if command == "train-ppo" else .2)
        if command == "train-ppo":
            p.add_argument("--updates", type=positive, default=100)
            p.add_argument("--boards-per-update", type=positive, default=16)
            p.add_argument("--epochs", type=positive, default=4)
            p.add_argument("--minibatch-size", type=positive, default=128)
            p.add_argument("--entropy-coef", type=float, default=.001)
            p.add_argument("--snapshot-every", type=positive, default=5)
            p.add_argument("--pool-capacity", type=positive, default=8)
            p.add_argument("--eval-boards", type=positive, default=32)
        else:
            p.add_argument("--steps", type=positive, default=1000)
            p.add_argument("--batch-size", type=positive, default=64)
            p.add_argument("--eval-states", type=positive, default=1024)
            p.add_argument("--value-coef", type=float, default=.1)
            if command == "distill":
                p.add_argument("--port", type=positive, default=18765)
                p.add_argument("--samples", type=positive, default=8)
                p.add_argument("--min-ply", type=int, default=28)
                p.add_argument("--temperature", type=float, default=50.)
        p.set_defaults(handler=handler)
    p = commands.add_parser("evaluate")
    p.add_argument("--data", default="data/cardplay/valid")
    p.add_argument("--policy", choices=["neural", "random", "dds", "hybrid"], default="neural")
    p.add_argument("--model")
    p.add_argument("--opponent", choices=["neural", "random", "dds", "hybrid"], default="random")
    p.add_argument("--opponent-model")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-threads", type=positive, default=1)
    p.add_argument("--count", type=positive, default=32)
    p.add_argument("--batch-size", type=positive, default=16)
    p.add_argument("--seed", type=int, default=714)
    p.add_argument("--port", type=positive, default=18765)
    p.add_argument("--samples", type=positive, default=32)
    p.add_argument("--supervised", action="store_true")
    p.add_argument("--states", type=positive, default=1024)
    p.add_argument("--output")
    p.set_defaults(handler=evaluate)
    p = commands.add_parser("demo", help="Bidding checkpoint → actual 52-card neural play → score")
    p.add_argument("--bidding-model", default="data/baselines/brl-sl.pt")
    p.add_argument("--model", required=True)
    p.add_argument("--boards", default="data/boards/valid.npz")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-threads", type=positive, default=1)
    p.add_argument("--output")
    p.set_defaults(handler=demo)
    return root


def main():
    p = parser()
    args = p.parse_args()
    if getattr(args, "resume", False) and args.init:
        p.error("--resume and --init are mutually exclusive")
    for name in ("lr", "temperature"):
        if hasattr(args, name) and (not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            p.error(f"--{name} must be finite and positive")
    for name in ("belief_coef", "value_coef", "entropy_coef"):
        if hasattr(args, name) and (not np.isfinite(getattr(args, name)) or getattr(args, name) < 0):
            p.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if hasattr(args, "min_ply") and not 0 <= args.min_ply <= 51:
        p.error("--min-ply must be in 0..51")
    args.handler(args)


if __name__ == "__main__":
    main()
