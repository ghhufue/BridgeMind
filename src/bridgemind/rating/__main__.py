"""python -m bridgemind.rating: init, register, evaluate, leaderboard."""

import argparse
import json
from pathlib import Path

import torch

from .system import RatingSystem, initialize, specification, writer_lock, KINDS


def parser():
    root = argparse.ArgumentParser(description="Versioned cardplay rating: random=0, full-information DDS=100")
    root.add_argument("--root", default="runs/cardplay-rating-v1", help="Versioned benchmark directory")
    root.add_argument("--port", type=int, default=18765, help="Existing independent DDS service")
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="Freeze boards, baseline panel, runtime and solver identity")
    p.add_argument("--name", default="cardplay-rating-v1")
    p.add_argument("--data", required=True, help="Existing PlayData dataset")
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[20261026, 20261027, 20261028])
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--purpose", choices=["tracking", "confirmation"], default="tracking")
    p.add_argument("--exclude-data", action="append", default=[])
    p = sub.add_parser("register", help="Append an immutable opponent; never change the scoring panel")
    p.add_argument("--id", required=True)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--samples", type=int)
    p.add_argument("--training-data", action="append", default=[], help="Declare known training datasets for overlap rejection")
    p = sub.add_parser("evaluate", help="Rate opponents, reusing validated paired match caches")
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--id", nargs="+")
    choice.add_argument("--all", action="store_true")
    sub.add_parser("leaderboard", help="Validate and export completed ratings")
    p = sub.add_parser("export", help="Copy protocol, roster, reports and match hashes for review")
    p.add_argument("--output", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "init":
        result = initialize(args.root, args.data, name=args.name, count=args.count, seeds=args.seeds,
                            device=args.device, port=args.port, draws=args.draws, purpose=args.purpose,
                            exclude_data=args.exclude_data)
        print(json.dumps({"benchmark": result["name"], "boards": result["boards"], "root": str(Path(args.root).resolve())}))
        return
    with writer_lock(args.root):
        system = RatingSystem(args.root, port=args.port)
        if args.command == "register":
            spec = specification(args.id, args.kind, checkpoint=args.checkpoint, samples=args.samples, training_data=args.training_data)
            system.register(spec)
            print(json.dumps({"registered": spec["id"], "kind": spec["kind"]}))
        elif args.command == "evaluate":
            names = list(dict.fromkeys(["random", "oracle", *(sorted(system.roster) if args.all else args.id)]))
            for name in names:
                report = system.evaluate(name)
                print(json.dumps({"id": name, **report["score"]}), flush=True)
                system.leaderboard()
        elif args.command == "export":
            print(json.dumps(system.export(args.output)))
        else:
            print(json.dumps(system.leaderboard(), indent=2))


if __name__ == "__main__":
    main()
