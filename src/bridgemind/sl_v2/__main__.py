"""Prepare and run mixed SL v2 without changing legacy trainers or rating suites."""

import argparse
import json

from bridgemind.dds import SolverClient
from .training import prepare, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run", required=True)
    prep.add_argument("--train", default="data/cardplay-expanded-v1/train")
    prep.add_argument("--valid", default="data/cardplay-expanded-v1/valid")
    prep.add_argument("--init", default="runs/play-sl-long-20261025/best.pt")
    prep.add_argument("--port", type=int, default=18766)
    prep.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    worker = sub.add_parser("train")
    worker.add_argument("--run", required=True)
    worker.add_argument("--seconds", type=float, default=7200)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--max-steps", type=int, default=10**9, help="Test/smoke step cap; time budget remains enforced")
    args = p.parse_args()
    if args.command == "prepare":
        with SolverClient(port=args.port) as solver:
            config = prepare(args.run, args.train, args.valid, args.init, solver, device=args.device)
        print(json.dumps({"run": args.run, "curriculum": config["curriculum_audit"]}), flush=True)
    else:
        print(json.dumps(train(args.run, seconds=args.seconds, resume=args.resume, max_steps=args.max_steps)), flush=True)


if __name__ == "__main__":
    main()
