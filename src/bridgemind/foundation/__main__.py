"""Build and run certified basic-skill tests, separately from match ratings."""

import argparse
import json
from pathlib import Path

import torch

from bridgemind.dds import SolverClient
from bridgemind.rating.system import writer_lock
from .system import build_suite, evaluate, export, install_suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/cardplay-foundations-v1")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", default="cardplay-foundations-v1")
    b.add_argument("--port", type=int, default=18765)
    b.add_argument("--seeds", type=int, nargs="+", default=[1701, 1709, 1723])
    e = sub.add_parser("evaluate")
    e.add_argument("--id", required=True)
    e.add_argument("--model")
    e.add_argument("--policy", choices=("neural", "rule-basic", "cash-public-master", "random"), default="neural")
    e.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    e.add_argument("--batch-size", type=int, default=64)
    x = sub.add_parser("export")
    x.add_argument("--output", default="artifacts/cardplay-foundations-v1")
    i = sub.add_parser("install", help="Install an exported fixture suite without DDS or old prediction caches")
    i.add_argument("--source", default="artifacts/cardplay-foundations-v1")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.command == "build":
        with SolverClient(port=args.port) as solver:
            result = build_suite(args.root, solver, name=args.name, seeds=args.seeds)
        print(json.dumps({"suite": result["name"], "base_candidates": len(result["bases"])}))
    elif args.command == "install":
        print(json.dumps(install_suite(args.source, args.root)))
    else:
        with writer_lock(args.root):
            if args.command == "evaluate":
                result = evaluate(args.root, name=args.id, model=args.model, recipe=args.policy,
                                  device=args.device, batch_size=args.batch_size)
                print(json.dumps({k: v for k, v in result["summary"].items() if k not in ("families", "by_permutation_basic_percent")}))
            else:
                print(json.dumps(export(args.root, args.output)))


if __name__ == "__main__":
    main()
