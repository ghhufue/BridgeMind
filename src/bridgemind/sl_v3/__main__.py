import argparse
import json
from bridgemind.dds import SolverClient
from .training import prepare, train


def main():
    parser = argparse.ArgumentParser(description="SL v3 expanded-data mixed training")
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--train", default="data/cardplay-sl-v3-50k/train")
    p.add_argument("--valid", default="data/cardplay-expanded-v1/valid")
    p.add_argument("--initial", default="runs/play-sl-v2-20261026-2h/best.pt")
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--curriculum", default="runs/sl-v3-curriculum-20261026")
    p = sub.add_parser("train")
    p.add_argument("--run", required=True)
    p.add_argument("--seconds", type=float, default=7200)
    p.add_argument("--max-steps", type=int, default=10**9)
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        with SolverClient(port=args.port) as solver:
            report = prepare(args.run, args.train, args.valid, args.initial, solver, prepared_curriculum=args.curriculum)
    else:
        report = train(args.run, seconds=args.seconds, resume=args.resume, max_steps=args.max_steps)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
