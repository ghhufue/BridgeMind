"""python -m bridgemind.training --help"""

import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch

from .baseline import convert_brl, fetch_baseline
from .data import BoardSet, SupervisedData, decode_pgx, make_splits, prepare_supervised
from .evaluation import evaluate_duplicate, supervised_metrics
from .model import load_model
from .runner import assert_disjoint, data_fingerprint, sha256, train_ppo, train_sl


ASSETS = {
    "dds_results_2.5M.npy": (
        "https://huggingface.co/datasets/sotetsuk/dds_dataset/resolve/main/dds_results_2.5M.npy",
        "186bb2833befd4a2354bd8eda9b42795ff2669177a23e998870d7a8ef65b3dca"),
    **{f"openspiel-{name}.txt": (
        f"https://storage.googleapis.com/openspiel-data/bridge/{name}.txt", digest) for name, digest in {
            "train": "59b0fb726b8887271a51f97af6e4bab425a54c0b712bf83e544df5ede95d4bbb",
            "valid": "682a1543bb27fcbd1d8aeba6cfd7bbc81bd5efe568d152d2f0ad436ce4febed8",
            "test": "41ee45364d71ec2608c474eda52d3c8d4302c6ca46cb23bcbaf66a8ac6386352",
            "README": "51aa62f10deb7104a71342ed75a046e7c51a465180ba81770cc52190cdab5a03"}.items()}}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def fetch(args):
    root = Path(args.root)
    (root / "downloads").mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in ASSETS.items():
        path = root / "downloads" / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve(url, temporary)
            if sha256(temporary) != digest:
                raise ValueError(f"Source SHA256 mismatch: {name}")
            temporary.replace(path)
        if sha256(path) != digest:
            raise ValueError(f"Source SHA256 mismatch: {name}")
    source = fetch_baseline(root / "baselines" / "brl-sl.pkl")
    metadata = convert_brl(source, root / "baselines" / "brl-sl.pt")
    manifest = {"assets": {name: {"url": url, "sha256": digest} for name, (url, digest) in ASSETS.items()},
                "baseline": metadata}
    (root / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def prepare_boards(args):
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty boards destination")
    if sha256(args.source) != ASSETS["dds_results_2.5M.npy"][1]:
        raise ValueError("Expected pinned 2.5M public dataset")
    data = np.load(args.source, mmap_mode="r", allow_pickle=False)
    count = sum(args.sizes)
    if data.shape != (2, 2500000, 4) or count > data.shape[1]:
        raise ValueError("Invalid source shape or split sizes")
    ix = np.random.default_rng(args.seed).choice(data.shape[1], count, replace=False)
    owners, tables = decode_pgx(data[0, ix], data[1, ix])
    sets = make_splits(owners, tables, tuple(args.sizes), seed=args.seed)
    verified = min(args.verify, len(sets["train"]))
    if verified:
        from bridgemind.dds import NativeDdsBackend
        with NativeDdsBackend() as solver:
            for i in range(verified):
                deal, table, _, _ = sets["train"].board(i)
                np.testing.assert_array_equal(solver.solve_table(deal).tricks, table.tricks)
    for name, dataset in sets.items():
        dataset.save(destination / f"{name}.npz")
    manifest = {"source": ASSETS["dds_results_2.5M.npy"][0], "source_sha256": sha256(args.source),
                "seed": args.seed, "sizes": dict(zip(sets, args.sizes)), "latest_dds_verified_boards": verified,
                "label_origin": "public precomputed Pgx DDS; not all recomputed with 3.1.0"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


def prepare_sl(args):
    root, board_root, output = Path(args.source), Path(args.boards), Path(args.output)
    excluded = set()
    for split in ("train", "valid", "test"):
        excluded.update(BoardSet.load(board_root / f"{split}.npz").ids)
    # Hold-out data takes priority. Re-running a partial import checks prior outputs.
    for split, count in (("valid", args.valid_boards), ("test", args.test_boards), ("train", args.train_boards)):
        directory = output / split
        source = root / f"openspiel-{split}.txt"
        digest = sha256(source)
        if digest != ASSETS[source.name][1]:
            raise ValueError(f"Source SHA256 mismatch: {source.name}")
        if (directory / "manifest.json").exists():
            metadata = json.loads((directory / "manifest.json").read_text())
            ids = np.load(directory / "board-ids.npy", allow_pickle=False)
            if metadata["boards"] != count or metadata.get("source_sha256", digest) != digest:
                raise ValueError("Existing supervised split configuration differs")
            assert_disjoint(excluded, ids)
            metadata["source_sha256"] = digest
        else:
            metadata = prepare_supervised(source, directory, max_boards=count, excluded_ids=excluded)
            metadata["source_sha256"] = digest
            ids = np.load(directory / "board-ids.npy", allow_pickle=False)
        if metadata["boards"] != count:
            raise ValueError("Source exhausted before reaching requested board count")
        (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        excluded.update(ids)
        print(json.dumps({"split": split, "boards": metadata["boards"], "states": metadata["states"]}), flush=True)


def generate_boards(args):
    from bridgemind.dds import SolverClient
    from .offline import generate_tables
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty boards destination")
    with SolverClient(port=args.port) as solver:
        info = solver.get_info()
        owners, tables = generate_tables(solver, sum(args.sizes), seed=args.seed, batch_size=args.batch_size,
                                        progress=lambda n, total: print(json.dumps({"solved": n, "total": total}), flush=True))
    for split, boards in make_splits(owners, tables, tuple(args.sizes), args.seed).items():
        boards.save(destination / f"{split}.npz")
    manifest = {"source": "new_uniform_random_deals", "seed": args.seed, "sizes": args.sizes, "solver": info}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def audit(args):
    groups, report = [], {}
    for name in ("train", "valid", "test"):
        path = Path(args.boards) / f"{name}.npz"
        boards = BoardSet.load(path)
        groups.append(boards.ids)
        report[f"boards/{name}"] = {"boards": len(boards), "sha256": data_fingerprint(path)}
    if args.supervised:
        for name in ("train", "valid", "test"):
            path = Path(args.supervised) / name
            data = SupervisedData(path)
            ids = np.load(path / "board-ids.npy", allow_pickle=False)
            groups.append(ids)
            report[f"supervised/{name}"] = {"boards": len(ids), "states": data.manifest["states"], "sha256": data_fingerprint(path)}
    assert_disjoint(*groups)
    print(json.dumps({"disjoint": True, "datasets": report}, indent=2))


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    model, metadata = load_model(args.model)
    if args.supervised:
        report = supervised_metrics(model, SupervisedData(args.supervised), count=args.states)
    else:
        if not args.opponent or not args.boards:
            raise ValueError("Duplicate evaluation needs --opponent and --boards")
        opponent, _ = load_model(args.opponent)
        report = evaluate_duplicate(model, opponent, BoardSet.load(args.boards), count=args.count, seed=args.seed)
    report.update(model_sha256=sha256(args.model), model_kind=metadata.get("kind"))
    if args.boards:
        report["boards_sha256"] = sha256(args.boards)
        report["opponent_sha256"] = sha256(args.opponent)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("imps", "board_ids")}, indent=2))


def parser():
    root = argparse.ArgumentParser(description="BridgeMind bidding training (SL + duplicate IMP PPO)")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch", help="Fetch and hash-check public corpora / convert brl baseline")
    p.add_argument("--root", default="data")
    p.set_defaults(handler=fetch)
    p = commands.add_parser("prepare-boards", help="Import public packed tables; randomize dealer/vulnerability")
    p.add_argument("--source", default="data/downloads/dds_results_2.5M.npy")
    p.add_argument("--output", default="data/boards")
    p.add_argument("--sizes", type=positive, nargs=3, default=[100000, 10000, 20000], metavar=("TRAIN", "VALID", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify", type=int, default=32, help="Recompute this many train tables with local DDS; 0 disables")
    p.set_defaults(handler=prepare_boards)
    p = commands.add_parser("generate-boards", help="Generate fresh random deals through the independent DDS TCP service")
    p.add_argument("--output", required=True)
    p.add_argument("--sizes", type=positive, nargs=3, default=[1000, 100, 100], metavar=("TRAIN", "VALID", "TEST"))
    p.add_argument("--port", type=positive, default=18765)
    p.add_argument("--batch-size", type=positive, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(handler=generate_boards)
    p = commands.add_parser("prepare-sl", help="Replay WBridge5 auctions into legal private observations")
    p.add_argument("--source", default="data/downloads")
    p.add_argument("--boards", default="data/boards")
    p.add_argument("--output", default="data/supervised")
    for split, count in (("train", 100000), ("valid", 10000), ("test", 10000)):
        p.add_argument(f"--{split}-boards", type=positive, default=count)
    p.set_defaults(handler=prepare_sl)
    p = commands.add_parser("audit", help="Validate datasets and reject any cross-split board overlap")
    p.add_argument("--boards", default="data/boards")
    p.add_argument("--supervised", default="data/supervised")
    p.set_defaults(handler=audit)
    for name, handler in (("train-sl", train_sl), ("train-ppo", train_ppo)):
        p = commands.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--train", default="data/supervised/train" if name == "train-sl" else "data/boards/train.npz")
        p.add_argument("--valid", default="data/supervised/valid" if name == "train-sl" else "data/boards/valid.npz")
        p.add_argument("--init", help="Optional tensor checkpoint; random start otherwise")
        p.add_argument("--resume", action="store_true", help="Resume RUN/last.pt; targets are total steps/updates")
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--cpu-threads", type=positive, default=1)
        p.add_argument("--hidden-size", type=positive, default=1024, help="Used only for random initialization")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--lr", type=float, default=1e-4 if name == "train-sl" else 1e-5)
        p.add_argument("--eval-every", type=positive, default=50 if name == "train-sl" else 5)
        if name == "train-sl":
            p.add_argument("--steps", type=positive, default=1000)
            p.add_argument("--batch-size", type=positive, default=1024)
            p.add_argument("--eval-states", type=positive, default=8192)
        else:
            p.add_argument("--updates", type=positive, default=100)
            p.add_argument("--boards-per-update", type=positive, default=128)
            p.add_argument("--epochs", type=positive, default=4)
            p.add_argument("--minibatch-size", type=positive, default=1024)
            p.add_argument("--entropy-coef", type=float, default=.001)
            p.add_argument("--snapshot-every", type=positive, default=5)
            p.add_argument("--pool-capacity", type=positive, default=8)
            p.add_argument("--eval-boards", type=positive, default=128)
        p.set_defaults(handler=handler)
    p = commands.add_parser("evaluate", help="Explicit evaluation; use VALID while tuning, TEST only for final reporting")
    p.add_argument("--model", required=True)
    p.add_argument("--opponent")
    p.add_argument("--boards")
    p.add_argument("--supervised")
    p.add_argument("--count", type=positive, help="Default: all boards in supplied file")
    p.add_argument("--states", type=positive, default=8192)
    p.add_argument("--seed", type=int, default=714)
    p.add_argument("--cpu-threads", type=positive, default=1)
    p.add_argument("--output")
    p.set_defaults(handler=evaluate)
    return root


def main():
    command_parser = parser()
    args = command_parser.parse_args()
    if args.command == "evaluate" and args.supervised and (args.boards or args.opponent):
        command_parser.error("Choose supervised evaluation or duplicate evaluation")
    if getattr(args, "resume", False) and args.init:
        command_parser.error("--resume and --init are mutually exclusive")
    if getattr(args, "verify", 0) < 0:
        command_parser.error("--verify must be nonnegative")
    if hasattr(args, "lr") and (not np.isfinite(args.lr) or args.lr <= 0):
        command_parser.error("--lr must be finite and positive")
    if hasattr(args, "entropy_coef") and (not np.isfinite(args.entropy_coef) or args.entropy_coef < 0):
        command_parser.error("--entropy-coef must be finite and nonnegative")
    args.handler(args)


if __name__ == "__main__":
    main()
