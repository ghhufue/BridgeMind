"""Prepare a fresh local tracking benchmark and register existing play models.

python examples/cardplay_rating_setup.py --phase prepare
python examples/cardplay_rating_setup.py --phase register
python examples/cardplay_rating_setup.py --phase evaluate
Reuses checked local downloads; never trains or downloads models.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.runner import fingerprint
from bridgemind.rating.system import RatingSystem, initialize, read, write, specification, writer_lock
from bridgemind.training.__main__ import ASSETS
from bridgemind.training.runner import sha256
from cardplay_long_sl_eval import collect_report_ids


ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    data = ROOT/"data"/args.name
    audit_path = ROOT/"data"/(args.name+"-exclusion-audit.json")
    snapshot = ROOT/"data"/(args.name+"-excluded.npy")
    source = ROOT/"data/downloads/openspiel-test.txt"
    if sha256(source) != ASSETS[source.name][1]:
        raise ValueError("Original source checksum changed")
    if snapshot.exists():
        audit = read(audit_path)
        if audit["excluded_ids_sha256"] != sha256(snapshot):
            raise ValueError("Exclusion snapshot changed")
        excluded = set(np.load(snapshot, allow_pickle=False))
    else:
        excluded, sources = set(), {}
        source_ids = ROOT/"data/supervised/test/board-ids.npy"
        paths = sorted(set((ROOT/"data").rglob("*board*ids*.npy")) |
                       set((ROOT/"data").rglob("*excluded*.npy")))
        paths += sorted((ROOT/"artifacts").rglob("*board*ids*.npy"))
        for path in paths:
            if path == source_ids or path.is_relative_to(data):
                continue
            ids = set(np.load(path, allow_pickle=False))
            excluded.update(ids)
            sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"data/boards").glob("*.npz")):
            with np.load(path, allow_pickle=False) as values:
                ids = set(values["ids"])
            excluded.update(ids)
            sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        for folder in (ROOT/"artifacts", ROOT/"runs"):
            for path in sorted(folder.rglob("*.json")):
                if path.is_relative_to(args.root):
                    continue
                value = read(path)
                ids = collect_report_ids(value)
                # Explicit frozen rating suites have an array of board records.
                if path.name == "boards.json" and isinstance(value, list):
                    ids.update(b["id"].encode() for b in value)
                if ids:
                    excluded.update(ids)
                    sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        np.save(snapshot, np.asarray(sorted(excluded), dtype="S64"), allow_pickle=False)
        audit = {"sources": sources, "excluded_ids_sha256": sha256(snapshot), "excluded_boards": len(excluded)}
        write(audit_path, audit)
    allowed = set(np.load(ROOT/"data/supervised/test/board-ids.npy", allow_pickle=False))-excluded
    if not (data/"manifest.json").exists():
        prepare_dataset(source, data, allowed_ids=allowed, max_boards=args.count,
                        system_id=1, system_description="WBridge5 SAYC; fresh fixed tracking benchmark, north dealer, neither vulnerable")
    imported = PlayData(data)
    if len(imported.ids) != args.count or not set(imported.ids).issubset(allowed):
        raise ValueError("Fresh rating dataset differs from frozen exclusion/count")
    audit.update(eligible_boards=len(allowed), selected_boards=len(imported.ids),
                 selected_ids_sha256=sha256(data/"board-ids.npy"), prior_intersection=0,
                 source_sha256=sha256(source), dataset_fingerprint=fingerprint(data))
    write(audit_path, audit)
    if not (args.root/"protocol.json").exists():
        initialize(args.root, data, name=args.name, count=args.count, seeds=args.seeds,
                   device=args.device, port=args.port, draws=args.draws, source_audit=audit)
    system = RatingSystem(args.root, port=args.port)
    if (system.protocol["source"]["fingerprint"] != fingerprint(data) or
            system.protocol["boards"] != args.count or system.protocol["seeds"] != args.seeds):
        raise ValueError("Existing benchmark differs from setup arguments")
    print(f"Prepared {len(imported.ids)} fresh tracking deals, prior intersection 0", flush=True)


def register(args):
    models = {"sl-pilot": "runs/play-sl-pilot/best.pt", "ppo-pilot": "runs/play-ppo-pilot/best.pt",
              "sl-pilot-last": "runs/play-sl-pilot/last.pt", "ppo-pilot-last": "runs/play-ppo-pilot/last.pt",
              "distill-pilot": "runs/play-distill-pilot/best.pt", "sl-long-42": "runs/play-sl-long-20261025/best.pt"}
    for seed in (42, 43, 44):
        for kind in ("sl", "ppo"):
            models[f"{kind}-expanded-{seed}"] = f"runs/cardplay-expanded-v1/{kind}-{seed}/best.pt"
    for seed in (43, 44):
        models[f"sl-long-{seed}"] = f"runs/cardplay-long-sl-replicas-v1/sl-{seed}/best.pt"
    with writer_lock(args.root):
        system = RatingSystem(args.root, port=args.port)
        for name, relative in models.items():
            path = ROOT/relative
            if not path.is_file():
                print(f"Missing local model, skipped: {relative}", flush=True)
                continue
            train = ROOT/("data/cardplay/train" if "pilot" in name else "data/cardplay-expanded-v1/train")
            system.register(specification(name, "neural", checkpoint=path, training_data=[train]))
            print(f"Registered {name}", flush=True)
        if (ROOT/models["sl-long-42"]).is_file():
            system.register(specification("hybrid-long-42-8", "hybrid", checkpoint=ROOT/models["sl-long-42"], samples=8,
                                          training_data=[ROOT/"data/cardplay-expanded-v1/train"]))


def run_match(root, port, pair):
    """Parent holds the writer lock; workers own distinct canonical match files."""
    torch.set_num_threads(1)
    system = RatingSystem(root, port=port)
    system.match(*pair)
    return pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["prepare", "register", "evaluate", "all"], default="prepare")
    p.add_argument("--name", default="cardplay-rating-v1")
    p.add_argument("--root", type=Path, default=ROOT/"runs/cardplay-rating-v1")
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261026, 20261027, 20261028])
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--id", nargs="+", help="Evaluate selected registered IDs; omit for entire roster")
    p.add_argument("--workers", type=int, default=1, help="Independent match processes under one parent writer lock")
    args = p.parse_args()
    args.root = args.root.resolve()
    if not 1 <= args.workers <= 8:
        p.error("--workers must be in 1..8")
    torch.set_num_threads(1)
    if args.phase in ("prepare", "all"):
        prepare(args)
    if args.phase in ("register", "all"):
        register(args)
    if args.phase in ("evaluate", "all"):
        with writer_lock(args.root):
            system = RatingSystem(args.root, port=args.port)
            names = list(dict.fromkeys(["random", "oracle", *(args.id or sorted(system.roster))]))
            if args.workers > 1:
                pairs = sorted({tuple(sorted((name, item["id"]))) for name in names for item in system.protocol["panel"]})
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(run_match, args.root, args.port, pair) for pair in pairs]
                    for future in as_completed(futures):
                        print(f"Match ready: {future.result()}", flush=True)
            for name in names:
                report = system.evaluate(name)
                print(f"{name}: {report['score']['rating']:.3f} {report['score']['rating_95_ci']}", flush=True)
                system.leaderboard()


if __name__ == "__main__":
    main()
