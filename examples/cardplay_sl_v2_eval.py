"""Freeze SL v2 and confirm it once on fresh deals; never select on test results."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.evaluation import evaluate_pairs
from bridgemind.cardplay.model import load_model
from bridgemind.cardplay.runner import fingerprint
from bridgemind.rating.policies import RulePolicy
from bridgemind.rating.system import digest, implementation, now, read, runtime, write, writer_lock
from bridgemind.sl_v2.data import orbit_key
from bridgemind.training.__main__ import ASSETS
from bridgemind.training.data import board_ids
from bridgemind.training.rollout import imps
from bridgemind.training.runner import sha256
from cardplay_long_sl_eval import collect_report_ids


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/cardplay-sl-v2-20261026/confirmation"
DATA = ROOT/"data/cardplay-sl-v2-confirmation-v1"
RUN = ROOT/"runs/play-sl-v2-20261026-2h"
COUNT, SEED = 2048, 20261029


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    else:
        write(path, value)


def freeze():
    completed = read(RUN/"report.json")
    if completed["state"] != "completed" or sha256(RUN/"best.pt") != completed["best_model_sha256"]:
        raise ValueError("Training candidate is not the completed frozen best")
    protocol = {"name": "cardplay-sl-v2-confirmation-v1", "boards": COUNT, "seed": SEED,
                "batch_size": 32, "device": "cuda", "primary_opponents": ["sl-long-42", "rule-basic"],
                "candidate": {"path": str(RUN/"best.pt"), "sha256": sha256(RUN/"best.pt"), "step": completed["best_step"]},
                "initial": {"path": str(RUN/"init.pt"), "sha256": sha256(RUN/"init.pt")},
                "implementation_sha256": implementation(), "script_sha256": sha256(__file__), "runtime": runtime("cuda"),
                "training_protocol_sha256": sha256(RUN/"protocol.json"), "curriculum_sha256": sha256(RUN/"curriculum.json"),
                "exclusion": "All declared data ID/exclusion snapshots and historical report IDs, plus suit/seat orbits of known played datasets, curriculum and published fixtures",
                "design": "Same full deal and recorded auction; candidate plays both partnerships in paired rooms; greedy neural; actual 52-card play",
                "bootstrap": {"draws": 10000, "unit": "distinct full deal", "seed": SEED,
                              "primary_interval": "97.5% per comparison; Bonferroni family coverage >=95% for two comparisons"},
                "decision": "Claim improvement over both only if both primary 97.5% lower bounds >0; report failures equally",
                "selection": "Candidate fixed before importing confirmation deals; no checkpoint selection or training based on these results",
                "scope": "One training seed; WBridge5 SAYC, north dealer, neither vulnerable; not human rank or bidding strength"}
    immutable(ART/"protocol.json", protocol)
    return protocol


def prepare():
    audit_path, snapshot = ART/"exclusion-audit.json", DATA.parent/"cardplay-sl-v2-confirmation-excluded.npy"
    source = ROOT/"data/downloads/openspiel-test.txt"
    if sha256(source) != ASSETS[source.name][1]:
        raise ValueError("Source checksum changed")
    if audit_path.exists():
        audit = read(audit_path)
        if sha256(snapshot) != audit["excluded_ids_sha256"]:
            raise ValueError("Exclusion snapshot changed")
        excluded = set(np.load(snapshot, allow_pickle=False))
        allowed = {s.encode() for s in audit["eligible_board_ids"]}
    else:
        excluded, sources, banned_orbits = set(), {}, set()
        pool = ROOT/"data/supervised/test/board-ids.npy"
        paths = sorted(set((ROOT/"data").rglob("*board*ids*.npy")) | set((ROOT/"data").rglob("*excluded*.npy")) |
                       set((ROOT/"artifacts").rglob("*board*ids*.npy")))
        for path in paths:
            if path == pool or path == snapshot or path.is_relative_to(DATA):
                continue
            ids = set(np.load(path, allow_pickle=False))
            excluded.update(ids)
            sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"data/boards").glob("*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                ids = set(arrays["ids"])
            excluded.update(ids)
            sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        for folder in (ROOT/"runs", ROOT/"artifacts"):
            for path in sorted(folder.rglob("*.json")):
                if path.is_relative_to(ART):
                    continue
                value = read(path)
                ids = collect_report_ids(value)
                if path.name == "boards.json" and isinstance(value, list):
                    ids.update(b["id"].encode() for b in value)
                    banned_orbits.update(orbit_key(b["owners"]) for b in value)
                if ids:
                    excluded.update(ids)
                    sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"data").rglob("games.jsonl")):
            if path.is_relative_to(DATA):
                continue
            for line in path.read_text().splitlines():
                game = json.loads(line)
                banned_orbits.add(orbit_key(game["owners"]))
            sources[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "purpose": "full-deal suit/seat orbit audit"}
        for split in read(RUN/"curriculum.json").values():
            banned_orbits.update(orbit_key(c["board"]["owners"]) for c in split)
        suite = read(ROOT/"runs/cardplay-foundations-v1/suite.json")
        banned_orbits.update(orbit_key(b["board"]["owners"]) for b in suite["bases"])
        banned_orbits.update(orbit_key(c["board"]["owners"]) for c in read(ROOT/"artifacts/cardplay-diagnostics-v1/tactical-cases.json"))
        allowed = set(np.load(pool, allow_pickle=False))-excluded
        orbit_rejected, eligible = 0, set()
        with source.open() as stream:
            for line in stream:
                actions = np.fromstring(line, dtype=np.int64, sep=" ")
                owners = np.empty(52, dtype=np.uint8)
                owners[actions[:52]] = np.arange(52) % 4
                identity = board_ids(owners[None])[0]
                if identity in allowed:
                    if orbit_key(owners) in banned_orbits:
                        orbit_rejected += 1
                    else:
                        eligible.add(identity)
        allowed = eligible
        if len(allowed) < COUNT:
            raise ValueError("Not enough unused confirmation deals")
        np.save(snapshot, np.asarray(sorted(excluded), dtype="S64"), allow_pickle=False)
        audit = {"sources": sources, "excluded_ids_sha256": sha256(snapshot), "excluded_boards": len(excluded),
                 "banned_deal_orbits": len(banned_orbits), "orbit_rejected": orbit_rejected,
                 "eligible_board_ids": sorted(x.decode() for x in allowed), "eligible_count": len(allowed),
                 "source_sha256": sha256(source), "created_at_utc": now()}
        immutable(audit_path, audit)
    if not (DATA/"manifest.json").exists():
        print(f"Importing {COUNT} untouched deals from {len(allowed)} eligible", flush=True)
        prepare_dataset(source, DATA, allowed_ids=allowed, max_boards=COUNT, system_id=1,
                        system_description="WBridge5 SAYC; frozen SL v2 independent confirmation; north dealer, neither vulnerable")
    data = PlayData(DATA)
    games = data.load_games()
    if len(games) != COUNT or not set(data.ids).issubset(allowed) or set(data.ids) & excluded:
        raise ValueError("Confirmation board identity/count audit failed")
    if len({orbit_key(g["owners"]) for g in games}) != COUNT:
        raise ValueError("Confirmation contains suit/seat duplicate deals")
    immutable(ART/"data-audit.json", {"boards": COUNT, "states": data.manifest["states"], "fingerprint": fingerprint(DATA),
              "board_ids_sha256": sha256(DATA/"board-ids.npy"), "prior_id_intersection": 0,
              "prior_orbit_intersection": 0, "exclusion_audit_sha256": sha256(audit_path)})
    return data


def bootstrap_summary(result):
    values = np.asarray(result["imps"], dtype=np.float64)
    if len(values) != COUNT or any(imps(d) != v for d, v in zip(result["raw_differences"], values)):
        raise ValueError("Per-board IMP audit failed")
    rng = np.random.default_rng(SEED)
    draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(10000)]
    return {"mean_imps": float(values.mean()), "boards": COUNT,
            "bootstrap_95_ci": np.quantile(draws, [.025, .975]).tolist(),
            "primary_97_5_ci": np.quantile(draws, [.0125, .9875]).tolist(),
            "wins": int((values>0).sum()), "ties": int((values==0).sum()), "losses": int((values<0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "prepare", "confirm"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    ART.mkdir(parents=True, exist_ok=True)
    with writer_lock(ART):
        started = time.perf_counter()
        protocol = freeze()
        write(ART/"status.json", {"state": "running", "phase": "prepare", "updated_at_utc": now()})
        data = prepare()
        if args.phase == "prepare":
            print("Frozen protocol and independent data ready", flush=True)
            return
        current = load_model(protocol["candidate"]["path"], "cuda")[0].eval()
        initial = load_model(protocol["initial"]["path"], "cuda")[0].eval()
        summaries = {}
        for name, opponent in (("sl-long-42", initial), ("rule-basic", RulePolicy())):
            output = ART/f"vs-{name}.json"
            write(ART/"status.json", {"state": "running", "phase": f"vs-{name}", "updated_at_utc": now()})
            if output.exists():
                result = read(output)
                if result["protocol_sha256"] != digest(protocol) or result["dataset_fingerprint"] != fingerprint(DATA):
                    raise ValueError("Cached confirmation belongs to a different protocol/data")
            else:
                print(f"Confirming frozen candidate vs {name}: {COUNT} paired deals", flush=True)
                result = evaluate_pairs(current, opponent, data, count=COUNT, seed=SEED, batch_size=32)
                result.update(protocol_sha256=digest(protocol), dataset_fingerprint=fingerprint(DATA),
                              model_sha256=protocol["candidate"]["sha256"], opponent=name)
                write(output, result)
            if set(result["board_ids"]) != {x.decode() for x in data.ids} or len(set(result["board_ids"])) != COUNT:
                raise ValueError("Confirmation used wrong boards")
            summaries[name] = bootstrap_summary(result)
            print(json.dumps({"opponent": name, **summaries[name]}), flush=True)
        for key in ("candidate", "initial"):
            if sha256(protocol[key]["path"]) != protocol[key]["sha256"]:
                raise ValueError("Frozen model changed during confirmation")
        report = {"protocol_sha256": digest(protocol), "data_audit": read(ART/"data-audit.json"),
                  "results": summaries, "both_primary_lower_bounds_positive": all(r["primary_97_5_ci"][0]>0 for r in summaries.values()),
                  "scope": protocol["scope"], "seconds": time.perf_counter()-started, "finished_at_utc": now()}
        write(ART/"report.json", report)
        write(ART/"status.json", {"state": "completed", "phase": "saved", "updated_at_utc": now()})
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
