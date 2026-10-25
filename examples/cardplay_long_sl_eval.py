"""Freeze the 20-minute SL candidate and test on 2048 previously unused boards.

python examples/cardplay_long_sl_eval.py --phase all
The conditional gate is registered before importing or evaluating the holdout.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.model import read_checkpoint
from bridgemind.cardplay.runner import fingerprint
from bridgemind.training.__main__ import ASSETS
from bridgemind.training.rollout import imps
from bridgemind.training.runner import assert_disjoint, provenance, sha256

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/cardplay-long-sl-20261025"
DATA = ROOT/"data/cardplay-long-sl-holdout-v1"
LONG_RUN = ROOT/"runs/play-sl-long-20261025"
SEED = 20261026
MODELS = {"candidate": "runs/play-sl-long-20261025/best.pt",
          "initial-sl": "runs/cardplay-expanded-v1/sl-42/best.pt",
          "old-ppo": "runs/play-ppo-pilot/last.pt"}
PROTOCOL = {
    "name": "cardplay-long-sl-holdout-v1", "date": "2026-10-25",
    "candidate": "50,000-step SL checkpoint selected by validation NLL before this experiment",
    "boards": 2048, "evaluation_seed": SEED, "batch_size": 32,
    "primary_opponents": ["initial-sl", "old-ppo"], "secondary_opponent": "random",
    "selection": "No model selection or retraining based on these holdout results",
    "test_exclusion": "All existing cardplay splits, bidding training/validation splits, bidding board libraries and historical evaluation board IDs",
    "inference": "Greedy neural; paired rooms; same recorded auction and actual 52-card play",
    "gate": "Continue to training-seed replication only if both primary 97.5% board-bootstrap lower bounds exceed zero (Bonferroni-style correction for two primary comparisons)",
    "bootstrap": {"draws": 10000, "seed": SEED, "unit": "distinct full deal"},
    "if_gate_passes": "Replicate two additional training seeds with the same 52,928 update budget, learning rate and validation-only checkpoint selection; confirm on a separate fresh holdout before search comparison",
    "if_gate_fails": "Do not expand PPO or automatically train replicas; analyse imitation targets, seed sensitivity and validation-versus-play gap",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen experiment differs: {path}")
    else:
        write(path, value)


def collect_report_ids(value):
    """Only explicit board_ids fields count; avoid guessing IDs from arbitrary text."""
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "board_ids":
                if not isinstance(item, list) or any(not isinstance(s, str) or len(s) != 64
                                                    or any(c not in "0123456789abcdef" for c in s) for s in item):
                    raise ValueError("Malformed historical evaluation board IDs")
                result.update(s.encode() for s in item)
            else:
                result.update(collect_report_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(collect_report_ids(item))
    return result


def freeze():
    immutable(ART/"protocol.json", PROTOCOL)
    training_report = json.loads((LONG_RUN/"report.json").read_text())
    frozen = {}
    for name, relative in MODELS.items():
        checkpoint = read_checkpoint(ROOT/relative)
        frozen[name] = {"path": relative, "sha256": sha256(ROOT/relative),
                        "step": checkpoint["metadata"]["step"], "config": checkpoint["config"]}
    if (frozen["candidate"]["step"] != 50000
            or frozen["candidate"]["sha256"] != training_report["best_model_sha256"]
            or frozen["initial-sl"]["sha256"] != training_report["initial_model_sha256"]):
        raise ValueError("Training-selected model or initialization changed")
    immutable(ART/"frozen-models.json", {"protocol_sha256": sha256(ART/"protocol.json"), "models": frozen})
    for name in ("metrics.jsonl", "report.json"):
        destination = ART/("training-"+name)
        if destination.exists() and sha256(destination) != sha256(LONG_RUN/name):
            raise ValueError("Original training record changed")
        if not destination.exists():
            shutil.copyfile(LONG_RUN/name, destination)


def prepare():
    exclusion_path = ROOT/"data/cardplay-long-sl-excluded-ids.npy"
    if exclusion_path.exists():
        excluded = set(np.load(exclusion_path, allow_pickle=False))
        exclusion = json.loads((ART/"exclusion-audit.json").read_text())
        if exclusion["excluded_ids_sha256"] != sha256(exclusion_path):
            raise ValueError("Frozen exclusion snapshot changed")
    else:
        excluded, sources = set(), {}
        paths = [ROOT/f"data/{base}/{split}/board-ids.npy"
                 for base in ("cardplay", "cardplay-expanded-v1") for split in ("train", "valid", "test")]
        paths += [ROOT/f"data/supervised/{split}/board-ids.npy" for split in ("train", "valid")]
        for path in paths:
            ids = set(np.load(path, allow_pickle=False))
            excluded.update(ids)
            sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"data/boards").glob("*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                ids = set(arrays["ids"])
            excluded.update(ids)
            sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        for directory in (ROOT/"artifacts", ROOT/"runs"):
            for path in sorted(directory.rglob("*.json")):
                if path.is_relative_to(ART):
                    continue
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                ids = collect_report_ids(value)
                if ids:
                    excluded.update(ids)
                    sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        exclusion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(exclusion_path, np.asarray(sorted(excluded), dtype="S64"), allow_pickle=False)
        exclusion = {"sources": sources, "distinct_excluded_boards": len(excluded),
                     "excluded_ids_sha256": sha256(exclusion_path)}
        immutable(ART/"exclusion-audit.json", exclusion)
    allowed = set(np.load(ROOT/"data/supervised/test/board-ids.npy", allow_pickle=False))-excluded
    source = ROOT/"data/downloads/openspiel-test.txt"
    if sha256(source) != ASSETS[source.name][1]:
        raise ValueError("Original holdout source SHA256 differs")
    if not (DATA/"manifest.json").exists():
        print(f"Importing 2048 untouched boards from {len(allowed)} eligible IDs", flush=True)
        prepare_dataset(source, DATA, allowed_ids=allowed, max_boards=2048,
                        system_id=1, system_description="WBridge5 SAYC; fresh holdout after expanded-v1")
    data = PlayData(DATA)
    data.load_games()  # Validates replay identities against original ownership.
    if data.manifest["boards"] != 2048 or not set(data.ids).issubset(allowed):
        raise ValueError("Fresh holdout includes excluded or unexpected boards")
    assert_disjoint(data.ids, np.asarray(sorted(excluded), dtype="S64"))
    immutable(ART/"data-audit.json", {"boards": len(data.ids), "states": data.manifest["states"],
              "source_sha256": sha256(source), "fingerprint": fingerprint(DATA),
              "excluded_intersection": 0, "eligible_ids": len(allowed),
              "board_ids_sha256": sha256(DATA/"board-ids.npy")})
    if not (ART/"holdout-board-ids.npy").exists():
        shutil.copyfile(DATA/"board-ids.npy", ART/"holdout-board-ids.npy")


def evaluate():
    frozen = json.loads((ART/"frozen-models.json").read_text())["models"]
    for model in frozen.values():
        if sha256(ROOT/model["path"]) != model["sha256"]:
            raise ValueError("A frozen model changed")
    data_hash = fingerprint(DATA)
    for opponent in ("initial-sl", "old-ppo", "random"):
        output = ART/f"vs-{opponent}.json"
        enemy = frozen.get(opponent)
        if output.exists():
            old = json.loads(output.read_text())
            if (old["dataset_fingerprint"] != data_hash or old["model_sha256"] != frozen["candidate"]["sha256"]
                    or old.get("opponent_model_sha256") != (enemy["sha256"] if enemy else None)
                    or old["boards"] != 2048 or old["seed"] != SEED):
                raise ValueError("Existing result does not match frozen experiment")
            continue
        arguments = ["evaluate", "--data", str(DATA), "--model", str(ROOT/MODELS["candidate"]),
                     "--opponent", "neural" if enemy else "random", "--count", "2048",
                     "--batch-size", "32", "--seed", str(SEED), "--device", "cuda", "--output", str(output)]
        if enemy:
            arguments += ["--opponent-model", str(ROOT/enemy["path"])]
        print(f"Fresh paired holdout: candidate vs {opponent}", flush=True)
        subprocess.run([sys.executable, "-m", "bridgemind.cardplay", *arguments], cwd=ROOT, check=True)


def report():
    output = {"protocol": PROTOCOL, "provenance": provenance(), "results": {}}
    expected = {s.decode() for s in PlayData(DATA).ids}
    passed = []
    for opponent in ("initial-sl", "old-ppo", "random"):
        result = json.loads((ART/f"vs-{opponent}.json").read_text())
        if (len(result["board_ids"]) != 2048 or set(result["board_ids"]) != expected
                or len(result["imps"]) != 2048 or len(result["raw_differences"]) != 2048
                or result["imps"] != [imps(x) for x in result["raw_differences"]]):
            raise ValueError("Invalid per-board paired result")
        values = np.asarray(result["imps"], np.float64)
        if not np.isclose(result["mean_imps"], values.mean()):
            raise ValueError("Reported mean IMP disagrees with per-board values")
        rng = np.random.default_rng(SEED)
        samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(10000)])
        interval = np.quantile(samples, [.0125, .9875]).tolist()
        stats = {key: result[key] for key in ("boards", "mean_imps", "bootstrap_95_ci", "wins", "ties", "losses", "seconds")}
        stats["bootstrap_97_5_ci"] = interval
        stats["mean_declarer_trick_difference"] = float(np.mean(np.asarray(result["current_as_declarer_tricks"])
                                                               -result["opponent_as_declarer_tricks"]))
        output["results"][opponent] = stats
        if opponent in PROTOCOL["primary_opponents"]:
            passed.append(interval[0] > 0)
    output["gate_passed"] = all(passed)
    output["decision"] = PROTOCOL["if_gate_passes"] if all(passed) else PROTOCOL["if_gate_fails"]
    output["limits"] = ["One trained candidate seed; board intervals do not measure training-seed uncertainty",
                        "Fixed recorded auction, North dealer, neither side vulnerable; no bidding improvement claim",
                        "This holdout is now consumed; future model tuning needs a separate final holdout"]
    destination = ART/"report.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if {k: v for k, v in previous.items() if k != "provenance"} != {k: v for k, v in output.items() if k != "provenance"}:
            raise ValueError("Frozen gate outcomes changed")
        # Preserve the original gate hash even when Git provenance changes later.
        output = previous
    else:
        write(destination, output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "evaluate", "report", "all"], default="all")
    args = parser.parse_args()
    freeze()
    for name, action in (("prepare", prepare), ("evaluate", evaluate), ("report", report)):
        if args.phase in (name, "all"):
            action()
