"""Run the fixed expanded cardplay experiment; hold out test until models freeze.

python examples/cardplay_experiment.py --phase train
python examples/cardplay_experiment.py --phase evaluate
python examples/cardplay_experiment.py --phase report
Existing runs resume with their original configuration. Existing evaluation files
are reused only after their frozen model and dataset hashes have been verified.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from bridgemind.cardplay.data import PlayData
from bridgemind.cardplay.model import read_checkpoint
from bridgemind.cardplay.runner import fingerprint
from bridgemind.training.runner import assert_disjoint, provenance, sha256
from bridgemind.training.rollout import imps

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cardplay-expanded-v1"
ARTIFACTS = ROOT / "artifacts/cardplay-expanded-v1"
DATA = ROOT / "data/cardplay-expanded-v1"
SEEDS = [42, 43, 44]
PROTOCOL = {
    "schema_version": 1, "name": "cardplay-expanded-v1", "date": "2026-10-25",
    "seeds": SEEDS, "data_boards": {"train": 10000, "valid": 1024, "test": 1024},
    "sl": {"steps": 1000, "batch_size": 64, "eval_every": 100, "eval_states": 4096},
    "ppo": {"updates": 30, "boards_per_update": 32, "epochs": 2,
            "eval_every": 5, "eval_boards": 128, "snapshot_every": 5},
    "sl_init": "runs/play-sl-pilot/best.pt",
    "selection": {"sl": "minimum validation imitation NLL", "ppo": "maximum validation IMP vs own SL anchor"},
    "evaluation": {"split": "test", "boards": 1024, "seed": 20261025, "batch_size": 32,
                   "primary": "PPO best vs own SL best, mean IMP across three training seeds",
                   "secondary_opponents": ["random", "previous PPO pilot"],
                   "policy": "greedy neural; actual 52-card paired play; fixed recorded auction",
                   "rules": "Freeze all six candidates before any test evaluation; no test-based model selection or retraining"},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen experiment record differs: {path}")
    else:
        write_json(path, value)


def run_cli(*arguments):
    subprocess.run([sys.executable, "-m", "bridgemind.cardplay", *map(str, arguments)], cwd=ROOT, check=True)


def register():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    immutable_json(ARTIFACTS / "protocol.json", PROTOCOL)
    immutable_json(ARTIFACTS / "initial-model.json", {
        "path": PROTOCOL["sl_init"], "sha256": sha256(ROOT/PROTOCOL["sl_init"]),
        "shared_initialization_for_seeds": SEEDS,
        "training_randomness": "minibatch draws and stochastic PPO rollouts"})
    groups, audit = [], {}
    for split, expected in PROTOCOL["data_boards"].items():
        data = PlayData(DATA / split)
        assert data.manifest["boards"] == expected
        allowed = np.load(ROOT / f"data/supervised/{split}/board-ids.npy", allow_pickle=False)
        assert set(data.ids).issubset(set(allowed))
        groups.append(data.ids)
        audit[split] = {"boards": len(data.ids), "states": data.manifest["states"],
                        "fingerprint": fingerprint(DATA / split), "bidding_split_subset": True}
    assert_disjoint(*groups)
    immutable_json(ARTIFACTS / "data-audit.json", {"disjoint": True, "datasets": audit})


def train():
    if (ARTIFACTS / "frozen-models.json").exists():
        raise ValueError("Models have been frozen for independent testing; use a new experiment to retrain")
    for seed in SEEDS:
        for kind in ("sl", "ppo"):
            directory = RUN / f"{kind}-{seed}"
            common = ["--run", directory, "--train", DATA/"train", "--valid", DATA/"valid",
                      "--seed", seed, "--device", "cuda"]
            if (directory / "last.pt").exists():
                common += ["--resume"]
            else:
                init = ROOT / PROTOCOL["sl_init"] if kind == "sl" else RUN / f"sl-{seed}/best.pt"
                common += ["--init", init]
            config = PROTOCOL[kind]
            for key, value in config.items():
                common += ["--"+key.replace("_", "-"), value]
            print(f"Training {kind} seed {seed}", flush=True)
            run_cli("train-"+kind, *common)
    frozen = {"protocol_sha256": sha256(ARTIFACTS/"protocol.json"), "models": {}}
    for seed in SEEDS:
        for kind in ("sl", "ppo"):
            directory = RUN / f"{kind}-{seed}"
            checkpoint = read_checkpoint(directory/"best.pt")
            frozen["models"][f"{kind}-{seed}"] = {
                "path": (directory/"best.pt").relative_to(ROOT).as_posix(),
                "sha256": sha256(directory/"best.pt"), "selected_step": checkpoint["metadata"]["step"],
                "last_step": read_checkpoint(directory/"last.pt")["metadata"]["step"]}
            shutil.copyfile(directory/"metrics.jsonl", ARTIFACTS/f"{kind}-{seed}-metrics.jsonl")
    frozen["previous_ppo"] = {"path": "runs/play-ppo-pilot/last.pt", "sha256": sha256(ROOT/"runs/play-ppo-pilot/last.pt")}
    immutable_json(ARTIFACTS/"frozen-models.json", frozen)
    print("All candidates frozen; independent test can now run", flush=True)


def evaluate():
    frozen = json.loads((ARTIFACTS/"frozen-models.json").read_text())
    if frozen["protocol_sha256"] != sha256(ARTIFACTS/"protocol.json"):
        raise ValueError("Frozen protocol changed")
    previous = frozen["previous_ppo"]
    for model in [*frozen["models"].values(), previous]:
        if sha256(ROOT/model["path"]) != model["sha256"]:
            raise ValueError("Frozen model changed")
    test_fingerprint = fingerprint(DATA/"test")
    for seed in SEEDS:
        for kind in ("sl", "ppo"):
            model = frozen["models"][f"{kind}-{seed}"]
            opponents = {"random": None, "previous-ppo": previous}
            if kind == "ppo":
                opponents["own-sl"] = frozen["models"][f"sl-{seed}"]
            for label, opponent in opponents.items():
                path = ARTIFACTS / f"{kind}-{seed}-vs-{label}.json"
                if path.exists():
                    report = json.loads(path.read_text())
                    if (report["model_sha256"] != model["sha256"] or report["dataset_fingerprint"] != test_fingerprint
                            or report.get("opponent_model_sha256") != (opponent["sha256"] if opponent else None)
                            or report["boards"] != 1024 or report["seed"] != 20261025):
                        raise ValueError("Existing evaluation differs from frozen protocol")
                    continue
                print(f"Independent test: {kind}-{seed} vs {label}", flush=True)
                arguments = ["evaluate", "--data", DATA/"test", "--model", ROOT/model["path"],
                             "--count", 1024, "--seed", 20261025, "--batch-size", 32,
                             "--device", "cuda", "--output", path, "--opponent", "neural" if opponent else "random"]
                if opponent:
                    arguments += ["--opponent-model", ROOT/opponent["path"]]
                run_cli(*arguments)


def seed_board_summary(reports):
    if any(p["board_ids"] != reports[0]["board_ids"] for p in reports):
        raise ValueError("Seed comparisons need identical board order")
    matrix = np.asarray([p["imps"] for p in reports], np.float64)
    rng = np.random.default_rng(20261025)
    bootstrap = []
    for _ in range(5000):
        rows = rng.integers(0, len(reports), len(reports))
        columns = rng.integers(0, matrix.shape[1], matrix.shape[1])
        bootstrap.append(matrix[rows][:, columns].mean())
    return {"mean_imps": float(matrix.mean()), "per_seed_mean_imps": matrix.mean(1).tolist(),
            "seed_and_board_bootstrap_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "training_seeds": len(reports), "distinct_boards": matrix.shape[1],
            "method": "Resample training seeds and shared board columns independently; 5000 draws; only three seeds"}


def report():
    summary = {"protocol": PROTOCOL, "provenance": provenance(),
               "data_audit": json.loads((ARTIFACTS/"data-audit.json").read_text()),
               "frozen_models": json.loads((ARTIFACTS/"frozen-models.json").read_text()),
               "training": {}, "workload": {}, "results": {}, "primary": {}}
    sl_draws, paired_boards, transitions = 0, 0, 0
    for seed in SEEDS:
        sl = [json.loads(line) for line in (ARTIFACTS/f"sl-{seed}-metrics.jsonl").read_text().splitlines()]
        ppo = [json.loads(line) for line in (ARTIFACTS/f"ppo-{seed}-metrics.jsonl").read_text().splitlines()]
        selected = summary["frozen_models"]["models"][f"sl-{seed}"]["selected_step"]
        best = next(row for row in reversed(sl) if row["step"] == selected)
        summary["training"][str(seed)] = {"sl_initial_validation": sl[0]["validation"],
                                          "sl_best_validation": best["validation"],
                                          "sl_selected_step": selected,
                                          "ppo_selected_update": summary["frozen_models"]["models"][f"ppo-{seed}"]["selected_step"]}
        sl_draws += sl[-1]["step"]*PROTOCOL["sl"]["batch_size"]
        paired_boards += sum(row.get("rollout", {}).get("boards", 0) for row in ppo)
        transitions += sum(row.get("train", {}).get("transitions", 0) for row in ppo)
    summary["workload"] = {"sl_decision_draws": sl_draws, "ppo_paired_board_draws": paired_boards,
                           "ppo_played_rooms": paired_boards*2, "ppo_on_policy_transitions": transitions,
                           "note": "Training draws may repeat boards; test uses 1024 distinct boards shared across matchups"}
    expected_ids = {identity.decode() for identity in PlayData(DATA/"test").ids}
    for kind in ("sl", "ppo"):
        for seed in SEEDS:
            for opponent in (["random", "previous-ppo", "own-sl"] if kind == "ppo" else ["random", "previous-ppo"]):
                name = f"{kind}-{seed}-vs-{opponent}"
                result = json.loads((ARTIFACTS/(name+".json")).read_text())
                if (len(result["board_ids"]) != 1024 or set(result["board_ids"]) != expected_ids
                        or len(result["imps"]) != 1024 or len(result["raw_differences"]) != 1024
                        or result["imps"] != [imps(value) for value in result["raw_differences"]]
                        or not np.isclose(result["mean_imps"], np.mean(result["imps"]))):
                    raise ValueError(f"Invalid independent paired results: {name}")
                summary["results"][name] = {key: result[key] for key in (
                    "boards", "mean_imps", "bootstrap_95_ci", "wins", "ties", "losses", "seconds")}
                summary["results"][name]["mean_declarer_trick_gain"] = float(np.mean(
                    np.asarray(result["current_as_declarer_tricks"])-result["opponent_as_declarer_tricks"]))
    primary = [json.loads((ARTIFACTS/f"ppo-{seed}-vs-own-sl.json").read_text()) for seed in SEEDS]
    summary["primary"] = seed_board_summary(primary)
    summary["workload"]["test_paired_matchups"] = sum(result["boards"] for result in summary["results"].values())
    summary["workload"]["test_played_rooms"] = 2*summary["workload"]["test_paired_matchups"]
    summary["secondary"] = {}
    for kind in ("sl", "ppo"):
        for opponent in ("random", "previous-ppo"):
            reports = [json.loads((ARTIFACTS/f"{kind}-{seed}-vs-{opponent}.json").read_text()) for seed in SEEDS]
            summary["secondary"][f"{kind}-vs-{opponent}"] = seed_board_summary(reports)
    summary["secondary_note"] = "Exploratory; no multiple-comparison adjustment"
    write_json(ARTIFACTS/"report.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "evaluate", "report", "all"], default="all")
    args = parser.parse_args()
    register()
    started = time.perf_counter()
    for name, action in (("train", train), ("evaluate", evaluate), ("report", report)):
        if args.phase in (name, "all"):
            action()
    print(f"Completed {args.phase} in {time.perf_counter()-started:.1f}s", flush=True)
