"""Conditional seed replication confirmation and observation-restricted DDS comparison.

Run after launching the two fixed-budget replica trainers documented in protocol.
--phase all prepares untouched data, benchmarks search while replicas train, waits
for their final checkpoints, freezes all candidates, confirms them, and reports.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from cardplay_long_sl import process_running
from cardplay_long_sl_eval import ROOT, ART, MODELS, write, immutable
from cardplay_experiment import seed_board_summary
from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.evaluation import evaluate_pairs
from bridgemind.cardplay.model import read_checkpoint
from bridgemind.cardplay.policy import NeuralPlayPolicy, HybridPlayPolicy
from bridgemind.cardplay.runner import fingerprint
from bridgemind.cardplay.search import SamplingDdsPolicy
from bridgemind.dds import SolverClient
from bridgemind.training.rollout import imps
from bridgemind.training.runner import assert_disjoint, provenance, sha256

REPLICAS = ROOT/"runs/cardplay-long-sl-replicas-v1"
OUTPUT = ART/"replication"
CONFIRM = ROOT/"data/cardplay-long-sl-confirm-v1"
SEARCH = ROOT/"data/cardplay-long-sl-search-v1"
PRIMARY = ROOT/"data/cardplay-long-sl-holdout-v1"
SEEDS = [42, 43, 44]


def verify_gate():
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    if sha256(ART/"report.json") != protocol["gate_report_sha256"]:
        raise ValueError("Original gate record changed")
    if not json.loads((ART/"report.json").read_text())["gate_passed"]:
        raise ValueError("The pre-registered replication gate was not passed")
    for initial in protocol["initial_models"].values():
        if sha256(ROOT/initial["path"]) != initial["sha256"]:
            raise ValueError("Replica initialization changed")


def prepare():
    exclusion_path = ROOT/"data/cardplay-long-sl-excluded-ids.npy"
    if sha256(exclusion_path) != json.loads((ART/"exclusion-audit.json").read_text())["excluded_ids_sha256"]:
        raise ValueError("Original exclusion snapshot changed")
    excluded = set(np.load(exclusion_path, allow_pickle=False))
    excluded.update(PlayData(PRIMARY).ids)
    source = ROOT/"data/downloads/openspiel-test.txt"
    if sha256(source) != json.loads((ART/"data-audit.json").read_text())["source_sha256"]:
        raise ValueError("Original test trajectory source changed")
    eligible = set(np.load(ROOT/"data/supervised/test/board-ids.npy", allow_pickle=False))
    audits, groups = {}, [np.asarray(sorted(excluded), dtype="S64")]
    for directory, count in ((CONFIRM, 2048), (SEARCH, 32)):
        allowed = eligible-excluded
        if not (directory/"manifest.json").exists():
            print(f"Preparing {count} separate untouched boards: {directory.name}", flush=True)
            prepare_dataset(source, directory, allowed_ids=allowed, max_boards=count,
                            system_id=1, system_description="WBridge5 SAYC; separate long-SL confirmation/search")
        data = PlayData(directory)
        data.load_games()
        if data.manifest["boards"] != count or not set(data.ids).issubset(allowed):
            raise ValueError("Conditional holdout includes excluded cards")
        groups.append(data.ids)
        excluded.update(data.ids)
        audits[directory.name] = {"boards": count, "states": data.manifest["states"],
                                  "fingerprint": fingerprint(directory), "excluded_intersection": 0,
                                  "source_sha256": sha256(source)}
        copy = OUTPUT/(directory.name+"-board-ids.npy")
        if not copy.exists():
            shutil.copyfile(directory/"board-ids.npy", copy)
    assert_disjoint(*groups)
    immutable(OUTPUT/"data-audit.json", {"mutually_disjoint": True, "datasets": audits})


class TimedPolicy:
    """Measure complete, unbatched observation-to-decision wall time."""
    def __init__(self, policy):
        self.policy, self.seconds, self.forced = policy, [], []

    def predict(self, observation):
        start = time.perf_counter()
        result = self.policy.predict(observation)
        self.seconds.append(time.perf_counter()-start)
        self.forced.append(len(observation["legal_card_ids"]) == 1)
        return result

    def predict_batch(self, observations):
        return [self.predict(obs) for obs in observations]

    def metrics(self):
        values = np.asarray(self.seconds)
        decision = values[~np.asarray(self.forced)]
        return {"decisions": len(values), "mean_ms": float(values.mean()*1000),
                "p95_ms": float(np.quantile(values, .95)*1000),
                "nonforced_decisions": len(decision), "nonforced_mean_ms": float(decision.mean()*1000),
                "timing": "unbatched end-to-end, including sampling, RPC and device synchronization; replicas train concurrently"}


def search():
    frozen = json.loads((ART/"frozen-models.json").read_text())["models"]["candidate"]
    if sha256(ROOT/frozen["path"]) != frozen["sha256"]:
        raise ValueError("Original long SL checkpoint changed")
    if (OUTPUT/"search-report.json").exists():
        old = json.loads((OUTPUT/"search-report.json").read_text())
        if old["model_sha256"] != frozen["sha256"] or old["dataset_fingerprint"] != fingerprint(SEARCH):
            raise ValueError("Existing search experiment differs")
        return
    torch.set_num_threads(1)
    ready_path = REPLICAS/f"dds-ready-{uuid.uuid4().hex}.json"
    with (REPLICAS/"dds-stdout.log").open("w") as out, (REPLICAS/"dds-stderr.log").open("w") as err:
        service = subprocess.Popen([sys.executable, "-u", "-m", "bridgemind.dds.server", "--workers", "4",
                                    "--port", "0", "--ready-file", str(ready_path)], cwd=ROOT, stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        ready = None
        try:
            limit = time.monotonic()+30
            while not ready_path.exists():
                if service.poll() is not None or time.monotonic() > limit:
                    raise RuntimeError("DDS server did not become ready; inspect dds-stderr.log")
                time.sleep(.05)
            ready = json.loads(ready_path.read_text())
            report = {"boards": 32, "samples": 8, "model_sha256": frozen["sha256"],
                      "dataset_fingerprint": fingerprint(SEARCH), "results": {}, "limits":
                      ["32-board exploratory panel; no hyperparameter or model selection",
                       "Uniform/learned PIMC optimize sampled raw score, not exact imperfect-information value",
                       "Latency measured while two neural training jobs run on the same GPU"]}
            data = PlayData(SEARCH)
            for variant in ("neural-self", "uniform-dds", "learned-hybrid"):
                output = OUTPUT/("search-"+variant+".json")
                if output.exists():
                    result = json.loads(output.read_text())
                    if (result["model_sha256"] != frozen["sha256"]
                            or result["dataset_fingerprint"] != fingerprint(SEARCH)
                            or result["boards"] != 32 or result["seed"] != 20261028):
                        raise ValueError("Existing search variant differs from the frozen experiment")
                    report["results"][variant] = {k: v for k, v in result.items() if k not in (
                        "imps", "raw_differences", "board_ids", "current_as_declarer_tricks", "opponent_as_declarer_tricks")}
                    continue
                print("Search comparison:", variant, flush=True)
                with SolverClient(port=ready["port"]) as solver:
                    neural = NeuralPlayPolicy(ROOT/frozen["path"], device="cuda")
                    enemy = TimedPolicy(NeuralPlayPolicy(ROOT/frozen["path"], device="cuda"))
                    if variant == "neural-self":
                        policy = neural
                    elif variant == "uniform-dds":
                        policy = SamplingDdsPolicy(solver, samples=8, seed=20261028)
                    else:
                        policy = HybridPlayPolicy(neural, SamplingDdsPolicy(solver, samples=8, seed=20261028))
                    timed = TimedPolicy(policy)
                    start = time.perf_counter()
                    result = evaluate_pairs(timed, enemy, data, count=32, seed=20261028, batch_size=8)
                    result.update(variant=variant, seconds=time.perf_counter()-start, latency=timed.metrics(),
                                  model_sha256=frozen["sha256"], dataset_fingerprint=fingerprint(SEARCH),
                                  neural_action_selection="greedy", samples_per_nonforced_search=0 if variant == "neural-self" else 8)
                    if variant == "neural-self" and any(result["imps"]):
                        raise ValueError("Neural self-play failed the identical-policy symmetry check")
                    write(output, result)
                    report["results"][variant] = {k: v for k, v in result.items() if k not in (
                        "imps", "raw_differences", "board_ids", "current_as_declarer_tricks", "opponent_as_declarer_tricks")}
            # The pure neural baseline may run past the server's idle-connection
            # timeout. Query metadata through a new connection; never resubmit work.
            with SolverClient(port=ready["port"]) as metadata:
                report["solver"], report["solver_stats"] = metadata.get_info(), metadata.get_stats()
            write(OUTPUT/"search-report.json", report)
        finally:
            if ready:
                try:
                    os.kill(ready["process_id"], signal.SIGTERM)
                except OSError:
                    pass
            if service.poll() is None:
                service.terminate()
            service.wait(timeout=15)


def training_progress():
    progress = {}
    for seed in (43, 44):
        directory = REPLICAS/f"sl-{seed}"
        metrics = directory/"metrics.jsonl"
        events = [json.loads(line) for line in metrics.read_text().splitlines() if line] if metrics.exists() else []
        updates = [row for row in events if "validation" in row]
        progress[str(seed)] = ({"step": updates[-1]["step"], "validation": updates[-1]["validation"]}
                               if updates else {"step": 0, "validation": None})
    return progress


def wait_for_replicas(status):
    deadline = time.monotonic()+7200
    while True:
        progress = training_progress()
        status.update(phase="training replicas", progress=progress)
        write(REPLICAS/"status.json", status)
        completed = []
        for seed in (43, 44):
            directory = REPLICAS/f"sl-{seed}"
            checkpoint = directory/"last.pt"
            last_step = read_checkpoint(checkpoint)["metadata"]["step"] if checkpoint.exists() else 0
            completed.append(last_step == 52928)
            launcher = json.loads((directory/"launcher.json").read_text(encoding="utf-8-sig"))
            if last_step < 52928 and not process_running(launcher["launcher_process_id"]):
                raise RuntimeError(f"Replica {seed} stopped early; inspect its stderr.log and resume")
        if all(completed):
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Replica training exceeded the two-hour monitoring bound")
        time.sleep(20)


def freeze():
    frozen = {}
    for seed in SEEDS:
        path = ROOT/MODELS["candidate"] if seed == 42 else REPLICAS/f"sl-{seed}/best.pt"
        checkpoint = read_checkpoint(path)
        frozen[str(seed)] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                              "selected_step": checkpoint["metadata"]["step"],
                              "validation": checkpoint["metadata"]["validation"]}
        if seed != 42:
            shutil.copyfile(REPLICAS/f"sl-{seed}/metrics.jsonl", OUTPUT/f"sl-{seed}-metrics.jsonl")
    immutable(OUTPUT/"frozen-models.json", {"models": frozen, "protocol_sha256": sha256(OUTPUT/"protocol.json")})


def confirm():
    frozen = json.loads((OUTPUT/"frozen-models.json").read_text())["models"]
    data_hash = fingerprint(CONFIRM)
    for seed in SEEDS:
        model = frozen[str(seed)]
        if sha256(ROOT/model["path"]) != model["sha256"]:
            raise ValueError("A confirmation model changed")
        for opponent in ("initial-sl", "old-ppo", "random"):
            enemy = (ROOT/f"runs/cardplay-expanded-v1/sl-{seed}/best.pt" if opponent == "initial-sl"
                     else ROOT/MODELS["old-ppo"] if opponent == "old-ppo" else None)
            output = OUTPUT/f"sl-{seed}-vs-{opponent}.json"
            if output.exists():
                old = json.loads(output.read_text())
                if (old["model_sha256"] != model["sha256"] or old["dataset_fingerprint"] != data_hash
                        or old.get("opponent_model_sha256") != (sha256(enemy) if enemy else None)
                        or old["seed"] != 20261027 or old["boards"] != 2048):
                    raise ValueError("Existing confirmation differs")
                continue
            print(f"Independent seed confirmation: {seed} vs {opponent}", flush=True)
            arguments = ["evaluate", "--data", str(CONFIRM), "--model", str(ROOT/model["path"]),
                         "--opponent", "neural" if enemy else "random", "--count", "2048", "--batch-size", "32",
                         "--seed", "20261027", "--device", "cuda", "--output", str(output)]
            if enemy:
                arguments += ["--opponent-model", str(enemy)]
            subprocess.run([sys.executable, "-m", "bridgemind.cardplay", *arguments], cwd=ROOT, check=True)


def report():
    summary = {"provenance": provenance(), "training_seeds": SEEDS, "confirmation_boards": 2048,
               "results": {}, "limits": ["Only three training seeds; seed uncertainty is coarsely estimated",
                                          "Confirmation and search panels are disjoint from initial gate panel",
                                          "No PPO expansion has been performed; this tests SL and search only"]}
    expected = {x.decode() for x in PlayData(CONFIRM).ids}
    for opponent in ("initial-sl", "old-ppo", "random"):
        reports = [json.loads((OUTPUT/f"sl-{seed}-vs-{opponent}.json").read_text()) for seed in SEEDS]
        for result in reports:
            if (len(result["board_ids"]) != 2048 or set(result["board_ids"]) != expected
                    or result["imps"] != [imps(x) for x in result["raw_differences"]]):
                raise ValueError("Invalid seed-confirmation board or IMP records")
        summary["results"][opponent] = seed_board_summary(reports)
    summary["search"] = json.loads((OUTPUT/"search-report.json").read_text())
    write(OUTPUT/"report.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "search", "confirm", "report", "all"], default="all")
    args = parser.parse_args()
    verify_gate()
    status_path = REPLICAS/"status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous["state"] == "running" and process_running(previous["process_id"]):
            raise ValueError("A confirmation pipeline is already running; inspect status instead of launching a duplicate")
    status = {"state": "running", "phase": args.phase, "process_id": os.getpid(),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write(REPLICAS/"status.json", status)
    try:
        if args.phase in ("prepare", "all"):
            prepare()
        if args.phase in ("search", "all"):
            status["phase"] = "search comparison while replicas train"
            write(REPLICAS/"status.json", status)
            search()
        if args.phase in ("confirm", "all"):
            wait_for_replicas(status)
            freeze()
            status["phase"] = "independent confirmation"
            write(REPLICAS/"status.json", status)
            confirm()
        if args.phase in ("report", "all"):
            report()
        status.update(state="completed", phase=args.phase+" completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(REPLICAS/"status.json", status)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write(REPLICAS/"status.json", status)
        raise
