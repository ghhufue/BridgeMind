"""Validation-only search/latency selection, then a single fresh 1024-board test.

python examples/cardplay_search_benchmark.py --phase all
Progress and logs: runs/cardplay-search-selection-v1/
Frozen protocol and results: artifacts/cardplay-search-selection-v1/
"""

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from cardplay_long_sl import process_running
from cardplay_long_sl_eval import collect_report_ids
from bridgemind.cardplay.benchmark import TimedPlayPolicy, choose_search_configuration
from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.evaluation import play_pairs
from bridgemind.cardplay.model import load_model
from bridgemind.cardplay.policy import NeuralPlayPolicy, HybridPlayPolicy
from bridgemind.cardplay.runner import fingerprint
from bridgemind.cardplay.search import SamplingDdsPolicy
from bridgemind.dds import SolverClient
from bridgemind.training.__main__ import ASSETS
from bridgemind.training.rollout import imps
from bridgemind.training.runner import assert_disjoint, provenance, sha256

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/cardplay-search-selection-v1"
RUN = ROOT/"runs/cardplay-search-selection-v1"
DATA = ROOT/"data/cardplay-search-selection-v1"
VALID = ROOT/"data/cardplay-expanded-v1/valid"
SEED = 20261030
NEURAL = {"id": "neural", "kind": "neural", "samples": 0}
CONFIGS = [NEURAL, *[{"id": f"{kind}-{count}", "kind": kind, "samples": count}
                     for kind in ("uniform", "hybrid") for count in (8, 16, 32)]]
PROTOCOL = {
    "name": "cardplay-search-selection-v1", "date": "2026-10-25", "configs": CONFIGS,
    "model_selection": "Lowest validation NLL among the three long-SL frozen checkpoints; no test-based seed choice",
    "validation_boards": 128, "validation_seed": SEED,
    "selection": {"mean_budget_ms": 100., "p95_budget_ms": 250., "tie_band_imps": .25},
    "selection_rule": "Filter by nonforced latency budgets; among positive search means within 0.25 IMP of best, choose lowest mean nonforced latency. Otherwise retain neural",
    "confirmation_boards": 1024, "confirmation_seed": SEED+1,
    "confirmation": "Freeze one configuration before test; compare only that configuration to the fixed neural baseline",
    "runtime": {"device": "cuda", "cpu_threads": 1, "DDS_workers": 4,
                "fresh_DDS_service_per_trial": True, "NN_batch": 1, "inference": "greedy neural",
                "NN_warmup": "3 forwards on a validation board outside the tuning panel; excluded from timing",
                "no_concurrent_training": True, "search_utility": "expected sampled raw duplicate score"},
    "uncertainty": "2000 full-deal bootstrap draws; validation intervals are exploratory after configuration selection",
    "limits": "Single frozen SL seed and fixed recorded auction; no strength rating against humans or full WBridge5 engine",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen record changed: {path}")
    else:
        write(path, value)


def status(**values):
    value = json.loads((RUN/"status.json").read_text())
    value.update(values, updated_at_utc=datetime.now(timezone.utc).isoformat())
    write(RUN/"status.json", value)


def gpu_state():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
                                      "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                         "--format=csv,noheader"], text=True).strip()
    foreign_python = []
    for line in processes.splitlines():
        if "," not in line:
            continue
        pid, name = line.split(",", 1)
        if "python" in name.lower() and pid.strip().isdigit() and int(pid) != os.getpid():
            foreign_python.append(line)
    if foreign_python:
        raise RuntimeError("Another Python CUDA process is active; stop concurrent training before clean latency measurement: "+str(foreign_python))
    return {"gpu": output, "compute_processes": processes, "other_python_CUDA_processes": foreign_python}


def freeze():
    immutable(ART/"protocol.json", PROTOCOL)
    previous = json.loads((ROOT/"artifacts/cardplay-long-sl-20261025/replication/frozen-models.json").read_text())
    seed, candidate = min(previous["models"].items(), key=lambda item: item[1]["validation"]["nll"])
    if sha256(ROOT/candidate["path"]) != candidate["sha256"]:
        raise ValueError("Validation-selected long SL changed")
    implementation = {name: sha256(ROOT/"src/bridgemind/cardplay"/name)
                      for name in ("benchmark.py", "sampling.py", "search.py", "policy.py")}
    immutable(ART/"frozen-baseline.json", {"seed": seed, "model": candidate,
              "protocol_sha256": sha256(ART/"protocol.json"), "implementation_sha256": implementation,
              "validation_fingerprint": fingerprint(VALID)})


def prepare():
    exclusion_path = ROOT/"data/cardplay-search-selection-v1-excluded.npy"
    if exclusion_path.exists():
        audit = json.loads((ART/"exclusion-audit.json").read_text())
        if sha256(exclusion_path) != audit["excluded_ids_sha256"]:
            raise ValueError("Exclusion snapshot changed")
        excluded = set(np.load(exclusion_path, allow_pickle=False))
    else:
        excluded, sources = set(), {}
        for path in sorted((ROOT/"data").rglob("board-ids.npy")):
            if path == ROOT/"data/supervised/test/board-ids.npy" or path.is_relative_to(DATA):
                continue
            ids = set(np.load(path, allow_pickle=False))
            excluded.update(ids)
            sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"data").rglob("*.npz")):
            if path.name not in ("train.npz", "valid.npz", "test.npz"):
                continue
            with np.load(path, allow_pickle=False) as arrays:
                ids = set(arrays["ids"]) if "ids" in arrays.files else set()
            if ids:
                excluded.update(ids)
                sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"artifacts").rglob("*board-ids.npy")):
            if path.is_relative_to(ART):
                continue
            ids = set(np.load(path, allow_pickle=False))
            excluded.update(ids)
            sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        for directory in (ROOT/"artifacts", ROOT/"runs"):
            for path in sorted(directory.rglob("*.json")):
                if path.is_relative_to(ART) or path.is_relative_to(RUN):
                    continue
                ids = collect_report_ids(json.loads(path.read_text(encoding="utf-8-sig")))
                if ids:
                    excluded.update(ids)
                    sources[path.relative_to(ROOT).as_posix()] = {"ids": len(ids), "sha256": sha256(path)}
        np.save(exclusion_path, np.asarray(sorted(excluded), dtype="S64"), allow_pickle=False)
        immutable(ART/"exclusion-audit.json", {"distinct_excluded_ids": len(excluded), "sources": sources,
                  "excluded_ids_sha256": sha256(exclusion_path)})
    allowed = set(np.load(ROOT/"data/supervised/test/board-ids.npy", allow_pickle=False))-excluded
    source = ROOT/"data/downloads/openspiel-test.txt"
    if sha256(source) != ASSETS[source.name][1]:
        raise ValueError("Official trajectory source hash differs")
    if not (DATA/"manifest.json").exists():
        status(phase="preparing independent holdout", eligible_boards=len(allowed))
        prepare_dataset(source, DATA, allowed_ids=allowed, max_boards=1024,
                        system_id=1, system_description="WBridge5 SAYC; new search-selection holdout")
    data = PlayData(DATA)
    data.load_games()
    if data.manifest["boards"] != 1024 or not set(data.ids).issubset(allowed):
        raise ValueError("Holdout contains excluded boards")
    assert_disjoint(data.ids, np.asarray(sorted(excluded), dtype="S64"))
    valid = PlayData(VALID)
    order = np.random.default_rng(SEED).permutation(len(valid.load_games()))
    panel = np.asarray([valid.load_games()[int(i)]["id"].encode() for i in order[:128]], dtype="S64")
    assert_disjoint(panel, data.ids)
    for name, ids in (("holdout-board-ids.npy", data.ids), ("validation-board-ids.npy", panel)):
        path = ART/name
        if path.exists():
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), ids)
        else:
            np.save(path, ids, allow_pickle=False)
    immutable(ART/"data-audit.json", {"eligible_before_import": len(allowed), "holdout_boards": 1024,
              "holdout_states": data.manifest["states"], "holdout_fingerprint": fingerprint(DATA),
              "validation_boards": 128, "validation_fingerprint": fingerprint(VALID),
              "holdout_excluded_intersection": 0, "holdout_validation_intersection": 0,
              "holdout_ids_sha256": sha256(ART/"holdout-board-ids.npy")})


@contextmanager
def dds_service(label):
    ready_path = RUN/f"dds-{label}-{uuid.uuid4().hex}.json"
    with (RUN/f"dds-{label}-stdout.log").open("w") as out, (RUN/f"dds-{label}-stderr.log").open("w") as err:
        process = subprocess.Popen([sys.executable, "-u", "-m", "bridgemind.dds.server", "--workers", "4",
                                    "--port", "0", "--ready-file", str(ready_path)], cwd=ROOT, stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        ready = None
        try:
            deadline = time.monotonic()+30
            while not ready_path.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("DDS startup failed; inspect "+str(RUN/f"dds-{label}-stderr.log"))
                time.sleep(.05)
            ready = json.loads(ready_path.read_text())
            yield ready["port"]
        finally:
            if ready:
                try:
                    os.kill(ready["process_id"], signal.SIGTERM)
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)


def verify_result(result, config, directory, count, seed):
    baseline = json.loads((ART/"frozen-baseline.json").read_text())
    audit = json.loads((ART/"data-audit.json").read_text())
    expected_hash = audit["holdout_fingerprint"] if directory == DATA else audit["validation_fingerprint"]
    if fingerprint(directory) != expected_hash:
        raise ValueError("Frozen evaluation dataset changed")
    expected_order = np.random.default_rng(seed).permutation(len(PlayData(directory).load_games()))[:count]
    games = PlayData(directory).load_games()
    expected_ids = [games[int(i)]["id"] for i in expected_order]
    if (result["config"] != config or result["model_sha256"] != baseline["model"]["sha256"]
            or result["dataset_fingerprint"] != fingerprint(directory) or result["seed"] != seed
            or result["boards"] != count or result["board_ids"] != expected_ids
            or len(result["imps"]) != count or len(result["raw_differences"]) != count
            or result["imps"] != [imps(x) for x in result["raw_differences"]]
            or not np.isclose(np.mean(result["imps"]), result["mean_imps"])):
        raise ValueError("Result does not match frozen model, data, configuration and board order")


def match(config, directory, count, seed, phase):
    label = f"{phase}-{config['id']}"
    output = ART/(label+".json")
    if output.exists():
        result = json.loads(output.read_text())
        verify_result(result, config, directory, count, seed)
        return result
    idle = gpu_state()
    baseline = json.loads((ART/"frozen-baseline.json").read_text())
    if sha256(ROOT/baseline["model"]["path"]) != baseline["model"]["sha256"]:
        raise ValueError("Frozen neural weights changed")
    model, _ = load_model(ROOT/baseline["model"]["path"], "cuda")
    model.eval()
    neural = NeuralPlayPolicy(model=model)
    valid = PlayData(VALID)
    warm_index = int(np.random.default_rng(SEED).permutation(len(valid.load_games()))[128])
    warm = valid.environment(warm_index).observation()
    for _ in range(3):
        neural.predict(warm)
    data = PlayData(directory)
    order = np.random.default_rng(seed).permutation(len(data.load_games()))[:count]
    rng = np.random.default_rng(seed+100)
    with ExitStack() as stack:
        port, solver = None, None
        if config["kind"] != "neural":
            port = stack.enter_context(dds_service(label))
            solver = stack.enter_context(SolverClient(port=port))
            search = SamplingDdsPolicy(solver, samples=config["samples"], seed=seed+10)
            actor = search if config["kind"] == "uniform" else HybridPlayPolicy(neural, search)
        else:
            actor = neural
        current, enemy = TimedPlayPolicy(actor), TimedPlayPolicy(neural)
        rewards, raw, own, other = [], [], [], []
        start = time.perf_counter()
        status(phase=phase, trial=config["id"], processed_boards=0, trial_boards=count)
        for begin in range(0, len(order), 8):
            _, batch = play_pairs(current, [enemy], data, order[begin:begin+8], rng)
            rewards.extend(batch["imps"]); raw.extend(batch["raw_differences"])
            own.extend(batch["current_as_declarer_tricks"]); other.extend(batch["opponent_as_declarer_tricks"])
            status(phase=phase, trial=config["id"], processed_boards=len(rewards), trial_boards=count,
                   trial_elapsed_seconds=time.perf_counter()-start)
        values = np.asarray(rewards, np.float64)
        bootstrap_rng = np.random.default_rng(seed)
        boot = [bootstrap_rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
        result = {"config": config, "boards": len(values), "seed": seed, "mean_imps": float(values.mean()),
                  "bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(),
                  "standard_error": float(values.std(ddof=1)/np.sqrt(len(values))),
                  "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
                  "imps": rewards, "raw_differences": raw, "current_as_declarer_tricks": own,
                  "opponent_as_declarer_tricks": other, "board_ids": [data.load_games()[int(i)]["id"] for i in order],
                  "seconds": time.perf_counter()-start, "latency": current.metrics(),
                  "model_sha256": baseline["model"]["sha256"], "dataset_fingerprint": fingerprint(directory),
                  "runtime_before_trial": idle, "evaluation_model": "played_out", "opponent": "frozen neural, greedy, batch=1"}
        if config["kind"] == "neural" and any(rewards):
            raise ValueError("Identical greedy neural policies did not tie")
        if solver:
            with SolverClient(port=port) as metadata:
                result["solver"], result["solver_stats"] = metadata.get_info(), metadata.get_stats()
        verify_result(result, config, directory, count, seed)
        write(output, result)
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"trial": label, "mean_imps": result["mean_imps"], "ci": result["bootstrap_95_ci"],
                      "nonforced_latency": result["latency"]["nonforced"], "seconds": result["seconds"]}), flush=True)
    return result


def tune():
    # Do not inspect any final-test outcome until this complete validation grid is frozen.
    ordering = np.random.default_rng(SEED+50).permutation(len(CONFIGS)-1)
    trials = [match(NEURAL, VALID, 128, SEED, "validation")]
    for index in ordering:
        trials.append(match(CONFIGS[int(index)+1], VALID, 128, SEED, "validation"))
    chosen = choose_search_configuration(trials, **PROTOCOL["selection"])
    chosen.update(protocol_sha256=sha256(ART/"protocol.json"), baseline_sha256=sha256(ART/"frozen-baseline.json"),
                  validation_result_sha256={t["config"]["id"]: sha256(ART/f"validation-{t['config']['id']}.json") for t in trials})
    immutable(ART/"selected-configuration.json", chosen)
    print("Frozen validation-selected configuration:", chosen["config"], flush=True)


def confirm():
    chosen = json.loads((ART/"selected-configuration.json").read_text())
    if chosen["protocol_sha256"] != sha256(ART/"protocol.json") or chosen["baseline_sha256"] != sha256(ART/"frozen-baseline.json"):
        raise ValueError("Selection provenance changed")
    for name, digest in chosen["validation_result_sha256"].items():
        if sha256(ART/f"validation-{name}.json") != digest:
            raise ValueError("Selection validation results changed")
    match(chosen["config"], DATA, 1024, SEED+1, "confirmation")


def report():
    chosen = json.loads((ART/"selected-configuration.json").read_text())
    final = json.loads((ART/f"confirmation-{chosen['config']['id']}.json").read_text())
    verify_result(final, chosen["config"], DATA, 1024, SEED+1)
    summary = {"protocol": PROTOCOL, "provenance": provenance(), "selected": chosen,
               "validation": {}, "confirmation": {k: v for k, v in final.items() if k not in (
                   "imps", "raw_differences", "board_ids", "current_as_declarer_tricks", "opponent_as_declarer_tricks")},
               "limits": ["Only the validation-selected search recipe receives the final 1024-board test",
                          "Single frozen SL seed; search randomness uses one held-out seed",
                          "No distillation, new PPO training or human-level rating in this experiment",
                          "This holdout is consumed; future tuning requires another holdout"]}
    for config in CONFIGS:
        trial = json.loads((ART/f"validation-{config['id']}.json").read_text())
        summary["validation"][config["id"]] = {k: v for k, v in trial.items() if k not in (
            "imps", "raw_differences", "board_ids", "current_as_declarer_tricks", "opponent_as_declarer_tricks")}
    write(ART/"report.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "tune", "confirm", "report", "all"], default="all")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/"status.json").exists():
        previous = json.loads((RUN/"status.json").read_text())
        if previous["state"] == "running" and process_running(previous["process_id"]):
            raise ValueError("Benchmark worker already running; inspect status instead of duplicating it")
    write(RUN/"status.json", {"state": "running", "phase": args.phase, "process_id": os.getpid(),
                              "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        torch.set_num_threads(1)
        gpu_state()
        freeze()
        for name, action in (("prepare", prepare), ("tune", tune), ("confirm", confirm), ("report", report)):
            if args.phase in (name, "all"):
                action()
        status(state="completed", phase=args.phase+" completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
