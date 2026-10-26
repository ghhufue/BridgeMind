"""Run search, equal-step pilots, two-hour SL v3, then frozen independent acceptance."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from cardplay_long_sl import process_running
from bridgemind.rating.system import read, write, digest, now, writer_lock
from bridgemind.training.runner import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"runs/cardplay-sl-v3-experiment-20261026"
COMMON = ROOT/"runs/sl-v3-common-20261026"
FORMAL = ROOT/"runs/play-sl-v3-20261026-2h"
SEARCH = ROOT/"runs/cardplay-sl-v3-search-20261026"
ART = ROOT/"artifacts/cardplay-sl-v3-20261026"


class StopRequested(Exception):
    pass


def check_stop():
    if (RUN/"STOP").exists():
        raise StopRequested("Experiment STOP requested")


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen experiment differs: {path}")
    else:
        write(path, value)


def clone_run(source, target, fraction):
    source, target = Path(source), Path(target)
    config = read(source/"protocol.json")
    config.update(curriculum_fraction=fraction, real_samples_per_batch=config["batch_size"]-int(round(config["batch_size"]*fraction)))
    if target.exists():
        if read(target/"protocol.json") != config or sha256(target/"init.pt") != config["initial_sha256"]:
            raise ValueError("Existing run differs from the frozen pilot/formal recipe")
        return
    target.mkdir(parents=True)
    for name in ("init.pt", "curriculum.json"):
        shutil.copyfile(source/name, target/name)
    write(target/"protocol.json", config)


def select_pilot(candidates):
    if not candidates:
        raise ValueError("No pilot candidates")
    best = max(c["mean_imps"] for c in candidates)
    near = [c for c in candidates if c["mean_imps"] >= best-.25]
    return min(near, key=lambda c: (-c["novel_accuracy"], c["augmented_nll"], c["fraction"]))


def status(phase, **extra):
    write(RUN/"status.json", {"state": "running", "phase": phase, "process_id": os.getpid(), "updated_at_utc": now(), **extra})
    print(json.dumps({"phase": phase, **extra}), flush=True)


def child(label, args):
    check_stop()
    status(label, command=args)
    with (RUN/(label+".stdout.log")).open("a", encoding="utf-8") as out, (RUN/(label+".stderr.log")).open("a", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; inspect its stderr log")


def wait_for_search():
    if not (SEARCH/"status.json").exists():
        child("search", ["examples/cardplay_sl_v3_search.py"])
    else:
        status("waiting_for_clean_search", active_run=str(SEARCH))
        while True:
            state = read(SEARCH/"status.json")
            if state["state"] == "completed":
                break
            if state["state"] != "running" or not process_running(state["process_id"]):
                child("search-resume", ["examples/cardplay_sl_v3_search.py"])
                break
            time.sleep(5)
    report = read(ART/"search/report.json")
    if read(ART/"search/protocol.json")["script_sha256"] != sha256(ROOT/"examples/cardplay_sl_v3_search.py"):
        raise ValueError("Search worker changed")
    return report


def evaluate_candidate():
    import numpy as np
    import torch
    from bridgemind.cardplay.data import PlayData
    from bridgemind.cardplay.evaluation import evaluate_pairs
    from bridgemind.cardplay.model import load_model
    from bridgemind.cardplay.runner import fingerprint
    from bridgemind.engine.scoring import score_contract
    from bridgemind.dds.models import SEATS
    from bridgemind.rating.policies import RulePolicy
    from bridgemind.training.rollout import imps
    torch.set_num_threads(1)
    completed = read(FORMAL/"report.json")
    if completed["state"] != "completed" or sha256(FORMAL/"best.pt") != completed["best_model_sha256"]:
        raise ValueError("Formal training did not complete with a frozen candidate")
    frozen = RUN/"candidate.pt"
    if not frozen.exists():
        shutil.copyfile(FORMAL/"best.pt", frozen)
    if sha256(frozen) != completed["best_model_sha256"]:
        raise ValueError("Frozen acceptance candidate changed")
    data = PlayData(ROOT/"data/cardplay-sl-v3-confirm-v1")
    reservation = read(ART/"reservation/report.json")
    if fingerprint(data.directory) != reservation["sl"]["fingerprint"]:
        raise ValueError("Reserved SL confirmation changed")
    immutable(ART/"acceptance/protocol.json", {"candidate_sha256": sha256(frozen),
        "initial_sha256": sha256(COMMON/"init.pt"), "data_fingerprint": fingerprint(data.directory),
        "boards": 1024, "seed": 20261102, "primary_opponents": ["sl-v2", "rule-basic"],
        "decision": "both 97.5% deal-bootstrap lower bounds >0 and foundation basic score does not regress",
        "selection": "one checkpoint fixed before reading this confirmation; do not reselect afterward"})
    candidate = load_model(frozen, "cuda")[0].eval()
    initial = load_model(COMMON/"init.pt", "cuda")[0].eval()
    games = {g["id"]: (i, g) for i, g in enumerate(data.load_games())}
    contracts = {identity: data.environment(i).contract for identity, (i, _) in games.items()}
    summaries = {}
    for name, opponent in (("sl-v2", initial), ("rule-basic", RulePolicy())):
        status("independent_acceptance", opponent=name, active_run=str(FORMAL))
        output = ART/f"acceptance/vs-{name}.json"
        if output.exists():
            result = read(output)
            if result["candidate_sha256"] != sha256(frozen) or result["dataset_fingerprint"] != fingerprint(data.directory):
                raise ValueError("Confirmation cache belongs to another candidate/data")
        else:
            result = evaluate_pairs(candidate, opponent, data, count=1024, seed=20261102, batch_size=32)
            result.update(candidate_sha256=sha256(frozen), dataset_fingerprint=fingerprint(data.directory))
            write(output, result)
        if len(result["board_ids"]) != 1024 or set(result["board_ids"]) != set(games):
            raise ValueError("Incomplete/wrong confirmation IDs")
        for identity, own, other, raw, imp in zip(result["board_ids"], result["current_as_declarer_tricks"],
                result["opponent_as_declarer_tricks"], result["raw_differences"], result["imps"], strict=True):
            contract = contracts[identity]
            vulnerable = bool(games[identity][1]["vulnerable"][SEATS.index(contract.declarer) % 2])
            difference = score_contract(contract, int(own), vulnerable)-score_contract(contract, int(other), vulnerable)
            if difference != raw or imps(difference) != imp:
                raise ValueError("Independent terminal score/IMP recomputation failed")
        values = np.asarray(result["imps"], dtype=float)
        rng = np.random.default_rng(20261102)
        boot = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(10000)]
        summaries[name] = {"mean_imps": float(values.mean()), "bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(),
                           "primary_97_5_ci": np.quantile(boot, [.0125, .9875]).tolist(),
                           "wins": int((values>0).sum()), "ties": int((values==0).sum()), "losses": int((values<0).sum())}
    child("foundation", ["-m", "bridgemind.foundation", "evaluate", "--id", "sl-v3-20261026", "--model", str(frozen)])
    child("foundation-export", ["-m", "bridgemind.foundation", "export"])
    foundation = read(ROOT/"runs/cardplay-foundations-v1/reports/sl-v3-20261026.json")
    write(ART/"acceptance/foundation.json", foundation)
    from bridgemind.rating.system import RatingSystem, specification
    from bridgemind.training.data import board_ids
    from cardplay_search_benchmark import dds_service
    import cardplay_search_benchmark as benchmark
    benchmark.RUN = RUN
    with dds_service("final-rating") as port, writer_lock(ROOT/"runs/cardplay-rating-v1"):
        rater = RatingSystem(ROOT/"runs/cardplay-rating-v1", port=port)
        spec = specification("sl-v3-20261026", "neural", checkpoint=frozen, training_data=[ROOT/"data/cardplay-sl-v3-50k/train"])
        ids = set(spec["training_ids"])
        inherited = []
        for path in (COMMON/"curriculum.json", ROOT/"runs/play-sl-v2-20261026-2h/curriculum.json"):
            cases = read(path)["train"]
            ids.update(board_ids(np.asarray([c["board"]["owners"]], dtype=np.uint8))[0].decode() for c in cases)
            inherited.append({"path": str(path), "sha256": sha256(path), "training_cases": len(cases)})
        spec["training_ids"], spec["declared_curricula"] = sorted(ids), inherited
        rater.register(spec)
        rating = rater.evaluate(spec["id"])
        rater.export(ROOT/"artifacts/cardplay-rating-v1")
    write(ART/"acceptance/rating.json", rating)
    report = {"candidate_sha256": sha256(frozen), "results": summaries, "rating": rating["score"],
              "foundation_basic_percent": foundation["summary"]["all_suits_basic_percent"],
              "promotion_gate_passed": all(s["primary_97_5_ci"][0] > 0 for s in summaries.values()) and foundation["summary"]["all_suits_basic_percent"] >= 100.,
              "scope": "single training seed, fixed WBridge5 SAYC recorded contracts; no human rank, no new-template test",
              "finished_at_utc": now()}
    write(ART/"acceptance/report.json", report)
    for name in ("protocol.json", "report.json", "metrics.jsonl"):
        shutil.copyfile(FORMAL/name, ART/("training-"+name))
    return report


def main():
    RUN.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    with writer_lock(RUN):
        config = read(COMMON/"protocol.json")
        protocol = {"name": "cardplay-sl-v3-20261026", "script_sha256": sha256(__file__),
                    "common_protocol_sha256": digest(config), "initial_sha256": config["initial_sha256"],
                    "pilot_fractions": [.125, .25], "pilot_steps": 2000, "pilot_max_seconds": 900,
                    "pilot_selection": "within 0.25 IMP of best validation: highest novel accuracy, then lowest augmented NLL, then lower fraction",
                    "formal_seconds": 7200, "formal_initialization": "original SL v2; no pilot weight continuation",
                    "final_evaluation": "reserved 1024 deals vs SL v2 and rule; frozen foundation/rating; no automatic deployment",
                    "next_stage": "only consider distillation if search passes its separate holdout; compare teacher to final SL v3 first; RL deferred"}
        immutable(ART/"experiment-protocol.json", protocol)
        try:
            search = wait_for_search()
            check_stop()
            pilots = []
            for fraction in protocol["pilot_fractions"]:
                check_stop()
                label = f"pilot-{int(1000*fraction):03d}"
                root = RUN/label
                clone_run(COMMON, root, fraction)
                if not (root/"report.json").exists():
                    arguments = ["-m", "bridgemind.sl_v3", "train", "--run", str(root), "--seconds", "900", "--max-steps", "2000"]
                    if (root/"last.pt").exists():
                        arguments.append("--resume")
                    child(label, arguments)
                if read(root/"report.json")["state"] == "stopped":
                    raise StopRequested("Pilot STOP requested")
                from bridgemind.cardplay.model import read_checkpoint
                meta = read_checkpoint(root/"best.pt")["metadata"]
                validation = meta["validation"]
                pilots.append({"fraction": fraction, "mean_imps": validation["duplicate_vs_initial"]["mean_imps"],
                               "novel_accuracy": validation["curriculum"]["novel_macro_accuracy"],
                               "augmented_nll": validation["real_augmented"]["nll"], "best_step": meta["step"],
                               "checkpoint_sha256": sha256(root/"best.pt")})
            selected = select_pilot(pilots)
            check_stop()
            immutable(ART/"selected-training.json", {"pilots": pilots, "selected": selected, "selection_scope": "validation only; no independent confirmation outcomes read"})
            clone_run(COMMON, FORMAL, selected["fraction"])
            if not (FORMAL/"report.json").exists():
                arguments = ["-m", "bridgemind.sl_v3", "train", "--run", str(FORMAL), "--seconds", "7200"]
                if (FORMAL/"last.pt").exists():
                    arguments.append("--resume")
                child("formal-training", arguments)
            if read(FORMAL/"report.json")["state"] == "stopped":
                raise StopRequested("Formal training STOP requested")
            check_stop()
            accepted = evaluate_candidate()
            write(ART/"report.json", {"search": search, "selected_training": selected, "acceptance": accepted,
                  "automatic_distillation_started": False, "automatic_rl_started": False,
                  "next_decision": "Compare selected search teacher against frozen SL v3 on validation before another training stage"})
            write(RUN/"status.json", {"state": "completed", "phase": "saved", "process_id": os.getpid(), "finished_at_utc": now(),
                                      "formal_run": str(FORMAL), "promotion_gate_passed": accepted["promotion_gate_passed"]})
        except StopRequested as error:
            write(RUN/"status.json", {"state": "stopped", "phase": "saved", "process_id": os.getpid(),
                                      "reason": str(error), "updated_at_utc": now()})
        except BaseException as error:
            write(RUN/"status.json", {"state": "failed", "process_id": os.getpid(), "error": f"{type(error).__name__}: {error}",
                                      "traceback": traceback.format_exc(), "updated_at_utc": now()})
            raise


if __name__ == "__main__":
    main()
