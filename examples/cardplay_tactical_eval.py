"""Evaluate any frozen play checkpoint on an existing diagnostic tactical suite."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from bridgemind.cardplay.policy import NeuralPlayPolicy
from bridgemind.dds import SolverClient
from bridgemind.diagnostics.cardplay import evaluate_case
from bridgemind.rating.system import read, write, solver_identity, now
from bridgemind.training.runner import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="artifacts/cardplay-diagnostics-v1/tactical-cases.json")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--port", type=int, default=18765)
    args = p.parse_args()
    torch.set_num_threads(1)
    identity = {"model_sha256": sha256(args.model), "suite_sha256": sha256(args.suite),
                "evaluator_sha256": sha256(Path(__file__).parents[1]/"src/bridgemind/diagnostics/cardplay.py"),
                "device": args.device}
    if Path(args.output).exists() and read(args.output)["identity"] != identity:
        raise ValueError("Choose a new output file for a different model, suite or evaluator")
    neural = NeuralPlayPolicy(args.model, device=args.device)
    rows, families = [], defaultdict(lambda: {"correct": 0, "variants": 0})
    with SolverClient(port=args.port) as solver:
        solver_info = solver_identity(solver)
        for stored in read(args.suite):
            case = {k: stored[k] for k in ("case_id", "family", "board", "prefix", "suggested_good", "suggested_bad", "scope")}
            result = evaluate_case(case, neural, solver)
            rows.append({k: result[k] for k in ("case_id", "family", "model_card", "rule_card", "model_dds_regret", "hidden_world_audit")})
            families[result["family"]]["variants"] += 1
            families[result["family"]]["correct"] += result["model_dds_regret"] == 0
    report = {"identity": identity, "created_at_utc": now(), "solver": solver_info, "families": dict(families), "cases": rows,
              "scope": "diagnostic template variants; not a new strength rating or independent final test"}
    write(args.output, report)
    print(report["families"])


if __name__ == "__main__":
    main()
