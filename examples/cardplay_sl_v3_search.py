"""Evaluate four SL v2 search recipes before GPU training, using a frozen gate."""

import os
from pathlib import Path
import torch
import numpy as np

import cardplay_search_benchmark as engine
from bridgemind.rating.system import read, write, digest, now, implementation, writer_lock
from bridgemind.cardplay.runner import fingerprint
from bridgemind.training.runner import sha256
from bridgemind.cardplay.benchmark import choose_search_configuration

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/cardplay-sl-v3-20261026/search"
RUN = ROOT/"runs/cardplay-sl-v3-search-20261026"
SEED = 20261030
CONFIGS = [{"id": "neural", "kind": "neural", "samples": 0},
           *[{"id": f"{kind}-{count}", "kind": kind, "samples": count} for kind in ("uniform", "hybrid") for count in (8, 32)]]


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen search record changed: {path}")
    write(path, value)


def main():
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    with writer_lock(RUN):
        engine.ART, engine.RUN = ART, RUN
        engine.DATA = ROOT/"data/cardplay-sl-v3-search-confirm-v1"
        engine.VALID = ROOT/"data/cardplay-expanded-v1/valid"
        engine.SEED = SEED
        checkpoint = ROOT/"runs/play-sl-v2-20261026-2h/best.pt"
        protocol = {"candidate_sha256": sha256(checkpoint), "configs": CONFIGS, "validation_boards": 64,
                    "confirmation_boards": 128, "validation_seed": SEED, "confirmation_seed": SEED+1,
                    "selection": {"mean_budget_ms": 100., "p95_budget_ms": 250., "tie_band_imps": .25},
                    "gate": "distillation candidate only if held-out IMP 95% lower bound >0; no automatic RL",
                    "runtime": "CUDA batch=1, one CPU torch thread, fresh 4-worker DDS per trial; no concurrent training",
                    "implementation_sha256": implementation(), "script_sha256": sha256(__file__),
                    "engine_sha256": sha256(engine.__file__), "limits": "PIMC strategy fusion; single SL v2 seed; validation selection is exploratory"}
        immutable(ART/"protocol.json", protocol)
        immutable(ART/"frozen-baseline.json", {"model": {"path": str(checkpoint), "sha256": sha256(checkpoint)}, "protocol_sha256": sha256(ART/"protocol.json")})
        reservation = read(ROOT/"artifacts/cardplay-sl-v3-20261026/reservation/report.json")
        if fingerprint(engine.DATA) != reservation["search"]["fingerprint"]:
            raise ValueError("Reserved search confirmation changed")
        immutable(ART/"data-audit.json", {"validation_fingerprint": fingerprint(engine.VALID), "holdout_fingerprint": fingerprint(engine.DATA)})
        write(RUN/"status.json", {"state": "running", "phase": "validation", "process_id": os.getpid(), "started_at_utc": now()})
        try:
            engine.gpu_state()
            order = [0, *list(np.random.default_rng(SEED+50).permutation(np.arange(1, len(CONFIGS))))]
            trials = [engine.match(CONFIGS[int(i)], engine.VALID, 64, SEED, "validation") for i in order]
            selected = choose_search_configuration(trials, **protocol["selection"])
            selected.update(validation_result_sha256={t["config"]["id"]: sha256(ART/f"validation-{t['config']['id']}.json") for t in trials},
                            protocol_sha256=digest(protocol))
            immutable(ART/"selected-configuration.json", selected)
            final = engine.match(selected["config"], engine.DATA, 128, SEED+1, "confirmation")
            gate = selected["config"]["kind"] != "neural" and final["bootstrap_95_ci"][0] > 0
            report = {"selected": selected, "confirmation": {k: v for k, v in final.items() if k not in ("imps", "raw_differences", "board_ids", "current_as_declarer_tricks", "opponent_as_declarer_tricks")},
                      "distillation_gate_passed": gate, "rl_authorized_by_this_result": False,
                      "scope": "Search teacher validated against SL v2; must compare with final SL v3 before distilling into SL v3"}
            write(ART/"report.json", report)
            engine.status(state="completed", phase="search complete", finished_at_utc=now())
        except BaseException as error:
            engine.status(state="failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
