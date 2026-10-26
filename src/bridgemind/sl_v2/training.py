"""Time-bounded SL v2, leaving frozen rating and foundation implementations intact."""

import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from bridgemind.cardplay.data import PlayData
from bridgemind.cardplay.encoding import collate
from bridgemind.cardplay.model import load_model, read_checkpoint, save_checkpoint, restore_training
from bridgemind.cardplay.runner import fingerprint
from bridgemind.cardplay.evaluation import evaluate_pairs
from bridgemind.foundation.cases import PERMUTATIONS, build_base
from bridgemind.foundation.system import load_suite, code_identity
from bridgemind.rating.system import digest, read, write, now, runtime, writer_lock, solver_identity
from bridgemind.training.runner import sha256, assert_disjoint, provenance
from .data import augment, orbit_key, prepare_courses, Curriculum


def implementation_id():
    return digest({"sl_v2": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                   "foundation_and_dependencies": code_identity()})


def prepare(root, train_path, valid_path, initial, solver, *, foundation_root="runs/cardplay-foundations-v1",
            train_seeds=tuple(range(260001, 260033)), valid_seeds=tuple(range(270001, 270009)),
            device="cuda", lr=1e-5, batch_size=64, curriculum_fraction=.25, eval_states=4096,
            eval_boards=128, eval_interval=300., save_interval=120., seed=20261026):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Choose a new SL v2 run directory")
    if batch_size < 2 or not 0 < curriculum_fraction < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid mixed-training parameters")
    if not 0 < int(round(batch_size*curriculum_fraction)) < batch_size:
        raise ValueError("Mixed batch must contain both sources")
    train, valid = PlayData(train_path), PlayData(valid_path)
    assert_disjoint(train.ids, valid.ids)
    foundation = load_suite(foundation_root)
    # Audit full-deal orbits, not merely raw IDs: training augmentation cannot
    # transform a training hand into a validation or published foundation hand.
    real_train_orbits = {orbit_key(g["owners"]) for g in train.load_games()}
    real_valid_orbits = {orbit_key(g["owners"]) for g in valid.load_games()}
    if real_train_orbits & real_valid_orbits:
        raise ValueError("Real train/validation overlap under suit or seat symmetry")
    published = {orbit_key(b["board"]["owners"]) for b in foundation["bases"]}
    for path in (Path("artifacts/cardplay-diagnostics-v1/tactical-cases.json"),):
        if path.exists():
            published.update(orbit_key(c["board"]["owners"]) for c in read(path))
    rating = Path("runs/cardplay-rating-v1/boards.json")
    if rating.exists():
        published.update(orbit_key(b["owners"]) for b in read(rating))
    if real_train_orbits & published:
        raise ValueError("Real training collides with a published diagnostic/test deal")
    courses, audit = prepare_courses(solver, train_seeds=train_seeds, valid_seeds=valid_seeds,
                                    excluded_orbits=real_train_orbits | real_valid_orbits | published)
    config = {"schema_version": 1, "kind": "play_sl_v2", "created_at_utc": now(), "implementation_sha256": implementation_id(),
              "environment": runtime(device), "initial_source": str(Path(initial).resolve()), "initial_sha256": sha256(initial),
              "train": {"path": str(Path(train_path).resolve()), "fingerprint": fingerprint(train_path)},
              "valid": {"path": str(Path(valid_path).resolve()), "fingerprint": fingerprint(valid_path)},
              "curriculum_sha256": digest(courses), "curriculum_audit": audit,
              "orbit_audit": {"real_train_orbits": len(real_train_orbits), "real_valid_orbits": len(real_valid_orbits),
                              "published_orbits": len(published), "train_valid_orbit_intersection": 0,
                              "training_published_orbit_intersection": 0},
              "seed": seed, "lr": lr, "batch_size": batch_size, "curriculum_fraction": curriculum_fraction,
              "real_samples_per_batch": batch_size-int(round(batch_size*curriculum_fraction)),
              "eval_states": eval_states, "eval_boards": eval_boards, "eval_interval_seconds": eval_interval,
              "save_interval_seconds": save_interval, "solver_for_curriculum": solver_identity(solver),
              "loss": "set-valued policy likelihood on all rows; belief*.2/value*.1 on real rows only",
              "augmentation": "real rows: half original recorded views, half uniform-24 transformed play with legal synthetic contract summary/system=unknown; curriculum: uniform-24; masks, ownership and score targets transformed",
              "selection": "best validation IMP vs frozen initial model, provided curriculum macro does not fall below initialization; ties: curriculum then real augmented NLL",
              "limits": ["Published foundation templates are development tasks; new-seed splits are variant-held-out, not unseen templates",
                         "No independent test or rating boards are used for training/checkpoint selection",
                         "Real validation is reused historical validation, not a new final test"]}
    root.mkdir(parents=True, exist_ok=False)
    write(root/"curriculum.json", courses)
    shutil.copyfile(initial, root/"init.pt")
    if sha256(root/"init.pt") != config["initial_sha256"]:
        raise ValueError("Initial checkpoint snapshot mismatch")
    write(root/"protocol.json", config)
    return config


def mixed_update(model, optimizer, rows):
    device = next(model.parameters()).device
    batch = collate([r["encoded"] for r in rows], device)
    dist, values, beliefs = model.distribution(batch)
    accepted = torch.as_tensor(np.stack([r["accepted"] for r in rows]), device=device).bool()
    if (accepted & ~batch["legal"].bool()).any() or not accepted.any(-1).all():
        raise ValueError("Mixed SL target includes illegal cards or empty answer set")
    # A set loss accepts every certified optimum without forcing arbitrary ties.
    set_nll = -torch.logsumexp(dist.logits.masked_fill(~accepted, -torch.inf), -1)
    is_course = torch.as_tensor([r.get("curriculum", False) for r in rows], device=device)
    real = ~is_course
    if not real.any() or not is_course.any():
        raise ValueError("Each mixed batch needs real and curriculum rows")
    owners = torch.as_tensor(np.stack([r["owners"] for r in rows]), device=device).long()
    returns = torch.as_tensor([r["return"] for r in rows], device=device).float()
    unknown = batch["unknown"][real]
    per_card = F.cross_entropy(beliefs[real].transpose(1, 2), owners[real], reduction="none")
    belief = (per_card*unknown).sum()/unknown.sum().clamp_min(1)
    value = F.mse_loss(values[real], returns[real])
    loss = set_nll.mean()+.2*belief+.1*value
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    correct = accepted.gather(1, dist.probs.argmax(-1, keepdim=True)).squeeze(1)
    return {"loss": float(loss.detach()), "real_nll": float(set_nll[real].mean().detach()),
            "curriculum_nll": float(set_nll[is_course].mean().detach()), "belief_loss": float(belief.detach()),
            "value_loss": float(value.detach()), "real_accuracy": float(correct[real].float().mean()),
            "curriculum_accuracy": float(correct[is_course].float().mean()), "gradient_norm": float(norm)}


def batches(train, curriculum, config, rng):
    n = config["batch_size"]
    real_count = config["real_samples_per_batch"]
    real_rows = train.sample(real_count, rng)
    rows = [augment(row, "CDHS") if i < real_count//2 else
            augment(row, PERMUTATIONS[int(rng.integers(24))], contract_only=True) for i, row in enumerate(real_rows)]
    rows += [augment(r, PERMUTATIONS[int(rng.integers(24))]) for r in curriculum.sample(n-real_count, rng)]
    return [rows[i] for i in rng.permutation(len(rows))]


def validate(model, real_data, curriculum, config, anchor):
    model.eval()
    rng = np.random.default_rng(77123)
    real_rows = real_data.sample(config["eval_states"], rng)
    for r in real_rows:
        r["accepted"] = np.zeros(52, dtype=bool); r["accepted"][r["action"]] = True
    augmented = [augment(r, PERMUTATIONS[int(rng.integers(24))], contract_only=True) for r in real_rows]
    course_rows = []
    # All 24 suit transformations, not just a cherry-picked orientation.
    for row in curriculum.rows:
        course_rows.extend(augment(row, p) for p in PERMUTATIONS)
    def metrics(rows):
        nll, correct = 0., 0
        by_family = {}
        device = next(model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                subset = rows[start:start+128]
                dist, _, _ = model.distribution(collate([r["encoded"] for r in subset], device))
                accepted = torch.as_tensor(np.stack([r["accepted"] for r in subset]), device=device).bool()
                losses = -torch.logsumexp(dist.logits.masked_fill(~accepted, -torch.inf), -1)
                hits = accepted.gather(1, dist.probs.argmax(-1, keepdim=True)).squeeze(1).cpu().tolist()
                nll += float(losses.sum()); correct += sum(hits)
                for row, hit in zip(subset, hits):
                    if "family" in row:
                        stat = by_family.setdefault(row["family"], [0, 0]); stat[0] += hit; stat[1] += 1
        return {"nll": nll/len(rows), "accuracy": correct/len(rows), "states": len(rows),
                "by_family": {k: v[0]/v[1] for k, v in by_family.items()}}
    canonical, shifted, tactical = metrics(real_rows), metrics(augmented), metrics(course_rows)
    tactical["macro_accuracy"] = float(np.mean(list(tactical["by_family"].values())))
    if config["eval_boards"]:
        match = evaluate_pairs(model, anchor, real_data, count=config["eval_boards"], seed=20261026, batch_size=16)
        match = {k: v for k, v in match.items() if k not in ("board_ids", "imps", "raw_differences", "current_as_declarer_tricks", "opponent_as_declarer_tricks")}
    else:
        match = {"boards": 0, "mean_imps": 0., "scope": "disabled in deterministic unit/smoke tests"}
    return {"real_canonical": canonical, "real_augmented": shifted, "curriculum": tactical,
            "duplicate_vs_initial": match, "scope": "validation for selection; not independent final strength evidence"}


def train(root, *, seconds=7200, resume=False, max_steps=10**9):
    root = Path(root).resolve()
    with writer_lock(root):
        config = read(root/"protocol.json")
        if config["implementation_sha256"] != implementation_id() or config["environment"] != runtime(config["environment"]["device"]):
            raise ValueError("SL v2 code/runtime changed; choose a new run")
        if sha256(root/"init.pt") != config["initial_sha256"]:
            raise ValueError("Frozen initialization changed")
        courses = read(root/"curriculum.json")
        if digest(courses) != config["curriculum_sha256"]:
            raise ValueError("Curriculum changed")
        for split in ("train", "valid"):
            if fingerprint(config[split]["path"]) != config[split]["fingerprint"]:
                raise ValueError("Real data changed")
        if seconds <= 0 or max_steps < 1:
            raise ValueError("Positive time/step budget required")
        if (root/"last.pt").exists() and not resume:
            raise ValueError("Run already trained; use --resume")
        torch.set_num_threads(1); torch.manual_seed(config["seed"])
        rng = np.random.default_rng(config["seed"])
        device = config["environment"]["device"]
        model, metadata = load_model(root/("last.pt" if resume else "init.pt"), device)
        anchor = load_model(root/"init.pt", device)[0].eval().requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        if resume:
            if metadata["v2_protocol_sha256"] != digest(config):
                raise ValueError("Checkpoint protocol mismatch")
            restore_training(root/"last.pt", optimizer, rng)
        else:
            metadata = {}
        real_train, real_valid = PlayData(config["train"]["path"]), PlayData(config["valid"]["path"])
        course_train, course_valid = Curriculum(courses["train"]), Curriculum(courses["valid"])
        step, consumed = metadata.get("step", 0), metadata.get("elapsed_seconds", 0.)
        tick = time.monotonic()
        def elapsed():
            return consumed+time.monotonic()-tick
        baseline = metadata.get("baseline")
        best_key = tuple(metadata.get("best_key", (-float("inf"), -float("inf"), -float("inf"))))
        best_course_key = tuple(metadata.get("best_course_key", (-float("inf"), -float("inf"))))
        status = {"state": "running", "phase": "initializing", "process_id": os.getpid(), "started_at_utc": now(),
                  "expected_finish_utc": (datetime.now(timezone.utc)+timedelta(seconds=max(0, seconds-consumed))).isoformat(),
                  "time_budget_seconds": seconds, "run": str(root), "device": device, "step": step,
                  "protocol_sha256": digest(config), "training_mix": {"real": config["real_samples_per_batch"], "curriculum": config["batch_size"]-config["real_samples_per_batch"]}}
        write(root/"status.json", status)
        def event(record):
            row = {"time_utc": now(), "step": step, "elapsed_seconds": elapsed(), **record}
            with (root/"metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        def save_last():
            save_checkpoint(root/"last.pt", model, metadata={"kind": "play_sl_v2", "step": step, "elapsed_seconds": elapsed(),
                "v2_protocol_sha256": digest(config), "baseline": baseline, "best_key": best_key, "best_course_key": best_course_key,
                "validation": last_validation, "provenance": provenance()}, optimizer=optimizer, rng=rng)
        def evaluate_and_save():
            nonlocal best_key, best_course_key, last_validation, baseline
            status.update(phase="validation", step=step, elapsed_seconds=elapsed()); write(root/"status.json", status)
            last_validation = validate(model, real_valid, course_valid, config, anchor)
            if baseline is None:
                baseline = last_validation
            skill = last_validation["curriculum"]["macro_accuracy"]
            real_nll = last_validation["real_augmented"]["nll"]
            imps = last_validation["duplicate_vs_initial"]["mean_imps"]
            key = (imps, skill, -real_nll)
            eligible = skill+1e-12 >= baseline["curriculum"]["macro_accuracy"]
            meta = {"kind": "play_sl_v2", "step": step, "validation": last_validation, "v2_protocol_sha256": digest(config)}
            if eligible and key > best_key:
                best_key = key; save_checkpoint(root/"best.pt", model, metadata=meta)
            course_key = (skill, -real_nll)
            if course_key > best_course_key:
                best_course_key = course_key; save_checkpoint(root/"best-curriculum.pt", model, metadata=meta)
            event({"event": "validation", "validation": last_validation, "eligible_for_best": eligible})
            save_last()
        last_validation = metadata.get("validation")
        last_eval = last_save = last_status = time.monotonic()
        try:
            if not resume:
                evaluate_and_save()
            last_eval = last_save = last_status = time.monotonic()
            model.train()
            while elapsed() < seconds and step < max_steps and not (root/"STOP").exists():
                rows = batches(real_train, course_train, config, rng)
                metrics = mixed_update(model, optimizer, rows)
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise ValueError("Non-finite SL v2 training metrics")
                step += 1
                current = time.monotonic()
                if current-last_status >= 30 or step == 1:
                    status.update(phase="training", step=step, elapsed_seconds=elapsed(), updated_at_utc=now(), train=metrics)
                    write(root/"status.json", status); event({"event": "progress", "train": metrics})
                    last_status = current
                if current-last_eval >= config["eval_interval_seconds"]:
                    evaluate_and_save(); last_eval = last_save = time.monotonic(); model.train()
                elif current-last_save >= config["save_interval_seconds"]:
                    save_last(); last_save = time.monotonic()
            reason = "stop_requested" if (root/"STOP").exists() else "time_budget" if elapsed() >= seconds else "step_limit"
            evaluate_and_save()
            status.update(state="stopped" if reason == "stop_requested" else "completed", phase="saved", step=step,
                          elapsed_seconds=elapsed(), finished_at_utc=now(), stop_reason=reason,
                          best_step=read_checkpoint(root/"best.pt")["metadata"]["step"],
                          best_curriculum_step=read_checkpoint(root/"best-curriculum.pt")["metadata"]["step"],
                          best_model_sha256=sha256(root/"best.pt"), last_model_sha256=sha256(root/"last.pt"), validation=last_validation)
            write(root/"status.json", status); write(root/"report.json", status)
            return status
        except BaseException as error:
            status.update(state="failed", phase="error", step=step, elapsed_seconds=elapsed(), finished_at_utc=now(),
                          error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            write(root/"status.json", status)
            raise
