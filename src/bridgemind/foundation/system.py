"""Versioned foundational skill suites and tie-aware symmetry scorecards."""

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import torch

from bridgemind.cardplay.policy import NeuralPlayPolicy
from bridgemind.engine.play import card_id, card_name
from bridgemind.rating.system import digest, read, write, now, identifier, solver_identity, implementation, runtime
from bridgemind.training.runner import sha256
from .cases import TEMPLATES, ROLES, PERMUTATIONS, build_base, observe, card_map, transform_observation, transform_case
from .certificates import certify, public_action


def code_identity():
    from bridgemind.diagnostics import cardplay
    return digest({"foundation": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                   "diagnostic_dependency": sha256(cardplay.__file__), "engine_and_policies": implementation()})


def build_suite(root, solver, *, name="cardplay-foundations-v1", seeds=(1701, 1709, 1723)):
    root = Path(root)
    identifier(name)
    if root.exists():
        raise ValueError("Choose a new suite directory; existing suites are immutable")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Distinct nonnegative seeds required")
    bases = []
    for family in TEMPLATES:
        for role in ROLES:
            for seed in seeds:
                base = build_base(family, seed, role)
                obs = observe(base)
                if base["mechanical"]:
                    if len(obs["legal_card_ids"]) != 1:
                        raise ValueError("Mechanical follow-suit fixture must have one legal move")
                    certificate = {"status": "mechanical", "accepted_cards": [card_name(obs["legal_card_ids"][0])], "worlds": 0}
                else:
                    certificate = certify(obs, solver)
                base["certificate"] = certificate
                bases.append(base)
            print(f"Certified {family}/{role}", flush=True)
    graded = [b for b in bases if b["certificate"]["status"] == "certified" and not b["control"]]
    if {b["family"] for b in graded} != {k for k, v in TEMPLATES.items() if not v.get("mechanical") and not v.get("control")}:
        raise ValueError("At least one tactical family has no certified coverage")
    coverage = defaultdict(lambda: {"certified": 0, "mechanical": 0, "ungraded": 0, "reasons": defaultdict(int)})
    for b in bases:
        status = b["certificate"]["status"]
        coverage[b["family"]][status] += 1
        if status == "ungraded":
            coverage[b["family"]]["reasons"][b["certificate"]["reason"]] += 1
    suite = {"schema_version": 1, "name": name, "created_at_utc": now(), "implementation_sha256": code_identity(),
             "environment": runtime("cpu"),
             "solver": solver_identity(solver), "seeds": list(seeds), "bases": bases,
             "suit_permutations": list(PERMUTATIONS), "seat_rotations": [0, 1, 2, 3],
             "coverage": dict(coverage), "purpose": "development skill assessment",
             "label_contract": "universal feasible-world DDS optimum AND one public continuation witness against all enemy legal play",
             "symmetry_contract": "complete suit bijection including trump, auction, history, hands and masks; scores measured separately from correctness",
             "limits": ["8 tactical templates plus follow-suit and multiple-answer controls; not all bridge knowledge",
                        "2-4 trick endings embedded in legal complete games; synthetic development variants",
                        "No common optimum or no public continuation witness => ungraded",
                        "Template, role, seed and permutation variants are correlated, not independent test deals",
                        "Basic skill percentages are separate from random=0/DDS=100 match ratings"]}
    root.mkdir(parents=True, exist_ok=False)
    write(root/"suite.json", suite)
    write(root/"manifest.json", {"suite_sha256": digest(suite)})
    return suite


def load_suite(root):
    root = Path(root)
    suite = read(root/"suite.json")
    if read(root/"manifest.json")["suite_sha256"] != digest(suite):
        raise ValueError("Frozen suite changed")
    if suite["implementation_sha256"] != code_identity():
        raise ValueError("Foundational evaluator changed; create a new suite version")
    if suite["environment"] != runtime("cpu"):
        raise ValueError("Foundational runtime changed; use the recorded environment or a new suite version")
    return suite


def install_suite(source, destination):
    """Install only frozen fixtures/proofs; do not import unshipped prediction caches."""
    suite = load_suite(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write(destination/"suite.json", suite)
    write(destination/"manifest.json", {"suite_sha256": digest(suite)})
    return {"suite": suite["name"], "destination": str(destination.resolve()), "DDS_required": False}


def variants(suite):
    for base in suite["bases"]:
        if base["certificate"]["status"] == "ungraded":
            continue
        observation = observe(base)
        for rotation in suite["seat_rotations"]:
            for permutation in suite["suit_permutations"]:
                obs = transform_observation(observation, permutation, rotation)
                # Policy-visible request IDs must not disclose the tactical family.
                obs["state_id"] = digest([base["id"], permutation, rotation])
                yield base, permutation, rotation, obs


class PublicPolicy:
    def __init__(self, recipe, seed=912):
        self.recipe, self.seed = recipe, seed

    def predict_batch(self, observations, **kwargs):
        out = []
        for obs in observations:
            probs = np.zeros(52)
            if self.recipe == "random":
                legal = obs["legal_card_ids"]
                probs[legal] = 1./len(legal)
                rng = np.random.default_rng(int(digest([self.seed, obs["state_id"]])[:16], 16))
                action = int(rng.choice(legal))
            else:
                action = public_action(obs, self.recipe)
                probs[action] = 1.
            out.append({"state_id": obs["state_id"], "card_id": action, "probabilities": probs.tolist()})
        return out


def evaluate_rows(suite, policy, *, batch_size=64):
    """Only public observations cross the policy boundary; labels remain outside."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive batch size required")
    rows, batch = [], []

    def flush():
        predictions = policy.predict_batch([v[3] for v in batch], return_beliefs=True)
        if len(predictions) != len(batch):
            raise ValueError("Incomplete policy batch")
        for (base, permutation, rotation, obs), prediction in zip(batch, predictions):
            if prediction.get("state_id") != obs["state_id"]:
                raise ValueError("Stale foundation policy result")
            action = prediction["card_id"]
            if type(action) is not int or not 0 <= action < 52:
                raise ValueError("Invalid action index")
            probs = np.asarray(prediction["probabilities"], dtype=float)
            if probs.shape != (52,) or not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1., atol=1e-5):
                raise ValueError("Expected a normalized finite 52-card policy distribution")
            mapping = card_map(permutation)
            canonical = int(np.flatnonzero(mapping == action)[0])
            aligned = probs[mapping]
            accepted = base["certificate"]["accepted_cards"]
            legal = action in obs["legal_card_ids"]
            row = {"base_id": base["id"], "family": base["family"], "role": base["role"],
                   "mechanical": base["mechanical"], "control": base["control"],
                   "permutation": permutation, "rotation": rotation, "chosen": card_name(action),
                   "canonical_chosen": card_name(canonical), "legal": legal,
                   "correct": legal and card_name(canonical) in accepted,
                   "accepted_canonical": accepted, "accepted_probability": float(sum(aligned[card_id(c)] for c in accepted)),
                   "canonical_probabilities": {card_name(c): float(v) for c, v in enumerate(aligned) if v > 0},
                   "worst_feasible_world_trick_regret": base["certificate"].get("worst_world_trick_regret", {}).get(card_name(canonical))}
            rows.append(row)
        batch.clear()

    for item in variants(suite):
        batch.append(item)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return rows


def percentage(rows):
    return 100.*sum(r["correct"] for r in rows)/len(rows) if rows else None


def macro(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    return float(np.mean([percentage(items) for items in grouped.values()])) if grouped else None


def summarize(rows):
    if not rows:
        raise ValueError("No evaluated cases")
    tactical = [r for r in rows if not r["mechanical"] and not r["control"]]
    family = {}
    for name in sorted({r["family"] for r in rows}):
        items = [r for r in rows if r["family"] == name]
        canonical = [r for r in items if r["permutation"] == "CDHS" and r["rotation"] == 0]
        family[name] = {"base_positions": len(canonical), "canonical_correct": sum(r["correct"] for r in canonical),
                        "canonical_percent": percentage(canonical), "transformed_correct": sum(r["correct"] for r in items),
                        "transformed_total": len(items), "transformed_percent": percentage(items),
                        "excluded_from_basic_macro": items[0]["mechanical"] or items[0]["control"]}
    groups = defaultdict(list)
    for row in tactical:
        groups[row["base_id"], row["rotation"]].append(row)
    tv, exact, all_correct = [], [], []
    for key, items in groups.items():
        reference = next(r for r in items if r["permutation"] == "CDHS")
        for item in items:
            if item["permutation"] == "CDHS":
                continue
            keys = set(reference["canonical_probabilities"]) | set(item["canonical_probabilities"])
            tv.append(.5*sum(abs(reference["canonical_probabilities"].get(k, 0)-item["canonical_probabilities"].get(k, 0)) for k in keys))
            exact.append(reference["canonical_chosen"] == item["canonical_chosen"])
        all_correct.append(all(r["correct"] for r in items))
    by_permutation = {p: macro([r for r in tactical if r["permutation"] == p]) for p in PERMUTATIONS}
    by_role = {role: macro([r for r in tactical if r["role"] == role]) for role in ROLES}
    seat_groups = defaultdict(list)
    for row in tactical:
        seat_groups[row["base_id"], row["permutation"]].append(row)
    seat_same = [len({r["canonical_chosen"] for r in group}) == 1 for group in seat_groups.values()]
    values = [v for v in by_permutation.values() if v is not None]
    return {"canonical_basic_percent": macro([r for r in tactical if r["permutation"] == "CDHS" and r["rotation"] == 0]),
            "all_suits_basic_percent": macro(tactical), "worst_suit_basic_percent": min(values),
            "best_suit_basic_percent": max(values), "suit_accuracy_gap_points": max(values)-min(values),
            "suit_distribution_equivariance_percent": 100.*(1-float(np.mean(tv))),
            "exact_suit_choice_agreement_percent": 100.*float(np.mean(exact)),
            "all_suits_pass_position_percent": 100.*float(np.mean(all_correct)),
            "seat_rotation_choice_consistency_percent": 100.*float(np.mean(seat_same)),
            "legal_choice_percent": 100.*sum(r["legal"] for r in rows)/len(rows),
            "evaluated_variants": len(rows), "tactical_base_positions": len({r["base_id"] for r in tactical}),
            "families": family, "by_role_basic_percent": by_role, "by_permutation_basic_percent": by_permutation,
            "interpretation": "Correctness and invariance are separate; consistent wrong answers do not imply basic competence. Equivalent accepted moves are never graded wrong.",
            "statistical_scope": "Correlated template/rank/role/permutation variants; no IID confidence interval or human-level rating"}


def evaluate(root, *, name, model=None, recipe="neural", device="cuda", batch_size=64):
    root = Path(root)
    identifier(name)
    suite = load_suite(root)
    if (recipe == "neural") != (model is not None):
        raise ValueError("Only neural assessment requires a checkpoint")
    if recipe not in ("neural", "rule-basic", "cash-public-master", "random"):
        raise ValueError("Unknown assessment policy")
    identity = {"suite_sha256": digest(suite), "implementation_sha256": code_identity(), "policy": recipe,
                "model_sha256": sha256(model) if model else None, "device": device if model else "cpu",
                "batch_size": batch_size, "torch": str(torch.__version__), "numpy": str(np.__version__)}
    report_path = root/"reports"/(name+".json")
    predictions_path = root/"predictions"/(name+".jsonl")
    if report_path.exists():
        report = read(report_path)
        if report["identity"] != identity or report["predictions_sha256"] != sha256(predictions_path):
            raise ValueError("Existing assessment differs or prediction cache changed; use a new report ID")
        return report
    policy = NeuralPlayPolicy(model, device=device) if model else PublicPolicy(recipe)
    rows = evaluate_rows(suite, policy, batch_size=batch_size)
    summary = summarize(rows)
    predictions_path.parent.mkdir(exist_ok=True)
    temp = predictions_path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    temp.replace(predictions_path)
    failed = [r for r in rows if not r["correct"] and not r["mechanical"]]
    failed.sort(key=lambda r: (-(r["worst_feasible_world_trick_regret"] or 0), r["accepted_probability"]))
    # Representative failures: one per template/role/seed, rather than 24 repeats.
    examples, used = [], set()
    bases = {b["id"]: b for b in suite["bases"]}
    for row in failed:
        if row["base_id"] in used:
            continue
        used.add(row["base_id"])
        case = transform_case(bases[row["base_id"]], row["permutation"], row["rotation"])
        case["certificate_transport"] = {"base_id": row["base_id"], "suit_permutation": row["permutation"], "seat_rotation": row["rotation"]}
        examples.append({"prediction": row, "case": case})
        if len(examples) == 24:
            break
    report = {"schema_version": 1, "id": name, "suite": suite["name"], "created_at_utc": now(), "identity": identity,
              "predictions_sha256": sha256(predictions_path), "summary": summary,
              "representative_failures": examples, "limits": suite["limits"]}
    write(report_path, report)
    return report


def export(root, output):
    root, output = Path(root), Path(output)
    suite = load_suite(root)
    write(output/"suite.json", suite)
    write(output/"manifest.json", read(root/"manifest.json"))
    reports = []
    for path in sorted((root/"reports").glob("*.json")):
        report = read(path)
        if report["identity"]["suite_sha256"] != digest(suite):
            raise ValueError("Assessment report belongs to another suite")
        if report["predictions_sha256"] != sha256(root/"predictions"/(report["id"]+".jsonl")):
            raise ValueError("Prediction records changed")
        write(output/"reports"/path.name, report)
        reports.append(report)
    lines = ["# 基础能力与花色一致性评估", "", "基础正确率与花色一致性分开显示；不对应随机0/DDS100的对战牌力分。", "",
             "| 策略 | 原始花色基础正确率 | 全花色基础正确率 | 最差置换 | 分布一致性 | 所有花色均答对的位置 |", "|---|---:|---:|---:|---:|---:|"]
    for report in reports:
        s = report["summary"]
        lines.append(f"| {report['id']} | {s['canonical_basic_percent']:.2f}% | {s['all_suits_basic_percent']:.2f}% | {s['worst_suit_basic_percent']:.2f}% | {s['suit_distribution_equivariance_percent']:.2f}% | {s['all_suits_pass_position_percent']:.2f}% |")
    lines.extend(["", "仅纳入已获得全部可行暗牌世界与可见续打策略证书的题；未认证题不判错。", "", "模板变体彼此相关，结果用于开发与回归；强制跟牌与多答案控制题不计入基础能力宏平均。"])
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"reports": len(reports), "suite_sha256": digest(suite)}
