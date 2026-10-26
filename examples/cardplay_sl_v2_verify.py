"""Recompute completed SL v2 evaluation results from per-deal and per-case records."""

from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from bridgemind.cardplay.data import PlayData
from bridgemind.cardplay.runner import fingerprint
from bridgemind.engine.scoring import score_contract
from bridgemind.dds.models import SEATS
from bridgemind.rating.system import RatingSystem, digest, implementation, now, read, write, writer_lock
from bridgemind.sl_v2.data import orbit_key
from bridgemind.training.rollout import imps
from bridgemind.training.runner import sha256


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/cardplay-sl-v2-20261026"
NAME = "sl-v2-20261026"


def verify_foundation():
    root = ROOT/"runs/cardplay-foundations-v1"
    report = read(root/"reports"/(NAME+".json"))
    prediction = root/"predictions"/(NAME+".jsonl")
    suite = read(root/"suite.json")
    assert sha256(prediction) == report["predictions_sha256"]
    assert digest(suite) == report["identity"]["suite_sha256"]
    bases = {b["id"]: b for b in suite["bases"]}
    rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    coordinates, groups, by_family = set(), defaultdict(list), defaultdict(list)
    for row in rows:
        key = (row["base_id"], row["permutation"], row["rotation"])
        assert key not in coordinates
        coordinates.add(key)
        accepted = bases[row["base_id"]]["certificate"]["accepted_cards"]
        assert set(accepted) == set(row["accepted_canonical"])
        correct = row["legal"] and row["canonical_chosen"] in accepted
        assert row["correct"] == correct
        assert abs(sum(row["canonical_probabilities"].values())-1) < 1e-5
        if not row["mechanical"] and not row["control"]:
            groups[row["base_id"], row["rotation"]].append(row)
            by_family[row["family"]].append(correct)
    assert len(rows) == 8064
    television = []
    for items in groups.values():
        assert len(items) == 24
        reference = next(r for r in items if r["permutation"] == "CDHS")["canonical_probabilities"]
        for row in items:
            if row["permutation"] != "CDHS":
                other = row["canonical_probabilities"]
                television.append(.5*sum(abs(reference.get(k, 0)-other.get(k, 0)) for k in reference.keys() | other.keys()))
    accuracy = 100*np.mean([np.mean(v) for v in by_family.values()])
    equivariance = 100*(1-np.mean(television))
    assert np.isclose(accuracy, report["summary"]["all_suits_basic_percent"])
    assert np.isclose(equivariance, report["summary"]["suit_distribution_equivariance_percent"])
    shutil.copyfile(root/"reports"/(NAME+".json"), ART/"foundation-report.json")
    return {"variants": len(rows), "basic_macro_percent": float(accuracy), "distribution_equivariance_percent": float(equivariance),
            "prediction_sha256": sha256(prediction), "independently_recomputed": True}


def verify_confirmation():
    art = ART/"confirmation"
    protocol, report = read(art/"protocol.json"), read(art/"report.json")
    assert protocol["script_sha256"] == sha256(ROOT/"examples/cardplay_sl_v2_eval.py")
    assert protocol["implementation_sha256"] == implementation()
    data = PlayData(ROOT/"data/cardplay-sl-v2-confirmation-v1")
    assert fingerprint(data.directory) == report["data_audit"]["fingerprint"]
    audit = read(art/"exclusion-audit.json")
    assert sha256(art/"exclusion-audit.json") == report["data_audit"]["exclusion_audit_sha256"]
    excluded = np.load(ROOT/"data/cardplay-sl-v2-confirmation-excluded.npy", allow_pickle=False)
    assert not set(data.ids).intersection(excluded)
    courses = read(ROOT/"runs/play-sl-v2-20261026-2h/curriculum.json")
    used_orbits = {orbit_key(c["board"]["owners"]) for cases in courses.values() for c in cases}
    confirm_orbits = {orbit_key(g["owners"]) for g in data.load_games()}
    assert not used_orbits.intersection(confirm_orbits) and len(confirm_orbits) == 2048
    indices = {g["id"]: i for i, g in enumerate(data.load_games())}
    contracts = {identity: data.environment(i).contract for identity, i in indices.items()}
    details = {}
    for opponent in protocol["primary_opponents"]:
        result = read(art/f"vs-{opponent}.json")
        assert result["protocol_sha256"] == digest(protocol)
        assert result["model_sha256"] == protocol["candidate"]["sha256"]
        assert len(result["board_ids"]) == 2048 and set(result["board_ids"]) == set(indices)
        for identity, own, other, raw, imp in zip(result["board_ids"], result["current_as_declarer_tricks"],
                result["opponent_as_declarer_tricks"], result["raw_differences"], result["imps"], strict=True):
            contract = contracts[identity]
            game = data.load_games()[indices[identity]]
            vulnerable = bool(game["vulnerable"][SEATS.index(contract.declarer)%2])
            difference = score_contract(contract, int(own), vulnerable)-score_contract(contract, int(other), vulnerable)
            assert raw == difference and imp == imps(difference)
        values = np.asarray(result["imps"])
        rng = np.random.default_rng(protocol["bootstrap"]["seed"])
        draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(10000)]
        ci = np.quantile(draws, [.0125, .9875])
        stat = report["results"][opponent]
        assert np.isclose(values.mean(), stat["mean_imps"]) and np.allclose(ci, stat["primary_97_5_ci"])
        assert (stat["wins"], stat["ties"], stat["losses"]) == (int((values>0).sum()), int((values==0).sum()), int((values<0).sum()))
        details[opponent] = {"verified_raw_scores_and_imps": len(values), "mean_imps": float(values.mean()), "primary_97_5_ci": ci.tolist()}
    return details


def verify_rating():
    with writer_lock(ROOT/"runs/cardplay-rating-v1"):
        system = RatingSystem(ROOT/"runs/cardplay-rating-v1", port=18766)
        panels = {}
        for name in (NAME, "sl-long-42", "random", "oracle"):
            pieces = []
            for item in system.protocol["panel"]:
                path = system.root/"matches"/("--".join(sorted((name, item["id"])))+".json")
                assert len(read(path)["records"]) == 384
                values, _ = system.match(name, item["id"])
                pieces.append(values*item["weight"])
            panels[name] = np.sum(pieces, axis=0)
        random, oracle = panels["random"], panels["oracle"]
        score = 100*(panels[NAME].mean()-random.mean())/(oracle.mean()-random.mean())
        reported = read(system.root/"reports"/(NAME+".json"))
        assert np.isclose(score, reported["score"]["rating"])
        rng, draws = np.random.default_rng(20261026), []
        for _ in range(2000):
            coords = np.ix_(rng.integers(3, size=3), rng.integers(128, size=128))
            draws.append(100*(panels[NAME][coords]-panels["sl-long-42"][coords]).mean()/(oracle[coords]-random[coords]).mean())
        gain = 100*(panels[NAME]-panels["sl-long-42"]).mean()/(oracle-random).mean()
        shutil.copyfile(system.root/"reports"/(NAME+".json"), ART/"rating-report.json")
        system.export(ROOT/"artifacts/cardplay-rating-v1")
    return {"rating": float(score), "gain_over_sl_long_42": float(gain), "gain_joint_bootstrap_95_ci": np.quantile(draws, [.025, .975]).tolist(),
            "scope": "tracking panel; shared seed rows and deal columns; not an independent final test"}


def main():
    torch.set_num_threads(1)
    completed = read(ROOT/"runs/play-sl-v2-20261026-2h/report.json")
    assert sha256(ROOT/"runs/play-sl-v2-20261026-2h/best.pt") == completed["best_model_sha256"]
    output = {"verified_at_utc": now(), "model_sha256": completed["best_model_sha256"],
              "foundation": verify_foundation(), "confirmation": verify_confirmation(), "rating": verify_rating()}
    write(ART/"evaluation-verification.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
