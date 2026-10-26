"""Reserve distinct unseen search/training confirmation folds before selection."""

from pathlib import Path
import json
import numpy as np

from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.runner import fingerprint
from bridgemind.rating.system import read, write
from bridgemind.sl_v2.data import orbit_key
from bridgemind.training.__main__ import ASSETS
from bridgemind.training.runner import sha256
from cardplay_long_sl_eval import collect_report_ids

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/cardplay-sl-v3-20261026/reservation"
DATA = ROOT/"data/cardplay-sl-v3-reserved-v1"
SEARCH = ROOT/"data/cardplay-sl-v3-search-confirm-v1"
CONFIRM = ROOT/"data/cardplay-sl-v3-confirm-v1"


def main():
    source = ROOT/"data/downloads/openspiel-test.txt"
    if sha256(source) != ASSETS[source.name][1]:
        raise ValueError("Original test source changed")
    snapshot = ROOT/"data/cardplay-sl-v3-excluded.npy"
    audit_path = ART/"exclusion-audit.json"
    if audit_path.exists():
        audit = read(audit_path)
        if audit["excluded_ids_sha256"] != sha256(snapshot):
            raise ValueError("Exclusion snapshot changed")
        allowed = {s.encode() for s in audit["eligible_board_ids"]}
    else:
        excluded, sources, orbits = set(), {}, set()
        for path in sorted(set((ROOT/"data").rglob("*board*ids*.npy")) | set((ROOT/"data").rglob("*excluded*.npy"))):
            if path == ROOT/"data/supervised/test/board-ids.npy" or path in (snapshot,) or any(path.is_relative_to(p) for p in (DATA, SEARCH, CONFIRM)):
                continue
            ids = set(np.load(path, allow_pickle=False)); excluded.update(ids)
            sources[str(path.relative_to(ROOT))] = {"ids": len(ids), "sha256": sha256(path)}
        for path in sorted((ROOT/"data/boards").glob("*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                ids = set(arrays["ids"])
            excluded.update(ids)
        for directory in (ROOT/"runs", ROOT/"artifacts"):
            for path in directory.rglob("*.json"):
                if path.is_relative_to(ART):
                    continue
                value = read(path)
                excluded.update(collect_report_ids(value))
                if path.name == "boards.json" and isinstance(value, list):
                    excluded.update(b["id"].encode() for b in value)
                    orbits.update(orbit_key(b["owners"]) for b in value)
        for path in (ROOT/"data").rglob("games.jsonl"):
            if any(path.is_relative_to(p) for p in (DATA, SEARCH, CONFIRM)):
                continue
            for line in path.read_text().splitlines():
                orbits.add(orbit_key(json.loads(line)["owners"]))
        for path in (ROOT/"runs/play-sl-v2-20261026-2h/curriculum.json", ROOT/"runs/sl-v3-curriculum-20261026/curriculum.json"):
            if path.exists():
                orbits.update(orbit_key(c["board"]["owners"]) for split in read(path).values() for c in split)
        allowed = set(np.load(ROOT/"data/supervised/test/board-ids.npy", allow_pickle=False))-excluded
        from bridgemind.training.data import board_ids
        eligible, rejected = set(), 0
        with source.open() as stream:
            for line in stream:
                actions = np.fromstring(line, sep=" ", dtype=np.int64)
                owners = np.empty(52, np.uint8); owners[actions[:52]] = np.arange(52) % 4
                identity = board_ids(owners[None])[0]
                if identity in allowed:
                    if orbit_key(owners) in orbits:
                        rejected += 1
                    else:
                        eligible.add(identity)
        if len(eligible) < 1152:
            raise ValueError("Fewer than 1152 unused deals remain")
        np.save(snapshot, np.asarray(sorted(excluded), dtype="S64"), allow_pickle=False)
        audit = {"sources": sources, "excluded_ids_sha256": sha256(snapshot), "excluded_ids": len(excluded),
                 "banned_orbits": len(orbits), "orbit_rejected": rejected, "eligible_count": len(eligible),
                 "eligible_board_ids": sorted(x.decode() for x in eligible), "source_sha256": sha256(source)}
        write(audit_path, audit); allowed = eligible
    if not (DATA/"manifest.json").exists():
        prepare_dataset(source, DATA, allowed_ids=allowed, max_boards=1152, system_id=1,
                        system_description="SL v3 reserved unseen deals before search/pilot/model selection")
    games = PlayData(DATA).load_games()
    if len(games) != 1152 or not {g["id"].encode() for g in games}.issubset(allowed):
        raise ValueError("Reserved pool differs from frozen audit")
    for path, subset in ((SEARCH, games[:128]), (CONFIRM, games[128:])):
        selected = {g["id"].encode() for g in subset}
        if not (path/"manifest.json").exists():
            prepare_dataset(source, path, allowed_ids=selected, max_boards=len(selected), system_id=1,
                            system_description="SL v3 independent confirmation; reserved before selection")
        if set(PlayData(path).ids) != selected:
            raise ValueError("Confirmation split changed")
    a, b = PlayData(SEARCH), PlayData(CONFIRM)
    if set(a.ids) & set(b.ids):
        raise ValueError("Search and SL confirmation overlap")
    report = {"search": {"path": str(SEARCH), "boards": 128, "fingerprint": fingerprint(SEARCH)},
              "sl": {"path": str(CONFIRM), "boards": 1024, "fingerprint": fingerprint(CONFIRM)},
              "between_splits_intersection": 0, "historical_ids_and_orbits_intersection": 0,
              "exclusion_audit_sha256": sha256(audit_path), "remaining_unused_pool_after_reservation": len(allowed)-1152}
    target = ART/"report.json"
    if target.exists() and read(target) != report:
        raise ValueError("Reservation report changed")
    write(target, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
