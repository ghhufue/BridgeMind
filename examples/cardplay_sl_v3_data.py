"""Expand recorded play to 50k training deals without using any validation/test fold."""

import json
from pathlib import Path
import numpy as np

from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.cardplay.runner import fingerprint
from bridgemind.rating.system import read, write
from bridgemind.training.__main__ import ASSETS
from bridgemind.training.runner import sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/cardplay-sl-v3-50k/train"
ART = ROOT/"artifacts/cardplay-sl-v3-20261026"


def main():
    source = ROOT/"data/downloads/openspiel-train.txt"
    if sha256(source) != ASSETS[source.name][1]:
        raise ValueError("Original training source changed")
    allowed = set(np.load(ROOT/"data/supervised/train/board-ids.npy", allow_pickle=False))
    excluded = set()
    for split in ("valid", "test"):
        excluded.update(np.load(ROOT/f"data/supervised/{split}/board-ids.npy", allow_pickle=False))
    if allowed & excluded:
        raise ValueError("Source training and validation/test folds overlap")
    if not (DATA/"manifest.json").exists():
        print("Importing 50,000 training deals / 2,600,000 decisions; all recorded actions replay-validated", flush=True)
        prepare_dataset(source, DATA, allowed_ids=allowed, max_boards=50000, system_id=1,
                        system_description="WBridge5 SAYC; SL v3 50k training only, N dealer, neither vulnerable")
    data = PlayData(DATA)
    data.load_games()
    if len(data.ids) != 50000 or not set(data.ids).issubset(allowed) or set(data.ids) & excluded:
        raise ValueError("Expanded training ID audit failed")
    old = set(PlayData(ROOT/"data/cardplay-expanded-v1/train").ids)
    report = {"boards": 50000, "states": data.manifest["states"], "source_sha256": sha256(source),
              "dataset_fingerprint": fingerprint(DATA), "board_ids_sha256": sha256(DATA/"board-ids.npy"),
              "validation_test_id_intersection": 0, "retained_v2_training_deals": len(old & set(data.ids)),
              "source_pool": "data/supervised/train only; no independent evaluation fold used as training"}
    path = ART/"data-audit.json"
    if path.exists() and read(path) != report:
        raise ValueError("Existing 50k audit changed")
    write(path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
