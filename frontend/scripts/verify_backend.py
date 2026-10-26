"""Cross-check the browser's bridge rules against the existing OpenSpiel backend.

Run from the repository root with .venv/Scripts/python.exe frontend/scripts/verify_backend.py.
No DDS process, GPU, models, or inference is needed.
"""
import json
import shutil
import subprocess
from pathlib import Path

from bridgemind.engine.bidding import BridgeBiddingEnv
from bridgemind.engine.play import BridgePlayEnv, card_name
from bridgemind.engine.scoring import Contract, score_contract


def main():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [shutil.which("node") or "node", "scripts/export-crosscheck.mjs"],
        cwd=root / "frontend", check=True, capture_output=True, encoding="utf-8",
    )
    data = json.loads(completed.stdout)
    auction_steps = play_steps = 0
    doubles = redoubles = passed_out = 0
    for board in data["boards"]:
        initial = board["initial"]
        deal = {"hands": initial["initialHands"]}
        bidding = BridgeBiddingEnv(None, deal, dealer=initial["dealer"], vulnerable=initial["vulnerable"], prefetch=False)
        for entry in board["auction"]:
            assert bidding.observation()["player"] == entry["turn"]
            assert set(bidding.legal_calls()) == set(entry["legal"]), entry
            bidding.apply_call(entry["call"])
            doubles += entry["call"] == "X"
            redoubles += entry["call"] == "XX"
            auction_steps += 1
        assert bidding.is_terminal()
        if board["contract"] is None:
            assert bidding._contract is None
            assert board["result"]["nsScore"] == 0
            passed_out += 1
            continue
        contract = board["contract"]
        assert bidding._contract == Contract(contract["level"], contract["strain"], contract["declarer"], contract["doubled"])
        env = BridgePlayEnv(deal, dealer=initial["dealer"], vulnerable=initial["vulnerable"], auction=[entry["call"] for entry in board["auction"]])
        for index, entry in enumerate(board["play"]):
            assert env.to_play == entry["toPlay"]
            assert env.controller == entry["controller"]
            assert {card_name(i) for i in env.legal_card_ids()} == set(entry["legal"])
            env.apply_card(entry["card"])
            if index % 4 == 3:
                assert env._leader == board["tricks"][index // 4]["winner"]
            play_steps += 1
        assert env.is_terminal()
        assert env.result()["returns"][0] == board["result"]["nsScore"]
    for case in data["scores"]:
        c = case["contract"]
        expected = score_contract(Contract(c["level"], c["strain"], c["declarer"], c["doubled"]), case["tricks"], case["vulnerable"])
        assert expected == case["score"], case
    report = {
        "date": "2026-10-26", "reference": "Existing BridgeMind Python / OpenSpiel 2.0.2 backend",
        "boards": len(data["boards"]), "passed_out": passed_out, "auction_decisions": auction_steps,
        "doubles": doubles, "redoubles": redoubles, "card_decisions": play_steps,
        "score_cases": len(data["scores"]), "status": "passed",
    }
    output = root / "artifacts/frontend/validation-20261026.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
