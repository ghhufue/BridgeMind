"""Replay neural/rule matches, annotate DDS losses, and build legal tactical tests.

python -m bridgemind.diagnostics.cardplay --output artifacts/cardplay-diagnostics-v1
DDS annotations are privileged diagnostics, never policy inputs or new ratings.
"""

import argparse
import copy
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from bridgemind.cardplay.policy import NeuralPlayPolicy
from bridgemind.dds import SolverClient, Position, Deal, Options
from bridgemind.dds.models import SEATS, CARDS
from bridgemind.engine import BridgePlayEnv, Contract
from bridgemind.engine.play import card_id, card_name
from bridgemind.engine.scoring import score_contract
from bridgemind.rating.policies import RulePolicy, oracle_position, strength
from bridgemind.rating.system import RatingSystem, digest, read, write, solver_identity, now
from bridgemind.training.data import ownership, board_ids
from bridgemind.training.rollout import imps
from bridgemind.training.runner import sha256


SELECTED = (34, 41, 37, 51, 1, 79)
SELECTION_REASON = ("largest recorded loss", "large notrump loss", "typical 1NT loss",
                    "typical suit-contract loss", "tie control", "winning control")


def make_environment(board):
    kwargs = {"dealer": board["dealer"], "vulnerable": tuple(board["vulnerable"]), "system_id": board["system_id"]}
    from bridgemind.training.data import deal_from_ownership
    deal = deal_from_ownership(board["owners"])
    if board.get("auction_source") == "synthetic_contract":
        return BridgePlayEnv(deal, contract=Contract(**board["contract"]), **kwargs)
    return BridgePlayEnv(deal, auction=board["auction"], **kwargs)


def visible_facts(obs, action):
    """Behavioral facts only: ducking/overtaking can be strategically correct."""
    trick = obs["current_trick"]
    if not trick:
        return ["opening_lead" if not obs["play_history"] else "lead"]
    led, trump = trick[0]["card"][0], obs["contract"]["strain"]
    winner = max(trick, key=lambda e: strength(card_id(e["card"]), led, trump))
    current = strength(card_id(winner["card"]), led, trump)
    partner = SEATS.index(winner["seat"]) % 2 == SEATS.index(obs["to_play"]) % 2
    wins = [c for c in obs["legal_card_ids"] if strength(c, led, trump) > current]
    tags = []
    if partner and action in wins:
        tags.append("overtakes_currently_winning_partner")
    if not partner and wins and action not in wins:
        tags.append("declines_available_current_trick")
    if not partner and action in wins and any(strength(c, led, trump) < strength(action, led, trump) for c in wins):
        tags.append("spends_higher_current_winner")
    if len(trick) == 3:
        tags.append("fourth_hand")
    return tags


def annotate(solver, board, obs, actions):
    if len(obs["legal_card_ids"]) == 1:
        return {"forced": True}
    result = solver.solve_position(oracle_position(board, obs), Options(state_id=obs["state_id"]))
    moves = {card_id(m.card): m.tricks_for_evaluated_side for m in result.moves}
    if set(moves) != set(obs["legal_card_ids"]) or result.to_play != obs["to_play"]:
        raise ValueError("Diagnostic DDS position mismatch")
    contract = Contract(**obs["contract"])
    declarer_side, acting_side = SEATS.index(contract.declarer) % 2, SEATS.index(obs["to_play"]) % 2
    remaining = 13-sum(obs["tricks"])

    def value(tricks):
        total = obs["tricks"][declarer_side]+(tricks if acting_side == declarer_side else remaining-tricks)
        score = score_contract(contract, total, obs["vulnerable"][declarer_side])
        return total, score if acting_side == declarer_side else -score

    best_total, best_score = value(result.best_tricks)
    values = {}
    for name, card in actions.items():
        total, score = value(moves[card])
        values[name] = {"card": card_name(card), "future_team_tricks": moves[card],
                        "trick_regret": result.best_tricks-moves[card], "team_score_regret": best_score-score,
                        "dds_declarer_total": total,
                        "crosses_contract_threshold": (total >= contract.level+6) != (best_total >= contract.level+6)}
    return {"forced": False, "best_cards": sorted(result.best_cards, key=card_id), "best_team_tricks": result.best_tricks,
            "best_declarer_total": best_total, "actions": values,
            "scope": "true hidden deal, perfect continuation; not an information-limited error certificate"}


def play_room(board, room, neural, rule, solver=None, *, original=None, patch=None):
    env = make_environment(board)
    trace = []
    prefix = original or []
    for ply in range(52):
        obs = env.observation()
        model_turn = SEATS.index(obs["player"]) % 2 == room
        if patch is not None:
            action = prefix[ply]["chosen_id"] if ply < patch["ply"] else (
                patch["card_id"] if ply == patch["ply"] else (neural if model_turn else rule).predict(obs)["card_id"])
        else:
            estimate = neural.predict_batch([obs], return_beliefs=True)[0]
            model_action, rule_action = estimate["card_id"], rule.predict(obs)["card_id"]
            action = model_action if model_turn else rule_action
            legal = obs["legal_card_ids"]
            probabilities = estimate["probabilities"]
            top = sorted(legal, key=lambda c: -probabilities[c])[:3]
            entry = {"ply": ply, "trick": ply//4+1, "to_play": obs["to_play"], "controller": obs["player"],
                     "actor": "model" if model_turn else "rule", "role": "declarer" if obs["player"] == obs["contract"]["declarer"] else "defender",
                     "current_trick": obs["current_trick"], "visible_hands": obs["visible_hands"],
                     "remaining_counts": obs["remaining_counts"], "tricks_before": obs["tricks"],
                     "legal": [card_name(c) for c in legal], "model_card": card_name(model_action),
                     "rule_card": card_name(rule_action), "chosen": card_name(action), "chosen_id": action,
                     "model_top3": [{"card": card_name(c), "probability": probabilities[c]} for c in top],
                     "behavior": visible_facts(obs, action),
                     "dds": annotate(solver, board, obs, {"model": model_action, "rule": rule_action, "chosen": action})}
            trace.append(entry)
        env.apply_card(action)
    if not env.is_terminal():
        raise ValueError("Diagnostic game did not finish")
    return {"model_side": "NS" if room == 0 else "EW", "contract": env.contract.to_dict(), "result": env.result(), "trace": trace}


def analyze_board(board, neural, solver, expected, index):
    rule = RulePolicy()
    rooms = [play_room(board, i, neural, rule, solver) for i in (0, 1)]
    scores = [r["result"]["returns"][0] for r in rooms]
    baseline = imps(scores[0]-scores[1])
    if baseline != -expected["imps"] or scores[0]-scores[1] != -expected["raw_difference"]:
        raise ValueError("Fresh replay does not reproduce the frozen rating match")
    interventions = []
    for room_id, room in enumerate(rooms):
        errors = [e for e in room["trace"] if e["actor"] == "model" and not e["dds"]["forced"] and
                  e["dds"]["actions"]["model"]["trick_regret"] > 0]
        # Select at most two states, before seeing their counterfactual outcomes.
        selected = sorted(errors, key=lambda e: (-e["dds"]["actions"]["model"]["team_score_regret"], e["ply"]))[:2]
        for event in selected:
            alternatives = {"dds": event["dds"]["best_cards"][0]}
            if event["rule_card"] != event["model_card"]:
                alternatives["rule"] = event["rule_card"]
            seen = set()
            for source, card in alternatives.items():
                if card in seen:
                    continue
                seen.add(card)
                patched = play_room(board, room_id, neural, rule, original=room["trace"],
                                    patch={"ply": event["ply"], "card_id": card_id(card)})
                new_scores = list(scores)
                new_scores[room_id] = patched["result"]["returns"][0]
                interventions.append({"room": room_id, "ply": event["ply"], "role": event["role"],
                                      "original_card": event["chosen"], "alternative": card, "source": source,
                                      "behavior": event["behavior"], "dds_original_regret": event["dds"]["actions"]["model"],
                                      "original_declarer_tricks": room["result"]["declarer_tricks"],
                                      "patched_declarer_tricks": patched["result"]["declarer_tricks"],
                                      "model_team_score_gain": (new_scores[room_id]-scores[room_id])*(1 if room_id == 0 else -1),
                                      "paired_imps_before": baseline, "paired_imps_after": imps(new_scores[0]-new_scores[1]),
                                      "scope": "one move changed; all later decisions recomputed by original information-limited policies"})
    return {"index": index, "board_id": board["id"], "board": board, "model_imps": baseline,
            "rooms": rooms, "single_move_interventions": interventions}


TEMPLATES = {
    "win_now_keep_lead": {"hands": {"N": ["H2", "CA"], "E": ["HK", "D3"], "S": ["HA", "HQ"], "W": ["H3", "S4"]},
                          "leader": "N", "current": ["H2", "HK"], "level": 1, "good": "HA", "bad": "HQ"},
    "fourth_hand_take_trick": {"hands": {"N": ["H8", "CA", "C2"], "E": ["HK", "CQ", "D2"],
                                          "S": ["HA", "HQ", "C3"], "W": ["H3", "CK", "S2"]},
                               "leader": "W", "current": ["H3", "H8", "HK"], "level": 2, "good": "HA", "bad": "HQ"},
    "preserve_partner_entry": {"hands": {"N": ["HK", "HJ", "CA", "CQ"], "E": ["H3", "H4", "C2", "D2"],
                                           "S": ["HA", "H2", "CK", "C3"], "W": ["HQ", "HT", "H9", "S2"]},
                               "leader": "N", "current": ["HK", "H3"], "level": 3, "good": "H2", "bad": "HA"},
    "cheapest_winning_honor": {"hands": {"N": ["H3", "H4", "CA", "CQ"], "E": ["HJ", "HK", "C2", "D2"],
                                            "S": ["HA", "HQ", "CK", "C3"], "W": ["H2", "H9", "HT", "S2"]},
                                "leader": "W", "current": ["H9", "H3", "HJ"], "level": 2, "good": "HQ", "bad": "HA"},
}


def construct_case(family, suit_rotation=0, history_seed=1):
    """Embed a known small ending in 52 unique cards and a legal public prefix.

    Remaining suit counts are 1 mod 4, permitting whole-suit filler tricks.
    All filler tricks are explicitly legal; winners alternate before the final
    required leader. No public/private metadata is fabricated as recorded play.
    """
    template = TEMPLATES[family]
    suits = "CDHS"
    def translate(card):
        return suits[(suits.index(card[0])+suit_rotation) % 4]+card[1]
    reserved = {s: [translate(c) for c in cards] for s, cards in template["hands"].items()}
    used = {c for cards in reserved.values() for c in cards}
    groups = []
    # Preserve the complete prefix under suit permutation, not just the ending.
    for suit in (suits[(i+suit_rotation) % 4] for i in range(4)):
        available = [c for c in CARDS if c[0] == suit and c not in used]
        if len(available) % 4:
            raise ValueError("Tactical template cannot be embedded in whole-suit filler tricks")
        groups.extend(available[i:i+4] for i in range(0, len(available), 4))
    rng = np.random.default_rng(history_seed)
    rng.shuffle(groups)
    hands = copy.deepcopy(reserved)
    planned = []
    for i, group in enumerate(groups):
        winner = template["leader"] if i == len(groups)-1 else SEATS[i % 4]
        seats = [s for s in SEATS if s != winner]+[winner]
        assigned = dict(zip(seats, sorted(group, key=card_id)))
        for seat, card in assigned.items():
            hands[seat].append(card)
        planned.append(assigned)
    deal = Deal(hands)
    contract = Contract(template["level"], "NT", "S")
    env = BridgePlayEnv(deal, contract=contract)
    prefix = []
    for assigned in planned:
        for _ in range(4):
            card = assigned[env.to_play]
            env.apply_card(card); prefix.append(card)
    assert env.to_play == template["leader"]
    for card in template["current"]:
        card = translate(card)
        env.apply_card(card); prefix.append(card)
    owners = ownership(deal)
    board = {"id": board_ids(owners[None])[0].decode(), "owners": owners.tolist(), "contract": contract.to_dict(),
             "auction": env.auction, "auction_source": "synthetic_contract", "system_id": 0,
             "dealer": "N", "vulnerable": [False, False]}
    return {"case_id": f"{family}-s{suit_rotation}-h{history_seed}", "family": family, "board": board,
            "prefix": prefix, "suggested_good": translate(template["good"]), "suggested_bad": translate(template["bad"]),
            "scope": "constructed diagnostic, not an independent strength estimate or training data"}


def hidden_worlds(obs):
    """Enumerate all deals consistent with visible cards, capacities and voids.

    After dummy is exposed there are two hidden seats. Public auction and
    behavioral inference are intentionally not used to discard worlds.
    """
    visible = obs["visible_hands"]
    seats = [s for s in SEATS if s not in visible]
    if len(seats) != 2:
        raise ValueError("Exhaustive ending audit requires two hidden seats")
    unavailable = {c for cards in visible.values() for c in cards} | {e["card"] for e in obs["play_history"]}
    unknown = sorted(set(CARDS)-unavailable)
    if len(unknown) > 8:
        raise ValueError("Too many hidden cards for exhaustive diagnostic")
    for first in combinations(unknown, obs["remaining_counts"][seats[0]]):
        assigned = {seats[0]: list(first), seats[1]: sorted(set(unknown)-set(first))}
        if any(c[0] in obs["voids"][seat] for seat, cards in assigned.items() for c in cards):
            continue
        hands = {**visible, **assigned}
        yield Position(hands, obs["contract"]["strain"], obs["trick_leader"], [e["card"] for e in obs["current_trick"]])


def evaluate_case(case, neural, solver):
    env = make_environment(case["board"])
    for card in case["prefix"]:
        env.apply_card(card)
    obs = env.observation()
    estimate = neural.predict_batch([obs], return_beliefs=True)[0]
    chosen, rule = estimate["card_id"], RulePolicy().predict(obs)["card_id"]
    annotation = annotate(solver, case["board"], obs, {"model": chosen, "rule": rule})
    good, bad = card_id(case["suggested_good"]), card_id(case["suggested_bad"])
    margins = []
    for position in hidden_worlds(obs):
        result = solver.solve_position(position)
        values = {card_id(m.card): m.tricks_for_evaluated_side for m in result.moves}
        margins.append(values[good]-values[bad])
    if not margins:
        raise ValueError("No feasible hidden worlds")
    return {**case, "observation": {k: v for k, v in obs.items() if k != "state_id"},
            "model_card": card_name(chosen), "rule_card": card_name(rule),
            "model_choice_probability": estimate["probabilities"][chosen], "behavior": visible_facts(obs, chosen),
            "dds": annotation, "model_dds_regret": annotation["actions"]["model"]["trick_regret"],
            "hidden_world_audit": {"worlds": len(margins), "minimum_good_minus_bad_dd_tricks": min(margins),
                                   "maximum_good_minus_bad_dd_tricks": max(margins),
                                   "good_never_worse": min(margins) >= 0,
                                   "good_strictly_better_in_all_worlds": min(margins) > 0,
                                   "scope": "full-information continuation in every public-constraint-compatible world; not an imperfect-information optimality proof"}}


def run(args):
    torch.set_num_threads(1)
    system = RatingSystem(args.rating_root, port=args.port)
    spec = system.verify_entry("sl-long-42")
    neural = NeuralPlayPolicy(spec["checkpoint"], device=args.device)
    output = Path(args.output)
    with SolverClient(port=args.port) as solver:
        identity = solver_identity(solver)
        if identity != system.protocol["solver"]:
            raise ValueError("DDS differs from frozen rating")
        protocol = {"schema_version": 1, "created_at_utc": now(), "selected_indices": list(SELECTED),
                    "selection_reasons": list(SELECTION_REASON), "model_sha256": spec["checkpoint_sha256"],
                    "diagnostic_implementation_sha256": sha256(__file__), "rating_protocol_sha256": system.protocol_hash,
                    "solver": identity, "device": args.device, "seed": system.protocol["seeds"][0],
                    "selection_scope": "four selected losses, one tie and one win; exploratory, not representative sampling"}
        if (output/"protocol.json").exists():
            previous = read(output/"protocol.json")
            if {k: v for k, v in previous.items() if k != "created_at_utc"} != {k: v for k, v in protocol.items() if k != "created_at_utc"}:
                raise ValueError("Diagnostic output belongs to a different frozen run")
            protocol = previous
        else:
            write(output/"protocol.json", protocol)
        match = read(system.root/"matches/rule-basic--sl-long-42.json")
        by_id = {r["board_id"]: r for r in match["records"] if r["seed"] == protocol["seed"]}
        boards = []
        for index in SELECTED:
            path = output/"games"/f"board-{index:03d}.json"
            if path.exists():
                result = read(path)
            else:
                result = analyze_board(system.boards[index], neural, solver, by_id[system.boards[index]["id"]], index)
                write(path, result)
            boards.append(result)
            print(f"Board {index}: reproduced {result['model_imps']} IMP; {len(result['single_move_interventions'])} single-move interventions", flush=True)
        case_path = output/"tactical-cases.json"
        if case_path.exists():
            cases = read(case_path)
        else:
            cases = []
            for family in TEMPLATES:
                for rotation in range(4):
                    for history in (1, 2):
                        cases.append(evaluate_case(construct_case(family, rotation, history), neural, solver))
            write(case_path, cases)
        events = [e for b in boards for r in b["rooms"] for e in r["trace"] if not e["dds"]["forced"]]
        summary = {"boards": [{"index": b["index"], "board_id": b["board_id"], "contract": b["rooms"][0]["contract"],
                               "model_imps": b["model_imps"], "single_move_interventions": b["single_move_interventions"]} for b in boards],
                   "nonforced_diagnostics": {}, "tactical_families": {},
                   "limits": [protocol["selection_scope"], "DDS regret can depend on hidden information and is not automatically a policy error",
                              "Single-move patches use original policies for subsequent actions; outcomes conditional on those policies",
                              "24 tactical cases are variants of three templates, not 24 independent real deals",
                              "No training, weight changes or changes to the frozen rating benchmark"]}
        for actor in ("model", "rule"):
            items = [e for e in events if e["actor"] == actor]
            summary["nonforced_diagnostics"][actor] = {"decisions": len(items),
                "dds_positive_regret": sum(e["dds"]["actions"]["chosen"]["trick_regret"] > 0 for e in items),
                "sum_dds_trick_regret": sum(e["dds"]["actions"]["chosen"]["trick_regret"] for e in items),
                "behavior_counts": dict(Counter(tag for e in items for tag in e["behavior"]))}
            phases = {}
            for phase in ("fourth_hand", "lead", "second_third"):
                selected = [e for e in items if (not e["current_trick"] if phase == "lead" else
                            len(e["current_trick"]) == 3 if phase == "fourth_hand" else len(e["current_trick"]) in (1, 2))]
                phases[phase] = {"decisions": len(selected), "positive_dds_regret":
                                 sum(e["dds"]["actions"]["chosen"]["trick_regret"] > 0 for e in selected)}
            summary["nonforced_diagnostics"][actor]["by_phase"] = phases
        for family in TEMPLATES:
            items = [c for c in cases if c["family"] == family]
            summary["tactical_families"][family] = {"variants": len(items), "model_dds_optimal": sum(c["model_dds_regret"] == 0 for c in items),
                "rule_dds_optimal": sum(c["dds"]["actions"]["rule"]["trick_regret"] == 0 for c in items),
                "strict_public_world_dominance_cases": sum(c["hidden_world_audit"]["good_strictly_better_in_all_worlds"] for c in items)}
        summary["limits"][3] = f"{len(cases)} tactical variants of {len(TEMPLATES)} templates, not independent real deals"
        natural = []
        for board_index, room_index, ply in ((34, 0, 15), (34, 0, 19), (41, 0, 7), (37, 1, 23), (51, 0, 35), (51, 0, 12)):
            result = next(b for b in boards if b["index"] == board_index)
            trace = result["rooms"][room_index]["trace"]
            event = trace[ply]
            natural.append({"case_id": f"real-{board_index}-room{room_index}-ply{ply}", "board": result["board"],
                            "prefix": [e["chosen"] for e in trace[:ply]], "model_card": event["model_card"],
                            "rule_card": event["rule_card"], "actual_deal_dds_best": event["dds"]["best_cards"],
                            "model_top3": event["model_top3"], "behavior": event["behavior"],
                            "single_move_interventions": [i for i in result["single_move_interventions"] if i["room"] == room_index and i["ply"] == ply],
                            "scope": "selected regression from tracking boards; actual-deal DDS labels are not public-information optimality certificates; training on these consumes rating-board independence"})
        write(output/"real-regression-cases.json", natural)
        write(output/"summary.json", summary)
        print(summary["nonforced_diagnostics"], flush=True)
        print(summary["tactical_families"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rating-root", default="runs/cardplay-rating-v1")
    p.add_argument("--output", default="artifacts/cardplay-diagnostics-v1")
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()
