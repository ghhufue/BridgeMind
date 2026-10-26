"""Validate browser positions without leaking hidden DDS hands into the model."""

from dataclasses import dataclass
import threading

from bridgemind.dds import Options, Position, SolverClient
from bridgemind.dds.models import CARDS, RANKS, SEATS, STRAINS
from bridgemind.engine.bidding import CALLS
from bridgemind.engine.play import card_id

PROTOCOL = "bridgemind.analysis.v1"
COMMON = {"protocol", "stateId", "board", "dealer", "vulnerable", "contract", "toPlay", "controller",
          "auction", "currentTrick", "tricks", "legalCards", "information"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def next_seat(seat, offset=1):
    return SEATS[(SEATS.index(seat) + offset) % 4]


def same_side(a, b):
    return SEATS.index(a) % 2 == SEATS.index(b) % 2


def exact(value, fields, label):
    require(isinstance(value, dict) and set(value) == set(fields), f"Invalid {label} fields")


def cards(value):
    require(isinstance(value, list) and len(value) <= 13, "Hand must contain 0..13 cards")
    require(all(isinstance(c, str) and c in CARDS for c in value), "Invalid card")
    require(len(set(value)) == len(value), "Duplicate card")
    return list(value)


def validate_auction(request):
    auction = request["auction"]
    require(isinstance(auction, list) and 4 <= len(auction) <= 319, "Invalid auction length")
    highest, doubled, passes, last = -1, 0, 0, None
    first, complete = {}, False
    for index, entry in enumerate(auction):
        exact(entry, {"seat", "call"}, "auction entry")
        seat, call = entry["seat"], entry["call"]
        require(not complete and seat == next_seat(request["dealer"], index), "Auction turn mismatch")
        require(isinstance(call, str) and call in CALLS, "Unknown call")
        if call == "P":
            passes += 1
            complete = passes == (4 if highest == -1 else 3)
        elif call == "X":
            require(highest >= 0 and doubled == 0 and not same_side(seat, last), "Illegal double")
            doubled, passes = 1, 0
        elif call == "XX":
            require(highest >= 0 and doubled == 1 and same_side(seat, last), "Illegal redouble")
            doubled, passes = 2, 0
        else:
            order = CALLS.index(call) - 3
            require(order > highest, "Bid must increase")
            highest, last, doubled, passes = order, seat, 0, 0
            first.setdefault((SEATS.index(seat) % 2, call[1:]), seat)
    require(complete and highest >= 0, "A completed contract auction is required")
    strain = STRAINS[highest % 5]
    expected = {"level": highest // 5 + 1, "strain": strain,
                "declarer": first[(SEATS.index(last) % 2, strain)], "doubled": doubled}
    exact(request["contract"], expected, "contract")
    require(type(request["contract"]["level"]) is int and type(request["contract"]["doubled"]) is int,
            "Contract numbers must be integers")
    require(request["contract"] == expected, "Contract disagrees with auction")


def winner(entries, trump):
    led = entries[0]["card"][0]
    return max(entries, key=lambda e: (2 if e["card"][0] == trump else 1 if e["card"][0] == led else 0,
                                       RANKS.index(e["card"][1])))["seat"]


@dataclass
class ValidatedPosition:
    request: dict
    observation: dict
    legal: list
    dds_position: Position | None


def validate_request(request, source):
    require(source in ("ai", "dds"), "Unknown source")
    extra = {"hands"} if source == "dds" else {"hand", "dummy", "remainingCounts"}
    exact(request, COMMON | extra, "request")
    require(request["protocol"] == PROTOCOL, "Unsupported protocol")
    require(isinstance(request["stateId"], str) and 1 <= len(request["stateId"]) <= 256, "Invalid stateId")
    require(type(request["board"]) is int and request["board"] >= 1, "Invalid board")
    require(request["dealer"] in SEATS, "Invalid dealer")
    require(isinstance(request["vulnerable"], list) and len(request["vulnerable"]) == 2
            and all(type(v) is bool for v in request["vulnerable"]), "Invalid vulnerability")
    require(request["information"] == ("full" if source == "dds" else "visible"), "Information boundary mismatch")
    validate_auction(request)
    contract = request["contract"]
    declarer, trump = contract["declarer"], contract["strain"]
    dummy, leader = next_seat(declarer, 2), next_seat(declarer)
    completed, current = request["tricks"], request["currentTrick"]
    require(isinstance(completed, list) and len(completed) <= 12, "Invalid completed trick count")
    require(isinstance(current, list) and len(current) <= 3, "Current trick must have 0..3 cards")
    remaining, voids = dict.fromkeys(SEATS, 13), {s: set() for s in SEATS}
    history, seen, counts = [], set(), [0, 0]

    def visit(entries):
        led = None
        for offset, entry in enumerate(entries):
            exact(entry, {"seat", "card"}, "played card")
            seat, card = entry["seat"], entry["card"]
            require(seat == next_seat(leader, offset), "Play order disagrees with leader")
            require(isinstance(card, str) and card in CARDS and card not in seen, "Invalid or duplicate played card")
            require(card[0] not in voids[seat], "History contradicts a shown void")
            if led and card[0] != led:
                voids[seat].add(led)
            led = led or card[0]
            remaining[seat] -= 1
            require(remaining[seat] >= 0, "Too many cards from a seat")
            seen.add(card)
            history.append(dict(entry))

    for trick in completed:
        exact(trick, {"cards", "winner"}, "completed trick")
        require(isinstance(trick["cards"], list) and len(trick["cards"]) == 4, "Completed trick needs four cards")
        visit(trick["cards"])
        actual = winner(trick["cards"], trump)
        require(trick["winner"] == actual, "Incorrect trick winner")
        counts[SEATS.index(actual) % 2] += 1
        leader = actual
    visit(current)
    to_play = next_seat(leader, len(current))
    actor = declarer if to_play == dummy else to_play
    require(request["toPlay"] == to_play and request["controller"] == actor, "Incorrect player or dummy controller")
    if source == "dds":
        exact(request["hands"], SEATS, "hands")
        known = {s: cards(request["hands"][s]) for s in SEATS}
    else:
        exact(request["hand"], {"seat", "cards"}, "controller hand")
        require(request["hand"]["seat"] == actor, "Incorrect controller hand")
        known = {actor: cards(request["hand"]["cards"])}
        if history:
            exact(request["dummy"], {"seat", "cards"}, "dummy hand")
            require(request["dummy"]["seat"] == dummy, "Incorrect dummy seat")
            known[dummy] = cards(request["dummy"]["cards"])
        else:
            require(request["dummy"] is None, "Dummy must be hidden before opening lead")
        exact(request["remainingCounts"], SEATS, "remaining counts")
        require(all(type(v) is int for v in request["remainingCounts"].values())
                and request["remainingCounts"] == remaining, "Remaining counts disagree with history")
    for seat, hand in known.items():
        require(len(hand) == remaining[seat], "Hand size disagrees with history")
        require(all(c not in seen for c in hand), "Duplicate visible or played card")
        require(not any(c[0] in voids[seat] for c in hand), "Visible hand contradicts public void")
        seen.update(hand)
    require(to_play in known, "Acting hand must be visible")
    hand = known[to_play]
    following = [c for c in hand if current and c[0] == current[0]["card"][0]]
    legal = following or hand
    require(set(cards(request["legalCards"])) == set(legal) and legal, "Client legalCards disagree with hand")
    obs = {"state_id": request["stateId"], "player": actor, "to_play": to_play,
           "dealer": request["dealer"], "vulnerable": request["vulnerable"],
           "contract": {"level": contract["level"], "strain": trump, "declarer": declarer, "double_status": contract["doubled"]},
           "visible_hands": known if source == "ai" else {}, "dummy": dummy, "dummy_visible": bool(history),
           "remaining_counts": remaining, "auction": request["auction"], "auction_source": "recorded", "system_id": 0,
           "play_history": history, "trick_leader": leader, "current_trick": current,
           "tricks": counts, "voids": {s: sorted(v) for s, v in voids.items()}, "legal_card_ids": [card_id(c) for c in legal]}
    position = Position(known, trump, leader, [e["card"] for e in current]) if source == "dds" else None
    return ValidatedPosition(request, obs, legal, position)


class AnalysisService:
    def __init__(self, policy=None, *, model_name=None, model_sha256=None, bidder=None, bidding_name=None,
                 solver_port=18765, client_factory=SolverClient):
        self.policy, self.model_name = policy, model_name
        self.model_sha256, self.bidder, self.bidding_name = model_sha256, bidder, bidding_name
        self.solver_port, self.client_factory = solver_port, client_factory
        self.lock = threading.Lock()
        self.models = None

    def model_catalog(self):
        if self.models is not None:
            return self.models.catalog()
        return {"protocol": "bridgemind.models.v1",
                "play": [{"id": "default", "name": self.model_name or "默认出牌模型", "available": self.policy is not None}],
                "bidding": [{"id": "default", "name": self.bidding_name or "默认叫牌模型", "available": self.bidder is not None}]}

    def health(self):
        version = None
        try:
            with self.client_factory(port=self.solver_port, connect_timeout=1) as solver:
                info = solver.get_info()
                version = str(info.get("solver_version", info.get("version", "DDS")))
        except Exception:
            pass
        ready = self.policy is not None and version is not None
        return {"protocol": PROTOCOL, "service": "BridgeMind LAN", "status": "ready" if ready else "degraded",
                "services": {"ai": {"ready": self.policy is not None, "model": self.model_name},
                             "bidding": {"ready": self.bidder is not None, "model": self.bidding_name},
                             "dds": {"ready": version is not None, "version": version}}}

    def analyze(self, request, source):
        if isinstance(request, dict) and "modelId" in request:
            identity = request["modelId"]
            require(source in ("ai", "bidding") and isinstance(identity, str) and 0 < len(identity) <= 64,
                    "Invalid modelId")
            body = {key: value for key, value in request.items() if key != "modelId"}
            # Validate information boundaries before loading any alternate checkpoint.
            if source == "bidding":
                from .bidding import validate_bidding
                validate_bidding(body)
            else:
                validate_request(body, source)
            require(identity == "default" or self.models is not None, "Unknown modelId")
            selected = self if identity == "default" else self.models.resolve(source, identity)
            response = selected.analyze(body, source)
            return {**response, "modelId": identity}
        if source == "bidding":
            from .bidding import analyze_bidding
            return analyze_bidding(self, request)
        parsed = validate_request(request, source)
        if source == "ai":
            if self.policy is None:
                raise RuntimeError("AI model is not loaded")
            if not self.lock.acquire(timeout=1):
                raise TimeoutError("AI is busy; retry shortly")
            try:
                decision, diagnostics = self.inspect(parsed.observation)
            finally:
                self.lock.release()
            require(decision["card"] in parsed.legal, "Model returned an illegal card")
            response = {"protocol": PROTOCOL, "stateId": request["stateId"], "source": source,
                    "bestCards": [decision["card"]], "model": self.model_name,
                    "candidates": [{"card": c, "probability": float(decision["probabilities"][card_id(c)])} for c in parsed.legal]}
            if diagnostics is not None:
                response["diagnostics"] = diagnostics
            return response
        with self.client_factory(port=self.solver_port, connect_timeout=2) as solver:
            result = solver.solve_position(parsed.dds_position, Options(wait_timeout_ms=11000, queue_timeout_ms=1000,
                                                                       priority="interactive", state_id=request["stateId"]))
        return {"protocol": PROTOCOL, "stateId": request["stateId"], "source": source,
                "bestCards": list(result.best_cards), "model": "DDS 3.1.0",
                "candidates": [{"card": m.card, "tricks": m.tricks_for_evaluated_side} for m in result.moves]}

    def inspect(self, obs):
        """Read all three heads in one forward pass without altering frozen policy code."""
        if not hasattr(self.policy, "model"):
            return self.policy.predict_batch([obs], return_beliefs=True)[0], None
        import math
        import numpy as np
        from bridgemind.cardplay.encoding import encode_observation
        from bridgemind.cardplay.policy import act_batch
        from bridgemind.engine.play import card_name
        encoded = encode_observation(obs)
        actions, _, values, probabilities, beliefs = act_batch(
            self.policy.model, [encoded], self.policy.rng, stochastic=False, return_beliefs=True)
        value = float(values[0])
        require(math.isfinite(value), "Model value is not finite")
        declarer = SEATS.index(obs["contract"]["declarer"])
        absolute = beliefs[0][:, [(s-declarer) % 4 for s in range(4)]]
        known = {card: seat for seat, hand in obs["visible_hands"].items() for card in hand}
        played = {event["card"]: event["seat"] for event in obs["play_history"]}
        ownership = []
        for index in range(52):
            card = card_name(index)
            seat = known.get(card, played.get(card))
            distribution = np.eye(4)[SEATS.index(seat)] if seat else absolute[index]
            ownership.append({"card": card, "status": "played" if card in played else "known" if card in known else "unknown",
                              "probabilities": {s: float(distribution[i]) for i, s in enumerate(SEATS)}})
        hand = obs["visible_hands"][obs["to_play"]]
        diagnostics = {"modelSha256": self.model_sha256, "inference": "greedy_neural",
                       "actionProbabilities": [{"card": c, "legal": card_id(c) in obs["legal_card_ids"],
                                                "probability": float(probabilities[0][card_id(c)])} for c in hand],
                       "ownership": ownership, "ownershipMethod": "raw_network_head; known cards fixed from observation; no count projection",
                       "value": {"normalized": value, "predictedScore": value*7600,
                                 "perspective": "NS" if SEATS.index(obs["player"]) % 2 == 0 else "EW",
                                 "target": "actual_terminal_team_score/7600"},
                       "winProbability": None, "winProbabilityReason": "当前模型未训练胜率头；价值是终局本方得分，不能换算为胜率。"}
        return {"card": card_name(int(actions[0])), "probabilities": probabilities[0]}, diagnostics
