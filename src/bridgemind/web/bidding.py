"""Public auction validation and private-hand-only bidding inference."""

from bridgemind.dds.models import SEATS
from bridgemind.engine.bidding import CALLS
from .analysis import PROTOCOL, cards, exact, next_seat, require, same_side


def validate_bidding(request):
    exact(request, {"protocol", "stateId", "board", "dealer", "vulnerable", "turn", "auction", "hand", "legalCalls"}, "bidding request")
    require(request["protocol"] == PROTOCOL, "Unsupported protocol")
    require(isinstance(request["stateId"], str) and 0 < len(request["stateId"]) <= 4096, "Invalid stateId")
    require(type(request["board"]) is int and request["board"] > 0, "Invalid board")
    require(request["dealer"] in SEATS and request["turn"] in SEATS, "Invalid seat")
    require(isinstance(request["vulnerable"], list) and len(request["vulnerable"]) == 2 and
            all(type(v) is bool for v in request["vulnerable"]), "Invalid vulnerability")
    require(isinstance(request["auction"], list) and len(request["auction"]) < 319, "Invalid auction")
    highest, doubled, passes, bidder = -1, 0, 0, None
    complete = False

    def legal(seat):
        result = ["P"]
        if highest >= 0 and doubled == 0 and not same_side(seat, bidder):
            result.append("X")
        if highest >= 0 and doubled == 1 and same_side(seat, bidder):
            result.append("XX")
        return result + list(CALLS[3+highest+1:])

    for index, entry in enumerate(request["auction"]):
        exact(entry, {"seat", "call"}, "auction entry")
        seat = next_seat(request["dealer"], index)
        require(not complete and entry["seat"] == seat and entry["call"] in legal(seat), "Illegal auction prefix")
        call = entry["call"]
        if call == "P":
            passes += 1
            complete = passes == (4 if highest == -1 else 3)
        else:
            passes = 0
            if call in ("X", "XX"):
                doubled = 1 if call == "X" else 2
            else:
                highest, bidder, doubled = CALLS.index(call)-3, seat, 0
    require(not complete and request["turn"] == next_seat(request["dealer"], len(request["auction"])), "Auction is complete or wrong turn")
    exact(request["hand"], {"seat", "cards"}, "private hand")
    require(request["hand"]["seat"] == request["turn"], "Wrong private hand")
    hand = cards(request["hand"]["cards"])
    require(len(hand) == 13, "Bidding requires 13 cards")
    calls = legal(request["turn"])
    require(isinstance(request["legalCalls"], list) and len(request["legalCalls"]) == len(calls) and
            set(request["legalCalls"]) == set(calls), "Client legalCalls mismatch")
    return {"state_id": request["stateId"], "player": request["turn"], "dealer": request["dealer"],
            "hand": hand, "vulnerable": request["vulnerable"], "auction": request["auction"],
            "legal_call_ids": [CALLS.index(c) for c in calls]}


def analyze_bidding(service, request):
    obs = validate_bidding(request)
    if service.bidder is None:
        raise RuntimeError("Bidding model is not loaded")
    if not service.lock.acquire(timeout=1):
        raise TimeoutError("Model is busy")
    try:
        decision = service.bidder.predict(obs, stochastic=False)
    finally:
        service.lock.release()
    require(decision["call"] in request["legalCalls"], "Model returned an illegal call")
    return {"protocol": PROTOCOL, "stateId": request["stateId"], "source": "bidding",
            "call": decision["call"], "probabilities": decision["probabilities"], "model": service.bidding_name}
