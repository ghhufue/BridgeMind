"""Full card play with OpenSpiel rules and explicit, controller-private observations."""

import copy
import uuid

from bridgemind.dds import Deal
from bridgemind.dds.models import RANKS, SEATS, STRAINS
from .bidding import BridgeBiddingEnv
from .scoring import Contract, score_contract


def card_id(card):
    if not isinstance(card, str) or len(card) != 2 or card[0] not in "CDHS" or card[1] not in RANKS:
        raise ValueError("Expected a card such as SA, HT, C2")
    return RANKS.index(card[1]) * 4 + "CDHS".index(card[0])


def card_name(index):
    if type(index) is not int or not 0 <= index < 52:
        raise ValueError("Card ID must be an integer in 0..51")
    return "CDHS"[index % 4] + RANKS[index // 4]


def contract_auction(contract, dealer="N"):
    """A synthetic, legal auction for fixed-contract experiments (not bidding evidence)."""
    if dealer not in SEATS:
        raise ValueError("Invalid dealer")
    calls = ["P"] * ((SEATS.index(contract.declarer) - SEATS.index(dealer)) % 4)
    calls.append(f"{contract.level}{contract.strain}")
    if contract.double_status:
        calls.append("X")
    if contract.double_status == 2:
        calls.append("XX")
    return calls + ["P"] * 3


class BridgePlayEnv:
    def __init__(self, deal, *, auction=None, contract=None, dealer="N", vulnerable=(False, False), system_id=0):
        if (auction is None) == (contract is None):
            raise ValueError("Supply exactly one of auction or contract")
        self._deal = Deal.from_dict(deal)
        if type(system_id) is not int or not 0 <= system_id < 256:
            raise ValueError("system_id must be in 0..255 (0 means unknown)")
        self.system_id = system_id
        self.auction_source = "synthetic_contract" if contract is not None else "recorded"
        if contract is not None:
            contract = contract if isinstance(contract, Contract) else Contract(**contract)
            auction = contract_auction(contract, dealer)
        bidding = BridgeBiddingEnv(None, self._deal, dealer=dealer, vulnerable=vulnerable, prefetch=False)
        for call in auction:
            if isinstance(call, dict):
                if call.get("seat") != bidding.observation()["player"]:
                    raise ValueError("Auction seat does not match turn order")
                call = call["call"]
            bidding.apply_call(call)
        if not bidding.is_terminal():
            raise ValueError("Auction must be complete")
        if contract is not None and bidding._contract != contract:
            raise ValueError("Synthetic auction did not reproduce contract")
        self._state = bidding._state.clone()
        self.contract = bidding._contract
        self.dealer, self.vulnerable = dealer, tuple(vulnerable)
        self._auction = copy.deepcopy(bidding._auction)
        self._hands = {s: set(h) for s, h in self._deal.hands.items()}
        self._plays, self._current = [], []
        self._tricks = [0, 0]
        self._voids = {s: set() for s in SEATS}
        self._leader = SEATS[(SEATS.index(self.contract.declarer) + 1) % 4] if self.contract else None
        self._board_id = uuid.uuid4().hex

    @property
    def auction(self):
        return copy.deepcopy(self._auction)

    @property
    def to_play(self):
        if self.is_terminal():
            return None
        return SEATS[(SEATS.index(self._leader) + len(self._current)) % 4]

    @property
    def controller(self):
        actor = self.to_play
        if actor is None:
            return None
        declarer = SEATS.index(self.contract.declarer)
        return self.contract.declarer if SEATS.index(actor) % 2 == declarer % 2 else actor

    def is_terminal(self):
        return self._state.is_terminal()

    def legal_card_ids(self):
        return [] if self.is_terminal() else list(self._state.legal_actions())

    def observation(self):
        if self.is_terminal():
            raise ValueError("Card play is terminal")
        player = self.controller
        dummy = SEATS[(SEATS.index(self.contract.declarer) + 2) % 4]
        visible = {player}
        if self._plays:
            visible.add(dummy)
        if SEATS[self._state.current_player()] != player:
            raise RuntimeError("OpenSpiel controller disagrees with local turn tracking")
        return {"state_id": f"{self._board_id}:{len(self._plays)}", "player": player, "to_play": self.to_play,
                "contract": self.contract.to_dict(), "dealer": self.dealer, "vulnerable": list(self.vulnerable),
                "dummy": dummy, "dummy_visible": bool(self._plays),
                "visible_hands": {s: sorted(self._hands[s], key=card_id) for s in SEATS if s in visible},
                "remaining_counts": {s: len(self._hands[s]) for s in SEATS},
                "auction": copy.deepcopy(self._auction), "auction_source": self.auction_source, "system_id": self.system_id,
                "play_history": copy.deepcopy(self._plays), "trick_leader": self._leader,
                "current_trick": copy.deepcopy(self._current), "tricks": list(self._tricks),
                "voids": {s: sorted(self._voids[s]) for s in SEATS}, "legal_card_ids": self.legal_card_ids()}

    def apply_card(self, card):
        action = card_id(card) if isinstance(card, str) else card
        if type(action) is not int or action not in self.legal_card_ids():
            raise ValueError("Illegal card")
        name, actor = card_name(action), self.to_play
        if name not in self._hands[actor]:
            raise RuntimeError("OpenSpiel legal card is not in the acting hand")
        if self._current and name[0] != self._current[0]["card"][0]:
            self._voids[actor].add(self._current[0]["card"][0])
        self._state.apply_action(action)
        self._hands[actor].remove(name)
        record = {"seat": actor, "card": name}
        self._plays.append(record)
        self._current.append(record)
        if len(self._current) == 4:
            led = self._current[0]["card"][0]
            def strength(item):
                suit, rank = item["card"]
                return (2 if suit == self.contract.strain else 1 if suit == led else 0, RANKS.index(rank))
            winner = max(self._current, key=strength)["seat"]
            self._tricks[SEATS.index(winner) % 2] += 1
            self._leader, self._current = winner, []

    def result(self):
        if not self.is_terminal():
            raise ValueError("Card play has not finished")
        if self.contract is None:
            return {"contract": None, "declarer_tricks": None, "tricks": [0, 0], "returns": [0] * 4,
                    "evaluation_model": "played_out"}
        side = SEATS.index(self.contract.declarer) % 2
        score = score_contract(self.contract, self._tricks[side], self.vulnerable[side])
        returns = [score if i % 2 == side else -score for i in range(4)]
        if returns != self._state.returns():
            raise RuntimeError("OpenSpiel terminal scoring disagrees with local scoring")
        return {"contract": self.contract.to_dict(), "declarer_tricks": self._tricks[side],
                "tricks": list(self._tricks), "returns": returns, "evaluation_model": "played_out"}

    def training_owners(self):
        """Privileged supervision ONLY. Never put this in policy observations."""
        owners = [0] * 52
        for s, cards in self._deal.hands.items():
            for card in cards:
                owners[card_id(card)] = SEATS.index(s)
        return owners
