"""OpenSpiel auction rules, settled exclusively by the external DDS facade."""

import copy
import uuid

from bridgemind.dds import Deal, SolverError, SolverWaitTimeout
from bridgemind.dds.models import RANKS, SEATS, STRAINS
from .scoring import Contract, score_contract

CALLS = ("P", "X", "XX") + tuple(f"{level}{strain}" for level in range(1, 8) for strain in STRAINS)


def deal_into_state(state, deal):
    """OpenSpiel deals cards round-robin N/E/S/W, with rank*4+suit IDs."""
    deal = Deal.from_dict(deal)
    for index in range(13):
        for seat in SEATS:
            card = deal.hands[seat][index]
            state.apply_action(RANKS.index(card[1]) * 4 + "CDHS".index(card[0]))


class BridgeBiddingEnv:
    """One environment per rollout owner; not intended for concurrent mutation.

    Calls use IDs 0..37, NOT OpenSpiel's 52..89. The wrapper terminates at
    auction completion; the underlying OpenSpiel state would enter card play.
    """

    def __init__(self, solver, deal, *, dealer="N", vulnerable=(False, False), prefetch=True):
        try:
            import pyspiel
        except ImportError as error:
            raise ImportError("Install BridgeMind's bridge extra: pip install -e '.[bridge]'") from error
        if dealer not in SEATS or len(vulnerable) != 2 or any(type(v) is not bool for v in vulnerable):
            raise ValueError("Invalid dealer or NS/EW vulnerability")
        self._solver = solver
        self._pyspiel = pyspiel
        self.dealer = dealer
        self.vulnerable = tuple(vulnerable)
        self.prefetch = prefetch
        self._submission = None
        self.reset(deal)

    def reset(self, deal):
        self.close()
        self._deal = Deal.from_dict(deal)
        self._board_id = uuid.uuid4().hex
        index = SEATS.index(self.dealer)
        self._game = self._pyspiel.load_game("bridge", {
            "use_double_dummy_result": False, "dealer": index,
            "dealer_vul": self.vulnerable[index % 2], "non_dealer_vul": self.vulnerable[1 - index % 2]})
        self._state = self._game.new_initial_state()
        deal_into_state(self._state, self._deal)
        self._auction = []
        self._first_bidder = {}
        self._contract = None
        self._settled = None
        if self.prefetch:
            self._submit_table()
        return self.observation()

    def _submit_table(self):
        self._submission = self._solver.submit([{"item_id": "table", "operation": "table",
                                                  "payload": self._deal.to_dict(), "state_id": self._board_id}])

    def is_terminal(self):
        return self._state.is_terminal() or self._state.current_phase() == 2

    def legal_call_ids(self):
        return [] if self.is_terminal() else [a - 52 for a in self._state.legal_actions()]

    def legal_calls(self):
        return [CALLS[i] for i in self.legal_call_ids()]

    def observation(self):
        if self.is_terminal():
            raise ValueError("Auction is terminal")
        player = self._state.current_player()
        return {"state_id": f"{self._board_id}:{len(self._auction)}", "player": SEATS[player],
                "tensor": list(self._state.observation_tensor(player)),
                "legal_call_ids": self.legal_call_ids(), "auction": copy.deepcopy(self._auction)}

    def apply_call(self, call):
        if isinstance(call, str):
            try:
                call = CALLS.index(call)
            except ValueError as error:
                raise ValueError("Unknown call; use P, X, XX, 1C..7NT") from error
        if type(call) is not int or call not in self.legal_call_ids():
            raise ValueError("Illegal call")
        player = self._state.current_player()
        self._state.apply_action(call + 52)
        self._auction.append({"seat": SEATS[player], "call": CALLS[call]})
        if call >= 3:
            level, strain = 1 + (call - 3) // 5, STRAINS[(call - 3) % 5]
            first = self._first_bidder.setdefault((player % 2, strain), SEATS[player])
            self._contract = Contract(level, strain, first)
        elif call in (1, 2):
            previous = self._contract
            self._contract = Contract(previous.level, previous.strain, previous.declarer, call)

    def result(self, wait_timeout_ms=30000):
        if not self.is_terminal():
            raise ValueError("Auction is not terminal")
        if self._settled is not None:
            return copy.deepcopy(self._settled)
        if self._contract is None:
            self.close()
            self._settled = {"contract": None, "declarer_tricks": None,
                             "returns": [0, 0, 0, 0], "evaluation_model": "passed_out"}
        else:
            if self._submission is None:
                self._submit_table()
            job_id = self._submission["job_id"]
            snapshot = self._solver.wait(job_id, wait_timeout_ms)
            if not snapshot["complete"]:
                raise SolverWaitTimeout(job_id)
            item = snapshot["items"][0]
            if item["status"] != "succeeded":
                raise SolverError.from_dict(item["error"])
            if item["state_id"] != self._board_id:
                raise SolverError("INVALID_ARGUMENT", "Stale DDS board result")
            contract = self._contract
            seat = SEATS.index(contract.declarer)
            tricks = item["result"]["tricks"][STRAINS.index(contract.strain)][seat]
            score = score_contract(contract, tricks, self.vulnerable[seat % 2])
            self._settled = {"contract": contract.to_dict(), "declarer_tricks": tricks,
                             "returns": [score if i % 2 == seat % 2 else -score for i in range(4)],
                             "evaluation_model": "double_dummy", "exact": True,
                             "solver_version": item["solver_version"], "solver_commit": item["solver_commit"]}
        return copy.deepcopy(self._settled)

    def close(self):
        if self._submission is not None:
            try:
                self._solver.cancel(self._submission["job_id"])
            except SolverError:
                pass
            self._submission = None

