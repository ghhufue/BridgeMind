import pytest

pyspiel = pytest.importorskip("pyspiel")

from bridgemind.dds import ServiceConfig, SolverService, NativeDdsBackend, parse_pbn
from bridgemind.engine import BridgeBiddingEnv
from bridgemind.engine.scoring import Contract, score_contract


def test_external_dds_settles_auction_and_observation_is_private(deal):
    with SolverService(ServiceConfig(worker_count=2)) as solver:
        env = BridgeBiddingEnv(solver, deal, prefetch=True)
        before = env.observation()
        assert before["player"] == "N"
        assert "hands" not in before
        with pytest.raises(ValueError):
            env.apply_call("XX")
        assert env.observation() == before
        for call in ("1NT", "P", "P", "P"):
            env.apply_call(call)
        assert env.is_terminal()
        result = env.result()
        assert result["contract"] == {"level": 1, "strain": "NT", "declarer": "N", "double_status": 0}
        assert result["declarer_tricks"] == 6
        assert result["returns"] == [-50, 50, -50, 50]
        assert result["solver_version"] == "3.1.0"


def test_passed_out_does_not_require_solver(deal):
    class NoSolver:
        def submit(self, *args, **kwargs):
            raise AssertionError("Passed-out board should not be solved")
    env = BridgeBiddingEnv(NoSolver(), deal, prefetch=False)
    for _ in range(4):
        env.apply_call("P")
    assert env.result()["returns"] == [0, 0, 0, 0]


def test_declarer_is_first_bidder_and_reset_changes_board(deal):
    with SolverService(ServiceConfig(worker_count=1)) as solver:
        env = BridgeBiddingEnv(solver, deal, dealer="E", vulnerable=(False, True), prefetch=False)
        original = env.observation()["state_id"]
        for call in ("1H", "P", "2H", "P", "3H", "P", "4H", "P", "P", "P"):
            env.apply_call(call)
        result = env.result()
        assert result["contract"]["declarer"] == "E"
        assert result["returns"] == [400, -400, 400, -400]
        env.reset(deal)
        assert env.observation()["state_id"] != original


def test_native_tables_and_all_contract_scores_agree_with_openspiel(backend, deal):
    # Independent old DDS in the installed OpenSpiel wheel is a cross-check,
    # never the evaluation backend of BridgeBiddingEnv.
    from bridgemind.engine.bidding import deal_into_state
    table = backend.solve_table(deal)
    for vulnerable in ((False, False), (True, True), (True, False), (False, True)):
        game = pyspiel.load_game("bridge", {"use_double_dummy_result": True,
                                "dealer_vul": vulnerable[0], "non_dealer_vul": vulnerable[1]})
        state = game.new_initial_state()
        deal_into_state(state, deal)
        scores = state.score_by_contract()
        assert scores[0] == 0
        index = 1
        for level in range(1, 8):
            for strain_index, strain in enumerate(("C", "D", "H", "S", "NT")):
                for seat_index, seat in enumerate("NESW"):
                    for doubled in range(3):
                        contract = Contract(level, strain, seat, doubled)
                        score = score_contract(contract, table.tricks[strain_index][seat_index], vulnerable[seat_index % 2])
                        assert (score if seat_index % 2 == 0 else -score) == scores[index]
                        index += 1

