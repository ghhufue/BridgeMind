"""Exact augmentation, scoring, mixed loss isolation, and resumable time budgets."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bridgemind.dds import Deal
from bridgemind.dds.models import SEATS
from bridgemind.engine import BridgePlayEnv, Contract
from bridgemind.engine.play import card_id, card_name
from bridgemind.engine.scoring import score_contract
from bridgemind.cardplay.encoding import encode_observation
from bridgemind.cardplay.model import PlayModel, ModelConfig, save_checkpoint, read_checkpoint
from bridgemind.cardplay.data import prepare_dataset
from bridgemind.foundation.cases import build_base, observe, transform_observation, PERMUTATIONS, card_map
from bridgemind.training.data import ownership, board_ids, deal_from_ownership
from bridgemind.rating.system import read, write
from bridgemind.sl_v2.data import augment, orbit_key, Curriculum
from bridgemind.sl_v2.training import mixed_update, prepare, train


def example_row(role="declarer"):
    case = build_base("small_ruff_preserve_master", 1701, role)
    obs = observe(case)
    encoded = encode_observation(obs)
    sign = 1 if encoded["declarer"] else -1
    own = np.asarray(case["board"]["owners"], dtype=np.int16)
    decl = SEATS.index(obs["contract"]["declarer"])
    row = {"encoded": encoded, "action": obs["legal_card_ids"][0], "owners": ((own-decl) % 4).astype(np.uint8),
           "return": np.float32(sign*score_contract(Contract(**obs["contract"]), 8, False)/7600)}
    return row, obs


@pytest.mark.parametrize("role", ("declarer", "dummy", "defender"))
def test_encoded_suit_augmentation_exactly_matches_reencoding_public_observation(role):
    row, obs = example_row(role)
    saved = copy.deepcopy(row)
    for permutation in PERMUTATIONS:
        actual = augment(row, permutation)
        expected = encode_observation(transform_observation(obs, permutation))
        for key in expected:
            np.testing.assert_array_equal(actual["encoded"][key], expected[key])
        mapping = card_map(permutation)
        assert actual["action"] == mapping[row["action"]]
        np.testing.assert_array_equal(actual["owners"][mapping], row["owners"])
        assert actual["encoded"]["legal"][actual["action"]]
    np.testing.assert_array_equal(row["encoded"]["tokens"], saved["encoded"]["tokens"])


@pytest.mark.parametrize("vulnerable", (False, True))
@pytest.mark.parametrize("doubled", (0, 1, 2))
def test_value_label_recomputed_for_minor_to_major_and_defense_perspective(deal, vulnerable, doubled):
    env = BridgePlayEnv(deal, contract=Contract(4, "C", "N", doubled), vulnerable=(vulnerable, vulnerable))
    encoded = encode_observation(env.observation())  # Opening defender, negative declarer score.
    row = {"encoded": encoded, "action": env.legal_card_ids()[0], "owners": ownership(deal),
           "return": np.float32(-score_contract(env.contract, 11, vulnerable)/7600)}
    augmented = augment(row, "HDCS")  # C -> H
    assert augmented["return"] == pytest.approx(-score_contract(Contract(4, "H", "N", doubled), 11, vulnerable)/7600)


def test_real_auction_is_replaced_by_legal_synthetic_summary_when_suits_change(deal):
    env = BridgePlayEnv(deal, auction=["1C", "P", "1D", "P", "2C", "P", "P", "P"], system_id=1)
    prefix = []
    for _ in range(9):
        action = env.legal_card_ids()[0]
        prefix.append(action); env.apply_card(action)
    obs = env.observation(); encoded = encode_observation(obs)
    sign = 1 if encoded["declarer"] else -1
    row = {"encoded": encoded, "action": env.legal_card_ids()[0], "owners": ownership(deal),
           "return": sign*score_contract(env.contract, 8, False)/7600}
    permutation = "SHDC"  # Original 1C -> 1S, 1D -> 1H would be illegal.
    actual = augment(row, permutation, contract_only=True)
    mapping = card_map(permutation)
    owners = np.empty(52, dtype=np.uint8); owners[mapping] = ownership(deal)
    transformed = BridgePlayEnv(deal_from_ownership(owners), contract=Contract(2, "S", "N"))
    for action in prefix:
        transformed.apply_card(int(mapping[action]))
    expected = encode_observation(transformed.observation())
    for key in expected:
        np.testing.assert_array_equal(actual["encoded"][key], expected[key])
    assert actual["encoded"]["system"] == 0 and actual["encoded"]["meta"][30] == 0


def test_orbit_groups_all_suit_and_seat_equivalent_complete_deals():
    base = build_base("fourth_hand_take_trick", 1701, "declarer")
    owners = np.asarray(base["board"]["owners"], dtype=np.uint8)
    expected = orbit_key(owners)
    for permutation in PERMUTATIONS:
        other = np.empty(52, dtype=np.uint8); other[card_map(permutation)] = owners
        for rotation in range(4):
            assert orbit_key((other+rotation) % 4) == expected


def test_curriculum_placeholder_owners_and_values_never_affect_auxiliary_loss():
    real, _ = example_row()
    real = augment(real, "CDHS")
    course = copy.deepcopy(real); course["curriculum"] = True
    alternate = copy.deepcopy(course); alternate["owners"][:] = 255; alternate["return"] = 1e9
    model_a = PlayModel(ModelConfig(32, 1, 4))
    model_b = copy.deepcopy(model_a)
    a = mixed_update(model_a, torch.optim.Adam(model_a.parameters(), lr=1e-4), [real, course])
    b = mixed_update(model_b, torch.optim.Adam(model_b.parameters(), lr=1e-4), [real, alternate])
    assert a == b
    for name, value in model_a.state_dict().items():
        torch.testing.assert_close(value, model_b.state_dict()[name], rtol=0, atol=0)


def raw_trajectory(deal):
    from bridgemind.engine.bidding import CALLS
    from bridgemind.engine.play import contract_auction
    calls = contract_auction(Contract(3, "NT", "N"))
    actions = [card_id(deal.hands[s][i]) for i in range(13) for s in SEATS]
    actions += [52+CALLS.index(call) for call in calls]
    env = BridgePlayEnv(deal, contract=Contract(3, "NT", "N"))
    while not env.is_terminal():
        action = env.legal_card_ids()[0]; actions.append(action); env.apply_card(action)
    return " ".join(map(str, actions))+"\n"


def test_mixed_worker_cpu_resume_matches_uninterrupted_and_time_budget_stops(tmp_path, monkeypatch, deal, backend):
    import bridgemind.sl_v2.training as training
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "load_suite", lambda path: {"bases": []})
    paths = []
    hands = {s: list(h) for s, h in deal.hands.items()}
    hands["N"][0], hands["E"][0] = hands["E"][0], hands["N"][0]
    for label, d in (("train", deal), ("valid", Deal(hands))):
        source = tmp_path/(label+".txt"); source.write_text(raw_trajectory(d))
        path = tmp_path/label
        prepare_dataset(source, path, allowed_ids=board_ids(np.asarray([ownership(d)])), max_boards=1)
        paths.append(path)
    initial = tmp_path/"init.pt"
    torch.manual_seed(12)
    save_checkpoint(initial, PlayModel(ModelConfig(32, 1, 4)))
    class Solver:
        def solve_position(self, p):
            return backend.solve_position(p)
        def get_info(self):
            return {"solver_identity": backend.info["identity"], "solver_version": backend.info["version"]}
    def setup_run(name):
        root = tmp_path/name
        prepare(root, paths[0], paths[1], initial, Solver(), train_seeds=(8101,), valid_seeds=(9103,),
                device="cpu", batch_size=8, eval_states=8, eval_boards=0, eval_interval=10000, save_interval=10000)
        return root
    a, b, c = setup_run("a"), setup_run("b"), setup_run("c")
    train(a, seconds=1000, max_steps=4)
    train(b, seconds=1000, max_steps=2)
    train(b, seconds=1000, resume=True, max_steps=4)
    first, resumed = read_checkpoint(a/"last.pt"), read_checkpoint(b/"last.pt")
    for name, value in first["state_dict"].items():
        torch.testing.assert_close(value, resumed["state_dict"][name], rtol=0, atol=0)
    assert first["numpy_rng"] == resumed["numpy_rng"]
    timed = train(c, seconds=1e-9)
    assert timed["stop_reason"] == "time_budget" and timed["step"] == 0
    config = read(b/"protocol.json"); config["lr"] *= 2; write(b/"protocol.json", config)
    with pytest.raises(ValueError, match="protocol mismatch"):
        train(b, seconds=1000, resume=True, max_steps=5)
