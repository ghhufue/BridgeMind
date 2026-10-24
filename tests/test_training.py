import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bridgemind.training.encoding import encode480, encode_env
from bridgemind.training.data import BoardSet, decode_pgx, ownership, deal_from_ownership, make_splits
from bridgemind.training.model import BiddingModel, load_model, save_model
from bridgemind.training.rollout import collect_duplicate, imps
from bridgemind.training.optim import ppo_update, supervised_update
from bridgemind.training.evaluation import evaluate_duplicate
from bridgemind.training.league import OpponentPool
from bridgemind.engine import BridgeBiddingEnv
from bridgemind.dds import NativeDdsBackend


def test_encoder_matches_openspiel_for_random_legal_auctions(deal):
    for dealer in "NESW":
        env = BridgeBiddingEnv(None, deal, dealer=dealer, vulnerable=(True, False), prefetch=False)
        rng = random.Random(81)
        for _ in range(100):
            if env.is_terminal():
                break
            obs = env.observation()
            actor = obs["player"]
            encoded = encode480(deal.hands[actor], obs["auction"], actor, (True, False))
            np.testing.assert_array_equal(encoded, encode_env(env)[0])
            if dealer == "N":
                np.testing.assert_array_equal(encoded, np.asarray(obs["tensor"])[4:484])
            env.apply_call(rng.choice(env.legal_call_ids()))


def test_encoder_does_not_depend_on_hidden_hands(deal):
    a = encode480(deal.hands["N"], [], "N", (False, False))
    assert a.shape == (480,) and a.dtype == np.bool_
    assert a[428:].sum() == 13
    assert a[:4].tolist() == [True, False, True, False]


def test_precomputed_environment_and_dataset_roundtrip(deal, tmp_path):
    with NativeDdsBackend() as backend:
        table = backend.solve_table(deal)
    env = BridgeBiddingEnv(None, deal, table=table)
    for call in ("1NT", "P", "P", "P"):
        env.apply_call(call)
    assert env.result()["returns"] == [-50, 50, -50, 50]
    assert deal_from_ownership(ownership(deal)) == deal
    boards = BoardSet(np.array([ownership(deal)]), np.array([table.tricks]), np.array([0]), np.array([[False, False]]))
    boards.save(tmp_path / "boards.npz")
    loaded = BoardSet.load(tmp_path / "boards.npz")
    assert loaded.ids.tolist() == boards.ids.tolist()
    assert loaded.board(0)[0] == deal


def test_pgx_fixture_decodes_and_matches_dds():
    keys = np.array([[24494087, 53250751, 4828063, 22552142]], dtype=np.int32)
    values = np.array([[213810, 694955, 148274, 629419]], dtype=np.int32)
    owners, tables = decode_pgx(keys, values)
    with NativeDdsBackend() as backend:
        np.testing.assert_array_equal(tables[0], backend.solve_table(deal_from_ownership(owners[0])).tricks)


def test_splits_are_disjoint_and_duplicate_boards_rejected(deal):
    owners = np.tile(ownership(deal), (3, 1))
    with pytest.raises(ValueError, match="Duplicate"):
        make_splits(owners, np.zeros((3, 5, 4), dtype=np.uint8), (1, 1, 1), seed=3)


@pytest.mark.parametrize("difference, expected", [(0, 0), (10, 0), (20, 1), (40, 1), (50, 2), (600, 12), (4000, 24), (-600, -12)])
def test_imp_boundaries(difference, expected):
    assert imps(difference) == expected


def _boards(deal):
    with NativeDdsBackend() as backend:
        table = backend.solve_table(deal)
    return BoardSet(np.array([ownership(deal)]), np.array([table.tricks]), np.array([0]), np.array([[False, False]]))


def test_masking_duplicate_rewards_and_ppo_updates_only_current_policy(deal):
    torch.manual_seed(4)
    model = BiddingModel(hidden_size=32)
    opponent = BiddingModel(hidden_size=32)
    before_opponent = {k: v.clone() for k, v in opponent.state_dict().items()}
    batch, summary = collect_duplicate(model, [opponent], _boards(deal), [0], np.random.default_rng(8))
    assert len(batch["actions"]) > 0
    assert all(batch["masks"][i, a] for i, a in enumerate(batch["actions"]))
    np.testing.assert_array_equal(batch["returns"], np.full(len(batch["actions"]), summary["imps"][0] / 24))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metrics = ppo_update(model, optimizer, batch, epochs=1, minibatch_size=32, rng=np.random.default_rng(1))
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(before_opponent[k], v) for k, v in opponent.state_dict().items())


def test_supervised_loss_decreases_and_checkpoint_preserves_predictions(tmp_path):
    torch.manual_seed(3)
    model = BiddingModel(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    observations = np.zeros((16, 480), dtype=np.bool_)
    masks = np.ones((16, 38), dtype=np.bool_)
    labels = np.zeros(16, dtype=np.int64)
    first = supervised_update(model, optimizer, observations, masks, labels)["loss"]
    for _ in range(10):
        last = supervised_update(model, optimizer, observations, masks, labels)["loss"]
    assert last < first
    path = tmp_path / "model.pt"
    save_model(path, model, optimizer=optimizer, metadata={"step": 11})
    loaded, metadata = load_model(path)
    assert metadata["step"] == 11
    with torch.no_grad():
        torch.testing.assert_close(model(torch.from_numpy(observations).float())[0], loaded(torch.from_numpy(observations).float())[0])


def test_dealer_encoding_is_invariant_under_seat_rotation(deal):
    auction = [{"seat": "N", "call": "P"}, {"seat": "E", "call": "1C"},
               {"seat": "S", "call": "P"}, {"seat": "W", "call": "X"}]
    expected = encode480(deal.hands["N"], auction, "N", (True, False))
    for rotation in range(4):
        rotated = [{"seat": "NESW"[("NESW".index(r["seat"]) + rotation) % 4], "call": r["call"]} for r in auction]
        vuln = (False, True) if rotation % 2 else (True, False)
        np.testing.assert_array_equal(expected, encode480(deal.hands["N"], rotated, "NESW"[rotation], vuln))


def test_greedy_duplicate_evaluation_self_ties_and_opponent_swap_reverses(deal):
    torch.manual_seed(9)
    boards = _boards(deal)
    a, b = BiddingModel(32), BiddingModel(32)
    same = evaluate_duplicate(a, a, boards)
    assert same["imps"] == [0.] and same["bootstrap_95_ci"] == [0., 0.]
    ab = evaluate_duplicate(a, b, boards)
    ba = evaluate_duplicate(b, a, boards)
    assert ab["mean_imps"] == -ba["mean_imps"]


def test_league_snapshots_frozen_and_restore_is_exact(tmp_path):
    model = BiddingModel(32)
    pool = OpponentPool(tmp_path, model, capacity=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.actor.bias.add_(1)
    pool.snapshot(model, 1)
    opponents, names = pool.roster(np.random.default_rng(1))
    assert names.count("snapshot-00000001.pt") == 4 and names.count("anchor.pt") == 2
    assert all(not p.requires_grad for opponent in opponents for p in opponent.parameters())
    assert all(torch.equal(before[k], v) for k, v in pool.get("anchor.pt").state_dict().items())
    restored = OpponentPool(tmp_path, history=pool.history, capacity=2)
    assert restored.roster(np.random.default_rng(1))[1] == names


def test_baseline_rejects_unrecognized_pickle_before_loading(tmp_path):
    from bridgemind.training.baseline import load_brl_arrays
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"untrusted data")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_brl_arrays(path)


def test_prepare_supervised_preserves_legal_labels_and_excludes_boards(tmp_path):
    from bridgemind.training.data import prepare_supervised, SupervisedData
    # Round-robin distribution; four passes is a complete legal auction.
    source = tmp_path / "raw.txt"
    source.write_text(" ".join(map(str, [*range(52), 52, 52, 52, 52])) + "\n")
    destination = tmp_path / "sl"
    manifest = prepare_supervised(source, destination)
    assert manifest["states"] == 4
    observations, masks, labels = SupervisedData(destination).sample(50, np.random.default_rng(1))
    assert observations.shape == (50, 480) and all(labels == 0)
    assert masks[np.arange(50), labels].all()
    ids = np.load(destination / "board-ids.npy", allow_pickle=False)
    with pytest.raises(ValueError, match="No supervised boards"):
        prepare_supervised(source, tmp_path / "excluded", excluded_ids=ids)


def test_ppo_runner_resume_matches_uninterrupted_training(deal, tmp_path):
    from bridgemind.training.__main__ import parser
    from bridgemind.training.runner import train_ppo
    from bridgemind.training.data import BoardSet
    torch.set_num_threads(1)
    boards = _boards(deal)
    train_path, valid_path = tmp_path / "train.npz", tmp_path / "valid.npz"
    boards.save(train_path)
    # Swap N/E hands and table columns to create a distinct valid board.
    owners = boards.owners.copy()
    owners[boards.owners == 0], owners[boards.owners == 1] = 1, 0
    BoardSet(owners, boards.tables[:, :, [1, 0, 2, 3]], boards.dealers, boards.vulnerable).save(valid_path)
    common = ["train-ppo", "--train", str(train_path), "--valid", str(valid_path), "--hidden-size", "32",
              "--device", "cpu", "--boards-per-update", "1", "--eval-boards", "1", "--eval-every", "1",
              "--snapshot-every", "1", "--epochs", "1", "--minibatch-size", "16"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train_ppo(parser().parse_args([*common, "--run", str(full), "--updates", "2"]))
    train_ppo(parser().parse_args([*common, "--run", str(resumed), "--updates", "1"]))
    train_ppo(parser().parse_args([*common, "--run", str(resumed), "--updates", "2", "--resume"]))
    a, _ = load_model(full / "last.pt")
    b, _ = load_model(resumed / "last.pt")
    for k, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[k], rtol=0, atol=0)
    changed = parser().parse_args([*common, "--run", str(resumed), "--updates", "3", "--resume", "--lr", "0.001"])
    with pytest.raises(ValueError, match="mismatch"):
        train_ppo(changed)


def test_policy_facade_uses_only_private_observation_and_returns_legal_call(deal, tmp_path):
    from bridgemind.training.policy import BiddingPolicy
    path = tmp_path / "policy.pt"
    save_model(path, BiddingModel(32))
    policy = BiddingPolicy(path)
    env = BridgeBiddingEnv(None, deal, prefetch=False)
    obs = env.observation()
    result = policy.predict(obs)
    assert result["state_id"] == obs["state_id"]
    assert result["call_id"] in obs["legal_call_ids"]
    assert sum(result["probabilities"].values()) == pytest.approx(1.)
    assert "hands" not in obs and set(obs["hand"]) == set(deal.hands["N"])


def test_new_board_generator_checks_solver_failures_and_batch_order():
    from bridgemind.training.offline import generate_tables
    from bridgemind.dds import SolverError
    class Stub:
        def __init__(self, fail=False):
            self.calls, self.fail, self.boards = 0, fail, []
        def solve_tables(self, deals, options):
            self.calls += 1
            self.boards.extend(deals)
            assert options.priority == "offline"
            if self.fail:
                return [{"status": "failed", "error": {"code": "INTERNAL", "message": "solver failed"}}]
            return [{"status": "succeeded", "result": {"tricks": [[self.calls] * 4] * 5}} for _ in deals]
    solver = Stub()
    owners, tables = generate_tables(solver, 3, batch_size=2)
    np.testing.assert_array_equal(tables[:, 0, 0], [1, 1, 2])
    assert all(deal_from_ownership(owners[i]) == solver.boards[i] for i in range(3))
    with pytest.raises(SolverError, match="solver failed"):
        generate_tables(Stub(True), 1)
