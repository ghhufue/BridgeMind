import copy
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bridgemind.dds import Deal, NativeDdsBackend
from bridgemind.dds.models import SEATS
from bridgemind.engine import BridgeBiddingEnv, BridgePlayEnv, Contract
from bridgemind.engine.play import card_id, card_name
from bridgemind.cardplay.encoding import encode_observation, collate
from bridgemind.cardplay.model import PlayModel, ModelConfig, save_checkpoint, load_model
from bridgemind.cardplay.policy import NeuralPlayPolicy
from bridgemind.cardplay.sampling import DealSampler
from bridgemind.cardplay.data import PlayData, prepare_dataset
from bridgemind.training.data import ownership, board_ids


def _env(deal, **kwargs):
    return BridgePlayEnv(deal, contract=Contract(3, "NT", "N"), **kwargs)


def _public(obs):
    return {k: v for k, v in obs.items() if k != "state_id"}


@pytest.mark.parametrize("declarer,strain,double", [("N", "NT", 0), ("E", "S", 1), ("S", "H", 2), ("W", "D", 0), ("N", "C", 0)])
def test_full_play_tracks_rules_controllers_tricks_and_scores(deal, declarer, strain, double):
    env = BridgePlayEnv(deal, contract=Contract(2, strain, declarer, double), dealer="W", vulnerable=(True, False))
    rng = random.Random(40)
    for step in range(52):
        obs = env.observation()
        expected = declarer if SEATS.index(obs["to_play"]) % 2 == SEATS.index(declarer) % 2 else obs["to_play"]
        assert obs["player"] == expected
        assert obs["dummy_visible"] == (step > 0)
        assert sum(obs["remaining_counts"].values()) == 52-step
        assert obs["to_play"] in obs["visible_hands"]
        assert len(obs["visible_hands"]) == (1 if step == 0 else 2)
        env.apply_card(rng.choice(obs["legal_card_ids"]))
    result = env.result()
    assert result["evaluation_model"] == "played_out" and sum(result["tricks"]) == 13
    assert env.is_terminal() and result["returns"][0] == -result["returns"][1]
    with pytest.raises(ValueError):
        env.observation()


def test_bidding_handoff_and_passed_out_board(deal):
    bidding = BridgeBiddingEnv(None, deal, prefetch=False)
    with pytest.raises(ValueError):
        bidding.start_play()
    for c in ("1C", "P", "1H", "P", "2H", "P", "P", "P"):
        bidding.apply_call(c)
    env = bidding.start_play()
    assert env.contract == Contract(2, "H", "S")
    assert env.observation()["auction"][-1] == {"seat": "W", "call": "P"}
    passed = BridgePlayEnv(deal, auction=["P"]*4)
    assert passed.is_terminal() and passed.result()["returns"] == [0]*4


def test_private_observations_do_not_change_when_hidden_cards_swap(deal):
    a = _env(deal)
    hands = {s: list(h) for s, h in deal.hands.items()}
    hands["N"], hands["W"] = hands["W"], hands["N"]
    b = _env(Deal(hands))
    assert _public(a.observation()) == _public(b.observation())
    # After the opening lead N controls dummy, so E/W are the hidden seats.
    lead = a.legal_card_ids()[0]
    a.apply_card(lead)
    hands = {s: list(h) for s, h in deal.hands.items()}
    east = next(c for c in hands["E"] if card_id(c) != lead)
    west = hands["W"][0]
    hands["E"].remove(east); hands["E"].append(west)
    hands["W"].remove(west); hands["W"].append(east)
    b = _env(Deal(hands)); b.apply_card(lead)
    assert _public(a.observation()) == _public(b.observation())
    assert "hands" not in a.observation()
    model = PlayModel(ModelConfig(32, 1, 4)).eval()
    policy = NeuralPlayPolicy(model=model)
    pa = policy.predict_batch([a.observation()], return_beliefs=True)[0]
    pb = policy.predict_batch([b.observation()], return_beliefs=True)[0]
    np.testing.assert_allclose(pa["probabilities"], pb["probabilities"], rtol=0, atol=0)


def test_illegal_card_is_atomic_and_follow_suit_enforced(deal):
    env = _env(deal)
    env.apply_card(env.legal_card_ids()[0])
    obs = env.observation()
    illegal = next(i for i in range(52) if i not in obs["legal_card_ids"])
    with pytest.raises(ValueError):
        env.apply_card(illegal)
    assert env.observation() == obs
    suit = obs["current_trick"][0]["card"][0]
    following = [card_id(c) for c in obs["visible_hands"][obs["to_play"]] if c[0] == suit]
    if following:
        assert sorted(following) == obs["legal_card_ids"]


def test_sampler_preserves_capacities_known_hands_and_public_voids(deal):
    env = _env(deal)
    rng = np.random.default_rng(7)
    for step in range(48):
        obs = env.observation()
        if step in (0, 1, 17, 40, 47):
            positions = DealSampler(obs).sample(12, rng)
            for position in positions:
                assert position.to_play == obs["to_play"]
                assert set(position.legal_cards()) == {card_name(i) for i in obs["legal_card_ids"]}
                for seat in SEATS:
                    assert len(position.remaining_hands[seat]) == obs["remaining_counts"][seat]
                    assert all(c[0] not in obs["voids"][seat] for c in position.remaining_hands[seat])
                for seat, hand in obs["visible_hands"].items():
                    assert set(position.remaining_hands[seat]) == set(hand)
        env.apply_card(int(rng.choice(obs["legal_card_ids"])))


def test_sampler_conditions_learned_weights_and_rejects_impossible_constraints(deal):
    env = _env(deal)
    obs = env.observation()
    weights = np.zeros((52, 4))
    owners = env.training_owners()
    weights[np.arange(52), owners] = 1
    position = DealSampler(obs, weights).sample(1, np.random.default_rng(2))[0]
    assert position.remaining_hands == deal.hands
    with pytest.raises(ValueError, match="No hidden deal"):
        DealSampler(obs, np.zeros((52, 4)))
    broken = copy.deepcopy(obs)
    broken["remaining_counts"]["N"] -= 1
    with pytest.raises(ValueError):
        DealSampler(broken)


def test_dynamic_tokens_include_all_passes_padding_does_not_change_predictions(deal):
    torch.manual_seed(6)
    short = _env(deal)
    long = BridgePlayEnv(deal, auction=["1C", "P", "1H", "P", "1NT", "P", "3NT", "P", "P", "P"])
    a, b = encode_observation(short.observation()), encode_observation(long.observation())
    assert len(a["tokens"]) == 53+4 and len(b["tokens"]) == 53+10
    assert sum(a["tokens"][:, 0] == 2) == 4  # Three Pass tokens are retained.
    model = PlayModel(ModelConfig(32, 1, 4)).eval()
    with torch.no_grad():
        alone = model(collate([a]))[0][0]
        padded = model(collate([a, b]))[0][0]
    torch.testing.assert_close(alone, padded, atol=2e-6, rtol=2e-6)


def test_play_checkpoint_and_policy_mask(tmp_path, deal):
    torch.manual_seed(9)
    model = PlayModel(ModelConfig(32, 1, 4)).eval()
    path = tmp_path / "play.pt"
    save_checkpoint(path, model, metadata={"step": 1})
    loaded, metadata = load_model(path)
    assert metadata["step"] == 1
    obs = _env(deal).observation()
    a = NeuralPlayPolicy(model=model).predict_batch([obs], return_beliefs=True)[0]
    b = NeuralPlayPolicy(model=loaded).predict_batch([obs], return_beliefs=True)[0]
    assert a == b and a["card_id"] in obs["legal_card_ids"]
    assert sum(a["probabilities"]) == pytest.approx(1)
    assert all(a["probabilities"][i] == 0 for i in range(52) if i not in obs["legal_card_ids"])


def _raw_trajectory(deal, seed=9):
    from bridgemind.engine.bidding import CALLS
    from bridgemind.engine.play import contract_auction
    auction = contract_auction(Contract(3, "NT", "N"))
    cards = []
    for i in range(13):
        cards.extend(card_id(deal.hands[s][i]) for s in SEATS)
    cards.extend(52+CALLS.index(call) for call in auction)
    env, rng = _env(deal), np.random.default_rng(seed)
    while not env.is_terminal():
        action = int(rng.choice(env.legal_card_ids()))
        cards.append(action); env.apply_card(action)
    return " ".join(map(str, cards))+"\n"


@pytest.fixture
def play_datasets(tmp_path, deal):
    hands = {s: list(h) for s, h in deal.hands.items()}
    hands["N"][0], hands["E"][0] = hands["E"][0], hands["N"][0]
    alternate = Deal(hands)
    paths = []
    for name, board in (("train", deal), ("valid", alternate)):
        source = tmp_path/f"{name}.txt"
        source.write_text(_raw_trajectory(board), encoding="utf-8")
        output = tmp_path/name
        prepare_dataset(source, output, allowed_ids=board_ids(np.array([ownership(board)])), max_boards=1, shard_states=20)
        paths.append(output)
    return paths


def test_play_data_ragged_roundtrip_privileged_labels_and_no_future_prefix(play_datasets):
    data = PlayData(play_datasets[0])
    assert data.manifest["states"] == 52 and data.manifest["boards"] == 1
    for index in (0, 1, 21, 51):
        row = data.row(index)
        obs = data.environment(row["game"], row["ply"]).observation()
        assert row["action"] in obs["legal_card_ids"]
        assert (row["encoded"]["tokens"][:, 0] == 3).sum() == index
        for key, value in encode_observation(obs).items():
            np.testing.assert_array_equal(row["encoded"][key], value)
        assert "owners" not in collate([row["encoded"]])
        if index == 51:
            assert not row["encoded"]["unknown"].any()
    assert len(data.sample(8, np.random.default_rng(1), min_ply=50)) == 8
    for index in (-1, 52, 1.5):
        with pytest.raises(ValueError, match="State index"):
            data.row(index)
    for game, ply in ((-1, 0), (1, 0), (0, -1), (0, 53)):
        with pytest.raises(ValueError):
            data.environment(game, ply)
    assert data.environment(0, 52).is_terminal()


def test_supervised_play_optimizer_learns_and_handles_zero_unknown_cards(play_datasets):
    from bridgemind.cardplay.optim import supervised_update
    torch.set_num_threads(1); torch.manual_seed(12)
    data = PlayData(play_datasets[0])
    model = PlayModel(ModelConfig(32, 1, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    rows = [data.row(i) for i in (0, 1, 2, 3, 51)]
    first = supervised_update(model, optimizer, rows)["loss"]
    for _ in range(15):
        last = supervised_update(model, optimizer, rows)["loss"]
    assert last < first
    metrics = supervised_update(model, optimizer, [data.row(51)])
    assert metrics["belief_loss"] == 0 and np.isfinite(metrics["loss"])


def test_distillation_rejects_signed_or_nonfinite_targets(play_datasets):
    from bridgemind.cardplay.optim import supervised_update
    data = PlayData(play_datasets[0])
    row = next(data.row(i) for i in range(52) if data.row(i)["encoded"]["legal"].sum() > 1)
    model = PlayModel(ModelConfig(32, 1, 4))
    optimizer = torch.optim.Adam(model.parameters())
    legal = np.flatnonzero(row["encoded"]["legal"])
    for values in ((1.1, -.1), (np.nan, 0), (np.inf, 0)):
        target = np.zeros((1, 52), np.float32)
        target[0, legal[:2]] = values
        with pytest.raises(ValueError, match="Invalid legal distillation"):
            supervised_update(model, optimizer, [row], targets=target)


def test_supervised_evaluation_uses_requested_seed(play_datasets, tmp_path):
    import json
    from bridgemind.cardplay.__main__ import parser, evaluate
    path = tmp_path / "model.pt"
    save_checkpoint(path, PlayModel(ModelConfig(32, 1, 4)))
    output = tmp_path / "evaluation.json"
    args = parser().parse_args(["evaluate", "--data", str(play_datasets[0]), "--model", str(path),
                               "--supervised", "--states", "8", "--seed", "123", "--output", str(output),
                               "--device", "cpu"])
    evaluate(args)
    assert json.loads(output.read_text())["seed"] == 123


def test_actual_duplicate_self_tie_swap_symmetry_and_on_policy_only(play_datasets):
    from bridgemind.cardplay.evaluation import evaluate_pairs, play_pairs
    from bridgemind.cardplay.optim import ppo_update
    torch.set_num_threads(1); torch.manual_seed(13)
    data = PlayData(play_datasets[0])
    a, b = PlayModel(ModelConfig(32, 1, 4)), PlayModel(ModelConfig(32, 1, 4))
    a.eval(); b.eval()
    same = evaluate_pairs(a, a, data, count=1)
    assert same["imps"] == [0] and same["evaluation_model"] == "played_out"
    assert evaluate_pairs(a, b, data, count=1)["mean_imps"] == -evaluate_pairs(b, a, data, count=1)["mean_imps"]
    old_enemy = {k: v.clone() for k, v in b.state_dict().items()}
    records, summary = play_pairs(a, [b], data, [0], np.random.default_rng(5), collect=True, stochastic=True)
    assert len(records) == 52  # Each room contributes 26 current-team plays.
    assert all(row["return"] == summary["imps"][0]/24 for row in records)
    before = {k: v.clone() for k, v in a.state_dict().items()}
    optimizer = torch.optim.Adam(a.parameters(), lr=.001)
    metrics = ppo_update(a, optimizer, records, np.random.default_rng(7), epochs=1, minibatch_size=32)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(before[k], v) for k, v in a.state_dict().items())
    assert all(torch.equal(old_enemy[k], v) for k, v in b.state_dict().items())


@pytest.fixture
def play_solver_port():
    import threading
    from bridgemind.dds import SolverService, ServiceConfig
    from bridgemind.dds.server import SolverServer
    with SolverService(ServiceConfig(worker_count=2)) as service, SolverServer(service, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("command", ["train-sl", "train-ppo", "distill"])
def test_play_training_resume_matches_continuous_run(play_datasets, tmp_path, command, request):
    from bridgemind.cardplay.__main__ import parser
    from bridgemind.cardplay.runner import train_supervised, train_ppo
    train, valid = play_datasets
    common = [command, "--train", str(train), "--valid", str(valid), "--device", "cpu", "--d-model", "32",
              "--layers", "1", "--heads", "4", "--eval-every", "1"]
    if command in ("train-sl", "distill"):
        common += ["--batch-size", "8", "--eval-states", "16"]
        if command == "distill":
            common += ["--port", str(request.getfixturevalue("play_solver_port")), "--samples", "2", "--min-ply", "40"]
        target, run = "--steps", train_supervised
    else:
        common += ["--boards-per-update", "1", "--epochs", "1", "--minibatch-size", "32",
                   "--eval-boards", "1", "--snapshot-every", "1"]
        target, run = "--updates", train_ppo
    full, resumed = tmp_path/"full", tmp_path/"resumed"
    run(parser().parse_args([*common, "--run", str(full), target, "2"]))
    run(parser().parse_args([*common, "--run", str(resumed), target, "1"]))
    run(parser().parse_args([*common, "--run", str(resumed), target, "2", "--resume"]))
    a, _ = load_model(full/"last.pt"); b, _ = load_model(resumed/"last.pt")
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
    changed = parser().parse_args([*common, "--run", str(resumed), target, "3", "--resume", "--lr", ".003"])
    with pytest.raises(ValueError, match="mismatch"):
        run(changed)


@pytest.mark.parametrize("seed,plies,vulnerable", [(33, 40, (False, False)), (1, 40, (True, False)),
                                                   (3, 41, (True, True))])
def test_search_real_rpc_and_score_perspective(deal, seed, plies, vulnerable):
    import threading
    from bridgemind.dds import SolverService, ServiceConfig, SolverClient
    from bridgemind.dds.server import SolverServer
    from bridgemind.engine.scoring import score_contract
    from bridgemind.cardplay.search import SamplingDdsPolicy
    env, rng = _env(deal, vulnerable=vulnerable), np.random.default_rng(seed)
    for _ in range(plies):
        env.apply_card(int(rng.choice(env.legal_card_ids())))
    obs = env.observation()
    assert len(obs["legal_card_ids"]) > 1
    with SolverService(ServiceConfig(worker_count=2)) as service, SolverServer(service, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with SolverClient(port=server.server_address[1]) as client:
                result = SamplingDdsPolicy(client, samples=4, seed=27).predict(obs)
            samples = DealSampler(obs).sample(4, np.random.default_rng(27))
            expected = {card_name(i): [] for i in obs["legal_card_ids"]}
            with NativeDdsBackend() as backend:
                for position in samples:
                    solved = backend.solve_position(position)
                    declarer_side = SEATS.index(env.contract.declarer) % 2
                    actor_side = SEATS.index(obs["to_play"]) % 2
                    remaining = 13-sum(obs["tricks"])
                    for move in solved.moves:
                        future = move.tricks_for_evaluated_side
                        tricks = obs["tricks"][declarer_side]+(future if actor_side == declarer_side else remaining-future)
                        score = score_contract(env.contract, tricks, vulnerable[declarer_side])
                        expected[move.card].append(score if actor_side == declarer_side else -score)
            assert result["mean_team_scores"] == {card: float(np.mean(values)) for card, values in expected.items()}
            assert result["card_id"] in obs["legal_card_ids"]
        finally:
            server.shutdown(); thread.join(timeout=5)
