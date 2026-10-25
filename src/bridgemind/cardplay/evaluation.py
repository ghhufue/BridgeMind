"""Actual 52-card paired play. Fixed deal/auction; no DDS terminal substitution."""

import numpy as np
import torch
import torch.nn.functional as F

from bridgemind.dds.models import SEATS
from bridgemind.training.rollout import imps
from .encoding import collate, encode_observation
from .model import PlayModel
from .policy import act_batch


def supervised_metrics(model, data, *, count=1024, batch_size=64, seed=714):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    policy_loss, belief_loss, value_loss, correct, unknowns = 0., 0., 0., 0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            rows = data.sample(min(batch_size, count-start), rng)
            batch = collate([r["encoded"] for r in rows], device)
            distribution, values, beliefs = model.distribution(batch)
            actions = torch.as_tensor(np.asarray([r["action"] for r in rows]), device=device).long()
            owners = torch.as_tensor(np.asarray([r["owners"] for r in rows]), device=device).long()
            target = torch.as_tensor(np.asarray([r["return"] for r in rows]), device=device).float()
            policy_loss += float(-distribution.log_prob(actions).sum())
            correct += int((distribution.probs.argmax(-1) == actions).sum())
            unknown = batch["unknown"]
            loss = F.cross_entropy(beliefs.transpose(1, 2), owners, reduction="none")
            belief_loss += float((loss*unknown).sum())
            unknowns += int(unknown.sum())
            value_loss += float((values-target).square().sum())
    return {"states": count, "nll": policy_loss/count, "accuracy": correct/count,
            "belief_nll": belief_loss/max(unknowns, 1), "value_mse": value_loss/count,
            "seed": seed, "sampling": "fixed_seed_with_replacement"}


def play_pairs(current, opponents, data, indices, rng, *, collect=False, stochastic=False):
    if not len(indices) or not opponents:
        raise ValueError("Need boards and opponents")
    if collect and not isinstance(current, PlayModel):
        raise ValueError("PPO collection needs a current neural model")
    n = len(indices)
    enemy_ids = rng.integers(0, len(opponents), n)
    envs = [data.environment(index) for _ in range(2) for index in indices]
    if any(e.is_terminal() for e in envs):
        raise ValueError("Cardplay evaluation needs non-passed-out games")
    records = []
    owners = [np.asarray(e.training_owners(), dtype=np.int16) for e in envs] if collect else None
    for _ in range(53):
        active = [i for i, env in enumerate(envs) if not env.is_terminal()]
        if not active:
            break
        groups, observations = {}, {}
        for i in active:
            obs = envs[i].observation()
            observations[i] = obs
            room, local = i // n, i % n
            key = -1 if SEATS.index(obs["player"]) % 2 == room else int(enemy_ids[local])
            groups.setdefault(key, []).append(i)
        for key, members in groups.items():
            actor = current if key == -1 else opponents[key]
            obs = [observations[i] for i in members]
            if isinstance(actor, PlayModel):
                encoded = [encode_observation(o) for o in obs]
                actions, logs, values = act_batch(actor, encoded, rng, stochastic=stochastic)
            else:
                decisions = actor.predict_batch(obs) if hasattr(actor, "predict_batch") else [actor.predict(o) for o in obs]
                actions = [d["card_id"] for d in decisions]
                logs, values = [0.]*len(members), [0.]*len(members)
            for j, i in enumerate(members):
                if key == -1 and collect:
                    declarer = SEATS.index(obs[j]["contract"]["declarer"])
                    records.append({"encoded": encoded[j], "action": int(actions[j]), "logprob": float(logs[j]),
                                    "value": float(values[j]), "owners": ((owners[i]-declarer) % 4).astype(np.uint8),
                                    "board": i % n})
                envs[i].apply_card(int(actions[j]))
    else:
        raise RuntimeError("Cardplay exceeded 52 moves")
    results = [env.result() for env in envs]
    raw = [results[i]["returns"][0] - results[n+i]["returns"][0] for i in range(n)]
    rewards = [imps(difference) for difference in raw]
    own_declarer, enemy_declarer = [], []
    for i in range(n):
        declarer_side = SEATS.index(envs[i].contract.declarer) % 2
        own, enemy = results[i+declarer_side*n], results[i+(1-declarer_side)*n]
        own_declarer.append(own["declarer_tricks"])
        enemy_declarer.append(enemy["declarer_tricks"])
    for record in records:
        record["return"] = rewards[record["board"]] / 24.
    return records, {"boards": n, "imps": rewards, "raw_differences": raw, "mean_imps": float(np.mean(rewards)),
                     "current_as_declarer_tricks": own_declarer, "opponent_as_declarer_tricks": enemy_declarer,
                     "evaluation_model": "played_out", "value_target": "paired_IMP/24"}


def evaluate_pairs(current, opponent, data, *, count=128, seed=714, batch_size=16):
    games = data.load_games()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(games))[:min(count, len(games))]
    if not len(indices):
        raise ValueError("Empty evaluation set")
    if isinstance(current, PlayModel):
        current.eval()
    if isinstance(opponent, PlayModel):
        opponent.eval()
    rewards, raw, own, enemy = [], [], [], []
    for start in range(0, len(indices), batch_size):
        _, summary = play_pairs(current, [opponent], data, indices[start:start+batch_size], rng)
        rewards.extend(summary["imps"]); raw.extend(summary["raw_differences"])
        own.extend(summary["current_as_declarer_tricks"]); enemy.extend(summary["opponent_as_declarer_tricks"])
    values = np.asarray(rewards, np.float64)
    bootstrap = [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
    return {"boards": len(indices), "mean_imps": float(values.mean()),
            "bootstrap_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "standard_error": float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None,
            "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
            "imps": rewards, "raw_differences": raw, "current_as_declarer_tricks": own,
            "opponent_as_declarer_tricks": enemy, "board_ids": [games[i]["id"] for i in indices],
            "seed": seed, "action_selection": "greedy_for_neural", "evaluation_model": "played_out",
            "contract_source": "fixed_recorded_auction"}
