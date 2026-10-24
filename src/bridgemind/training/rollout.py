"""Paired duplicate boards. Only current-policy actions enter PPO's buffer."""

import numpy as np

from bridgemind.engine import BridgeBiddingEnv
from .encoding import encode_env
from .model import act_batch

# WBF/ACBL IMP lower bounds: first nonzero IMP starts at 20 raw points.
IMP_THRESHOLDS = np.array([20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600,
                           750, 900, 1100, 1300, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000])


def imps(difference):
    difference = int(difference)
    return int(np.sign(difference)) * int(np.searchsorted(IMP_THRESHOLDS, abs(difference), side="right"))


def collect_duplicate(policy, opponents, boards, indices, rng, *, stochastic=True, collect=True):
    if not opponents or not len(indices):
        raise ValueError("Need opponents and boards")
    opponent_ids = rng.integers(0, len(opponents), len(indices))
    envs, assignments = [], []
    for room in (0, 1):
        for local, index in enumerate(indices):
            deal, table, dealer, vulnerable = boards.board(index)
            envs.append(BridgeBiddingEnv(None, deal, dealer=dealer, vulnerable=vulnerable, table=table))
            assignments.append((room, local))
    records = []
    for turn in range(320):
        active = [i for i, env in enumerate(envs) if not env.is_terminal()]
        if not active:
            break
        groups = {}
        encoded = {}
        for i in active:
            room, local = assignments[i]
            player = envs[i].observation()["player"]
            is_current = "NESW".index(player) % 2 == room
            key = -1 if is_current else int(opponent_ids[local])
            groups.setdefault(key, []).append(i)
            encoded[i] = encode_env(envs[i])
        for key, members in groups.items():
            xs, masks = zip(*(encoded[i] for i in members))
            actor = policy if key == -1 else opponents[key]
            actions, logs, values = act_batch(actor, xs, masks, rng, stochastic)
            for j, i in enumerate(members):
                if key == -1 and collect:
                    records.append((encoded[i][0], encoded[i][1], int(actions[j]), float(logs[j]), float(values[j]), assignments[i][1]))
                envs[i].apply_call(int(actions[j]))
    else:
        raise RuntimeError("Auction exceeded the legal turn bound")
    scores = [env.result()["returns"][0] for env in envs]
    n = len(indices)
    differences = [scores[i] - scores[n+i] for i in range(n)]
    rewards = np.array([imps(d) for d in differences], dtype=np.float32)
    summary = {"imps": rewards.tolist(), "raw_differences": differences,
               "mean_imps": float(rewards.mean()), "boards": n,
               "standard_error": float(rewards.std(ddof=1) / np.sqrt(n)) if n > 1 else None}
    if not collect:
        return None, summary
    if not records:
        raise RuntimeError("No current-policy decisions collected")
    xs, masks, actions, logs, values, board_index = zip(*records)
    batch = {"observations": np.array(xs, dtype=np.bool_), "masks": np.array(masks, dtype=np.bool_),
             "actions": np.array(actions, np.int64), "logprobs": np.array(logs, np.float32),
             "values": np.array(values, np.float32), "returns": rewards[np.array(board_index)] / 24}
    return batch, summary
