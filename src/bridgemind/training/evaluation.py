"""Frozen-opponent duplicate evaluation and held-out supervised metrics."""

import numpy as np
import torch

from .rollout import collect_duplicate


def supervised_metrics(model, data, *, count=8192, batch_size=1024, seed=714):
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    loss, correct = 0., 0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            xs, masks, labels = data.sample(size, rng)
            distribution, _ = model.distribution(torch.as_tensor(xs, device=device), torch.as_tensor(masks, device=device))
            targets = torch.as_tensor(labels, device=device)
            loss += float(-distribution.log_prob(targets).sum())
            correct += int((distribution.probs.argmax(-1) == targets).sum())
    return {"nll": loss / count, "accuracy": correct / count, "states": count,
            "sampling": "fixed_seed_with_replacement", "seed": seed}


def evaluate_duplicate(policy, opponent, boards, *, count=None, seed=714, batch_size=64):
    count = len(boards) if count is None else min(count, len(boards))
    if count < 1:
        raise ValueError("Evaluation needs at least one board")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(boards))[:count]
    results = []
    for start in range(0, count, batch_size):
        _, summary = collect_duplicate(policy, [opponent], boards, indices[start:start+batch_size], rng,
                                       stochastic=False, collect=False)
        results.extend(summary["imps"])
    values = np.array(results, dtype=np.float64)
    # Resample complete paired boards, never individual seat actions or rooms.
    bootstrap = np.empty(2000)
    for i in range(len(bootstrap)):
        bootstrap[i] = rng.choice(values, count, replace=True).mean()
    return {"boards": count, "mean_imps": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else None,
            "bootstrap_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
            "imps": results, "board_ids": [boards.ids[i].decode() for i in indices],
            "seed": seed, "action_selection": "greedy", "evaluation_model": "double_dummy"}
