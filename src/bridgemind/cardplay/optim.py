"""Supervised imitation / belief learning, search distillation and PPO updates."""

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import collate


def belief_loss(logits, owners, unknown):
    losses = F.cross_entropy(logits.transpose(1, 2), owners.long(), reduction="none")
    return (losses * unknown).sum() / unknown.sum().clamp_min(1)


def supervised_update(model, optimizer, rows, *, belief_coef=.2, value_coef=.1, targets=None):
    device = next(model.parameters()).device
    batch = collate([r["encoded"] for r in rows], device)
    labels = torch.as_tensor(np.asarray([r["action"] for r in rows]), device=device).long()
    owners = torch.as_tensor(np.asarray([r["owners"] for r in rows]), device=device).long()
    returns = torch.as_tensor(np.asarray([r["return"] for r in rows]), device=device).float()
    distribution, values, beliefs = model.distribution(batch)
    if not batch["legal"].gather(1, labels[:, None]).all():
        raise ValueError("Illegal supervised label")
    if targets is None:
        policy = -distribution.log_prob(labels).mean()
    else:
        target = torch.as_tensor(np.asarray(targets), device=device).float()
        if (target.shape != batch["legal"].shape or not torch.isfinite(target).all() or (target < 0).any()
                or (target[~batch["legal"]] != 0).any()
                or not torch.allclose(target.sum(-1), torch.ones(len(rows), device=device))):
            raise ValueError("Invalid legal distillation distribution")
        policy = -(target * distribution.logits).sum(-1).mean()
    belief = belief_loss(beliefs, owners, batch["unknown"])
    value = F.mse_loss(values, returns)
    loss = policy + belief_coef * belief + value_coef * value
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "policy_loss": float(policy.detach()), "belief_loss": float(belief.detach()),
            "value_loss": float(value.detach()), "accuracy": float((distribution.probs.argmax(-1) == labels).float().mean()),
            "grad_norm": float(norm)}


def ppo_update(model, optimizer, records, rng, *, epochs=4, minibatch_size=128, entropy_coef=.001,
               belief_coef=.05, clip=.2, target_kl=.03):
    device = next(model.parameters()).device
    returns = np.asarray([r["return"] for r in records], np.float32)
    advantages = returns - np.asarray([r["value"] for r in records], np.float32)
    advantages = (advantages-advantages.mean()) / max(float(advantages.std()), 1e-8)
    totals, updates, stopped = {}, 0, False
    for _ in range(epochs):
        order = rng.permutation(len(records))
        for start in range(0, len(order), minibatch_size):
            indices = order[start:start+minibatch_size]
            rows = [records[i] for i in indices]
            batch = collate([r["encoded"] for r in rows], device)
            distribution, values, beliefs = model.distribution(batch)
            actions = torch.as_tensor([r["action"] for r in rows], device=device).long()
            old_logs = torch.as_tensor([r["logprob"] for r in rows], device=device)
            advantage = torch.as_tensor(advantages[indices], device=device)
            target = torch.as_tensor(returns[indices], device=device)
            logs = distribution.log_prob(actions)
            ratio = (logs-old_logs).exp()
            policy = -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean()
            value = .5 * (values-target).square().mean()
            entropy = distribution.entropy().mean()
            owners = torch.as_tensor(np.asarray([r["owners"] for r in rows]), device=device)
            belief = belief_loss(beliefs, owners, batch["unknown"])
            loss = policy + .5*value - entropy_coef*entropy + belief_coef*belief
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            kl = ((ratio-1)-(logs-old_logs)).mean()
            metrics = {"loss": float(loss.detach()), "policy_loss": float(policy.detach()), "value_loss": float(value.detach()),
                       "belief_loss": float(belief.detach()), "entropy": float(entropy.detach()),
                       "approx_kl": float(kl.detach()), "grad_norm": float(norm),
                       "clip_fraction": float(((ratio-1).abs() > clip).float().mean())}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value
            updates += 1
            if metrics["approx_kl"] > target_kl:
                stopped = True
                break
        if stopped:
            break
    return {**{key: value/updates for key, value in totals.items()}, "minibatches": updates,
            "transitions": len(records), "kl_early_stop": stopped}
