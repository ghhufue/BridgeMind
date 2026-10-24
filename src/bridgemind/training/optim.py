import numpy as np
import torch


def supervised_update(model, optimizer, observations, masks, labels):
    device = next(model.parameters()).device
    x, mask = torch.as_tensor(observations, device=device), torch.as_tensor(masks, device=device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    if not torch.all(mask.gather(1, labels[:, None])):
        raise ValueError("Supervised label is illegal")
    distribution, _ = model.distribution(x, mask)
    loss = -distribution.log_prob(labels).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss.item(), "accuracy": (distribution.probs.argmax(-1) == labels).float().mean().item(), "grad_norm": float(norm)}


def ppo_update(model, optimizer, batch, *, epochs=4, minibatch_size=1024, rng=None, clip=.2, entropy_coef=.001, value_coef=.5, target_kl=.03):
    rng = rng or np.random.default_rng()
    device = next(model.parameters()).device
    tensors = {key: torch.as_tensor(value, device=device) for key, value in batch.items()}
    advantages = tensors["returns"] - tensors["values"]
    if not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite advantages")
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    totals, updates = {}, 0
    stopped = False
    for _ in range(epochs):
        indices = rng.permutation(len(advantages))
        for start in range(0, len(indices), minibatch_size):
            ix = torch.as_tensor(indices[start:start + minibatch_size], device=device)
            distribution, values = model.distribution(tensors["observations"][ix], tensors["masks"][ix])
            logprob = distribution.log_prob(tensors["actions"][ix].long())
            logratio = logprob - tensors["logprobs"][ix]
            ratio = logratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[ix], ratio.clamp(1-clip, 1+clip) * advantages[ix]).mean()
            value_loss = .5 * (values - tensors["returns"][ix]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            metrics = {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                       "entropy": entropy.item(), "approx_kl": ((ratio-1)-logratio).mean().item(),
                       "clip_fraction": ((ratio-1).abs() > clip).float().mean().item(), "grad_norm": float(norm)}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value
            updates += 1
            if metrics["approx_kl"] > target_kl:
                stopped = True
                break
        if stopped:
            break
    return {**{k: v / updates for k, v in totals.items()}, "minibatches": updates,
            "kl_early_stop": float(stopped), "transitions": len(advantages)}
