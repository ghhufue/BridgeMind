"""PyTorch equivalent of brl's DeepMind MLP; portable tensor checkpoints."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .encoding import ENCODING_VERSION


class BiddingModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([nn.Linear(480 if i == 0 else hidden_size, hidden_size) for i in range(4)])
        self.actor = nn.Linear(hidden_size, 38)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        x = observations.float()
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.actor(x), self.critic(x).squeeze(-1)

    def distribution(self, observations, masks):
        if masks.shape[-1] != 38 or not torch.all(masks.any(dim=-1)):
            raise ValueError("Every nonterminal state needs a legal action")
        logits, value = self(observations)
        return torch.distributions.Categorical(logits=logits.masked_fill(~masks.bool(), -1e9)), value


def save_model(path, model, *, optimizer=None, metadata=None, rng=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "encoding": ENCODING_VERSION, "hidden_size": model.hidden_size,
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "metadata": metadata or {}, "torch_rng": torch.get_rng_state()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if rng is not None:
        payload["numpy_rng"] = rng.bit_generator.state
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1 or payload["encoding"] != ENCODING_VERSION:
        raise ValueError("Unsupported checkpoint")
    return payload


def load_model(path, device="cpu"):
    payload = read_checkpoint(path)
    model = BiddingModel(payload["hidden_size"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload["metadata"]


def restore_training(path, optimizer, rng):
    payload = read_checkpoint(path)
    if "optimizer" not in payload or "numpy_rng" not in payload:
        raise ValueError("Checkpoint has no resumable training state")
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"])
    if "cuda_rng" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["metadata"]


def act_batch(model, observations, masks, rng, stochastic=True):
    device = next(model.parameters()).device
    with torch.no_grad():
        distribution, values = model.distribution(torch.as_tensor(np.asarray(observations), device=device),
                                                  torch.as_tensor(np.asarray(masks), device=device))
        probabilities = distribution.probs.cpu().numpy()
        if stochastic:
            actions = np.array([rng.choice(38, p=p.astype(np.float64) / p.sum(dtype=np.float64)) for p in probabilities])
        else:
            actions = probabilities.argmax(axis=-1)
        logs = distribution.log_prob(torch.as_tensor(actions, device=device)).cpu().numpy()
        return actions, logs, values.cpu().numpy()
