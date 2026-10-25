"""Shared Transformer, declarer/defender policy heads, value and ownership heads."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .encoding import ENCODING_VERSION, META_SIZE, MAX_AUCTION


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    layers: int = 4
    heads: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()) or self.d_model % self.heads:
            raise ValueError("Positive model sizes required; d_model must be divisible by heads")


class PlayModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.d_model
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n in (4, 90, 5, MAX_AUCTION, 3)])
        self.metadata = nn.Linear(META_SIZE, d)
        self.system = nn.Embedding(256, d)
        layer = nn.TransformerEncoderLayer(d, self.config.heads, dim_feedforward=4*d, dropout=0.,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, self.config.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.declarer_head, self.defender_head = nn.Linear(d, 1), nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.belief_head = nn.Linear(d, 4)

    def forward(self, batch):
        tokens = batch["tokens"].long()
        x = sum(embedding(tokens[:, :, i]) for i, embedding in enumerate(self.embeddings))
        summary = self.metadata(batch["meta"].float()) + self.system(batch["system"].long())
        x = x.clone()
        x[:, 0] = x[:, 0] + summary
        x = self.encoder(x, src_key_padding_mask=batch["padding"].bool())
        cards = x[:, 1:53]
        logits = torch.where(batch["declarer"][:, None].bool(), self.declarer_head(cards).squeeze(-1),
                             self.defender_head(cards).squeeze(-1))
        return logits, self.value_head(x[:, 0]).squeeze(-1), self.belief_head(cards)

    def distribution(self, batch):
        if not batch["legal"].any(-1).all():
            raise ValueError("Every nonterminal observation needs a legal card")
        logits, values, beliefs = self(batch)
        distribution = torch.distributions.Categorical(logits=logits.masked_fill(~batch["legal"].bool(), -1e9))
        return distribution, values, beliefs


def save_checkpoint(path, model, *, metadata=None, optimizer=None, rng=None):
    payload = {"format_version": 1, "encoding": ENCODING_VERSION, "config": asdict(model.config),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "metadata": metadata or {}, "torch_rng": torch.get_rng_state()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if rng is not None:
        payload["numpy_rng"] = rng.bit_generator.state
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1 or payload["encoding"] != ENCODING_VERSION:
        raise ValueError("Unsupported cardplay checkpoint")
    return payload


def load_model(path, device="cpu"):
    checkpoint = read_checkpoint(path)
    model = PlayModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint["metadata"]


def restore_training(path, optimizer, rng):
    payload = read_checkpoint(path)
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
