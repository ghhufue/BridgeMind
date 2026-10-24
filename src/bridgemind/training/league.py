"""Immutable snapshots; active history is bounded, saved files remain resumable."""

import copy
from pathlib import Path

from .model import load_model, save_model


def frozen_cpu(model):
    result = copy.deepcopy(model).cpu().eval()
    result.requires_grad_(False)
    return result


class OpponentPool:
    def __init__(self, directory, anchor=None, *, capacity=8, history=None):
        if capacity < 2:
            raise ValueError("Opponent capacity must be at least two")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity, self.cache = capacity, {}
        self.history = list(history or [])
        if anchor is not None:
            if (self.directory / "anchor.pt").exists():
                raise ValueError("Opponent pool already initialized")
            save_model(self.directory / "anchor.pt", anchor, metadata={"kind": "frozen_anchor"})
            self.snapshot(anchor, 0)
        if not self.history or any(not (self.directory / name).is_file() for name in ["anchor.pt", *self.history]):
            raise ValueError("Incomplete opponent pool")

    def snapshot(self, model, update):
        name = f"snapshot-{update:08d}.pt"
        if (self.directory / name).exists():
            raise ValueError("Refusing to overwrite immutable opponent snapshot")
        save_model(self.directory / name, model, metadata={"update": update, "kind": "frozen_snapshot"})
        self.history = (self.history + [name])[-self.capacity:]
        self.cache = {key: value for key, value in self.cache.items() if key in ["anchor.pt", *self.history]}

    def get(self, name):
        if name not in self.cache:
            model, _ = load_model(self.directory / name)
            model.eval().requires_grad_(False)
            self.cache[name] = model
        return self.cache[name]

    def roster(self, rng):
        previous = self.history[:-1] or self.history
        selected = str(rng.choice(previous))
        # Uniform selection of these eight slots gives a 50/25/25 mixture.
        names = [self.history[-1]] * 4 + [selected] * 2 + ["anchor.pt"] * 2
        return [self.get(name) for name in names], names
