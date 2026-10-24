"""Restricted, pinned import of brl's public JAX checkpoint into PyTorch.

Original network/checkpoint: Haruka Kita and Sotetsu Koyamada, Apache-2.0.
Only the explicit NumPy reconstruction globals below are accepted; JAX code
is not imported or executed. The source Git blob hash is checked first.
"""

import hashlib
import io
from pathlib import Path
import pickle
import urllib.request

import numpy as np
import torch

from .model import BiddingModel, save_model

BRL_COMMIT = "fdd958ff8adcb44844d70ea2e233bed18d5a7b70"
BRL_SL_BLOB = "78cadbe49fa44bda1ec8dd79895f9d84fc418c13"
BRL_URL = f"https://raw.githubusercontent.com/harukaki/brl/{BRL_COMMIT}/bridge_models/model-sl.pkl"


def _jax_array(function, arguments, state, attributes):
    array = function(*arguments)
    array.__setstate__(state)
    return array


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("jax._src.array", "_reconstruct_array"): _jax_array,
                   ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
                   ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
                   ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported checkpoint global: {module}.{name}")
        return allowed[(module, name)]


def load_brl_arrays(path):
    data = Path(path).read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != BRL_SL_BLOB:
        raise ValueError("Public baseline Git blob hash mismatch")
    arrays = _RestrictedUnpickler(io.BytesIO(data)).load()
    expected = [(480, 1024)] + [(1024, 1024)] * 3 + [(1024, 38), (1024, 1)]
    for index, shape in enumerate(expected):
        name = "actor_critic/linear" + (f"_{index}" if index else "")
        if arrays[name]["w"].shape != shape or arrays[name]["b"].shape != (shape[1],):
            raise ValueError("Unexpected baseline shape")
    return arrays


def convert_brl(source, destination):
    arrays = load_brl_arrays(source)
    model = BiddingModel()
    with torch.no_grad():
        for index, layer in enumerate([*model.layers, model.actor, model.critic]):
            name = "actor_critic/linear" + (f"_{index}" if index else "")
            layer.weight.copy_(torch.from_numpy(arrays[name]["w"].T.copy()))
            layer.bias.copy_(torch.from_numpy(arrays[name]["b"].copy()))
    # Validate conversion against a direct NumPy evaluation of original weights.
    x = np.random.default_rng(4).integers(0, 2, (16, 480)).astype(np.float32)
    reference = x
    for index in range(4):
        name = "actor_critic/linear" + (f"_{index}" if index else "")
        reference = np.maximum(reference @ arrays[name]["w"] + arrays[name]["b"], 0)
    expected = reference @ arrays["actor_critic/linear_4"]["w"] + arrays["actor_critic/linear_4"]["b"]
    with torch.no_grad():
        actual = model(torch.from_numpy(x))[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    metadata = {"source": BRL_URL, "source_commit": BRL_COMMIT, "source_git_blob": BRL_SL_BLOB,
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "license": "Apache-2.0", "conversion_max_abs_error": float(np.abs(actual-expected).max()), "kind": "public_sl"}
    save_model(destination, model, metadata=metadata)
    return metadata


def fetch_baseline(destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(BRL_URL, temporary)
        load_brl_arrays(temporary)
        temporary.replace(path)
    load_brl_arrays(path)
    return path
