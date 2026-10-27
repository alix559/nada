"""Load a CLAP state dict onto the MAX module.

Checkpoint keys follow the Hugging Face `ClapModel` names. Three buffers are
not parameters of the graph: the RoBERTa position-id and token-type buffers,
which the forward pass recomputes, and BatchNorm's step counter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max.dtype import DType
from max.experimental.tensor import Tensor

_NUMPY = {
    DType.float32: np.float32,
    DType.float64: np.float64,
    DType.int64: np.int64,
    DType.int32: np.int32,
}


def usable(key: str) -> bool:
    """True when `key` is a tensor the MAX module stores."""
    if key.endswith("num_batches_tracked"):
        return False
    if key.endswith("embeddings.position_ids"):
        return False
    if key.endswith("embeddings.token_type_ids"):
        return False
    return True


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read a `.safetensors` or `.npz` checkpoint, dropping unused buffers."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        arrays = load_file(path)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
    else:
        raise ValueError(f"{path.name} is not a .safetensors or .npz checkpoint")
    return {key: value for key, value in arrays.items() if usable(key)}


def assign_parameters(module, arrays: dict[str, np.ndarray]) -> None:
    """Copy `arrays` into `module`. Names and shapes must match."""
    wanted = dict(module.parameters)
    missing = [name for name in wanted if name not in arrays]
    extra = [name for name in arrays if name not in wanted]
    if missing or extra:
        sample = (missing or extra)[:6]
        raise KeyError(
            f"checkpoint does not match the module "
            f"({len(missing)} missing, {len(extra)} extra, first {sample})"
        )
    cast = {name: _fit(arrays[name], tensor) for name, tensor in wanted.items()}
    module.load_state_dict(cast, strict=True)


def _fit(array: np.ndarray, tensor: Tensor) -> np.ndarray:
    """Cast to the parameter dtype and reshape rank-0 scales stored as length 1."""
    expected = tuple(int(dim) for dim in tensor.shape)
    array = np.ascontiguousarray(array, dtype=_NUMPY[tensor.dtype])
    if array.shape != expected:
        if array.size != int(np.prod(expected, dtype=int, initial=1)):
            raise ValueError(f"shape {array.shape} cannot fill {expected}")
        array = np.ascontiguousarray(array.reshape(expected))
    return array

