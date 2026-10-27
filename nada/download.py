"""Download `laion/clap-htsat-unfused` and write a numpy checkpoint.

The Hugging Face file is a PyTorch pickle. Conversion needs the `reference`
pixi environment, which has PyTorch. The runtime only needs the safetensors
file and the tokenizer.
"""

from __future__ import annotations

from pathlib import Path

from .core import repo_root
from .weights import usable

REPO = "laion/clap-htsat-unfused"


def download(root: Path | None = None) -> Path:
    """Fetch the tokenizer and config, convert the weights, and return the safetensors path."""
    from huggingface_hub import snapshot_download

    destination = (root or repo_root()) / "models" / "clap-htsat-unfused"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        local_dir=destination,
        allow_patterns=["*.json", "*.txt", "tokenizer.json", "vocab.json", "merges.txt", "pytorch_model.bin"],
    )
    output = (root or repo_root()) / "models" / "clap.safetensors"
    convert(destination / "pytorch_model.bin", output)
    return output


def convert(source: Path, output: Path) -> None:
    import torch
    from safetensors.numpy import save_file

    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(source, map_location="cpu", weights_only=False)
    arrays = {}
    for key, value in state.items():
        if not usable(key):
            continue
        array = value.detach().cpu().numpy()
        arrays[key] = np_contiguous(array)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(arrays, output)
    print(f"wrote {output} ({len(arrays)} tensors)")


def np_contiguous(array):
    import numpy as np

    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
    return np.ascontiguousarray(array)


if __name__ == "__main__":
    download()
