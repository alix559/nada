"""Find the checkout that holds the weights."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Checkout that holds `models/`, `pixi.toml`, and `requirements.txt`."""
    starts = []
    file = globals().get("__file__")
    if file:
        starts.append(Path(file).resolve().parent)
    starts.append(Path.cwd())
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / "pixi.toml").is_file() and (parent / "requirements.txt").is_file():
                return parent
    raise FileNotFoundError("nada checkout (pixi.toml) was not found")
