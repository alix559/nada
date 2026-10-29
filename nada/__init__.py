"""LAION CLAP, the unfused HTSAT and RoBERTa model, on Modular MAX

LAION [CLAP](https://github.com/LAION-AI/CLAP) on [Modular MAX](https://docs.modular.com/max/). This is the unfused model: an HTSAT-tiny audio tower and a RoBERTa text tower, projected into one embedding. The weights are the Hugging Face release `laion/clap-htsat-unfused`, which is the `630k-audioset-best` checkpoint. Feature fusion, the four-crop path for long audio, is not in this graph.

The layout follows `solid_lens`. `config` is the architecture, `weights` binds a checkpoint, and `compile_model` traces the module for one text length and the 10-second mel shape. The graph is a MAX module (`max.experimental.nn`): the same `forward` runs eagerly or compiled.

Modules:

- `nada.audio`: HTSAT audio tower, as the unfused `ClapAudioModel` runs it.
- `nada.config`: Shapes for the unfused LAION CLAP checkpoint.
- `nada.download`: Download `laion/clap-htsat-unfused` and write a numpy checkpoint.
- `nada.features`: Log-mel frontend for the unfused CLAP checkpoint.
- `nada.model`: Joint CLAP embedding.
- `nada.parity`: Compare the MAX graph with Hugging Face on a tiny CLAP of the same weights.
- `nada.text`: RoBERTa text tower, as `ClapTextModel` runs it at inference.
- `nada.weights`: Load a CLAP state dict onto the MAX module."""

__version__ = "0.1.0"

from .config import AudioConfig, ClapConfig, TextConfig, tiny_config, unfused_config
from .model import ClapModel, build, compile_model, contrastive_loss

__all__ = [
    "AudioConfig",
    "ClapConfig",
    "ClapModel",
    "TextConfig",
    "build",
    "compile_model",
    "contrastive_loss",
    "tiny_config",
    "unfused_config",
]
