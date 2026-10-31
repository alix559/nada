"""Wav2vec 2.0 speech recognition on Modular MAX

[Wav2vec 2.0](https://arxiv.org/abs/2006.11477) on [Modular MAX](https://docs.modular.com/max/). This is the base model fine-tuned on 960 hours of LibriSpeech for character-level CTC: a convolutional feature encoder, a convolutional position embedding, and a 12-layer transformer, with a linear head that scores 32 characters for every 20 ms of audio. The weights are the Hugging Face release `facebook/wav2vec2-base-960h`.

The layout follows `solid_lens`. `config` is the architecture, `weights` binds a checkpoint, and `compile_model` traces the module for one batch size and one clip length. The graph is a MAX module (`max.experimental.nn`): the same `forward` runs eagerly or compiled. Module and parameter names match the Hugging Face implementation, so its state dict loads without renaming.

Modules:

- `nada.audio`: Wav2vec 2.0 encoder, as `Wav2Vec2Model` runs it at inference.
- `nada.config`: Shapes for the wav2vec 2.0 base checkpoint.
- `nada.download`: Download `facebook/wav2vec2-base-960h` and write a numpy checkpoint.
- `nada.features`: Waveform preparation and CTC decoding for the wav2vec 2.0 checkpoint.
- `nada.layers`: Building blocks the MAX graph needs that are not in `max.experimental.nn`.
- `nada.model`: Wav2vec 2.0 with a CTC head.
- `nada.parity`: Compare the MAX graph with Hugging Face on a tiny wav2vec 2.0 of the same weights.
- `nada.weights`: Load a wav2vec 2.0 state dict onto the MAX module."""

__version__ = "0.1.0"

from .config import Wav2Vec2Config, base_960h_config, tiny_config
from .model import Wav2Vec2ForCTC, build, compile_model, transcribe

__all__ = [
    "Wav2Vec2Config",
    "Wav2Vec2ForCTC",
    "base_960h_config",
    "build",
    "compile_model",
    "tiny_config",
    "transcribe",
]
