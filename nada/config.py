"""Shapes for the unfused LAION CLAP checkpoint.

The audio tower is HTSAT-tiny: a Swin transformer on a log-mel spectrogram
that has been folded into a square. The text tower is RoBERTa-base. Both
project to 512 dimensions. These defaults match `laion/clap-htsat-unfused`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextConfig:
    """RoBERTa encoder used as CLAP's text tower."""

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 1
    projection_dim: int = 512


@dataclass(frozen=True)
class AudioConfig:
    """HTSAT-tiny encoder used as CLAP's audio tower.

    `mel_frames` is the time axis of the log-mel the feature extractor
    produces for a 10 second clip at 48 kHz. The tower stretches that axis
    to `spec_size * freq_ratio` before folding the spectrogram into a square.
    """

    spec_size: int = 256
    num_mel_bins: int = 64
    mel_frames: int = 1001
    patch_size: int = 4
    patch_stride: int = 4
    patch_embeds_hidden_size: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_attention_heads: tuple[int, ...] = (4, 8, 16, 32)
    window_size: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-5
    hidden_size: int = 768
    projection_dim: int = 512
    sample_rate: int = 48_000
    n_fft: int = 1024
    hop_length: int = 480
    frequency_min: float = 50.0
    frequency_max: float = 14_000.0
    max_length_s: int = 10

    @property
    def freq_ratio(self) -> int:
        return self.spec_size // self.num_mel_bins

    @property
    def spec_width(self) -> int:
        """Time axis after the bicubic stretch, before the spectrogram is folded."""
        return self.spec_size * self.freq_ratio

    @property
    def num_features(self) -> int:
        """Channel count after the last Swin stage. This is the pooled width."""
        return self.patch_embeds_hidden_size * 2 ** (len(self.depths) - 1)

    @property
    def max_samples(self) -> int:
        return self.max_length_s * self.sample_rate


@dataclass(frozen=True)
class ClapConfig:
    """Joint embedding model. Logit scales start at log(1/0.07), as in CLIP."""

    text: TextConfig = TextConfig()
    audio: AudioConfig = AudioConfig()
    logit_scale_init_value: float = 1.0 / 0.07


def unfused_config() -> ClapConfig:
    """Config of `laion/clap-htsat-unfused` (HTSAT-tiny, no feature fusion)."""
    return ClapConfig()


def tiny_config() -> ClapConfig:
    """A few blocks of the same wiring, for graph checks and parity tests."""
    return ClapConfig(
        text=TextConfig(
            vocab_size=50,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            projection_dim=8,
            layer_norm_eps=1e-12,
        ),
        audio=AudioConfig(
            spec_size=32,
            num_mel_bins=8,
            mel_frames=16,
            patch_embeds_hidden_size=16,
            depths=(2, 2),
            num_attention_heads=(4, 4),
            window_size=4,
            hidden_size=32,
            projection_dim=8,
            layer_norm_eps=1e-5,
        ),
    )
