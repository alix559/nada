"""Compare the MAX graph with Hugging Face on a tiny CLAP of the same weights.

The bicubic stretch and the log-mel frontend are checked first. The joint
model is then compiled once and compared on text embeddings, audio embeddings,
and both logit matrices.
"""

from __future__ import annotations

import numpy as np

from .audio import bicubic_map
from .config import tiny_config
from .features import log_mel
from .model import build, compile_model
from .weights import assign_parameters


def main() -> None:
    check_bicubic()
    check_mel()
    check_model()
    print("parity ok")


def check_bicubic() -> None:
    import torch
    import torch.nn.functional as torch_f

    source = torch.randn(1, 1, 16, 8)
    reference = torch_f.interpolate(source, (128, 8), mode="bicubic", align_corners=True)
    weight = bicubic_map(16, 128)
    flat = source.numpy().reshape(1, 16, 8).transpose(0, 2, 1)
    stretched = flat @ weight.T
    got = stretched.transpose(0, 2, 1).reshape(1, 1, 128, 8)
    np.testing.assert_allclose(got, reference.numpy(), atol=1e-5, rtol=1e-5)
    print("bicubic ok", float(np.max(np.abs(got - reference.numpy()))))


def check_mel() -> None:
    from transformers import ClapFeatureExtractor

    from .config import AudioConfig

    config = AudioConfig()
    time = np.arange(config.sample_rate, dtype=np.float64) / config.sample_rate
    waveform = np.sin(2 * np.pi * 440 * time)
    extractor = ClapFeatureExtractor(
        feature_size=config.num_mel_bins,
        sampling_rate=config.sample_rate,
        hop_length=config.hop_length,
        max_length_s=config.max_length_s,
        fft_window_size=config.n_fft,
        frequency_min=config.frequency_min,
        frequency_max=config.frequency_max,
        truncation="rand_trunc",
        padding="repeatpad",
    )
    reference = np.asarray(extractor(waveform, sampling_rate=config.sample_rate, return_tensors="np")["input_features"])
    got = log_mel(waveform, config)
    np.testing.assert_allclose(got, reference, atol=1e-4, rtol=1e-4)
    print("mel ok", got.shape, float(np.max(np.abs(got - reference))))


def check_model() -> None:
    import torch
    from max.experimental.tensor import Tensor
    from transformers import ClapAudioConfig, ClapConfig, ClapModel, ClapTextConfig

    config = tiny_config()
    reference = ClapModel(
        ClapConfig(
            text_config=ClapTextConfig(
                vocab_size=config.text.vocab_size,
                hidden_size=config.text.hidden_size,
                num_hidden_layers=config.text.num_hidden_layers,
                num_attention_heads=config.text.num_attention_heads,
                intermediate_size=config.text.intermediate_size,
                max_position_embeddings=config.text.max_position_embeddings,
                type_vocab_size=config.text.type_vocab_size,
                layer_norm_eps=config.text.layer_norm_eps,
                pad_token_id=config.text.pad_token_id,
                projection_dim=config.text.projection_dim,
            ),
            audio_config=ClapAudioConfig(
                spec_size=config.audio.spec_size,
                num_mel_bins=config.audio.num_mel_bins,
                patch_size=config.audio.patch_size,
                patch_stride=(config.audio.patch_stride, config.audio.patch_stride),
                patch_embeds_hidden_size=config.audio.patch_embeds_hidden_size,
                depths=list(config.audio.depths),
                num_attention_heads=list(config.audio.num_attention_heads),
                window_size=config.audio.window_size,
                mlp_ratio=config.audio.mlp_ratio,
                qkv_bias=config.audio.qkv_bias,
                layer_norm_eps=config.audio.layer_norm_eps,
                hidden_size=config.audio.hidden_size,
                projection_dim=config.audio.projection_dim,
                enable_fusion=False,
            ),
            projection_dim=config.text.projection_dim,
        )
    )
    reference.eval()
    model = build(config)
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in reference.state_dict().items()
    }
    assign_parameters(model, {key: value for key, value in arrays.items() if _keep(key)})

    generator = np.random.default_rng(1)
    ids = generator.integers(0, config.text.vocab_size, size=(2, 6), dtype=np.int64)
    ids[:, -1] = config.text.pad_token_id
    mask = np.ones((2, 6), dtype=np.float32)
    mask[:, -1] = 0
    features = generator.standard_normal((2, 1, config.audio.mel_frames, config.audio.num_mel_bins)).astype(np.float32)

    with torch.no_grad():
        output = reference(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(mask),
            input_features=torch.from_numpy(features),
        )
    compiled = compile_model(model, text_batch=2, text_length=6, audio_batch=2)
    text, audio, logits_text, logits_audio = compiled(
        Tensor(np.ascontiguousarray(ids)),
        Tensor(np.ascontiguousarray(mask)),
        Tensor(np.ascontiguousarray(features)),
    )
    pairs = {
        "text": (text.to_numpy(), output.text_embeds.numpy()),
        "audio": (audio.to_numpy(), output.audio_embeds.numpy()),
        "logits_text": (logits_text.to_numpy(), output.logits_per_text.numpy()),
        "logits_audio": (logits_audio.to_numpy(), output.logits_per_audio.numpy()),
    }
    for name, (got, expected) in pairs.items():
        delta = float(np.max(np.abs(np.asarray(got) - expected)))
        print(f"{name} max abs {delta:.3e}")
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def _keep(key: str) -> bool:
    from .weights import usable

    return usable(key)


if __name__ == "__main__":
    main()
