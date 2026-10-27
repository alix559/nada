"""LAION CLAP on Modular MAX.

The unfused model is an HTSAT audio tower and a RoBERTa text tower, both
projected into one embedding. `laion/clap-htsat-unfused` supplies the weights.
"""

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
