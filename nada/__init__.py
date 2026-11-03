"""Audio-JEPA audio embeddings on Modular MAX"""

__version__ = "0.1.0"

from .config import AudioJEPAConfig, pretrained_config, tiny_config
from .model import AudioJEPA, build, compile_model, embed

__all__ = [
    "AudioJEPA",
    "AudioJEPAConfig",
    "build",
    "compile_model",
    "embed",
    "pretrained_config",
    "tiny_config",
]
