"""Environment sound classification system built on Soundata + UrbanSound8K."""

from .config import TrainConfig
from .model import EnvSoundCNN

__all__ = ["TrainConfig", "EnvSoundCNN"]
