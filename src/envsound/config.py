from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    """Configuration values for training and preprocessing."""

    data_home: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed/mel")
    artifacts_dir: Path = Path("artifacts")
    sample_rate: int = 22_050
    clip_duration: float = 4.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    target_frames: int = 128
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    validation_split: float = 0.2
    augment_prob: float = 0.5
    limit_per_class: Optional[int] = None
    device: str = "auto"
    log_interval: int = 20

    def ensure_dirs(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 将 Path 对象转换为字符串，确保跨平台兼容性
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d
