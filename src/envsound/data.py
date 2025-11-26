from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TrainConfig

URBANSOUND_CLASS_NAMES: Sequence[str] = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


@dataclass
class ClipRecord:
    audio_path: Path
    class_id: int
    class_name: str
    fold: int
    slice_file_name: str
    start: float
    end: float


def ensure_dataset(data_home: Path, download: bool = False, validate: bool = False) -> Path:
    """Download (optional) and return the UrbanSound8K root directory."""
    try:
        import soundata
    except ImportError as exc:  # pragma: no cover - informative message
        raise ImportError(
            "soundata is required. Install the project dependencies first."
        ) from exc

    dataset = soundata.initialize("urbansound8k", data_home=str(data_home))
    if download:
        dataset.download()
    if validate:
        dataset.validate()

    base_path = Path(dataset.data_home)
    resolved = resolve_dataset_root(base_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"UrbanSound8K data not found under {base_path}. "
            "Run with --download to fetch it."
        )
    return resolved


def resolve_dataset_root(base_path: Path) -> Path:
    """Best-effort resolution of the actual dataset folder."""
    if (base_path / "audio").exists():
        return base_path
    for candidate in ("urbansound8k", "UrbanSound8K"):
        candidate_path = base_path / candidate
        if (candidate_path / "audio").exists():
            return candidate_path
    return base_path


def load_clip_records(dataset_root: Path, limit_per_class: Optional[int] = None) -> List[ClipRecord]:
    """Parse the UrbanSound8K metadata CSV and return clip records."""
    metadata_csv = dataset_root / "metadata" / "UrbanSound8K.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Could not locate metadata CSV at {metadata_csv}. "
            "Ensure the dataset is downloaded via soundata."
        )

    counts: Dict[int, int] = defaultdict(int)
    records: List[ClipRecord] = []
    with metadata_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row["classID"])
            if limit_per_class is not None and counts[class_id] >= limit_per_class:
                continue
            fold = int(row["fold"])
            audio_rel = Path("audio") / f"fold{fold}" / row["slice_file_name"]
            audio_path = dataset_root / audio_rel
            if not audio_path.exists():
                # Skip missing files but keep training going
                continue
            records.append(
                ClipRecord(
                    audio_path=audio_path,
                    class_id=class_id,
                    class_name=row["class"],
                    fold=fold,
                    slice_file_name=row["slice_file_name"],
                    start=float(row["start"]),
                    end=float(row["end"]),
                )
            )
            counts[class_id] += 1
    if not records:
        raise RuntimeError(
            "No audio records were loaded. Check that the dataset was downloaded correctly."
        )
    return records


def waveform_to_mel(
    y: np.ndarray,
    sr: int,
    config: TrainConfig,
) -> np.ndarray:
    """Convert waveform to a normalized Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.mel_fmin,
        fmax=config.mel_fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = pad_or_trim(mel_db, config.target_frames)
    mel_norm = normalize_spectrogram(mel_db)
    return mel_norm.astype(np.float32)


def pad_or_trim(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad (with minimum value) or trim a spectrogram to target frames."""
    if spec.shape[1] == target_frames:
        return spec
    if spec.shape[1] > target_frames:
        return spec[:, :target_frames]
    pad_amount = target_frames - spec.shape[1]
    pad_values = float(spec.min())
    return np.pad(spec, ((0, 0), (0, pad_amount)), mode="constant", constant_values=pad_values)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Min-max normalize spectrogram."""
    min_val = spec.min()
    max_val = spec.max()
    return (spec - min_val) / (max_val - min_val + 1e-8)


def _cache_path(processed_dir: Path, record: ClipRecord) -> Path:
    digest = hashlib.md5(str(record.audio_path).encode("utf-8")).hexdigest()
    return processed_dir / f"{digest}.npy"


def load_waveform(record: ClipRecord, config: TrainConfig) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(
        record.audio_path,
        sr=config.sample_rate,
        mono=True,
    )
    target_length = int(config.clip_duration * sr)
    if target_length > 0:
        y = librosa.util.fix_length(y, size=target_length)
    return y, sr


def augment_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply random time stretching and pitch shifting."""
    augmented = y.copy()
    if random.random() < 0.5:
        stretch_factor = random.uniform(0.8, 1.2)
        augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
    if random.random() < 0.5:
        n_steps = random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
    target_len = len(y)
    augmented = librosa.util.fix_length(augmented, size=target_len)
    return augmented


class UrbanSoundSpectrogramDataset(Dataset):
    """Torch Dataset returning Mel spectrogram tensors and labels."""

    def __init__(
        self,
        records: Sequence[ClipRecord],
        config: TrainConfig,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.augment = augment
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        mel = self._load_mel(record)
        tensor = torch.from_numpy(mel).unsqueeze(0)
        label = torch.tensor(record.class_id, dtype=torch.long)
        return tensor, label

    def _load_mel(self, record: ClipRecord) -> np.ndarray:
        cache_file = _cache_path(self.config.processed_dir, record)
        use_cache = not self.augment
        if use_cache and cache_file.exists():
            return np.load(cache_file)

        waveform, sr = load_waveform(record, self.config)
        if self.augment and random.random() < self.config.augment_prob:
            waveform = augment_waveform(waveform, sr)
        mel = waveform_to_mel(waveform, sr, self.config)

        if use_cache:
            np.save(cache_file, mel)
        return mel


def split_records(
    records: Sequence[ClipRecord],
    validation_split: float,
    seed: int = 42,
) -> Tuple[List[ClipRecord], List[ClipRecord]]:
    """Split records into train/validation lists while preserving class balance."""
    from sklearn.model_selection import train_test_split

    indices = list(range(len(records)))
    labels = [record.class_id for record in records]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_split,
        random_state=seed,
        stratify=labels,
    )
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    return train_records, val_records


def class_id_to_name(records: Iterable[ClipRecord]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for record in records:
        mapping.setdefault(record.class_id, record.class_name)
    return mapping
