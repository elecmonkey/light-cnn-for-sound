from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import TrainConfig
from .data import waveform_to_mel
from .model import EnvSoundCNN

PATH_FIELDS = {"data_home", "processed_dir", "artifacts_dir"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained EnvSoundCNN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True, help="Path to a WAV/OGG audio clip.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save-mel", type=Path, help="Optional path to save the Mel spectrogram image.")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return torch.load(path, map_location=device)


def config_from_checkpoint(data: Dict) -> TrainConfig:
    cfg_dict = data.get("config", {})
    base = TrainConfig()
    for key, value in cfg_dict.items():
        if not hasattr(base, key):
            continue
        if key in PATH_FIELDS and isinstance(value, str):
            setattr(base, key, Path(value))
        else:
            setattr(base, key, value)
    return base


def prepare_input_tensor(audio_path: Path, config: TrainConfig) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    target_length = int(config.clip_duration * sr)
    if target_length > 0:
        y = librosa.util.fix_length(y, target_length)
    mel = waveform_to_mel(y, sr, config)
    return mel


def plot_mel(spec: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    img = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Input Mel Spectrogram")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def format_predictions(
    probs: torch.Tensor,
    class_mapping: Dict[int, str],
    top_k: int,
) -> Sequence[str]:
    num_classes = probs.shape[1]
    k = min(top_k, num_classes)
    values, indices = torch.topk(probs, k=k, dim=1)
    formatted = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        label = class_mapping.get(idx, f"class_{idx}")
        formatted.append(f"{label}: {score * 100:.2f}%")
    return formatted


def get_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            raise RuntimeError("MPS requested but not available. Requires PyTorch built with MPS support.")
        return torch.device("mps")

    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = config_from_checkpoint(checkpoint)

    num_classes = checkpoint.get("num_classes")
    if num_classes is None:
        raise RuntimeError("Checkpoint missing 'num_classes'. Re-train the model to create a compatible checkpoint.")
    class_mapping = checkpoint.get("class_mapping", {i: f"class_{i}" for i in range(num_classes)})

    model = EnvSoundCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    mel = prepare_input_tensor(args.audio, config)
    if args.save_mel:
        plot_mel(mel, args.save_mel)

    tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

    formatted = format_predictions(probs, class_mapping, args.top_k)
    print(f"Predictions for {args.audio}:")
    for line in formatted:
        print(" -", line)


if __name__ == "__main__":
    main()
