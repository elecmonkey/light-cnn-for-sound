from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainConfig
from .data import (
    ClipRecord,
    UrbanSoundSpectrogramDataset,
    class_id_to_name,
    ensure_dataset,
    load_clip_records,
    split_records,
)
from .model import EnvSoundCNN


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train a CNN on UrbanSound8K using Soundata-managed data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-home", type=Path, default=defaults.data_home)
    parser.add_argument("--processed-dir", type=Path, default=defaults.processed_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--validation-split", type=float, default=defaults.validation_split)
    parser.add_argument("--limit-per-class", type=int, default=-1)
    parser.add_argument("--augment-prob", type=float, default=defaults.augment_prob)
    parser.add_argument("--clip-duration", type=float, default=defaults.clip_duration)
    parser.add_argument("--target-frames", type=int, default=defaults.target_frames)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--download", action="store_true", help="Download the dataset via soundata.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run soundata's dataset validation (slower, but useful after download).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    limit = None if args.limit_per_class is None or args.limit_per_class < 0 else args.limit_per_class
    config = TrainConfig(
        data_home=args.data_home,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        augment_prob=args.augment_prob,
        limit_per_class=limit,
        seed=args.seed,
        clip_duration=args.clip_duration,
        target_frames=args.target_frames,
        device=args.device,
    )
    config.ensure_dirs()
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def create_dataloaders(
    train_records: Sequence[ClipRecord],
    val_records: Sequence[ClipRecord],
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = UrbanSoundSpectrogramDataset(train_records, config, augment=True)
    val_dataset = UrbanSoundSpectrogramDataset(val_records, config, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)
    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    preds_all: List[int] = []
    targets_all: List[int] = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Val", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
            preds_all.extend(preds.cpu().tolist())
            targets_all.extend(targets.cpu().tolist())
    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, preds_all, targets_all


def save_history(history: List[Dict[str, float]], path: Path) -> None:
    data = {"epochs": history}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_conf_matrix(
    preds: Sequence[int],
    labels: Sequence[int],
    class_names: Dict[int, str],
    output_path: Path,
) -> None:
    classes_sorted = sorted(class_names.items(), key=lambda item: item[0])
    ordered_labels = [label for label, _ in classes_sorted]
    names = [name for _, name in classes_sorted]
    cm = confusion_matrix(labels, preds, labels=ordered_labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=True)
    ax.set_title("Validation Confusion Matrix (normalized)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)
    device = get_device(config.device)

    dataset_root = ensure_dataset(config.data_home, download=args.download, validate=args.validate)
    records = load_clip_records(dataset_root, limit_per_class=config.limit_per_class)
    class_names = class_id_to_name(records)
    train_records, val_records = split_records(records, config.validation_split, config.seed)
    train_loader, val_loader = create_dataloaders(train_records, val_records, config)

    num_classes = len({record.class_id for record in records})
    model = EnvSoundCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_acc = 0.0
    history: List[Dict[str, float]] = []
    best_preds: List[int] = []
    best_labels: List[int] = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:02d}/{config.num_epochs} "
            f"- train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} "
            f"- train_acc: {train_acc:.3f} val_acc: {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_preds = preds
            best_labels = labels
            checkpoint_path = config.artifacts_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "class_mapping": class_names,
                    "config": config.asdict(),
                },
                checkpoint_path,
            )
            print(f"Saved new best model to {checkpoint_path} (acc={best_acc:.3f})")

    save_history(history, config.artifacts_dir / "training_log.json")
    if best_preds and best_labels:
        plot_conf_matrix(
            preds=best_preds,
            labels=best_labels,
            class_names=class_names,
            output_path=config.artifacts_dir / "confusion_matrix.png",
        )
    print("Training complete. Artifacts written to", config.artifacts_dir)


if __name__ == "__main__":
    main()
