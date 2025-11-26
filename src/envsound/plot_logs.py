from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training/validation loss & accuracy curves from training_log.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log", type=Path, default=Path("artifacts/training_log.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training_curves.png"))
    return parser.parse_args()


def load_history(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    epochs = data.get("epochs")
    if not epochs:
        raise ValueError("Log file does not contain 'epochs' entries.")
    return epochs


def plot_curves(history: List[dict], output: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_acc"] for entry in history]
    val_acc = [entry["val_acc"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history = load_history(args.log)
    plot_curves(history, args.output)
    print(f"Saved curves to {args.output}")


if __name__ == "__main__":
    main()
