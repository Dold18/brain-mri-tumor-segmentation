"""Render training and validation learning curves from authentic training CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from brain_mri.storage_paths import BASELINE_RUN_DIR, require_ssd_if_targeted


def create_plots(history: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(history).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training history contains no epochs")
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["training_loss"]) for row in rows]
    validation = [(int(r["epoch"]), float(r["validation_loss"]),
                   float(r["validation_mean_dice"])) for r in rows
                  if r["validation_loss"] and r["validation_mean_dice"]]
    if not validation:
        raise ValueError("No validation metrics available to plot")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    loss_path = output / "loss_curves.png"
    dice_path = output / "validation_dice.png"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, train_loss, label="Training (balanced slice sampler)", linewidth=2)
    ax.plot([r[0] for r in validation], [r[1] for r in validation],
            label="Validation (all patient slices)", marker="o")
    ax.set(xlabel="Epoch", ylabel="Dice + BCE loss", title="Brain MRI Tumor Segmentation training history")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot([r[0] for r in validation], [r[2] for r in validation],
            marker="o", linewidth=2)
    ax.set(xlabel="Epoch", ylabel="Mean patient-level Dice",
           title="Brain MRI Tumor Segmentation validation segmentation quality", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dice_path, dpi=160)
    plt.close(fig)
    return loss_path, dice_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=BASELINE_RUN_DIR / "history.csv")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_RUN_DIR / "figures")
    args = parser.parse_args()
    for path in (args.history, args.output_dir):
        require_ssd_if_targeted(path)
    paths = create_plots(args.history, args.output_dir)
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
