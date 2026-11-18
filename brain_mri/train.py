"""Reproducible PyTorch training with immutable splits and patient-level validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .cache import PreprocessSettings
from .data import read_catalog
from .datasets import MRISliceDataset, balanced_sampler
from .losses import DiceBCELoss
from .metrics import volume_metrics
from .model import UNet2D
from .splits import validate_splits
from .storage_paths import DATASET_DIR, SPLIT_MANIFEST, CACHE_DIR, BASELINE_RUN_DIR, require_ssd_if_targeted


HISTORY_HEADER = ("epoch", "training_loss", "validation_loss", "validation_mean_dice",
                  "validation_mean_iou", "validation_mean_precision", "validation_mean_recall")


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training YAML must define a mapping")
    try:
        tr = config["training"]
        assert tr["optimizer"] == "AdamW" and tr["loss"] == "DiceBCELoss"
        assert int(tr["epochs"]) > 0 and int(tr["batch_size"]) > 0
        assert int(tr["samples_per_epoch"]) > 0 and int(tr["validation_every"]) > 0
        assert int(tr["patience_checks"]) > 0 and int(tr["num_workers"]) >= 0
        assert float(tr["learning_rate"]) > 0 and float(tr["weight_decay"]) >= 0
        assert 0 < float(tr["positive_fraction"]) < 1
        assert 0 < float(tr["threshold"]) < 1
        assert config["model"]["name"] == "UNet2D"
        assert int(config["model"]["base_channels"]) > 0
        assert int(config["dataset"]["modalities"]) > 0
        assert isinstance(config["seed"], int)
        PreprocessSettings(float(config["preprocessing"]["clip_low_percentile"]),
                           float(config["preprocessing"]["clip_high_percentile"]),
                           int(config["dataset"]["modalities"]))
    except (KeyError, AssertionError, TypeError, ValueError) as exc:
        raise ValueError("Invalid training configuration: check numeric and model settings") from exc
    return config


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if requested == "mps" and not (getattr(torch.backends, "mps", None)
                                   and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown training device: {requested}")
    return torch.device(requested)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def run_training(*, config_path: str | Path, dataset_dir: str | Path,
                 split_manifest: str | Path, cache_dir: str | Path,
                 output_dir: str | Path, device: str = "auto", resume: bool = False,
                 max_train_patients: int | None = None,
                 max_validation_patients: int | None = None) -> dict:
    """Train on cached MRI data. A subset run is always marked exploratory."""
    for path in (dataset_dir, split_manifest, cache_dir, output_dir):
        require_ssd_if_targeted(path)
    config = load_config(config_path)
    _seed(config["seed"])
    selected_device = select_device(device)
    catalog = read_catalog(dataset_dir)
    split_doc = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    all_ids = {record.case_id for record in catalog.cases}
    validate_splits(split_doc, all_ids)
    case_lookup = {r.case_id: r for r in catalog.cases}
    train_ids = list(split_doc["train"])
    val_ids = list(split_doc["validation"])
    if max_train_patients is not None:
        if max_train_patients < 1:
            raise ValueError("max_train_patients must be positive")
        train_ids = train_ids[:max_train_patients]
    if max_validation_patients is not None:
        if max_validation_patients < 1:
            raise ValueError("max_validation_patients must be positive")
        val_ids = val_ids[:max_validation_patients]
    exploratory = (len(train_ids) < len(split_doc["train"]) or
                   len(val_ids) < len(split_doc["validation"]))
    settings = PreprocessSettings(
        float(config["preprocessing"]["clip_low_percentile"]),
        float(config["preprocessing"]["clip_high_percentile"]),
        int(config["dataset"]["modalities"]),
    )
    train_set = MRISliceDataset([case_lookup[cid] for cid in train_ids], cache_dir,
                                settings, augment=True)
    val_set = MRISliceDataset([case_lookup[cid] for cid in val_ids], cache_dir, settings)
    tr = config["training"]
    batch_size = int(tr["batch_size"])
    num_workers = int(tr["num_workers"])
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=selected_device.type == "cuda")

    model = UNet2D(settings.modalities, int(config["model"]["base_channels"])).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["learning_rate"]),
                                   weight_decay=float(tr["weight_decay"]))
    loss_fn = DiceBCELoss()
    out = Path(output_dir)
    latest = out / "latest.pt"
    best_path = out / "best.pt"
    history_path = out / "history.csv"
    config_for_hash = json.loads(json.dumps(config))
    del config_for_hash["training"]["epochs"]
    provenance = {
        "config_hash": _hash(config_for_hash),
        "split_hash": _hash(split_doc),
        "train_ids_hash": _hash(train_ids),
        "validation_ids_hash": _hash(val_ids),
        "exploratory_subset": exploratory,
    }
    start_epoch = 1
    best_dice = -1.0
    patience = 0
    if resume:
        if not latest.is_file() or not history_path.is_file():
            raise FileNotFoundError("Resume requested but latest.pt/history.csv is missing")
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state.get("provenance") != provenance:
            raise ValueError("Cannot resume: model, configuration, or patient splits changed")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_dice = float(state["best_dice"])
        patience = int(state["patience"])
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != state["epoch"] or int(rows[-1]["epoch"]) != state["epoch"]:
            raise ValueError("History does not align with checkpoint; refusing unsafe resume")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Run folder {out} is not empty; choose a new folder or --resume")
        out.mkdir(parents=True, exist_ok=True)
        with history_path.open("x", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(HISTORY_HEADER)
        (out / "run_metadata.json").write_text(json.dumps({
            "provenance": provenance,
            "train_patients": len(train_ids), "validation_patients": len(val_ids),
            "training_slices": len(train_set), "validation_slices": len(val_set),
            "device": str(selected_device), "configuration": config,
            "model": "Brain MRI Tumor Segmentation custom UNet2D, binary whole-tumor-region segmentation",
            "note": "Exploratory subset: not a full baseline" if exploratory else
                    "Training/validation only; held-out test untouched",
        }, indent=2) + "\n", encoding="utf-8")
    if start_epoch > int(tr["epochs"]):
        raise ValueError("Checkpoint already completed the configured epoch count")

    print(f"Training on {selected_device}: {len(train_ids)} training / "
          f"{len(val_ids)} validation patients; exploratory_subset={exploratory}", flush=True)
    last_summary = {}
    for epoch in range(start_epoch, int(tr["epochs"]) + 1):
        _seed(config["seed"] + epoch)
        sampler = balanced_sampler(
            train_set, samples_per_epoch=int(tr["samples_per_epoch"]),
            positive_fraction=float(tr["positive_fraction"]), seed=config["seed"] + epoch,
        )
        loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=selected_device.type == "cuda")
        model.train()
        training_loss_total = 0.0
        training_samples = 0
        for batch in loader:
            x = batch["image"].to(selected_device, non_blocking=True)
            y = batch["mask"].to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            training_samples += x.shape[0]
            training_loss_total += float(loss.detach()) * x.shape[0]
        avg_training_loss = training_loss_total / training_samples
        # For compact initial runs, perform complete volume-level validation every N epochs.
        should_validate = (epoch % int(tr["validation_every"]) == 0 or
                           epoch == int(tr["epochs"]))
        last_summary = {}
        if should_validate:
            scores = volume_metrics(model, val_loader, selected_device,
                                    threshold=float(tr["threshold"]), loss_fn=loss_fn,
                                    expected_case_sizes=val_set.case_sizes)
            last_summary = {key: value for key, value in scores.items() if key != "cases"}
            score = float(scores["mean_dice"])
            improved = score > best_dice
            if improved:
                best_dice = score
                patience = 0
            else:
                patience += 1
            print(f"epoch {epoch} train_loss={avg_training_loss:.5f} "
                  f"val_loss={scores['validation_loss']:.5f} val_patient_dice={score:.5f}",
                  flush=True)
        else:
            improved = False
            print(f"epoch {epoch} train_loss={avg_training_loss:.5f} (validation skipped)",
                  flush=True)

        row = [epoch, avg_training_loss]
        for key in ("validation_loss", "mean_dice", "mean_iou", "mean_precision", "mean_recall"):
            row.append(last_summary.get(key, ""))
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        checkpoint = {
            "schema": 1, "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "best_dice": best_dice,
            "patience": patience, "provenance": provenance,
            "model_settings": {"in_channels": settings.modalities,
                               "base_channels": int(config["model"]["base_channels"])},
            "preprocessing": {
                "clip_low_percentile": settings.clip_low_percentile,
                "clip_high_percentile": settings.clip_high_percentile,
            },
            "threshold": float(tr["threshold"]),
        }
        if improved:
            _save_checkpoint(best_path, checkpoint)
        _save_checkpoint(latest, checkpoint)
        if should_validate and patience >= int(tr["patience_checks"]):
            print(f"Early stopping after {patience} validation checks without improvement")
            break
    return {"best_validation_dice": best_dice, "exploratory_subset": exploratory,
            "epochs_completed": epoch, "latest_checkpoint": str(latest),
            "best_checkpoint": str(best_path), "last_validation": last_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--splits", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_RUN_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-patients", type=int,
                        help="EXPLORATORY ONLY: restrict training to N cases")
    parser.add_argument("--max-validation-patients", type=int,
                        help="EXPLORATORY ONLY: restrict validation to N cases")
    args = parser.parse_args()
    results = run_training(config_path=args.config, dataset_dir=args.dataset_dir,
                           split_manifest=args.splits, cache_dir=args.cache_dir,
                           output_dir=args.output_dir, device=args.device, resume=args.resume,
                           max_train_patients=args.max_train_patients,
                           max_validation_patients=args.max_validation_patients)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
