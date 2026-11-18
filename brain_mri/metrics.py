"""Confusion-count metrics aggregated over complete patient volumes."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=bool)
        gold = np.asarray(target, dtype=bool)
        if pred.shape != gold.shape:
            raise ValueError("Prediction and ground truth must share a shape")
        self.tp += int(np.count_nonzero(pred & gold))
        self.fp += int(np.count_nonzero(pred & ~gold))
        self.fn += int(np.count_nonzero(~pred & gold))

    def result(self) -> dict[str, float | int]:
        d = 2 * self.tp + self.fp + self.fn
        u = self.tp + self.fp + self.fn
        # Empty ground truth / predictions: perfect only if both truly empty.
        dice = 2 * self.tp / d if d else 1.0
        iou = self.tp / u if u else 1.0
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else (
            1.0 if self.fn == 0 else 0.0)
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else (
            1.0 if self.fp == 0 else 0.0)
        return {"dice": dice, "iou": iou, "precision": precision,
                "recall": recall, "false_positive_voxels": self.fp,
                "true_positive_voxels": self.tp, "false_negative_voxels": self.fn}


def summarize_cases(per_case: dict[str, Confusion]) -> dict:
    if not per_case:
        raise ValueError("No patients supplied to metrics")
    result = {cid: metric.result() for cid, metric in sorted(per_case.items())}
    return {"mean_dice": float(np.mean([x["dice"] for x in result.values()])),
            "mean_iou": float(np.mean([x["iou"] for x in result.values()])),
            "mean_recall": float(np.mean([x["recall"] for x in result.values()])),
            "mean_precision": float(np.mean([x["precision"] for x in result.values()])),
            "cases": result}


@torch.inference_mode()
def volume_metrics(model: torch.nn.Module, loader, device: torch.device, *, threshold: float = 0.5,
                   loss_fn=None, expected_case_sizes: dict[str, int] | None = None):
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one")
    model.eval()
    confusions: dict[str, Confusion] = {}
    counts: dict[str, int] = {}
    loss_total = 0.0
    n_slices = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        if loss_fn is not None:
            loss_total += float(loss_fn(logits, batch["mask"].to(device))) * x.shape[0]
            n_slices += x.shape[0]
        prediction = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        truth = batch["mask"].cpu().numpy()
        for i, case_id in enumerate(batch["case_id"]):
            confusions.setdefault(case_id, Confusion()).add(prediction[i], truth[i])
            counts[case_id] = counts.get(case_id, 0) + 1
    if expected_case_sizes is not None and counts != expected_case_sizes:
        raise ValueError("Incomplete or duplicated validation slices; volume metrics invalid")
    results = summarize_cases(confusions)
    if loss_fn is not None:
        results["validation_loss"] = loss_total / n_slices
    return results
