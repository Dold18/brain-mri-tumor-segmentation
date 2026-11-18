"""2D slice datasets backed by memory-mapped MRI files per patient."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .cache import PreprocessSettings, case_directory, read_case_metadata
from .data import CaseRecord


class MRISliceDataset(Dataset):
    def __init__(self, records: list[CaseRecord], cache_root: str | Path,
                 settings: PreprocessSettings, *, max_open_cases: int = 2,
                 augment: bool = False) -> None:
        if not records:
            raise ValueError("Cannot construct an empty MRI dataset")
        if max_open_cases < 1:
            raise ValueError("max_open_cases must be positive")
        self.root = Path(cache_root)
        self.records = records
        self.max_open_cases = max_open_cases
        self.augment = augment
        self._open_arrays: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.samples: list[tuple[str, int]] = []
        self.positive: list[bool] = []
        self.case_sizes: dict[str, int] = {}
        for record in records:
            meta = read_case_metadata(self.root, record, settings)
            n, c, h, w = meta["image_shape"]
            if meta["mask_shape"] != [n, 1, h, w] or min(h, w) < 16:
                raise ValueError(f"Invalid cached shapes for {record.case_id}")
            if c != settings.modalities:
                raise ValueError(f"Incorrect channel count for {record.case_id}")
            positives = set(meta["foreground_slices"])
            self.case_sizes[record.case_id] = n
            for index in range(n):
                self.samples.append((record.case_id, index))
                self.positive.append(index in positives)

    def __len__(self) -> int:
        return len(self.samples)

    def _case_arrays(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if case_id in self._open_arrays:
            self._open_arrays.move_to_end(case_id)
            return self._open_arrays[case_id]
        folder = case_directory(self.root, case_id)
        arrays = (np.load(folder / "image.npy", mmap_mode="r"),
                  np.load(folder / "mask.npy", mmap_mode="r"))
        self._open_arrays[case_id] = arrays
        if len(self._open_arrays) > self.max_open_cases:
            self._open_arrays.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict:
        case_id, slice_index = self.samples[index]
        images, masks = self._case_arrays(case_id)
        # Copy out of read-only memmaps; avoid modifying the cached patient scan.
        x = torch.from_numpy(np.asarray(images[slice_index], dtype=np.float32).copy())
        y = torch.from_numpy(np.asarray(masks[slice_index], dtype=np.float32).copy())
        if self.augment:
            # Spatial flips are identical for input and target; no interpolation of labels.
            for axis in (-1, -2):
                if bool(torch.randint(0, 2, (1,)).item()):
                    x, y = torch.flip(x, (axis,)), torch.flip(y, (axis,))
        return {"image": x, "mask": y, "case_id": case_id, "slice_index": slice_index}


def balanced_sampler(dataset: MRISliceDataset, *, samples_per_epoch: int,
                     positive_fraction: float, seed: int) -> WeightedRandomSampler:
    if samples_per_epoch < 1 or not 0 < positive_fraction < 1:
        raise ValueError("samples_per_epoch must be positive and positive_fraction in (0, 1)")
    positive_count = sum(dataset.positive)
    negative_count = len(dataset) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Foreground-balanced sampling requires both positive and negative slices")
    weights = [positive_fraction / positive_count if positive else
               (1 - positive_fraction) / negative_count for positive in dataset.positive]
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                 samples_per_epoch, replacement=True, generator=gen)
