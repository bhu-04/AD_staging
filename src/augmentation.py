"""
Module 3 - Data Augmentation
==============================
Randomized flips, rotations, and intensity jitter "within anatomically
valid bounds", as specified in the project document's module table.

Bounds chosen deliberately conservative for brain MRI:
  - Horizontal flip only (left-right anatomical symmetry is a reasonable
    approximation for axial brain slices; vertical flip is NOT used since
    it would flip superior/inferior anatomy in an invalid way).
  - Small rotation (+/-10 degrees) to simulate minor head-tilt variation
    without warping the scan into an unrealistic pose.
  - Small intensity jitter (brightness/contrast) to simulate scanner/
    acquisition variability without altering tissue-contrast relationships
    enough to change what a class label should be.

Implemented as a torchvision-transform-compatible callable operating on the
standardized (H, W) float32 numpy array produced by `preprocessing.py`, so
it drops directly into the Dataset's __getitem__.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W) float32 array -> (1, H, W) float32 tensor."""
    return torch.from_numpy(arr).unsqueeze(0).float()


def get_train_transform(rotation_degrees: float = 10.0,
                         jitter_strength: float = 0.1):
    """Augmentations applied only to the training split."""
    return transforms.Compose([
        _to_tensor,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(brightness=jitter_strength, contrast=jitter_strength),
    ])


def get_eval_transform():
    """No augmentation for val/test -- deterministic evaluation."""
    return transforms.Compose([_to_tensor])


if __name__ == "__main__":
    from . import data_acquisition, preprocessing

    manifest = data_acquisition.build_manifest()
    sample_path = manifest.iloc[0]["filepath"]
    std_arr = preprocessing.preprocess_slice(sample_path)

    train_tf = get_train_transform()
    eval_tf = get_eval_transform()

    out_train = train_tf(std_arr)
    out_eval = eval_tf(std_arr)
    print(f"Train-augmented tensor shape: {tuple(out_train.shape)}, "
          f"dtype: {out_train.dtype}")
    print(f"Eval tensor shape: {tuple(out_eval.shape)}, dtype: {out_eval.dtype}")
