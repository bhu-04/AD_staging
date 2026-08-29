"""
Module 2 - MRI Preprocessing
==============================
Implements Algorithm 1 ("MRI Preprocessing") from the project document,
adapted to the actual input format.

Algorithm 1 as written assumes a raw NIfTI volume and performs
skull-stripping -> intensity normalization -> registration -> resampling.
The Kaggle release used here ships already skull-stripped, already
registered 2D axial slices (this is standard for OASIS-derived slice
datasets), so steps 1 and 3 of Algorithm 1 are pre-applied upstream by the
dataset publisher. What remains -- and what this module actually
implements -- is:

    1. Grayscale conversion (source JPEGs are RGB with the value
       replicated across channels; single-channel is the correct
       representation for MRI intensity).
    2. Intensity normalization to zero-mean / unit-variance
       (Algorithm 1, step 2), matching the doc's stated normalization
       target exactly.
    3. Resampling to a fixed target grid, i.e. resizing to a fixed
       (H, W) (Algorithm 1, step 4).

This keeps the module's input/output contract identical to what Algorithm 1
promises (raw slice in, standardized fixed-shape tensor out) even though
two of its four processing steps are inherited rather than re-implemented.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import config


def load_slice_as_array(filepath: str) -> np.ndarray:
    """Load a raw slice image and convert to single-channel float32 array."""
    img = Image.open(filepath).convert("L")  # grayscale
    return np.asarray(img, dtype=np.float32)


def resize_array(arr: np.ndarray, target_size=config.IMAGE_SIZE) -> np.ndarray:
    """Resample to the fixed target grid (Algorithm 1, step 4)."""
    img = Image.fromarray(arr)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # PIL wants (W, H)
    return np.asarray(img, dtype=np.float32)


def normalize_intensity(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean / unit-variance normalization (Algorithm 1, step 2)."""
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / (std + eps)


def preprocess_slice(filepath: str, target_size=config.IMAGE_SIZE) -> np.ndarray:
    """Full Algorithm-1-equivalent pipeline for one slice.

    Returns a standardized (H, W) float32 array, zero-mean/unit-variance,
    at the fixed target shape -- the V_std referenced in Algorithm 1.
    """
    arr = load_slice_as_array(filepath)
    arr = resize_array(arr, target_size)
    arr = normalize_intensity(arr)
    return arr


def quality_check(std_arr: np.ndarray, target_size=config.IMAGE_SIZE) -> bool:
    """Algorithm 1's 'Evaluation' step: verify output shape and that
    normalization actually produced a near-zero mean / near-unit variance
    array (rather than silently passing through a corrupt/blank image)."""
    shape_ok = std_arr.shape == tuple(target_size)
    stats_ok = abs(std_arr.mean()) < 1e-2 and 0.9 < std_arr.std() < 1.1
    return bool(shape_ok and stats_ok)


if __name__ == "__main__":
    from . import data_acquisition

    manifest = data_acquisition.build_manifest()
    sample_path = manifest.iloc[0]["filepath"]
    std = preprocess_slice(sample_path)
    print(f"Sample: {sample_path}")
    print(f"Output shape: {std.shape}, mean: {std.mean():.4f}, std: {std.std():.4f}")
    print(f"Quality check passed: {quality_check(std)}")
