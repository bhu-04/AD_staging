"""
Volumetric Data Loader (Data Management Layer)
================================================
Wraps the manifest + preprocessing + augmentation modules into a standard
PyTorch Dataset/DataLoader pair, matching the "Volumetric Data Loader"
component named in the project's architecture table (here loading 2D
standardized slices rather than 3D volumes, per the dataset-format note in
data_acquisition.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from . import config, preprocessing


class ADStagingDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, filepaths: list, transform=None):
        self.transform = transform
        subset = manifest[manifest["filepath"].isin(filepaths)].reset_index(drop=True)
        # Preserve filepath order as given (already shuffled upstream by the splitter)
        order = {p: i for i, p in enumerate(filepaths)}
        subset["_order"] = subset["filepath"].map(order)
        self.manifest = subset.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        std_arr = preprocessing.preprocess_slice(row["filepath"])
        if self.transform:
            x = self.transform(std_arr)
        else:
            x = torch.from_numpy(std_arr).unsqueeze(0).float()
        y = int(row["class_idx"])
        return x, y


def make_balanced_sampler(labels: list) -> WeightedRandomSampler:
    """Per-sample inverse-frequency weights for WeightedRandomSampler, so
    the TRAIN loader draws roughly equal numbers of CN/MCI/AD per epoch
    instead of relying on loss-weighting alone. Loss-weighting penalizes
    mistakes on rare classes more, but with a weak/undertrained model it's
    often not enough to stop it converging on "predict the class that's
    safest on average" -- oversampling fixes this at the batch-composition
    level instead, which is a much stronger signal."""
    labels = np.asarray(labels)
    class_counts = np.bincount(labels, minlength=len(config.CLASS_NAMES))
    class_counts = np.clip(class_counts, 1, None)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(manifest: pd.DataFrame, split_result: dict,
                       train_transform, eval_transform,
                       batch_size: int = config.BATCH_SIZE,
                       balance_train: bool = True) -> dict:
    splits = split_result["splits"]
    datasets = {
        "train": ADStagingDataset(manifest, splits["train"], transform=train_transform),
        "val": ADStagingDataset(manifest, splits["val"], transform=eval_transform),
        "test": ADStagingDataset(manifest, splits["test"], transform=eval_transform),
    }

    if balance_train:
        train_labels = datasets["train"].manifest["class_idx"].tolist()
        train_sampler = make_balanced_sampler(train_labels)
        train_loader = DataLoader(datasets["train"], batch_size=batch_size,
                                   sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size,
                                   shuffle=True, num_workers=0)

    loaders = {
        "train": train_loader,
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    return loaders


if __name__ == "__main__":
    from . import data_acquisition, splitting, augmentation

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )
    for name, loader in loaders.items():
        x, y = next(iter(loader))
        print(f"{name}: {len(loader.dataset)} samples, batch x={tuple(x.shape)}, y={tuple(y.shape)}")