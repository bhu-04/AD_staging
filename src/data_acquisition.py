"""
Module 1 - Dataset Acquisition
===============================
Implements the "Dataset Acquisition" module from the project document,
adapted to the actual Kaggle release format.

The project doc specifies this module as retrieving raw NIfTI volumes from
OASIS/ADNI repositories. The Kaggle dataset actually referenced
(ninadaithal/imagesoasis) instead ships as already-extracted 2D axial JPEG
slices, one file per (subject, MRI session, scan run, slice index), grouped
into class folders. This module's job is therefore reframed as: walk the
local mirror of that release and produce a clean manifest -- the same
downstream contract (subject_id, cohort, label, volume_path) the rest of
the pipeline expects, regardless of whether the source was NIfTI volumes
or JPEG slices.

Filename convention observed in the dataset, e.g.:
    OAS1_0003_MR1_mpr-2_129.jpg
      OAS1_0003  -> subject_id
      MR1        -> session
      mpr-2      -> scan run
      129        -> slice index
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"(?P<subject_id>OAS\d+_\d+)_(?P<session>MR\d+)_(?P<scan>mpr-\d+)_(?P<slice_idx>\d+)"
)


def _parse_filename(filename: str) -> dict:
    """Extract subject/session/scan/slice identifiers from an OASIS filename.

    Falls back to using the whole stem as subject_id if the expected pattern
    isn't found, rather than silently dropping the file.
    """
    match = FILENAME_RE.search(filename)
    if match:
        d = match.groupdict()
        d["slice_idx"] = int(d["slice_idx"])
        return d
    logger.warning("Filename '%s' did not match expected OASIS pattern; "
                    "using filename stem as subject_id.", filename)
    stem = Path(filename).stem
    return {"subject_id": stem, "session": "unknown", "scan": "unknown", "slice_idx": -1}


def build_manifest(data_root: Path = config.DATA_ROOT,
                    folder_to_class: dict = config.RAW_FOLDER_TO_CLASS) -> pd.DataFrame:
    """Walk the raw dataset directory and build a manifest DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        filepath, raw_folder, class_label, class_idx,
        subject_id, session, scan, slice_idx
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root '{data_root}' does not exist. Upload/extract the "
            f"OASIS Kaggle subset there before running the pipeline."
        )

    rows = []
    skipped_folders = []
    for raw_folder, class_label in folder_to_class.items():
        folder_path = data_root / raw_folder
        if not folder_path.is_dir():
            skipped_folders.append(raw_folder)
            continue
        image_files = sorted(
            [p for p in folder_path.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        for img_path in image_files:
            ident = _parse_filename(img_path.name)
            rows.append({
                "filepath": str(img_path),
                "raw_folder": raw_folder,
                "class_label": class_label,
                "class_idx": config.CLASS_TO_IDX[class_label],
                "subject_id": ident["subject_id"],
                "session": ident["session"],
                "scan": ident["scan"],
                "slice_idx": ident["slice_idx"],
            })

    if skipped_folders:
        logger.warning("These expected raw folders were not found under %s: %s",
                        data_root, skipped_folders)

    if not rows:
        raise RuntimeError(
            f"No images found under {data_root}. Check that the folder names "
            f"match {list(folder_to_class.keys())}."
        )

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(["class_label", "subject_id", "session", "scan", "slice_idx"])
    manifest = manifest.reset_index(drop=True)
    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-class file and unique-subject counts -- the first sanity check
    anyone reviewing intermediate output should look at."""
    summary = manifest.groupby("class_label").agg(
        n_files=("filepath", "count"),
        n_subjects=("subject_id", "nunique"),
    ).reindex(config.CLASS_NAMES)
    return summary


def save_manifest(manifest: pd.DataFrame, path: Path = config.MANIFEST_CSV) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    logger.info("Manifest saved to %s (%d rows)", path, len(manifest))


if __name__ == "__main__":
    manifest = build_manifest()
    save_manifest(manifest)
    print(manifest.head())
    print()
    print("Per-class summary:")
    print(summarize_manifest(manifest))
