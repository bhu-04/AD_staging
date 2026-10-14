"""
Central configuration for the Uncertainty-Aware sMRI AD Staging pipeline.

Only the parameters needed for the currently-implemented slice of the
project (see README.md for module-by-module status) live here. Values are
deliberately small (image size, epochs) because this run is meant to prove
the pipeline is *correct end to end*, not to reach publication-grade
accuracy on a 480-image demo subset.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "oasis_subset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MANIFEST_CSV = OUTPUT_ROOT / "manifests" / "manifest.csv"
SPLITS_JSON = OUTPUT_ROOT / "manifests" / "splits.json"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PLOTS_DIR = OUTPUT_ROOT / "plots"

# ---------------------------------------------------------------------------
# Class labels (CN / MCI / AD collapse of the four OASIS-derived Kaggle
# folders). The Kaggle "imagesoasis" release exposes four folders; the
# project's staging target is CN / MCI / AD, so "Very Mild Dementia" is
# mapped to MCI and the two more severe folders are collapsed into AD.
# This mapping is a documented design decision, not an accident.
# ---------------------------------------------------------------------------
RAW_FOLDER_TO_CLASS = {
    "non demented": "CN",
    "very mild demented": "MCI",
    "Mild dementia": "AD",
    "Moderate dementia": "AD",
}
CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Preprocessing / splitting
# ---------------------------------------------------------------------------
IMAGE_SIZE = (64, 64)     # (H, W) after resize -- kept small because this
                          # sandbox has a single CPU core (no GPU); see
                          # README.md for the full benchmarking note.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 30               # was 10 -- too few for this CNN to reliably beat
                          # the majority-class baseline; short training was
                          # producing checkpoints that just predict the
                          # "safest" middle class (MCI) for everything
LEARNING_RATE = 1e-3
DROPOUT_P = 0.35          # shared by the classifier head and MC-Dropout

# ---------------------------------------------------------------------------
# Vision Transformer specific
# ---------------------------------------------------------------------------
VIT_PATCH_SIZE = 8
VIT_EMBED_DIM = 32
VIT_DEPTH = 3
VIT_NUM_HEADS = 4
VIT_MLP_DIM = 64

# ---------------------------------------------------------------------------
# Deep Ensembles
# ---------------------------------------------------------------------------
ENSEMBLE_SIZE = 3         # M in Algorithm 8 (kept small for CPU runtime)

# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------
MC_DROPOUT_SAMPLES = 30   # T in Algorithm 6 of the project document