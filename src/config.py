"""
Central configuration for the Uncertainty-Aware sMRI AD Staging pipeline.
"""

from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "oasis_subset"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MANIFEST_CSV = OUTPUT_ROOT / "manifests" / "manifest.csv"
SPLITS_JSON = OUTPUT_ROOT / "manifests" / "splits.json"

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PLOTS_DIR = OUTPUT_ROOT / "plots"


# ============================================================================
# CLASSES
# ============================================================================

RAW_FOLDER_TO_CLASS = {
    "non demented": "CN",
    "very mild demented": "MCI",
    "Mild dementia": "AD",
    "Moderate dementia": "AD",
}

CLASS_NAMES = [
    "CN",
    "MCI",
    "AD",
]

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


# ============================================================================
# IMAGE / DATA
# ============================================================================

IMAGE_SIZE = (64, 64)

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42

# IMPORTANT:
# The current demo dataset does not contain enough subjects for a genuine
# subject-level 70/15/15 split.
#
# Keep False for the tiny demo dataset.
# Set True when using the real/larger OASIS dataset.
REQUIRE_SUBJECT_LEVEL_SPLIT = True


# ============================================================================
# TRAINING
# ============================================================================

BATCH_SIZE = 32

# 15 is enough for the small development dataset.
# Early stopping will normally terminate earlier.
EPOCHS = 15

# Lower than the previous 3e-4 because the dataset is tiny.
LEARNING_RATE = 1e-4

# Slightly stronger regularization.
DROPOUT_P = 0.30

WEIGHT_DECAY = 1e-4


# ============================================================================
# EARLY STOPPING / DEGENERACY
# ============================================================================

# We don't need to wait 5 epochs before detecting collapse.
MIN_EPOCHS = 3

# Four stagnant epochs was wasting time on bad runs.
EARLY_STOPPING_PATIENCE = 3

# Maximum complete retries.
MAX_ATTEMPTS = 2

# If validation accuracy remains below this after a few epochs,
# the run is probably not learning.
MIN_USEFUL_VAL_ACC = 0.40


# ============================================================================
# VISION TRANSFORMER
# ============================================================================

# 64x64 -> 8x8 = 64 patches.
VIT_PATCH_SIZE = 8

VIT_EMBED_DIM = 32

VIT_DEPTH = 2

VIT_NUM_HEADS = 4

VIT_MLP_DIM = 64


# ============================================================================
# DEEP ENSEMBLES
# ============================================================================

# Keep small for CPU experimentation.
ENSEMBLE_SIZE = 2


# ============================================================================
# MC DROPOUT
# ============================================================================

# 10 was unnecessarily expensive during development.
MC_DROPOUT_SAMPLES = 5


# ============================================================================
# BEDL
# ============================================================================

BEDL_ANNEALING_EPOCHS = 5

BEDL_LEARNING_RATE = 5e-5

BEDL_MAX_ATTEMPTS = 1

BEDL_EVIDENCE_REGULARIZATION = 1e-3