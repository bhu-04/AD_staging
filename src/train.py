"""
Stable training for CNN2D / DenseNet2D / ViT2D.

Features:
- fresh model for every retry
- deterministic seeds
- validation-accuracy checkpointing
- early stopping
- fast degeneracy detection
- gradient clipping
- AdamW regularization
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DEVICE
# ============================================================================

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# EPOCH
# ============================================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []

    with torch.set_grad_enabled(train):

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = criterion(logits, y)

            if train:

                if not torch.isfinite(loss):
                    logger.warning(
                        "Non-finite loss detected. Stopping epoch."
                    )
                    return float("inf"), 0.0, np.array([])

                loss.backward()

                # Prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += x.size(0)

            all_preds.append(
                preds.detach().cpu()
            )

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
    else:
        all_preds = np.array([])

    avg_loss = total_loss / max(total, 1)

    accuracy = correct / max(total, 1)

    return avg_loss, accuracy, all_preds


# ============================================================================
# DEGENERACY
# ============================================================================

def _is_degenerate(
    val_preds: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
) -> bool:

    if val_preds is None or len(val_preds) == 0:
        return True

    unique_preds = len(np.unique(val_preds))

    # Complete collapse.
    if unique_preds <= 1:
        return True

    # Majority baseline.
    class_counts = np.bincount(
        val_labels,
        minlength=num_classes
    )

    majority_baseline = (
        class_counts.max()
        / max(len(val_labels), 1)
    )

    val_acc = (
        val_preds == val_labels
    ).mean()

    # Only mark as degenerate if it is actually no better
    # than the trivial classifier.
    return val_acc <= majority_baseline + 1e-9


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def compute_class_weights(
    loader,
    num_classes: int,
    device: str = "cpu",
):

    counts = torch.zeros(num_classes)

    for _, y in loader:

        counts += torch.bincount(
            y,
            minlength=num_classes
        ).float()

    counts = counts.clamp(min=1.0)

    weights = (
        counts.sum()
        / (num_classes * counts)
    )

    return weights.to(device)


# ============================================================================
# ONE TRAINING ATTEMPT
# ============================================================================

def _train_once(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    checkpoint_path,
    device,
    seed,
):

    set_seed(seed)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_preds = None

    patience_counter = 0

    # Validation labels are fixed.
    val_labels = torch.cat(
        [y for _, y in val_loader]
    ).numpy()

    for epoch in range(1, epochs + 1):

        train_loss, train_acc, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        val_loss, val_acc, val_preds = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = (
            val_acc > best_val_acc
            or (
                val_acc == best_val_acc
                and val_loss < best_val_loss
            )
        )

        if improved:

            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_preds = val_preds.copy()

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            patience_counter = 0

        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %2d/%2d | "
            "train_loss=%.4f train_acc=%.3f | "
            "val_loss=%.4f val_acc=%.3f | "
            "lr=%.2e%s",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            "  <- checkpoint saved"
            if improved else "",
        )

        # ------------------------------------------------------------
        # FAST DEGENERACY CHECK
        # ------------------------------------------------------------

        if epoch >= config.MIN_EPOCHS:

            if _is_degenerate(
                val_preds,
                val_labels,
                len(config.CLASS_NAMES),
            ):

                logger.warning(
                    "Degenerate model detected at epoch %d. "
                    "Stopping attempt immediately.",
                    epoch,
                )

                break

        # ------------------------------------------------------------
        # EARLY STOPPING
        # ------------------------------------------------------------

        if (
            epoch >= config.MIN_EPOCHS
            and patience_counter
            >= config.EARLY_STOPPING_PATIENCE
        ):

            logger.info(
                "Early stopping at epoch %d "
                "(no validation improvement for %d epochs).",
                epoch,
                config.EARLY_STOPPING_PATIENCE,
            )

            break

    degenerate = _is_degenerate(
        best_val_preds,
        val_labels,
        len(config.CLASS_NAMES),
    )

    history["best_val_acc"] = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["degenerate"] = bool(degenerate)
    history["seed"] = seed
    history["epochs_completed"] = len(
        history["train_loss"]
    )

    return history


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=config.EPOCHS,
    lr=config.LEARNING_RATE,
    checkpoint_path=config.CHECKPOINT_DIR / "cnn2d_best.pt",
    device=None,
    max_attempts=config.MAX_ATTEMPTS,
    base_seed=config.RANDOM_SEED,
):

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if device is None:
        device = get_device()

    model_cls = type(model)

    best_history = None
    best_state_dict = None

    for attempt in range(
        1,
        max_attempts + 1
    ):

        seed = base_seed + attempt - 1

        logger.info("=" * 60)

        logger.info(
            "Training attempt %d/%d | seed=%d | epochs=%d",
            attempt,
            max_attempts,
            seed,
            epochs,
        )

        logger.info("=" * 60)

        # ------------------------------------------------------------
        # CRITICAL:
        # Seed BEFORE model construction.
        # ------------------------------------------------------------

        set_seed(seed)

        model = model_cls(
            num_classes=len(config.CLASS_NAMES),
            dropout_p=config.DROPOUT_P,
        )

        history = _train_once(
            model,
            train_loader,
            val_loader,
            epochs,
            lr,
            checkpoint_path,
            device,
            seed,
        )

        # ------------------------------------------------------------
        # KEEP BEST ATTEMPT
        # ------------------------------------------------------------

        if best_history is None:

            choose = True

        elif (
            not history["degenerate"]
            and best_history["degenerate"]
        ):

            choose = True

        elif (
            history["degenerate"]
            == best_history["degenerate"]
            and history["best_val_acc"]
            > best_history["best_val_acc"]
        ):

            choose = True

        elif (
            history["degenerate"]
            == best_history["degenerate"]
            and history["best_val_acc"]
            == best_history["best_val_acc"]
            and history["best_val_loss"]
            < best_history["best_val_loss"]
        ):

            choose = True

        else:

            choose = False

        if choose:

            best_history = history

            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        # ------------------------------------------------------------
        # STOP RETRYING IF GOOD
        # ------------------------------------------------------------

        if not history["degenerate"]:

            logger.info(
                "Successful non-degenerate model found "
                "on attempt %d.",
                attempt,
            )

            break

        logger.warning(
            "Attempt %d produced a degenerate model.",
            attempt,
        )

    if best_history is None:
        raise RuntimeError(
            "No training attempt completed."
        )

    # Save the best attempt.
    torch.save(
        best_state_dict,
        checkpoint_path,
    )

    # ------------------------------------------------------------
    # SAVE HISTORY
    # ------------------------------------------------------------

    history_path = (
        config.METRICS_DIR
        / "training_history.json"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(history_path, "w") as f:

        json.dump(
            best_history,
            f,
            indent=2,
        )

    logger.info(
        "Training history saved to %s",
        history_path,
    )

    logger.info(
        "Best checkpoint | "
        "val_acc=%.3f | "
        "val_loss=%.4f | "
        "degenerate=%s | "
        "saved=%s",
        best_history["best_val_acc"],
        best_history["best_val_loss"],
        best_history["degenerate"],
        checkpoint_path,
    )

    return best_history