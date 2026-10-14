"""
Training Procedure (Algorithm 3, "Training Procedure / Optimization")
=======================================================================
Standard supervised training loop: shuffle -> forward -> cross-entropy loss
-> backprop -> Adam update -> validate -> checkpoint the best model ->
retrain from a new seed if the result is degenerate.

BUG FIX: the previous version of this module checkpointed by *lowest
validation loss*. On this tiny, easily-overfit dataset, validation loss is
often minimized very early -- sometimes at epoch 1, before the model has
learned to discriminate classes at all -- and then rises as the model
overfits even as validation *accuracy* keeps improving afterward.
Selecting by loss alone could therefore save a near-untrained checkpoint
that outputs close to the class prior for every input: near-uniform class
probabilities, low confidence, the same predicted class regardless of the
input image, and a Grad-CAM map dominated by generic edge/position
artifacts instead of anatomy (because the conv filters never learned
anything useful yet). That is exactly the "predicts one class for
everything, ~45% confidence" failure that showed up in one run.

Fix, in two parts:
  1. Checkpoint by *validation accuracy* first, tie-broken by lower loss
     -- not by loss alone.
  2. After training, check whether the saved checkpoint is degenerate
     (predicts only one class across the whole validation set, or scores
     at/below the naive majority-class baseline). If so, retrain from a
     new random seed (re-initializing the model), up to `max_attempts`
     times, keeping the best result found. This is logged explicitly
     either way, and the returned/saved history records whether the final
     checkpoint is degenerate so callers (and you) can tell at a glance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds = []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)
            all_preds.append(preds.detach().cpu())
    all_preds = torch.cat(all_preds).numpy() if all_preds else np.array([])
    return total_loss / n, correct / n, all_preds


def _is_degenerate(val_preds: np.ndarray, val_labels: np.ndarray, num_classes: int) -> bool:
    """A checkpoint is degenerate if it collapses to predicting a single
    class for the entire validation set (a hallmark of an undertrained or
    prior-only model), or scores at/below the naive majority-class
    baseline it could reach without looking at the image at all."""
    n_unique_preds = len(np.unique(val_preds))
    if n_unique_preds <= 1:
        return True
    majority_baseline = np.bincount(val_labels, minlength=num_classes).max() / max(len(val_labels), 1)
    val_acc = (val_preds == val_labels).mean()
    return val_acc <= majority_baseline + 1e-9


def compute_class_weights(loader, num_classes: int, device: str = "cpu") -> torch.Tensor:
    """Inverse-frequency class weights from the TRAIN loader only, so
    CrossEntropyLoss stops defaulting to the majority class (this dataset
    is ~50% AD / 25% CN / 25% MCI, which was silently biasing every
    backbone toward predicting AD regardless of the true class)."""
    counts = torch.zeros(num_classes)
    for _, y in loader:
        counts += torch.bincount(y, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)   # inverse frequency, normalized
    return weights.to(device)


def _train_once(model, train_loader, val_loader, epochs: int, lr: float,
                 checkpoint_path: Path, device: str, seed: int,
                 class_weights: torch.Tensor = None) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Decay LR when val_loss plateaus -- helps the model actually escape
    # the "predict the safe middle class" plateau within the epoch budget
    # instead of oscillating around it for the whole run.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_val_loss = -1.0, float("inf")
    best_val_preds = None

    # capture val labels once (val loader uses shuffle=False, so order is fixed)
    val_labels = torch.cat([y for _, y in val_loader]).numpy()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpoint by validation ACCURACY first, tie-broken by lower loss
        # -- not by loss alone (see module docstring for why).
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc, best_val_loss = val_acc, val_loss
            best_val_preds = val_preds
            torch.save(model.state_dict(), checkpoint_path)

        logger.info(
            "Epoch %2d/%2d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f%s",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            "  <- checkpoint saved" if improved else "",
        )

    degenerate = _is_degenerate(best_val_preds, val_labels, num_classes=len(config.CLASS_NAMES))
    history["best_val_acc"] = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["degenerate"] = bool(degenerate)
    history["seed"] = seed
    return history


def train_model(model, train_loader, val_loader,
                 epochs: int = config.EPOCHS,
                 lr: float = config.LEARNING_RATE,
                 checkpoint_path: Path = config.CHECKPOINT_DIR / "cnn2d_best.pt",
                 device: str = "cpu",
                 max_attempts: int = 3,
                 base_seed: int = config.RANDOM_SEED) -> dict:
    """Train `model`, retrying from a fresh random seed (re-initializing
    weights) up to `max_attempts` times if the resulting checkpoint is
    degenerate (see module docstring). Always leaves the best attempt's
    weights at `checkpoint_path` when done."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_cls = type(model)
    best_history = None
    best_state_dict = None

    # Computed once from the TRAIN split (not val/test), reused across
    # every retry attempt -- see compute_class_weights() docstring.
    class_weights = compute_class_weights(train_loader, num_classes=len(config.CLASS_NAMES), device=device)
    logger.info("Class weights (inverse frequency, order=%s): %s",
                config.CLASS_NAMES, [round(w, 3) for w in class_weights.tolist()])

    for attempt in range(1, max_attempts + 1):
        seed = base_seed + attempt - 1
        attempt_epochs = epochs
        if attempt > 1:
            # BUG FIX: retrying with only a new random seed and identical
            # hyperparameters rarely escapes a degenerate result if the
            # root cause is "not enough training", not "bad luck" -- it
            # was just re-rolling the same failure. Each retry now also
            # gets more epochs and slightly lower dropout, so it's an
            # actually-different attempt, not the same run again.
            attempt_epochs = int(epochs * (1 + 0.5 * (attempt - 1)))
            retry_dropout = max(0.1, config.DROPOUT_P - 0.1 * (attempt - 1))
            model = model_cls(num_classes=len(config.CLASS_NAMES), dropout_p=retry_dropout)
            logger.warning(
                "Attempt %d/%d: previous attempt's checkpoint was degenerate "
                "(collapsed to a single predicted class / at-or-below the "
                "majority-class baseline). Re-initializing with seed=%d, "
                "epochs=%d (was %d), dropout=%.2f (was %.2f).",
                attempt, max_attempts, seed, attempt_epochs, epochs,
                retry_dropout, config.DROPOUT_P,
            )

        history = _train_once(model, train_loader, val_loader, attempt_epochs, lr, checkpoint_path, device, seed,
                               class_weights=class_weights)

        if best_history is None or (
            history["degenerate"] is False and best_history["degenerate"] is True
        ) or (
            history["degenerate"] == best_history["degenerate"]
            and history["best_val_acc"] > best_history["best_val_acc"]
        ):
            best_history = history
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if not history["degenerate"]:
            break  # good result found, no need to burn more attempts

    if best_history["degenerate"]:
        logger.warning(
            "All %d attempt(s) produced a degenerate model for this backbone "
            "(best val_acc=%.3f). This is a known risk with this tiny demo "
            "dataset -- treat results from this backbone with extra caution.",
            max_attempts, best_history["best_val_acc"],
        )
    torch.save(best_state_dict, checkpoint_path)

    history_path = config.METRICS_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(best_history, f, indent=2)
    logger.info("Training history saved to %s", history_path)
    logger.info(
        "Best checkpoint (val_acc=%.3f, val_loss=%.4f, degenerate=%s) saved to %s",
        best_history["best_val_acc"], best_history["best_val_loss"],
        best_history["degenerate"], checkpoint_path,
    )

    return best_history


if __name__ == "__main__":
    from . import data_acquisition, splitting, augmentation, dataset
    from .models.cnn2d import CNN2D

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    model = CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    history = train_model(model, loaders["train"], loaders["val"])