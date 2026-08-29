"""
Module - Deep Ensembles (Algorithm 8)
========================================
Implements Algorithm 8 exactly as specified: train M independently
initialized models (different random seeds, different data shuffling),
then at inference use disagreement across the M members' softmax outputs
as the uncertainty signal -- no stochastic test-time sampling required
(unlike MC Dropout), each member gives one deterministic prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..train import train_model

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_ensemble(model_fn, train_loader, val_loader,
                    M: int = 5, epochs: int = config.EPOCHS,
                    checkpoint_dir: Path = config.CHECKPOINT_DIR / "ensemble",
                    base_seed: int = config.RANDOM_SEED) -> list:
    """Train M independently-seeded copies of model_fn() and checkpoint each.

    model_fn: a zero-arg callable returning a fresh, untrained nn.Module
              instance (e.g. lambda: CNN2D(num_classes=3)).
    Returns the list of checkpoint paths, one per ensemble member.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = []
    for m in range(M):
        # BUG FIX: previously this seed was only used for the outer
        # torch.manual_seed() call below and never reached train_model's
        # internal retry logic, which defaulted base_seed to
        # config.RANDOM_SEED for every member -- so any member that needed
        # a retry (degenerate first attempt) used the exact same retry
        # seeds as every other member, producing bit-identical, non-diverse
        # "ensemble" members. Each member now gets its own seed block
        # (spaced by 1000, well clear of max_attempts) and that block is
        # passed all the way through to train_model via base_seed.
        seed_block_start = base_seed + m * 1000
        torch.manual_seed(seed_block_start)
        logger.info("Training ensemble member %d/%d (seed block starting at %d)",
                    m + 1, M, seed_block_start)
        model = model_fn()
        ckpt_path = checkpoint_dir / f"member_{m}.pt"
        train_model(model, train_loader, val_loader, epochs=epochs, checkpoint_path=ckpt_path,
                    base_seed=seed_block_start)
        checkpoint_paths.append(ckpt_path)

    logger.info("Ensemble training complete: %d members saved to %s", M, checkpoint_dir)
    return checkpoint_paths


@torch.no_grad()
def deep_ensemble_predict(model_fn, checkpoint_paths: list, x: torch.Tensor,
                           device: str = "cpu") -> dict:
    """Single-batch version of deep_ensemble_predict_dataset -- same
    (deterministic, one forward pass per member) mechanics, used by
    visualize_pipeline.py so the visual demo can use the real ensemble
    instead of silently substituting a single member's checkpoint."""
    members = []
    for ckpt_path in checkpoint_paths:
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        members.append(model)

    x = x.to(device)
    member_probs = [torch.softmax(m(x), dim=1) for m in members]
    probs_stack = torch.stack(member_probs, dim=0)  # (M, B, C)

    p_mean = probs_stack.mean(dim=0)
    y_pred = p_mean.argmax(dim=1)
    if probs_stack.shape[0] > 1:
        variance = probs_stack.var(dim=0, unbiased=False).mean(dim=1)
    else:
        variance = torch.zeros(
            probs_stack.shape[1],
            device=probs_stack.device
    )
    eps = 1e-12
    entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)
    return {"p_mean": p_mean, "y_pred": y_pred, "variance": variance, "entropy": entropy}


@torch.no_grad()
def deep_ensemble_predict_dataset(model_fn, checkpoint_paths: list, loader,
                                   device: str = "cpu") -> dict:
    """Run every ensemble member once (deterministic forward pass, eval
    mode, no dropout sampling) over the dataset and combine into the same
    result schema mc_dropout_predict_dataset uses, so evaluate.py works
    unmodified against either uncertainty method."""
    members = []
    for ckpt_path in checkpoint_paths:
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        members.append(model)

    all_p_mean, all_variance, all_entropy, all_y_true, all_y_pred = [], [], [], [], []

    for x, y in loader:
        x = x.to(device)
        member_probs = []
        for model in members:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            member_probs.append(probs)
        probs_stack = torch.stack(member_probs, dim=0)  # (M, B, C)

        p_mean = probs_stack.mean(dim=0)
        y_pred = p_mean.argmax(dim=1)
        variance = probs_stack.var(dim=0).mean(dim=1)
        eps = 1e-12
        entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)

        all_p_mean.append(p_mean.cpu())
        all_y_pred.append(y_pred.cpu())
        all_variance.append(variance.cpu())
        all_entropy.append(entropy.cpu())
        all_y_true.append(y)

    return {
        "p_mean": torch.cat(all_p_mean).numpy(),
        "y_pred": torch.cat(all_y_pred).numpy(),
        "variance": torch.cat(all_variance).numpy(),
        "entropy": torch.cat(all_entropy).numpy(),
        "y_true": torch.cat(all_y_true).numpy(),
    }


if __name__ == "__main__":
    from .. import data_acquisition, splitting, augmentation, dataset
    from ..models.cnn2d import CNN2D

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    model_fn = lambda: CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    ckpts = train_ensemble(model_fn, loaders["train"], loaders["val"], M=3, epochs=6)
    results = deep_ensemble_predict_dataset(model_fn, ckpts, loaders["test"])
    print(f"Ensemble of {len(ckpts)} members on {len(results['y_true'])} test samples")
    print(f"Mean predictive variance: {results['variance'].mean():.5f}")
    print(f"Mean predictive entropy:  {results['entropy'].mean():.5f}")