"""
Module - Monte Carlo Dropout (Algorithm 6)
=============================================
Implements Algorithm 6 exactly as specified: T stochastic forward passes at
inference time with dropout layers active while every other layer
(BatchNorm included) stays in eval mode, using the spread of the resulting
predictions as the uncertainty signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


def enable_mc_dropout(model: nn.Module) -> None:
    """Set the whole model to eval mode, then re-enable train-mode (active)
    behaviour on Dropout layers only -- matching Algorithm 6's
    Initialization step precisely."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor,
                        T: int = config.MC_DROPOUT_SAMPLES) -> dict:
    """Run T stochastic forward passes for a batch and return the
    predictive mean, predicted label, and predictive variance
    (uncertainty score), per Algorithm 6.

    Returns
    -------
    dict with keys:
        p_mean:  (B, num_classes) mean softmax probability across T passes
        y_pred:  (B,) argmax(p_mean)
        variance: (B,) scalar per-sample uncertainty = mean class-wise
                  variance across T stochastic passes
        entropy: (B,) predictive entropy of p_mean (reported alongside
                  variance since Algorithm 9 / the evaluation layer expects
                  predictive entropy as a calibration input too)
    """
    enable_mc_dropout(model)

    probs_over_t = []
    for _ in range(T):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs_over_t.append(probs)
    probs_stack = torch.stack(probs_over_t, dim=0)  # (T, B, C)

    p_mean = probs_stack.mean(dim=0)                # (B, C)
    y_pred = p_mean.argmax(dim=1)                   # (B,)

    # mean of per-class variance across the T passes -> single scalar/sample
    variance = probs_stack.var(dim=0).mean(dim=1)   # (B,)

    eps = 1e-12
    entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)  # (B,)

    return {
        "p_mean": p_mean,
        "y_pred": y_pred,
        "variance": variance,
        "entropy": entropy,
    }


def mc_dropout_predict_dataset(model: nn.Module, loader, device: str = "cpu",
                                T: int = config.MC_DROPOUT_SAMPLES) -> dict:
    """Run MC Dropout inference over an entire DataLoader and concatenate
    results, alongside ground-truth labels."""
    model.to(device)
    all_p_mean, all_y_pred, all_variance, all_entropy, all_y_true = [], [], [], [], []

    for x, y in loader:
        x = x.to(device)
        out = mc_dropout_predict(model, x, T=T)
        all_p_mean.append(out["p_mean"].cpu())
        all_y_pred.append(out["y_pred"].cpu())
        all_variance.append(out["variance"].cpu())
        all_entropy.append(out["entropy"].cpu())
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

    model = CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    ckpt = config.CHECKPOINT_DIR / "cnn2d_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    results = mc_dropout_predict_dataset(model, loaders["test"], T=config.MC_DROPOUT_SAMPLES)
    print(f"Ran MC Dropout with T={config.MC_DROPOUT_SAMPLES} on {len(results['y_true'])} test samples")
    print(f"Mean predictive variance: {results['variance'].mean():.5f}")
    print(f"Mean predictive entropy:  {results['entropy'].mean():.5f}")
