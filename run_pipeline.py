"""End-to-end runner for the Uncertainty-Aware sMRI-Based AD Staging
pipeline -- full scope (all 12 modules, all 9 architecture-uncertainty
combinations).

Usage:
    python run_pipeline.py

Runs, in order:
  1. Dataset Acquisition
  2. Dataset Splitting
  3. Preprocessing + Augmentation (inline, via the DataLoaders)
  4. Full Comparative Study: 3D CNN / DenseNet-121 / ViT backbones x
     Monte Carlo Dropout / Deep Ensembles / Bayesian Evidential Deep
     Learning -- all 9 combinations, trained and evaluated under
     identical conditions
                             -> outputs/checkpoints/*
                             -> outputs/metrics/combination_comparison.csv
                             -> outputs/metrics/statistical_comparison.csv
                             -> outputs/metrics/decision_layer_recommendation.json
  5. Visual output pipeline for the RECOMMENDED combination, using that
     combination's REAL prediction mechanism (not always MC Dropout --
     see visualize_pipeline.build_predict_fn)
                             -> outputs/plots/sample_pipeline/*.png

Every stage prints its own intermediate output so progress can be
verified step by step, not just at the final report.
"""
import json

from src import config, data_acquisition, splitting, augmentation, dataset
from src import comparative_evaluation as ce
from src import visualize_pipeline as viz
from src.models.cnn2d import CNN2D
from src.models.densenet2d import DenseNet2D
from src.models.vit2d import ViT2D

MODEL_CLASSES = {"CNN2D": CNN2D, "DenseNet2D": DenseNet2D, "ViT2D": ViT2D}
# Backbones whose penultimate feature map is spatial (B,C,H,W) and therefore
# Grad-CAM-compatible for the "classified image" panel.
CONV_BACKBONES = {"CNN2D", "DenseNet2D"}


def main():
    print("=" * 70)
    print("STAGE 1/5: Dataset Acquisition")
    print("=" * 70)
    manifest = data_acquisition.build_manifest()
    data_acquisition.save_manifest(manifest)
    print(data_acquisition.summarize_manifest(manifest))

    print("\n" + "=" * 70)
    print("STAGE 2/5: Dataset Splitting")
    print("=" * 70)
    split_result = splitting.subject_stratified_split(manifest)
    leakage_ok = splitting.assert_no_subject_leakage(manifest, split_result)
    print(f"No-subject-leakage assertion passed: {leakage_ok}")
    print(splitting.class_distribution_check(manifest, split_result))
    splitting.save_splits(split_result)

    print("\n" + "=" * 70)
    print("STAGE 3/5: Building DataLoaders (Preprocessing + Augmentation applied inline)")
    print("=" * 70)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )
    for name, loader in loaders.items():
        print(f"  {name}: {len(loader.dataset)} samples")

    print("\n" + "=" * 70)
    print("STAGE 4/5: Full Comparative Study (9 combinations:")
    print("3D CNN / DenseNet-121 / ViT  x  MC Dropout / Deep Ensembles / BEDL)")
    print("=" * 70)
    study = ce.run_comparative_study(loaders)
    print("\nComparison table (all 9 combinations):")
    print(study["comparison_df"].round(4).to_string())
    print("\nStatistical comparison vs. best combination (paired t-test + Wilcoxon):")
    print(study["stats_df"].round(4).to_string())
    print("\nDecision Layer recommendation:")
    print(json.dumps(study["decision"], indent=2))
    ce.plot_comparison_bars(study["comparison_df"], config.PLOTS_DIR / "combination_comparison.png")

    print("\n" + "=" * 70)
    print("STAGE 5/5: Visual Output Pipeline for the RECOMMENDED combination")
    print("(1. input image  2. preprocessed image  3. classified image  4. final output)")
    print("=" * 70)
    best_combo = study["decision"]["recommended_combination"]
    best_backbone = best_combo.split("+")[0]
    viz_combo = best_combo

    if best_backbone not in CONV_BACKBONES:
        # Grad-CAM (the "classified image" panel) needs a spatial feature
        # map; ViT2D doesn't expose one the same way, so fall back to the
        # best conv-based backbone's combination for this figure and say
        # so explicitly. The comparison table/decision above still reflect
        # the TRUE recommendation -- this only affects which combination
        # gets the image walkthrough.
        conv_rows = study["comparison_df"][
            study["comparison_df"].index.str.split("+").str[0].isin(CONV_BACKBONES)
        ]
        viz_combo = conv_rows["accuracy"].idxmax()
        print(f"  Recommended combination '{best_combo}' has no spatial feature map "
              f"for Grad-CAM; using '{viz_combo}' for the visual walkthrough instead.")

    # BUG FIX: build the ACTUAL prediction mechanism for viz_combo (MC
    # Dropout / Deep Ensembles / BEDL) instead of always loading a single
    # plain backbone checkpoint and running MC Dropout on it regardless of
    # what was recommended.
    predict_fn, gradcam_model = viz.build_predict_fn(viz_combo, study["checkpoints"], MODEL_CLASSES)

    results = viz.visualize_samples_per_class(manifest, gradcam_model, predict_fn=predict_fn, n_per_class=5)
    for r in results:
        print(f"  {r['predicted_class']} (true: {r['true_class']}), "
              f"confidence={r['confidence']:.1%}, review={r['needs_review']} "
              f"-> {r['figure_path']}")

    print("\n" + "=" * 70)
    print("Pipeline complete. See outputs/ for manifests, splits, checkpoints,")
    print("comparison table, statistical comparison, decision recommendation,")
    print("and the 4-panel visual walkthroughs in outputs/plots/sample_pipeline/.")
    print("=" * 70)


if __name__ == "__main__":
    main()