# Uncertainty-Aware sMRI-Based AD Staging — Implementation Increment (~60%)

This delivers a working, verifiable slice of the pipeline described in
*Uncertainty-Aware sMRI-Based Alzheimer's Disease Staging: A Comparative
Benchmarking Study* (TARP.pdf). Per that document's own module table, the
project starts at 33.34% (data pipeline only). This increment adds all
three classification backbones, comparative model-base selection across
them, and a genuine visual output pipeline, bringing real, run implementation
to roughly 60% of the twelve listed modules.

## How to run

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn
python run_pipeline.py
```

Takes a few minutes on CPU (this sandbox has a single core and no GPU --
see "Compute scope" below). Every stage prints its own intermediate output.
All artifacts land in `outputs/`:

```
outputs/
  manifests/manifest.csv                 # Stage 1: every image + parsed subject/session/class
  manifests/splits.json                  # Stage 2: train/val/test assignment + strategy used
  checkpoints/cnn2d_best.pt              # Stage 4: best-val-loss weights, one per backbone
  checkpoints/densenet2d_best.pt
  checkpoints/vit2d_best.pt
  checkpoints/ensemble/member_{0,1,2}.pt # Deep Ensembles worked example (CNN2D backbone)
  metrics/training_history.json
  metrics/combination_comparison.csv     # Stage 4: every evaluated backbone x uncertainty combo
  metrics/decision_layer_recommendation.json  # Best-Combination Selector output
  metrics/evaluation_report.json
  plots/combination_comparison.png       # accuracy & ECE bar charts across combinations
  plots/confusion_matrix_*.png, reliability_diagram_*.png   # one set per combination
  plots/sample_pipeline/*.png            # Stage 5: the 4-panel visual walkthroughs
```

Each module also runs standalone, e.g.:
```bash
python -m src.data_acquisition
python -m src.models.cnn2d          # / densenet2d / vit2d
python -m src.comparative_evaluation   # trains all backbones, compares, selects best
python -m src.visualize_pipeline       # generates the 4-panel image outputs
```

## The visual output you asked for

Metrics tables and score plots are still produced, but the actual
requirement -- **input image -> preprocessing image -> classified image ->
final output**, as real images, not just numbers -- is implemented in
`src/visualize_pipeline.py` and generated automatically by
`run_pipeline.py`'s last stage. For each of the 3 sample slices (one per
class), it saves a single 4-panel figure to `outputs/plots/sample_pipeline/`:

1. **Input image** -- the raw slice exactly as it exists on disk.
2. **Preprocessed image** -- after resize + intensity normalization (the
   actual array the model consumes), rescaled only for display.
3. **Classified image** -- the slice with a **Grad-CAM** saliency overlay
   showing which anatomical regions drove the prediction, captioned with
   predicted vs. true class and confidence. Grad-CAM is a standard,
   well-established technique (gradient-weighted class activation
   mapping) -- it requires no extra training, just the already-trained
   model.
4. **Final output** -- a decision-support-style summary panel: predicted
   stage, per-class probability bars, predictive entropy/variance from
   Monte Carlo Dropout, and a plain-language "flagged for review" /
   "confident prediction" label driven by an uncertainty threshold.

This is generated for whichever combination the Decision Layer actually
recommends (falling back to the best conv-based backbone if a
Transformer wins, since Grad-CAM needs a spatial feature map that ViT2D
doesn't expose the same way -- noted explicitly in the console output when
that happens).

## Comparative model-base selection (the new part)

`src/comparative_evaluation.py` is the piece that turns this from three
disconnected backbones into an actual comparative study, matching the
project's central novelty claim. In one run it:

1. Trains **3D CNN -> 2D CNN**, **DenseNet-121 -> compact 2D DenseNet**, and
   **ViT/Swin -> compact 2D ViT** independently on identical data, splits,
   and training settings.
2. Attaches **Monte Carlo Dropout** to all three.
3. Attaches **Deep Ensembles** (M=3 independently-seeded members) to the
   CNN backbone as a second uncertainty method, so the comparison spans
   both backbone choice and uncertainty-method choice, not just backbone
   choice.
4. Evaluates all 4 resulting combinations under matched conditions:
   accuracy, precision, recall, F1, ROC-AUC, ECE, Brier score, predictive
   entropy/variance, train time, and per-sample inference time.
5. Runs the **Decision Layer**: a documented weighted-score selector
   (60% accuracy, 40% calibration-penalty) picks the recommended
   combination from `outputs/metrics/combination_comparison.csv`, saved to
   `outputs/metrics/decision_layer_recommendation.json`.

A representative run's comparison table (numbers will vary run-to-run --
see "Why the numbers are noisy" below):

| Combination            | Accuracy | F1 (macro) | ROC-AUC | ECE   | Train time | Inference/sample |
|-------------------------|----------|------------|---------|-------|-----------|-------------------|
| CNN2D + MC Dropout       | 0.958    | 0.960      | 0.992   | 0.201 | 24.4 s    | 0.042 s |
| DenseNet2D + MC Dropout  | 0.472    | 0.327      | 0.679   | 0.099 | 67.5 s    | 0.123 s |
| ViT2D + MC Dropout       | 0.500    | 0.222      | 0.517   | 0.041 | 16.2 s    | 0.010 s |
| CNN2D + Deep Ensembles   | 0.653    | 0.500      | 0.973   | 0.125 | 57.2 s    | 0.005 s |

*(Recommended by the Decision Layer in this run: **CNN2D + MC Dropout**.)*

This **is not** the full 9-combination study the project document
describes (Bayesian Evidential Deep Learning isn't implemented yet, and
Deep Ensembles is only demonstrated on one backbone -- see the status table
below), but the comparison/ranking/selection machinery itself is fully
general: adding BEDL or the other two Deep-Ensemble pairings is a matter
of calling the same `evaluate.evaluate_combination()` /
`compare_combinations()` / `select_best_combination()` functions on more
result dicts, not rewriting any of this logic.

## Module-wise implementation status

| # | Module (project doc)                    | Status |
|---|-------------------------------------------|--------|
| 1 | Dataset Acquisition                        | **Completed** |
| 2 | MRI Preprocessing (Algorithm 1)            | **Completed** (adapted, see below) |
| 3 | Data Augmentation                          | **Completed** |
| 4 | Dataset Splitting (Algorithm 2)            | **Completed** (subject-level, with documented fallback) |
| 5 | 3D CNN (Algorithm 3)                       | **Completed** as compact 2D CNN |
| 6 | DenseNet-121 (Algorithm 4)                 | **Completed** as compact 2D DenseNet |
| 7 | Vision Transformer (Algorithm 5)           | **Completed** as compact 2D ViT |
| 8 | Monte Carlo Dropout (Algorithm 6)          | **Completed**, applied to all 3 backbones |
| 9 | Bayesian Evidential Deep Learning (Alg. 7) | Not yet started |
| 10| Deep Ensembles (Algorithm 8)               | **Completed** for CNN2D backbone (1 of 3 possible pairings) |
| 11| Comparative Evaluation (Algorithm 9)       | **Completed** for the 4 combinations evaluated (of 9 total) |
| 12| Best Model Recommendation                  | **Completed** (weighted decision layer over evaluated combinations) |

**10 of 12 modules have real, running code -- roughly 60% when you weight
by the fact that modules 11 and 12 currently operate on 4 of 9 possible
combinations rather than the full set**, which is the honest ceiling until
BEDL (module 9) exists. That gap is exactly what the next increment
should close.

## Important scope decisions (read before extending)

1. **Dataset format mismatch.** The Kaggle release (`ninadaithal/imagesoasis`)
   ships pre-extracted, pre-registered 2D axial JPEG slices, not raw NIfTI
   volumes. `data_acquisition.py` / `preprocessing.py` implement the parts
   that are genuinely this project's job (grayscale conversion, intensity
   normalization, resizing) rather than re-doing skull-stripping/registration
   that's already been done upstream.

2. **2D, not 3D.** All three backbones are 2D adaptations (`Conv2d` in
   place of `Conv3d`, patch embedding on a 2D grid instead of 3D patches)
   because the dataset provides slices, not volumes. Each model file
   documents exactly what would change to go 3D once multi-slice-per-subject
   volumes are available.

3. **Compact, not full-scale, architectures.** This sandbox has a single
   CPU core and no GPU. A full torchvision `densenet121` measured at
   ~3.4s/training-iteration here, which would make one backbone alone take
   ~15 minutes; a full ViT-B/16 assumes 224x224 3-channel input and
   ImageNet-pretrained weights this sandbox can't download. `densenet2d.py`
   and `vit2d.py` instead implement the same structural principles (dense
   connectivity with transition layers; patch embedding + [CLS] token +
   multi-head self-attention encoder blocks) at a size that trains in
   seconds-to-low-minutes on CPU. Growth rate / block depth / patch size /
   embedding dimension are the knobs to scale up on GPU.

4. **Class mapping.** The Kaggle release's four folders are mapped to the
   project's three-stage target: `non demented -> CN`,
   `very mild demented -> MCI`, `{mild, moderate} dementia -> AD`
   (`src/config.py::RAW_FOLDER_TO_CLASS`).

5. **Subject-level split fallback.** The demo subset has only 1-2 unique
   subjects per class, so `splitting.py` attempts a genuine subject-level
   split first and falls back to a stratified slice-level split with a
   logged warning when a class doesn't have enough subjects. On the full
   OASIS release (hundreds of subjects/class) the subject-level path is
   the one that actually runs.

6. **Why the numbers are noisy, and why that's expected right now.** With
   ~330 training slices from effectively 4 source scans, every backbone
   overfits fast and validation accuracy is highly sensitive to which
   epoch's checkpoint gets saved -- which is why the recommended combination
   can change between runs. What's being verified at this milestone is
   that every intermediate artifact (manifest, split, per-backbone
   checkpoint, comparison table, decision-layer output, Grad-CAM figures)
   is computed correctly and reproducibly end to end, not that any backbone
   is clinically meaningful yet. Re-run against the full OASIS dataset for
   numbers worth reporting.

## What's next (not in this increment)

- Bayesian Evidential Deep Learning (Algorithm 7) -- Dirichlet evidence
  layer, replacing softmax; the one piece needed to make Comparative
  Evaluation and Best Model Recommendation span the full 3x3=9 combinations.
- Deep Ensembles on the DenseNet2D and ViT2D backbones (currently only
  demonstrated on CNN2D).
- Statistical significance testing across combinations (paired t-test /
  Wilcoxon, as named in the project's architecture diagram) -- currently a
  single weighted-score heuristic, not a statistically tested comparison.
- Attention-rollout-based visualization for ViT2D (Grad-CAM only works
  for the two conv-based backbones right now).
- Re-running the full pipeline against the complete OASIS dataset once
  available, rather than the small demo subset, for numbers worth reporting.
