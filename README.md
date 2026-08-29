# Uncertainty-Aware sMRI-Based AD Staging — Full Implementation (100%)

Complete implementation of the pipeline described in *Uncertainty-Aware
sMRI-Based Alzheimer's Disease Staging: A Comparative Benchmarking Study*
(TARP.pdf), adapted from raw NIfTI/3D volumes to the actual data format
available (2D axial slices from the OASIS Kaggle release) and from
GPU-scale architectures to CPU-trainable compact equivalents — both
substitutions are documented in detail below and in the source files
themselves. All 12 modules and all 9 architecture-uncertainty combinations
named in the project document have real, running code.

## How to run

```bash
pip install torch torchvision scikit-learn scipy pandas matplotlib seaborn
python run_pipeline.py
```

Trains 3 backbones and attaches 3 uncertainty methods to each (9
combinations total), so a full run takes longer than earlier increments —
budget several minutes to a few tens of minutes on CPU depending on
retries (see "Known limitations" below). Every stage prints its own
intermediate output. All artifacts land in `outputs/`:

Each module also runs standalone, e.g.:
```bash
python -m src.data_acquisition
python -m src.models.cnn2d              # / densenet2d / vit2d
python -m src.uncertainty.mc_dropout
python -m src.uncertainty.deep_ensembles
python -m src.uncertainty.bedl
python -m src.comparative_evaluation      # trains+evaluates all 9 combinations, compares, selects best
python -m src.visualize_pipeline
```

## Module-wise implementation status — 12 of 12 complete

| # | Module (project doc)                       | Status | Where |
|---|----------------------------------------------|--------|-------|
| 1 | Dataset Acquisition                           | **Completed** | `src/data_acquisition.py` |
| 2 | MRI Preprocessing (Algorithm 1)               | **Completed** (adapted, see below) | `src/preprocessing.py` |
| 3 | Data Augmentation                             | **Completed** | `src/augmentation.py` |
| 4 | Dataset Splitting (Algorithm 2)               | **Completed** (subject-level, documented fallback) | `src/splitting.py` |
| 5 | 3D CNN (Algorithm 3)                          | **Completed** as compact 2D CNN | `src/models/cnn2d.py` |
| 6 | DenseNet-121 (Algorithm 4)                    | **Completed** as compact 2D DenseNet | `src/models/densenet2d.py` |
| 7 | Vision Transformer (Algorithm 5)              | **Completed** as compact 2D ViT | `src/models/vit2d.py` |
| 8 | Monte Carlo Dropout (Algorithm 6)             | **Completed**, all 3 backbones | `src/uncertainty/mc_dropout.py` |
| 9 | Bayesian Evidential Deep Learning (Algorithm 7)| **Completed**, all 3 backbones | `src/uncertainty/bedl.py` |
| 10| Deep Ensembles (Algorithm 8)                  | **Completed**, all 3 backbones | `src/uncertainty/deep_ensembles.py` |
| 11| Comparative Evaluation (Algorithm 9)          | **Completed**, all 9 combinations + statistical comparator | `src/evaluate.py`, `src/comparative_evaluation.py` |
| 12| Best Model Recommendation / Decision Layer    | **Completed**, weighted score + significance testing | `src/evaluate.py::select_best_combination` |

Training, comparison, and the visual output pipeline are all orchestrated
end to end by `run_pipeline.py`.

## What "100%" means here, precisely

Every module the project document lists now has real, executable code
covering its stated purpose, and the benchmarking framework runs the full
3×3 = 9 architecture-uncertainty grid the document's novelty section
describes — not a subset. Three deliberate, documented substitutions
carry through every module (unchanged from earlier increments, repeated
here for completeness):

1. **2D slices, not 3D volumes.** The Kaggle release actually used
   (`ninadaithal/imagesoasis`) ships pre-extracted, pre-registered 2D
   axial JPEG slices rather than raw NIfTI volumes. Every "3D" module in
   the doc (3D CNN, DenseNet-121 (3D), skull-stripping/registration steps)
   is implemented as its 2D-input equivalent, with the exact
   3D-conversion path documented in each file's docstring.
2. **Compact, not full-scale, architectures.** Sized to train on a single
   CPU core in reasonable time — full torchvision `densenet121` and
   ImageNet-pretrained ViT-B/16 were both measured/considered and
   rejected for this sandbox (see `densenet2d.py`/`vit2d.py` docstrings).
   Growth rate / depth / patch size are the parameters to scale up on GPU.
3. **BEDL reuses the same three backbone classes**, changing only the
   loss function (evidential loss, Sensoy et al. 2018) and the inference
   interpretation of the same raw logits (softplus → Dirichlet evidence
   instead of softmax → probability) — no new architecture was needed,
   which is also why it was straightforward to extend to all three
   backbones rather than just one.

## What's new in this increment

- **Bayesian Evidential Deep Learning (Algorithm 7)** — `src/uncertainty/bedl.py`.
  Implements the full evidential loss (expected cross-entropy under the
  Dirichlet + annealed KL regularizer penalizing misleading evidence),
  training with the same accuracy-based checkpointing and degenerate-model
  retry logic as `train.py`, and inference that reports both an epistemic
  (vacuity, `K/S`) and aleatoric uncertainty score per prediction.
- **Deep Ensembles extended to all 3 backbones** (previously CNN2D only).
- **Class-weighted loss** (`train.py::compute_class_weights`,
  `bedl.py`'s `class_weights` argument) — this dataset is ~50% AD / 25%
  CN / 25% MCI, which was silently biasing every backbone toward
  predicting AD regardless of the true class; both the cross-entropy and
  evidential losses now weight each sample by inverse class frequency,
  computed from the train split only.
- **Statistical Comparator** (`evaluate.py::statistical_comparison`) —
  paired t-test and Wilcoxon signed-rank test between the top-accuracy
  combination and every runner-up, computed on matched per-sample
  correctness over the shared test split. The Decision Layer now reports,
  for its recommended combination, how many of the other 8 combinations
  it's statistically significantly better than (`p<0.05`), not just a
  raw score — matching the project document's "Statistical Comparator"
  component in the Evaluation Layer.
- **Visualization dispatch bug fixed** — the 4-panel visual walkthrough
  previously always ran plain MC Dropout on a single backbone checkpoint
  regardless of which combination the Decision Layer actually recommended,
  so a recommended Deep Ensemble or BEDL combination's demo silently showed
  a different (sometimes much worse) model's behavior instead. `visualize_pipeline.build_predict_fn()`
  now dispatches to the real mechanism — MC Dropout, Deep Ensembles average,
  or BEDL — for whichever combination is actually being shown.

## The visual output (input → preprocessed → classified → final output)

Unchanged in mechanism from earlier increments, now driven by whichever
combination is actually recommended (see bug fix above). For each sample:

1. **Input image** — the raw slice exactly as it exists on disk.
2. **Preprocessed image** — after resize + intensity normalization (the
   actual array the model consumes), rescaled only for display.
3. **Classified image** — Grad-CAM saliency overlay (needs a spatial
   feature map, so it runs against a representative conv-based model:
   the backbone itself for MC Dropout/BEDL, ensemble member 0 for Deep
   Ensembles — noted in `visualize_pipeline.py`; ViT2D recommendations
   fall back to the best conv-based combination for this panel only).
4. **Final output** — predicted stage, per-class probability bars, and
   the real uncertainty score of the actual recommended mechanism
   (predictive variance/entropy for MC Dropout and Deep Ensembles;
   vacuity + aleatoric uncertainty for BEDL), plus a plain-language
   "flagged for review" / "confident prediction" label.

## Known limitations (dataset, not code)

- **Demo subset is tiny**: ~4 unique subjects total across CN/MCI/AD, so
  `splitting.py` falls back to a stratified slice-level split (logged
  explicitly) rather than the genuine subject-level split the project
  calls for. On the full OASIS release (hundreds of subjects/class) the
  subject-level path is the one that actually runs, with zero code changes.
- **Training stability on this tiny dataset**: validation loss can swing
  wildly between epochs (a handful of misclassified slices among ~70
  validation samples move the loss a lot), which is why every training
  function here checkpoints by **validation accuracy**, not loss, and
  retries from a fresh seed if a run collapses to the majority-class
  baseline. This makes results reproducible enough to trust the pipeline
  itself, but backbone-vs-backbone accuracy numbers will still vary
  noticeably run to run — check your own
  `outputs/metrics/combination_comparison.csv` for current numbers rather
  than treating any single run's table as definitive.
- **Real accuracy gains from here are a data problem, not a missing
  module**: every module the project specifies is implemented; the
  ceiling on accuracy with ~4 source subjects is a sample-size limit, not
  a gap in the pipeline. Re-running the same code against the full OASIS
  dataset is the next step for numbers worth reporting, not further
  module development.

