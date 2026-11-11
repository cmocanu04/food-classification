# Food-11 Training Plan (explicit)

Deadline: **Sunday 15 Nov, 23:59** (leaderboard + code + report on Moodle).
Today: 11 Nov -> 4 days. Plan: experiments 11-13 Nov, selection 13 Nov,
final train + submission 13-14 Nov, buffer 15 Nov.

Interpretation of current results (EfficientNetB0, 64x64):

| Setup            | train | val  | Diagnosis |
|------------------|-------|------|-----------|
| head only        | 62%   | 51%  | underfits: 64x64 destroys the texture signal the frozen ImageNet features need |
| all layers       | 94%   | 72%  | 22-point gap: overfits (13k images, no augmentation) |

So fine-tuning everything is clearly the right direction. The two fixes
(resolution 224 + augmentation) attack the *underfitting* and the
*overfitting* at the same time. Expected val macro-F1 after both fixes:
~0.78-0.84.

---

## 0. Fixed experimental protocol (identical for every architecture)

Fair comparison requires holding everything except the architecture fixed:

- **Split:** 80/20, stratified per class, `rng(42)` (fixed seed, same for all
  networks). Names: `trainData` / `valData`, labels `labelsTrain` / `labelsVal`.
- **Resolution:** 64x64 for the experiments (fast iteration), then
  **224x224 for the final training** (native ImageNet size). WARNING:
  architecture rankings can shift between 64 and 224 - before the final
  training, run one confirmation of the top-2 candidates at 224.
  LeNet stays at 64x64 as the from-scratch baseline.
- **Preprocessing:** identical resize + `im2single` path at train and predict
  time (this is what `predict_test_set.m` replicates).
- **Metrics:** accuracy AND macro F1 (F1 is the leaderboard metric).
  Macro F1 from the confusion matrix (per class c):
  `P = diag/(col sums)`, `R = diag/(row sums)`,
  `F1_c = 2PR/(P+R)` (0 if P+R=0), macro F1 = mean over 11 classes.
  Always report per-class F1 + confusion matrix (rice: 376 train images,
  noodles: 587 -> expect these to be the weak classes; that discussion is
  report material).

## 1. Data loading: augmentedImageDatastore

Implemented in `FoodClassification_database.m` (datastore-based, works at
any resolution): set `inputSize = [64 64 3]` for experiments and
`[224 224 3]` for the final training. In-memory 4-D arrays no longer fit
in RAM at 224 (13296 * 224 * 224 * 3 * 4 B ~ 8 GB); datastores resize on
the fly, feed `trainnet` directly, and give augmentation for free.

```matlab
% FoodClassification_database_224.m
clearvars; close all;
rng(42)                                    % FIXED seed: same split for all nets
inputSize = [224 224 3];

root = fullfile(fileparts(mfilename("fullpath")), "food-11", "food-11");
data = imageDatastore(fullfile(root, "train"), ...
    IncludeSubfolders=true, LabelSource="foldernames");
[trainData, valData] = splitEachLabel(data, 0.8, "randomized");
labelsTrain = trainData.Labels;
labelsVal   = valData.Labels;

% ---- augmentation: TRAIN only ----
augmenter = imageDataAugmenter( ...
    RandXReflection = true, ...            % horizontal flip: plates look the same mirrored
    RandRotation     = [-15 15], ...       % small rotations: camera angle
    RandXTranslation = [-10 10], ...       % food not always centered
    RandYTranslation = [-10 10], ...
    RandXScale       = [0.9 1.1], ...      % slight zoom in/out
    RandYScale       = [0.9 1.1]);
% NO vertical flip (upside-down plates are not natural), NO color jitter
% (imageDataAugmenter has no color ops; color is informative for food).

augTrain = augmentedImageDatastore(inputSize(1:2), trainData, ...
    DataAugmentation = augmenter);

% validation: resize ONLY, never augment
augVal = augmentedImageDatastore(inputSize(1:2), valData);
```

Notes:
- `augTrain` yields a *different* random variant of each image at every
  epoch -> effectively infinite data -> closes the 22-point overfitting gap.
- Augmentation is disabled on validation: val must measure generalization.
- The input layer of the pretrained net keeps its own Normalization
  (rescale-symmetric etc.), applied automatically after the datastore.

## 2. Architectures to compare

Frugality penalty = 0.3 / (1 + 1/R), R in millions of parameters
(w = 0.3). The penalty saturates: big nets are cheap in frugality terms
but must win F1 to justify themselves.

| Network          | ~params (11-class head) | penalty | break-even vs MobileNetV2 |
|------------------|-------------------------|---------|---------------------------|
| LeNet (baseline) | ~0.06 M                 | 0.005   | - (reference, from scratch, 64x64) |
| SqueezeNet       | ~1.2 M                  | 0.164   | can be 4.2% F1 WORSE and still win |
| MobileNetV2      | ~2.2 M                  | 0.206   | reference |
| EfficientNetB0   | ~4.0 M                  | 0.240   | needs +3.4% F1 over MobileNetV2 |
| ResNet-18        | ~11.2 M                 | 0.275   | needs +6.9% F1 over MobileNetV2 |

Minimum required by the challenge: 2 networks. Recommended set of 4 for a
strong report: LeNet (done) + MobileNetV2 + EfficientNetB0 + ResNet-18.
ResNet-18 mostly serves the frugality trade-off analysis in the report,
even if it does not win the leaderboard.

## 3. Two-stage transfer-learning recipe (exact hyperparameters)

Implemented as decommentable TEST blocks, one file per architecture:
`MobileNetV2_Implementation.m`, `EfficientNetB0_Implementation.m`,
`ResNet18_Implementation.m`. In each file, decomment ONE block and run:

- **Test 1 (head only):** backbone frozen (factor 0), head factor 10,
  adam, base LR 1e-4 (effective head LR 1e-3), batch 32, 10 epochs,
  patience 4, validation once per epoch.
- **Test 2 (partial fine-tune):** last K backbone layers unfrozen
  (K = 24 MobileNetV2 / 20 EfficientNetB0 / 15 ResNet-18, ~last block or
  stage + head), LR 1e-4 uniform on unfrozen layers, batch 32, 8 epochs,
  patience 3. Optional warm start from Test 1's saved net (lines provided).
- **Test 3 (full fine-tune):** all layers unfrozen, base LR 1e-5 with head
  factor 10 (head 1e-4, backbone 1e-5), batch 32, 8 epochs, patience 3.
  Optional warm start from Test 1's saved net (lines provided).

Each test saves its own .mat (e.g. `EfficientNetB0_Food11_test3_full.mat`)
and prints accuracy + macro F1 + per-class F1 + confusion matrix via
`evaluateFood11Metrics.m`. Run at least Tests 1 and 2/3 per architecture;
the 1-vs-2-vs-3 comparison is the ablation section of the report.

Your own earlier result (72% val full fine-tune vs 51% head-only at 64x64,
without augmentation) says fine-tuning the backbone is necessary. Test 2
(partial) is the sensible default; Test 3 (full) is kept because it may
still win once augmentation is on - let the numbers decide.

## 4. Experiment run sheet (what to test, in order)

Phase 1 - the 3 tests x 3 architectures at 64x64. Everything fixed except
the test block (this IS the main hyper-parameter study: freezing strategy).
`rng(42)` guarantees the same split for every run.

| #  | Script (decomment block) | Test | What varies | Purpose | Time (5050) |
|----|--------------------------|------|-------------|---------|-------------|
| 1  | MobileNetV2_Impl.        | 1    | head only, LR head 1e-3, 10 ep | baseline of the arch | ~10 min |
| 2  | MobileNetV2_Impl.        | 2    | partial (K=24), LR 1e-4, 8 ep  | partial vs head     | ~15 min |
| 3  | MobileNetV2_Impl.        | 3    | full, LR 1e-5/1e-4, 8 ep      | full vs partial     | ~20 min |
| 4  | EfficientNetB0_Impl.     | 1    | idem run 1                     | same on arch 2      | ~10 min |
| 5  | EfficientNetB0_Impl.     | 2    | partial (K=20), LR 1e-4        |                     | ~15 min |
| 6  | EfficientNetB0_Impl.     | 3    | full, LR 1e-5/1e-4             |                     | ~20 min |
| 7  | ResNet18_Implementation  | 1    | head only                      | same on arch 3      | ~10 min |
| 8  | ResNet18_Implementation  | 2    | partial (K=15), LR 1e-4        |                     | ~15 min |
| 9  | ResNet18_Implementation  | 3    | full, LR 1e-5/1e-4             |                     | ~20 min |

Decision after Phase 1: best (architecture, test) by val macro-F1 ->
compute Frug for the top candidates with the penalty table in section 2.

Phase 2 - LR sensitivity on the WINNER only (3 runs, one variable at a
time, everything else identical to the winning run):

| #   | Script | Test | What varies            | Values              | Purpose |
|-----|--------|------|------------------------|---------------------|---------|
| 10  | winner | same | learning rate only     | 0.5x / 1x / 5x the winning LR (e.g. 5e-6, 1e-5, 5e-5) | is the LR well chosen? |
| 11  | winner | same | warm start from Test 1 | ImageNet start vs Test-1 start | two-stage vs single-stage |

Phase 3 - resolution confirmation at 224x224 (database script: set
inputSize = [224 224 3]):

| #   | Script | Test | What varies | Purpose |
|-----|--------|------|-------------|---------|
| 12  | top-1 arch | its best test | resolution 64 -> 224 | confirm ranking holds |
| 13  | top-2 arch | its best test | resolution 64 -> 224 | catch a ranking flip |

Phase 4 - final training + submission:

| #   | What | Details |
|-----|------|---------|
| 14  | retrain winner on 100% of train (224) | same epochs as early stopping chose; no val split (or 5% monitoring) |
| 15  | predict_test_set.m -> predictions.json | sanity checks: 3097 entries, 11 class names, distribution mirrors train |
| 16  | leaderboard submission + method description | ONE submission, no duplicates |

Budget: Phase 1 = one evening; Phase 2 = ~1 h; Phase 3 = ~1-2 h;
Phase 4 = ~1 h + submission. Fits comfortably before Nov 15.

Tips:
- Run the table top to bottom and fill val accuracy + val macro-F1 for
  each row immediately after it finishes (the numbers ARE the report).
- Do not change two things between two runs. One variable at a time.
- If a run diverges (loss -> NaN, F1 ~ 0.09 = random), halve the LR and
  rerun that row; note it in the report.

Results log (fill as runs finish):

| #  | Arch | Test | val acc | val macro-F1 | notes |
|----|------|------|---------|--------------|-------|
| 1  |      |      |         |              |       |
| 2  |      |      |         |              |       |
| 3  |      |      |         |              |       |
| 4  |      |      |         |              |       |
| 5  |      |      |         |              |       |
| 6  |      |      |         |              |       |
| 7  |      |      |         |              |       |
| 8  |      |      |         |              |       |
| 9  |      |      |         |              |       |
| 10 |      |      |         |              |       |
| 11 |      |      |         |              |       |
| 12 |      |      |         |              |       |
| 13 |      |      |         |              |       |

## 5. Model selection

- Rank by **val macro-F1**, then compute `Frug = valF1 - penalty` with the
  table above. The leaderboard entry should be the network with the best
  *expected Frug*, not the best F1 (e.g. EffNetB0 must beat MobileNetV2 by
  more than 3.4% F1 to be worth submitting).
- Check the confusion matrices of the top two before deciding: if one wins
  mainly by fixing rice/noodles (the rare classes), macro F1 will reward it
  more than accuracy suggests.

## 6. Final training on 100% + submission

1. Take the winning recipe (architecture + stage A + stage B settings).
2. Retrain on **all 13296 train images** (no val split, or keep 5% just to
   monitor that training is stable). Use the *same number of epochs* that
   early stopping chose in the 80% runs, not more (no early stopping
   possible without val data).
3. `predict_test_set.m` on the final `.mat` -> `predictions.json`.
4. Sanity checks before upload:
   - 3097 entries, keys = test image names without extension
   - values are the 11 lowercase folder names, nothing else
   - predicted distribution roughly mirrors train proportions
   - `jsondecode(fileread("predictions.json"))` parses without error
5. One careful leaderboard submission with a short method description
   (architecture, params, F1, resolution, fine-tuning strategy). No
   duplicates - resubmission overwrites.

## 7. Report skeleton (10 pages max)

1. Problem, dataset, frugality metric (incl. the 0.843 break-even answer)
2. Protocol: split, seeds, metrics, augmentation policy
3. Baseline: LeNet from scratch (why it is weak: no transfer, 64x64)
4. Transfer learning: 2-stage recipe, what each stage does
5. Architecture comparison (the run-sheet table + confusion matrices)
6. Frugality analysis (penalty table, break-evens, final choice)
7. Per-class analysis (rice/noodles imbalance), errors, limitations
8. Conclusion + leaderboard result
