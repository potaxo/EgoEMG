# Single-participant offline calibration suite

The suite compares the frozen released EMGFormer-M against regularized offsets,
diagonal affine calibration and final angle-linear finger-row adaptation. Each
method independently fits reference labels or the released ResNet18 crop teacher.
It uses all eligible train intervals of wmh / episode_000020 / right hand, and a
fixed EMG-only evaluation pool from user, gesture and both intervals. This is a
preview-recording engineering experiment, not a complete-dataset benchmark.

Run from the existing checkout and environment:

```bash
cd /home/potaxo/research/egoemg/EgoEMG
export PYTHONDONTWRITEBYTECODE=1
/home/potaxo/miniconda3/envs/egoemg/bin/python \
  myself/scripts/analysis/single_subject_calibration_suite.py
```

Each invocation without `--run-dir` creates a readable directory such as
`2026-10-05_wmh_right_suite_full_01` under
`myself/outputs/single_subject_calibration_suite/`; conflicts increment the final
sequence. A tiny machinery check uses
`--limit-per-pool 4` in its own directory; its metrics are not the study results.
The full run uses the default zero limit. Default stride is 200 rows, EMG batch
size 2, teacher batch size 8, feature batch size 256 and RGB delta limit 50 ms.

Existing renamed runs and reports are indexed in `myself/outputs/CURRENT_RUNS.md`.
The old-to-new mapping is `myself/outputs/run_directory_renames.json`. Historical
logs, settings and source hashes remain as originally executed; changing workflow
code does not make those archived caches compatible with a new workflow version.

Phases are `manifest`, `emg`, `teacher`, `fit`, `evaluate`, `report`, or `all`.
Resume with the printed absolute run path and the same original arguments:

```bash
/home/potaxo/miniconda3/envs/egoemg/bin/python \
  myself/scripts/analysis/single_subject_calibration_suite.py \
  --run-dir /absolute/path/to/myself/outputs/single_subject_calibration_suite/RUN \
  --phase all
```

The identity includes workflow/helper/model/configuration hashes, checkpoint
hashes, normalization values, dataset manifest/metadata/vocabulary identities,
recording IDs, preprocessing, window/alignment and all fitting settings. Dataset
arrays are read in place on D: and are not hashed or duplicated. Completion
markers hash each phase's files; incompatible or corrupt completed caches are
rejected. A failed unmarked phase can be rerun; each finished condition also has
an identity and parameter digest. Changing suite/model/config source requires a
new run directory. No raw windows or crop database are copied into caches.

The manifest enumerates starts at each maximal row-level constant-split interval
boundary, stepping 200 while full 12,000-row support remains inside the interval.
Selection checks all EMG inputs, finger validity at the center and finite center
finger angles. Context labels need not be valid. Wrist validity is separate.
Fitting and evaluation input unions cannot intersect; chronological separation
is unnecessary. Overlaps within pools are allowed. Support duration is the
interval-union row count divided by the nominal 2,000 Hz; nominal center coverage
uses the union of `[center, center+stride)`. Timestamps need not be uniform.

Normalization uses `egoemg__filtered_paper_right` in
`assets/per_dataset_norm_stats_repro_filtered_paper_alias.json`, exactly as the
preserved smoke workflow. The original loader handles normalization, channel
order, radians and center targets. No augmentation or normalization estimation
is performed. The last actual linear layer's input includes preceding frozen
MLP operations. Its output is followed only by reshape/transpose and linear
interpolation. Every extracted center verifies that interpolating those features
and applying the released linear layer matches the original aligned prediction,
with `atol=1e-5, rtol=1e-4`. The observed maximum discrepancy is recorded.

The teacher strictly loads the locally verified ResNet18 architecture, bypassing
constructor ImageNet downloads because the checkpoint supplies all weights.
The existing per-episode LMDB reader uses `%08d_R`; stored RGB 256x256 crops are
not mirrored or resized. ImageNet normalization uses the loader's pixel-scale
means/stds. Each unique mapped train frame is read/inferred once. Eligibility
requires nonnegative mapping, nonstale data, finite absolute delta <= 50 ms,
a decoded/preprocessed crop and finite 20 finger predictions. Exclusions are
first-failure mapping, stale, nonfinite delta, delta tolerance, missing crop,
decode, preprocessing or nonfinite prediction. No reference agreement is used.
The tolerance is a first pass, not a synchronization guarantee. Evaluation does
not require RGB. An unavailable teacher leaves independent reference conditions
running and records exact blockers.

Offsets use lambda_b=1. Affine uses float64 normal equations, lambda_a=.01 and
lambda_b=1, with variance < 1e-6 rad² falling back to the offset solution. No
clipping is applied. Head fitting exposes only 20 finger rows/biases to Adam;
wrist rows do not exist in its optimizer. Each seed 42/43/44 starts from released
parameters, trains 50 epochs at lr=.001, weight_decay=0, with .001 mean squared
weight and bias deviations from initialization. The final epoch is scored.
No evaluation selection or early stopping occurs. Reference fits on T_rgb are
added if its coverage differs from T_all; otherwise primary results are reused.

`evaluation_predictions.npz` includes identical center IDs, reference labels,
masks and every condition/seed prediction. `manifest.npz` retains all pools and
center labels. `emg_cache.npz` contains compact center features, original
predictions and original final-layer parameters; `teacher_cache.npz` retains
separate teacher targets, unique-frame predictions and eligibility reasons.
Compact adapted parameters and per-epoch losses live in `parameters/`.

Inspect `report.md`, `primary_comparison.md`, `matched_comparison.md`,
`all_seeds.md`, the corresponding CSVs, `detailed_metrics.csv`,
`reference_variation.csv`, `coverage.json`, `gesture_coverage.csv`,
`coverage.md`, `rgb_interval_coverage.csv`, `rgb_gesture_coverage.csv`, teacher diagnostics,
contact sheet, and scientific plots. Per-joint and per-gesture plots use all
head seeds' mean. Traces take the first available active gesture per evaluation
split and up to 80 contiguous centers from its first occurrence, at index_mcp_fe; errors do not select traces.

MAE is absolute angular difference without wrapping/clipping/smoothing, averaged
across all valid finger entries, in radians and degrees. Detailed gesture rows
keep raw ID, name and active flag; no gesture macro-average is substituted for
the primary score. Head summaries use sample SD across seeds (ddof=1), not a
population confidence interval. Wrist predictions and metrics are checked for
exact equality across conditions. GPU memory reports peak allocated PyTorch
bytes after stage/seed reset, including resident tensors and transients, excluding
allocator-reserved/driver memory and other processes. Shared extraction/teacher
construction time is separated from per-method fitting time.

Focused verification, with all task caches under myself/:

```bash
/home/potaxo/miniconda3/envs/egoemg/bin/python -m pytest \
  myself/egoemg/tests/test_single_subject_calibration_suite.py \
  myself/egoemg/tests/test_preview_calibration_smoke.py \
  egoemg/tests/test_release_eval_helpers.py \
  egoemg/tests/test_egoemg_memmap_dataset.py egoemg/tests/test_metrics.py \
  egoemg/tests/test_rotary_attention_mask.py \
  -q -o cache_dir="$PWD/myself/.pytest_cache" \
  --basetemp="$PWD/myself/.cache/suite_pytest"
/home/potaxo/miniconda3/envs/egoemg/bin/python -m black --check \
  myself/scripts/analysis/single_subject_calibration_suite.py \
  myself/egoemg/tests/test_single_subject_calibration_suite.py
/home/potaxo/miniconda3/envs/egoemg/bin/python -m isort --profile black --check-only \
  myself/scripts/analysis/single_subject_calibration_suite.py \
  myself/egoemg/tests/test_single_subject_calibration_suite.py
```

Whole-recording FFT filtering is noncausal, and train intervals may follow
those evaluated. Checkpoint exposure of wmh is unresolved. Results do not show
unseen-user generalization, short-calibration retention, causal deployment or a
REACT reproduction. The older 15-second smoke workflow and artifacts are preserved.
