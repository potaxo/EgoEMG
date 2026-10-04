# One-participant offline preview calibration smoke test

Run from this checkout with the existing `egoemg` interpreter. This workflow
freezes released EMGFormer-M, fits a reference-supervised finger offset on earlier
centers, and scores the same later centers before and after correction. It does
not establish unseen-user performance or RGB calibration efficacy.

```bash
export EGOEMG_OUTPUT_DIR=/home/potaxo/research/egoemg/EgoEMG/myself/outputs/preview_calibration_smoke
export PYTHONPYCACHEPREFIX=/home/potaxo/research/egoemg/EgoEMG/myself/.cache/pycache
/home/potaxo/miniconda3/envs/egoemg/bin/python myself/scripts/analysis/preview_calibration_smoke.py \
  --data-root /mnt/d/egoemg/data/dataset_egoemg_preview \
  --checkpoint /home/potaxo/research/egoemg/EgoEMG/checkpoints/egoemg_emgformer_middle.ckpt \
  --output-dir "$EGOEMG_OUTPUT_DIR" \
  --episode episode_000020 --seed 42 \
  --calibration-seconds 15 --evaluation-seconds 30 \
  --regularization 1.0 --stride 200 --batch-size 2

/home/potaxo/miniconda3/envs/egoemg/bin/python -m pytest \
  myself/egoemg/tests/test_preview_calibration_smoke.py \
  egoemg/tests/test_egoemg_memmap_dataset.py \
  egoemg/tests/test_release_eval_helpers.py \
  egoemg/tests/test_metrics.py -q \
  -o cache_dir=/home/potaxo/research/egoemg/EgoEMG/myself/.pytest_cache
```

`--data-root` accepts the preview package or its absolute `data/memmap_data` path.
It never substitutes another dataset.

All added workflow files and generated results live under `myself/`, mirroring
the original repository layout. Original model/configuration/assets/checkpoints
remain inputs from the repository root. Earlier run artifacts were moved intact;
`myself/outputs/preview_calibration_smoke/relocation.json` maps their old paths to
current locations. Their saved paths and source hashes describe the original
execution, not a rerun after relocation. Historical artifacts are local outputs
and are excluded from Git.

Each invocation creates a timestamp/UUID
subdirectory below the output parent; a smaller batch can be used if GPU memory
is insufficient. No Trainer, training entrypoint, Hydra job, or logger is
instantiated. Hydra composition sets `train=false`, one device, and explicit run
and logger destinations inside that subdirectory; the resolved configuration is
saved there. Local base/fusion configurations otherwise default to training and
may write to `logs/` or the launch directory.

## Selection and alignment

Metadata episode names resolve participants; metadata subject integers are not
interpreted as row IDs. `frame_split_id` supplies row labels; episode split values
are ignored. The earliest contiguous train/valid/finite span is selected without
looking at prediction errors. Complete 12,000-row supports must fit inside the
episode and train block. The latest calibration end is at most the earliest
evaluation start. Windows within each phase can overlap. If the requested pair
does not fit anywhere in this episode, 5/15 seconds is attempted; failure reports
the train blocks rather than choosing another participant.

The nominal sample rate is 2,000 Hz (`EMG_SAMPLE_RATE` and the released FFT filter's
`FS`), even though stored timestamps have repeated/nonuniform increments. Stride
200 samples gives ten centers per nominal second. Durations define half-open
center intervals; the last sampled center precedes the interval end by up to one
stride. Input context and phase separation add rows beyond those intervals.

The matching local config is `emgformer/egoemg_emgformer_middle`: eight input
channels, slim TDS (strides 5, 2, 5), six transformer layers of dimension 256,
eight heads, FFN dimension 1,024, and a 22-output MLP. Strict loading reuses local
model APIs while ignoring obsolete checkpoint import paths and training init
paths. Parameters are frozen; loading and inference are deterministic, FP32,
`eval()` and `torch.inference_mode()` on one GPU, without jitter or augmentation.

The dataset loader supplies `emg_right_filtered_paper`, stored channels 0–7 in
their existing order, no channel interpolation/rotation, and transpose to C,T.
Normalization uses the released evaluation alias file
`assets/per_dataset_norm_stats_repro_filtered_paper_alias.json`, dataset name
`egoemg` as in `test_analysis`, and its eight-channel right-hand mean/std. No new
statistics are estimated. The standard training dataset config's
`egoemg_unified` normalization key is not used with this legacy alias asset.

A window produces 230 output time steps and left/right context 510/0. The pilot
uses `model.align_predictions` (linear interpolation, `align_corners=False`) to
the 11,490-row context-trimmed timeline, then extracts index 5,490 corresponding
to input center row 6,000. This avoids guessing a native output index. The
repository's center-only fast evaluation instead selects the native output
middle step; this pilot explicitly uses full-sequence resampling at the input
center and is not the pooled benchmark metric. Both phases use the same rule.

## Calibration, masks, and outputs

Indices 0–19 are the manifest's ordered finger angles, already in radians:
thumb CMC FE/AA, MCP FE, IP FE; then index/middle/ring/pinky MCP AA/FE, PIP FE,
DIP FE. Outputs 20/21 are wrist pitch/yaw; the loader converts stored degrees to
radians. Finger supervision uses `generated_label_valid[:,1]`, broadcast across
the 20 joints, plus finite references. Wrist scoring additionally requires
`mocap_right_wrist_angles_valid`; the existing general loader does not surface
this wrist mask. Wrist correction is always zero and wrist metrics are separate.

For each finger, `offset = mean(reference - prediction) / (1 + lambda)` on valid
calibration centers, with fixed `lambda=1.0`. A joint without observations gets
zero offset and a zero coverage count. Later predictions receive that fixed
offset. Angular MAE is absolute difference without wrapping, clipping, smoothing,
or representation conversion. Aggregates average all valid finger entries;
degrees are radians × 180/π. Negative `corrected - baseline` means improvement.
Improvement is not required for a successful smoke test.

Each run contains `settings.json`, `resolved_config.yaml`, `run.log`, `windows.csv`,
`offsets.json`, `metrics.json`, `per_joint.csv`, `center_predictions.npz`, and
`summary.md`. Provenance includes Git dirty state, interpreter/package versions,
device/seed, model configuration, small manifest/metadata and normalization
hashes, checkpoint SHA-256, preprocessing, selection, masks, units, and alignment.
No input windows or checkpoint copies are saved and no large data arrays are
hashed. Raw assets are read directly from the external disk.

Released filtering is whole-recording FFT processing and can use future samples.
Disjoint supports establish sample-selection isolation, not causal preprocessing.

## RGB readiness boundary

The preview's video/crop roots are siblings of `memmap_data`. The workflow reads
`image_head_frame_index` at the first selected calibration center and uses the
existing per-episode LMDB crop reader with key `frame:08d_R`. A decoded image is
saved as `right_hand_crop.png`; row/frame indices, stale/delta fields, paths,
readiness gaps and teacher checkpoint presence are in `rgb_readiness.json`.
The generic vision-enabled dataset constructor requires `episode_000000.lmdb`,
which the preview lacks; the existing per-episode reader is callable through the
EMG-only instance without changing global loader behavior.

`fusion/vision_resnet18` predicts the same 20 finger plus 2 wrist angles from one
ImageNet-normalized 256-pixel crop. Before RGB supervision, verify exact
frame/time alignment and stale tolerances, crop orientation/hand identity,
preprocessing, strict teacher loading, joint order/units, and wrist masking.
This run does not instantiate the teacher, download assets, or perform conversion.
