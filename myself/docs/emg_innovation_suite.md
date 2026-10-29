# EMG innovation suite (2026-10-29)

Status: completed locally and independently verified. This document records the
protocol and commands; results are in the [English report](../results/2026-10-29_emg_innovation/report.md)
and [snapshot](../results/2026-10-29_emg_innovation/README.md). Historical reports remain unchanged.

Execution uses the actual checkout (`main`, d7f55cf at initial inspection), without
switching branches. Windows D: was present but unmounted; restored read-only at
`/mnt/d`. Inputs and supplied checkpoints remain read-only. No bulk input or
feature copies are written. Frozen features are reused in bounded CPU RAM;
predictions, trained parameters, state and compact diagnostics stay in outputs.

## Fixed sequence

1. Strictly load released fusion and independent ResNet18 assets; verify their
   unmodified forward wrappers separately. Audit local normalization once.
2. Revalidate historical 12000-row eligibility, actual 7790-row EMG support,
   metadata, crops, stale/alignment and labels. Build complete execution IDs
   independently of original split. Seed 20261028 assigns whole executions to
   fit/validation; purge centers whose input support touches the other partition.
   Retain T_all for later retraining. Auxiliary endpoints are +/-200 nominal rows;
   mask invalid endpoints only. Never use evaluation errors for design.
3. Real-data tiny smoke covers all seven families and three conditions, reference
   cancellation, conditioned readout, adapter/auxiliary gradients, frozen states
   and serialization. Profile ordinary and difference branches.
4. Exploratory seed-42 absolute/delta and visual-residual probes on T_fit/T_val.
5. Seven families x three conditions x seeds 42/43/44, 30 epochs, Adam 1e-4,
   weight decay 1e-4, effective batch 128 (user-authorized hardware adjustment), deterministic FP32 with TF32 off.
   Difference-family B0 cells are explicitly analytical anchors (9 cells),
   leaving 54 optimization cells. Final epoch is primary.
6. Select exactly one B1 family by mean of three angle-weighted T_val MAEs.
   Within 0.001 degree of lowest: measured inference cost, then parameter count,
   then ID. Save selection_lock.json BEFORE any formal E label-error scoring.
7. Fixed E_dev(wmh), zbk and eligible wsj evaluation, B1 constant/three mismatch/
   raw-pre-encoder reverse interventions; B2 also scored on correct EMG.
8. Selected family and R0 retrained from original seed initialization on T_all,
   same 30 epochs, batch 128 and B0/B1/B2, deduplicating R0 and analytical B0 cells.
9. Independent NumPy float64 verification of saved predictions, weighted groups,
   seed summaries, six English figures and fixed-rule trajectories. Export only
   reviewed compact evidence with source hashes; update navigation.

Shared model: 128 hidden, 4 heads, five finger queries in repository joint order,
4x4 frozen independent-vision tokens with 2D positions, common global/anchor
conditioning, full ordered EMG decoder tokens with index positions (not claimed
physical receptive fields), common trainable adapter and output head, dropout 0.
R0 adds a temporal attention pool; R1/R2 differ only in reference subtraction.
Reference is physical zero through actual sparse layout/normalization/encoder;
both current-adapter branches keep gradients. R3/R6 add EMG-only pose; R4/R5
also add past/future changes, coefficients 1. Outputs remain absolute radians.

All donor mappings are same recording/hand/pool/original split bijections with
nonoverlapping full supports, preferring different executions; failures are
explicit. Training maps change reproducibly per epoch independently of training
seed; inference seeds 101/102/103. Recipient labels always remain unchanged.

Resume must verify source/config/center/asset hashes, preserve failures and restore
optimizer, epoch, order and Python/NumPy/CPU/CUDA RNG. Completed cells are reused
only after identity checks. No incomplete result is marked complete.

## Interpretation

G_method compares this round's R0_B1 with each B1. G_EMG compares each family's
B0 and B1; difference B0 is an anchor, with R1/R6/R5 ordinary B0 shown separately.
Decodeability, sensitivity and conditional increment are different questions.
P3 is offline local-change decoding because the window includes future context.
wmh E and zbk are development/frozen-review recordings, not unseen-user benchmarks.
Checkpoint participant exposure and original statistics bytes remain unresolved.

Commands and validated result links will be added as execution reaches each stage.

## Hardware adjustment authorized during execution

The user explicitly authorized a more aggressive GPU budget and adjustment of
effective batch size. Real T_fit-only hardware profiles compared encoding
16/64/128/256 and head training 32/64/128/256/512, without E errors. Encoding
128 was fastest (about 326 samples/s, 3.0 GiB reserved); 256 was slightly slower
and reserved 5.3 GiB. Head batch 128 substantially reduced epoch overhead while
retaining 24 updates per T_fit epoch (versus 6 at batch 512). All final main
conditions and exploratory probes therefore use effective batch 128; Adam,
learning rate and 30 epochs remain fixed. All final encoding uses batch 128.
FP32 batch-kernel differences were measured: EMG relative L2 about 1.1e-5;
visual anchor maximum error below 1e-6 rad. No precision change was made.

`2026-10-29_emg_innovation_full_01` was interrupted before selection or any formal
E scoring to apply this authorized change. Its checkpoints and failure record
remain intact. Final results use a new run from original seed initializations,
without reusing its fitted weights or probe results.

## Executable entry points

```bash
cd /home/potaxo/research/egoemg/EgoEMG
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/myself/.cache/matplotlib"
PY=/home/potaxo/miniconda3/envs/egoemg/bin/python
# New directories are required for smoke/full; defaults are explicit Preview paths.
$PY myself/scripts/analysis/emg_innovation_suite.py smoke --run-dir myself/outputs/NEW_innovation_smoke
$PY myself/scripts/analysis/emg_innovation_suite.py full --run-dir myself/outputs/NEW_innovation_full
# Resume identical execution code/config/assets, optimizer and RNG from last epoch.
$PY myself/scripts/analysis/emg_innovation_suite.py resume --run-dir myself/outputs/NEW_innovation_full
# After selection lock: independent endpoints, distributions, true pipeline timing,
# and full SHA-256 of only the consumed Preview payloads/crop LMDBs (no copying).
$PY myself/scripts/analysis/audit_emg_innovation.py --run-dir myself/outputs/NEW_innovation_full --time-pipeline --hash-payloads
# After training_complete.json, independent NumPy float64 metrics and figures.
$PY myself/scripts/analysis/summarize_emg_innovation.py summarize --run-dir myself/outputs/NEW_innovation_full
$PY myself/scripts/analysis/summarize_emg_innovation.py verify --run-dir myself/outputs/NEW_innovation_full
$PY myself/scripts/analysis/summarize_emg_innovation.py export --run-dir myself/outputs/NEW_innovation_full --snapshot myself/results/NEW_emg_innovation
```

The final authorized run is `myself/outputs/2026-10-29_emg_innovation_full_02`.
Smoke `2026-10-29_emg_innovation_smoke_03` covers the final code; smoke_02
covered the earlier batch32 version. Smoke_01's dataset API error was fixed and
its failure retained. Hardware profiles are separate from experimental results.

`state.json` lists every planned cell and records analytical versus actual
optimization. `cells/*/last.pt` includes optimizer, step/epoch, last order and all
Python/NumPy/CPU/CUDA RNG states. `orders/` records deterministic data orders and
B2 training maps, reused across families. `identity.json`, `executed_sources/`
and `dirty.patch` capture the code/config and supplied asset identity. A source
or configuration mismatch rejects resume; preserve the failed run and start a
new run for a material change. Batch features are reconstructed into RAM on
resume, never restored from disk. No completed/failed cell is silently relabeled.

`selection_lock.json` records all B1 T_val scores, unique final family, measured
cost tie-break, pool/map hashes, original-initialization T_all rules and timestamp.
`predictions/fit/` and `predictions/all/` keep recipient-aligned predictions for
all conditions and fixed interventions. `training_complete.json` means fitting
and scoring completed; `verified_complete.json` additionally records independent
metric verification. Curated export is separate and never includes full weights,
per-window predictions, caches, bulk features or logs.

Tests and formatting:

```bash
$PY -m pytest myself/egoemg/tests/test_emg_innovation_suite.py \
  myself/egoemg/tests/test_effective_emg_contribution.py \
  myself/egoemg/tests/test_visual_guided_pool_pilot.py \
  egoemg/tests/test_mid_fusion_forward_modes.py \
  egoemg/tests/test_mid_fusion_vision_lock.py -q \
  -o cache_dir=myself/.pytest_cache --basetemp=myself/.cache/innovation_tests
$PY -m black --check myself/egoemg/models/emg_innovation.py \
  myself/scripts/analysis/emg_innovation_suite.py \
  myself/scripts/analysis/summarize_emg_innovation.py \
  myself/scripts/analysis/audit_emg_innovation.py \
  myself/egoemg/tests/test_emg_innovation_suite.py
```

The hardware-only batch sweep can be repeated without fitting methods or scoring E:

```bash
$PY myself/scripts/analysis/profile_emg_innovation_encoding.py \
  --selection-run myself/outputs/2026-10-29_emg_innovation_full_02 \
  --run-dir myself/outputs/NEW_hardware_profile
```

This profiles real T_fit inputs, compares encoding outputs with batch16, and
records training throughput for ordinary and dynamic-difference heads. Its
profile weights are disposable timing artifacts, not experiment checkpoints.

## Completed execution and audit scope

Final identity: `c49b1625007ade78fc323ec0bcca509e1357c01d417aaf0966626a94778c69e0`.
All 63 primary cells and 18 T_all cells completed: 69 optimization cells,
12 analytical B0 cells, no missing/failed final-run cell. Final-related tests
passed 21 cases, including exact Adam/RNG resume continuation; real smoke
covered all 21 family/input combinations. Independent verification recomputed
all 243 primary recording/checkpoint predictions and weighted groups, checked
selection chronology, frozen states, donor maps and T_all initialization.
See snapshot verification.json for the machine-readable checks.

Only consumed Preview payloads were hashed (18 files); no unified scan or input
copy was performed. D: remains mounted read-only. Pipeline timing includes fresh
memmap/LMDB reads, backbones and head with OS caches warmed; cached-head timing
is separate. Snapshot figures were visually inspected. Changes remain local,
without commit or push.
