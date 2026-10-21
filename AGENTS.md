# EgoEMG repository guide

## Environment and paths

- WSL2; conda environment `egoemg`; RTX 4060 Laptop GPU.
- Repository: `/home/potaxo/research/egoemg/EgoEMG`
- Working branch: `reproduce/baseline`
- Preview data: `/mnt/d/egoemg/data/dataset_egoemg_preview`
  (Windows: `D:\egoemg\data\dataset_egoemg_preview`).
- Checkpoints: `/home/potaxo/research/egoemg/EgoEMG/checkpoints`
- Personal experiment outputs: repository-local `myself/outputs/<run-name>/`.

## Dataset storage rule

- Read inputs directly from user-designated paths, currently `/mnt/d/egoemg/data/`
  (Windows `D:\egoemg\data\`). Without explicit permission, do not copy, move, stage,
  extract, or bulk-cache inputs in the repository, `myself/`, WSL, or temporary
  directories. This covers memmaps, shards, videos, crop/LMDB databases,
  archives, and converted copies; speed or convenience is not authorization.
- If a path is unavailable, inspect the designated mount and directories first;
  never silently substitute data. Essential duplication requires explicit
  permission after explaining its purpose and expected size.
- Small inspection previews and experiment outputs belong under `myself/`,
  never as bulk input copies. Keep supplied checkpoints at their specified paths.

## Personal work directory rule

- Add personal scripts, docs, tests, and config deltas under `myself/`, mirroring
  repository paths: `scripts/analysis/`, `docs/`, `egoemg/tests/`, `config/`.
- Keep all generated artifacts, temporary files, and caches under `myself/`.
  Use distinct, descriptive `myself/outputs/<run-name>/` directories for full
  runs, including logs, metrics, plots, predictions, checkpoints, and Hydra output.
  Use `myself/.pytest_cache` and `myself/.cache/pycache`, or disable bytecode.
- Reuse original code, configs, normalization assets, and supplied checkpoints.
  Resolve the repository root separately from `myself/` so script relocation
  does not change input paths.
- Preserve user work, staged changes, dataset/checkpoint bytes, and unrelated
  files. Maintain this root `AGENTS.md` as needed; edit other original files when
  explicitly requested. For run renames, record old-to-new paths and preserve
  historical logs and provenance as executed.

```bash
conda activate egoemg
cd /home/potaxo/research/egoemg/EgoEMG
export EGOEMG_ROOT="$PWD"
export EGOEMG_PREVIEW_DIR=/mnt/d/egoemg/data/dataset_egoemg_preview
export EGOEMG_MEMMAP_DIR="$EGOEMG_PREVIEW_DIR/data/memmap_data"
export EGOEMG_CHECKPOINT_DIR="$PWD/checkpoints"
export EGOEMG_OUTPUT_DIR="$PWD/myself/outputs"
export PYTHONPYCACHEPREFIX="$PWD/myself/.cache/pycache"
```

## Research navigation and versioned results

- Keep prose concise. The root `README.md` starts with links to
  `myself/README.md` (goals, supported conclusions, limits, next steps) and
  `myself/results/README.md` (results index). Update navigation as research changes.
- `myself/docs/`: plans, protocols, reproduction commands, and workflow guidance;
  link results instead of duplicating numerical reports. Preserve protocol
  details already embedded in historical reports.
- `myself/results/<date>_<study>/`: curated Markdown reports, important aggregate
  CSV/JSON tables, and small explanatory figures from validated, completed runs.
  Link them from docs and the results index, distinguishing different protocols
  and older evidence. Keep links and figures usable on GitHub without outputs.
- Full runs stay under ignored `myself/outputs/`. Export selected evidence into
  new snapshots; never copy entire runs, unignore outputs/logs/caches/checkpoints,
  or alter historical runs, reports, logs, or provenance.
- Each snapshot needs a `README.md` (scope, valid runs, omissions, reproduction)
  and a manifest mapping files to original repository-relative paths and SHA-256.
  Preserve available execution hashes/settings. Record export date and checkout
  commit separately: neither current code nor export commit is execution identity.
  Label archived checks as historical, not newly run.
- Select by purpose, not extension. Exclude inputs, per-window predictions and
  features, parameter dumps/weights, bulk caches, full logs, Word/PDF duplicates,
  and large media.
- Before handoff, check links, snapshot/source hashes, and ignore rules; state
  whether changes are local, committed, or pushed. Local files become available
  through GitHub only after push.

## Commands

Validate preview memmaps:

```bash
python scripts/data/validate_memmap.py \
  --memmap-dir "$EGOEMG_MEMMAP_DIR" --no-checksums --allow-partial-sources
```

Evaluate released EMGFormer-M on preview data:

```bash
export EGOEMG_RUN_DIR="$EGOEMG_ROOT/myself/outputs/preview_emgformer_middle_evaluation"
python -m egoemg.test_analysis \
  experiment=emgformer/egoemg_emgformer_middle \
  egoemg_unified_memmap_dir="$EGOEMG_MEMMAP_DIR" \
  checkpoint="$EGOEMG_CHECKPOINT_DIR/egoemg_emgformer_middle.ckpt" \
  'trainer.devices=[0]' batch_size=8 num_workers=2 \
  logger.save_dir="$EGOEMG_RUN_DIR/logger" \
  hydra.run.dir="$EGOEMG_RUN_DIR/hydra"
```

Use matching configs, datasets, and checkpoints for other experiments;
config names depend on the checkout.

```bash
PYTHONPYCACHEPREFIX="$PWD/myself/.cache/pycache" python -m pytest egoemg/tests -q -o cache_dir="$PWD/myself/.pytest_cache"
PYTHONPYCACHEPREFIX="$PWD/myself/.cache/pycache" python -m pytest myself/egoemg/tests -q -o cache_dir="$PWD/myself/.pytest_cache"
```

## Project entry points

| Area | Location |
| --- | --- |
| Training / evaluation | `egoemg/train.py` / `egoemg/test_analysis.py` |
| Data loading / normalization | `egoemg/datasets/egoemg_memmap_dataset.py` / `egoemg/datamodule.py` |
| Checkpoint loading / metrics | `egoemg/lightning.py` |
| EMG experiments | `config/experiment/emgformer/` |
| Vision and fusion experiments | `config/experiment/fusion/` |
| Models | `egoemg/models/` |
| Scripts / tests / documentation | `scripts/` / `egoemg/tests/` / `docs/` |

Released EMG checkpoints are `egoemg_emgformer_{small,middle,large}.ckpt`;
vision checkpoints are `vision_resnet18.ckpt` and `vision_vit_small.ckpt`.
Python code follows nearby conventions: 4-space indentation, type hints,
and isort-formatted imports.
