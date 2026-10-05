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

- Keep dataset inputs on the user-designated data disk and read them directly
  from their specified paths (currently `/mnt/d/egoemg/data/`, Windows
  `D:\egoemg\data\`).
- Without explicit user permission, do not copy, move, stage, or extract large
  dataset files into the repository, `myself/`, the WSL filesystem, or temporary
  working directories. This includes memmap arrays, dataset shards, videos,
  crop/LMDB databases, archives, and bulk input caches or converted copies.
  Faster loading or convenience alone does not authorize local duplication.
- If a required data path is absent or inaccessible, inspect the designated
  mount and existing directories first. If local duplication is essential,
  explain the reason and expected storage size and obtain explicit permission
  before proceeding; do not silently substitute another dataset.
- Small inspection previews and experiment outputs (logs, metrics, plots,
  predictions, and newly trained checkpoints) remain under `myself/` according
  to the personal work directory rule. They must not serve as a bulk copy of
  dataset inputs. Existing supplied checkpoints remain at their specified paths.

## Personal work directory rule

- Put all newly added personal research/workflow files under repository-local
  `myself/`, mirroring their original repository-relative paths. For example,
  use `myself/scripts/analysis/`, `myself/docs/`, `myself/egoemg/tests/`, and
  `myself/config/` for new scripts, documents, tests, and config deltas.
- Put every generated artifact under `myself/` too: run logs, metrics, plots,
  prediction files, experiment checkpoints, Hydra/logger output, reports,
  temporary task files, and task-created caches. Use distinct run directories
  under `myself/outputs/`; direct pytest caches to `myself/.pytest_cache` and
  Python bytecode caches to `myself/.cache/pycache` (or disable bytecode writes).
- Name folders clearly to describe their purpose and contents.
- Reuse the original repository's code, configuration, normalization assets,
  and supplied checkpoints as inputs. Resolve the repository root separately
  from `myself/`; moving a script must not change its input asset paths.
- Preserve existing repository/user work and dataset/checkpoint bytes. Do not
  move unrelated existing files. Changes to this root `AGENTS.md` are allowed
  for maintaining these instructions; follow explicit user requests for edits
  to existing original files.
- When moving or renaming run artifacts, preserve historical provenance and logs
  as executed; record an old-to-new path mapping rather than rewriting history.

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

## Commands

Check a memmap dataset (preview shown here):

```bash
python scripts/data/validate_memmap.py \
  --memmap-dir "$EGOEMG_MEMMAP_DIR" --no-checksums --allow-partial-sources
```

Evaluate the released EMGFormer-M checkpoint (preview shown here):

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

For other experiments, select the matching local configuration, dataset, and
checkpoint. Configuration names may differ across checkout versions.

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
