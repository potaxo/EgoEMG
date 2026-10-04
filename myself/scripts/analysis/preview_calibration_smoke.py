#!/usr/bin/env python3
"""Frozen, offline right-hand reference-label calibration on one preview episode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


# Keep imported repository modules from creating caches outside personal work.
sys.dont_write_bytecode = True

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


PERSONAL_ROOT = Path(__file__).resolve().parents[2]
REPO = PERSONAL_ROOT.parent
sys.path.insert(0, str(REPO))

from egoemg.constants import EMG_SAMPLE_RATE, JOINTS
from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset


OUTPUT_PARENT = PERSONAL_ROOT / "outputs/preview_calibration_smoke"
EXPERIMENT = "emgformer/egoemg_emgformer_middle"


def runs(values: np.ndarray) -> list[tuple[int, int, Any]]:
    """Half-open constant runs, including both array boundaries."""
    if not len(values):
        return []
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [(int(a), int(b), values[a].item()) for a, b in zip(edges[:-1], edges[1:])]


def assert_isolation(
    starts: np.ndarray,
    phases: np.ndarray,
    window: int,
    block: tuple[int, int],
    split: np.ndarray,
    train_id: int,
) -> None:
    """Validate complete supports, not merely the split at each center."""
    ends = starts + window
    assert len(starts) and np.all(starts >= block[0]) and np.all(ends <= block[1])
    assert np.all(np.diff(starts) > 0)
    for a, b in zip(starts, ends):
        assert np.all(split[a:b] == train_id), f"Support [{a}, {b}) crosses split"
    cal, ev = phases == "calibration", phases == "evaluation"
    assert cal.any() and ev.any()
    assert ends[cal].max() <= starts[ev].min(), "Phase supports overlap"


def select_windows(
    split: np.ndarray,
    valid: np.ndarray,
    emg: np.ndarray,
    reference: np.ndarray,
    train_id: int,
    window: int,
    stride: int,
    calibration_s: float,
    evaluation_s: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Earliest fully valid span; only split/masks/finite inputs determine selection."""
    choices = [(calibration_s, evaluation_s)]
    if calibration_s >= 5 and evaluation_s >= 15 and choices[0] != (5, 15):
        choices.append((5.0, 15.0))
    train_runs = [(a, b) for a, b, s in runs(split) if s == train_id]
    diagnostics = []
    # Inspect only train blocks, never hash/scan all large dataset arrays.
    spans = []
    for a, b in train_runs:
        good = np.asarray(valid[a:b], dtype=bool).copy()
        good &= np.isfinite(emg[a:b]).all(axis=1)
        good &= np.isfinite(reference[a:b]).all(axis=1)
        spans.append((a, b, [(a + x, a + y) for x, y, v in runs(good) if v]))
        diagnostics.append({"start": a, "end": b, "valid_rows": int(good.sum())})
    for cal_s, ev_s in choices:
        cal_rows, ev_rows = round(cal_s * EMG_SAMPLE_RATE), round(
            ev_s * EMG_SAMPLE_RATE
        )
        ncal, nev = math.ceil(cal_rows / stride), math.ceil(ev_rows / stride)
        for a, b, valid_spans in spans:
            for first, last in valid_spans:
                cal = first + np.arange(ncal, dtype=np.int64) * stride
                ev = cal[-1] + window + np.arange(nev, dtype=np.int64) * stride
                if ev[-1] + window > last:
                    continue
                starts = np.r_[cal, ev]
                phases = np.array(["calibration"] * ncal + ["evaluation"] * nev)
                assert_isolation(starts, phases, window, (a, b), split, train_id)
                return (
                    starts,
                    phases,
                    {
                        "train_block_local": [a, b],
                        "valid_span_local": [first, last],
                        "calibration_seconds": cal_s,
                        "evaluation_seconds": ev_s,
                        "reduced_durations": (cal_s, ev_s) != choices[0],
                        "duration_convention": "half-open nominal center interval; stride-spaced samples",
                        "train_blocks": diagnostics,
                    },
                )
    raise ValueError(
        f"No valid isolated span for durations {choices}; train blocks: {diagnostics}"
    )


def fit_offsets(
    predictions: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize mean squared residual plus strength * offset squared per joint."""
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Regularization must be finite and nonnegative")
    mask = np.asarray(mask, dtype=bool)
    assert predictions.shape == reference.shape == mask.shape
    assert np.isfinite(predictions[mask]).all() and np.isfinite(reference[mask]).all()
    counts = mask.sum(axis=0)
    residual = np.where(
        mask, reference.astype(np.float64) - predictions.astype(np.float64), 0
    ).sum(axis=0)
    offsets = np.divide(
        residual,
        counts * (1 + strength),
        out=np.zeros(predictions.shape[1]),
        where=counts > 0,
    )
    return offsets, counts


def apply_offsets(predictions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return predictions + offsets[None, :]


def angular_mae(
    predictions: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> dict:
    assert predictions.shape == reference.shape == mask.shape
    selected = np.abs(
        predictions[mask].astype(np.float64) - reference[mask].astype(np.float64)
    )
    assert np.isfinite(selected).all()
    rad = float(selected.mean()) if selected.size else None
    return {
        "valid_entries": int(selected.size),
        "mae_rad": rad,
        "mae_deg": float(np.rad2deg(rad)) if rad is not None else None,
    }


def aligned_center(
    model: torch.nn.Module, raw: torch.Tensor, window: int
) -> torch.Tensor:
    """Use BaseModule's full-sequence alignment, then select the input center row."""
    center = window // 2 - model.left_context
    length = window - model.left_context - model.right_context
    assert 0 <= center < length
    return model.align_predictions(raw, length)[..., center]


class SelectedSupports(Dataset):
    """Local wrapper reusing the loader's preprocessing and direct center targets."""

    def __init__(
        self, base: EgoEmgMemmapDataset, episode_index: int, starts: np.ndarray
    ):
        self.base, self.episode_index, self.starts = base, episode_index, starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start = int(self.starts[index])
        sample = self.base._getitem_center_supervised_emg(
            self.episode_index, start, start + self.base.window_length
        )
        assert np.isfinite(sample["emg"]).all()
        return {k: sample[k] for k in ("emg", "joint_angles", "label_valid_mask")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def open_field(root: Path, manifest: dict, key: str) -> np.memmap:
    spec = manifest["fields"][key]
    return np.memmap(
        root / spec["filename"],
        mode="r",
        dtype=spec["dtype"],
        shape=tuple(spec["shape"]),
    )


def rgb_readiness(
    root: Path,
    manifest: dict,
    base: EgoEmgMemmapDataset,
    episode: str,
    ep_index: int,
    center: int,
    output: Path,
    checkpoint: Path,
) -> dict:
    """One crop through the existing reader; no teacher instantiation or inference."""
    crops = root.parent / "pre-crop_webcam_videoframes"
    videos = root.parent / "webcam_videos"
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "matching_config": "fusion/vision_resnet18",
        "target": "20 finger radians + 2 wrist radians",
        "crop_root": str(crops),
        "video_root": str(videos),
        "metadata_head_video": base._episode_head_video_path[ep_index],
        "video_asset": str(videos / f"{episode}_head_allintra.mp4"),
        "remaining_checks": [
            "Verify RGB/EMG timestamp mapping and stale/delta tolerance",
            "Verify right-hand crop identity, orientation and ImageNet preprocessing",
            "Strict-load teacher checkpoint before inference",
            "Confirm joint order, units, wrist mask and output-center convention",
        ],
        "reader_constructor_gap": "Vision-enabled loader requires episode_000000.lmdb, absent from preview; existing per-episode reader can read episode 20 directly",
    }
    result["video_exists"] = Path(result["video_asset"]).is_file()
    try:
        frame = int(open_field(root, manifest, "image_head_frame_index")[center])
        result.update(
            center_global=center,
            center_local=center - int(base._episode_start_idx[ep_index]),
            head_frame_index=frame,
            lmdb_key=f"{frame:08d}_R",
        )
        for key in (
            "image_head_stale",
            "image_head_delta_ms",
            "timestamp_us",
            "frame_index",
        ):
            result[key] = open_field(root, manifest, key)[center].item()
        # EMG-only construction avoids the preview-incompatible sample-LMDB check.
        base.per_episode_crops_dir = str(crops)
        base._pec_pid = os.getpid()
        base._pec_lmdb_cache = {}
        base._pec_lmdb_max_open = 1
        images = base._read_episode_crops(episode, np.array([frame]), "R")
        if images is None:
            result["crop_status"] = "missing mapped right-hand crop"
        else:
            from PIL import Image

            Image.fromarray(images[0]).save(output / "right_hand_crop.png")
            result.update(
                crop_status="decoded",
                crop_shape=list(images[0].shape),
                preview="right_hand_crop.png",
            )
    except (OSError, ValueError, KeyError, ImportError) as error:
        result["crop_status"] = f"{type(error).__name__}: {error}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Preview package or its memmap_data",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Repository-local output parent"
    )
    parser.add_argument("--episode", default="episode_000020")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-seconds", type=float, default=15)
    parser.add_argument("--evaluation-seconds", type=float, default=30)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument(
        "--stride", type=int, default=200, help="Rows (default 10Hz center scoring)"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--rgb-checkpoint", type=Path, default=REPO / "checkpoints/vision_resnet18.ckpt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.data_root, args.checkpoint, args.output_dir, args.rgb_checkpoint):
        if not path.is_absolute():
            raise ValueError(f"An absolute path is required: {path}")
    parent = args.output_dir.resolve()
    if not parent.is_relative_to(OUTPUT_PARENT.resolve()):
        raise ValueError(f"Output parent must be under {OUTPUT_PARENT}")
    for seconds in (args.calibration_seconds, args.evaluation_seconds):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Durations must be positive and finite")
    if args.stride <= 0 or args.batch_size <= 0 or args.regularization != 1.0:
        raise ValueError(
            "Positive stride/batch size and fixed regularization=1.0 required"
        )
    if (
        args.stride
        > min(args.calibration_seconds, args.evaluation_seconds) * EMG_SAMPLE_RATE
    ):
        raise ValueError("Stride exceeds a requested center interval")
    root = args.data_root.resolve()
    if not (root / "manifest.json").is_file():
        root = root / "data/memmap_data"
    if not (root / "manifest.json").is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing preview manifest at {root} or checkpoint {args.checkpoint}"
        )
    run = parent / (
        datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run / "run.log"),
        ],
        format="%(message)s",
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info("Run directory: %s", run)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        config = compose(
            config_name="base",
            overrides=[
                f"experiment={EXPERIMENT}",
                "train=false",
                "eval=true",
                "num_workers=0",
                f"egoemg_unified_memmap_dir={root}",
                f"checkpoint={args.checkpoint}",
                f"batch_size={args.batch_size}",
                "trainer.devices=[0]",
                f"hydra.run.dir={run / 'hydra'}",
                f"logger.save_dir={run / 'logger'}",
            ],
        )
    config.datamodule.per_dataset_norm_stats_path = str(
        REPO / "assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
    )
    ds_config = OmegaConf.create(
        OmegaConf.to_container(config.dataset.val[1], resolve=True)
    )
    ds_config.update(
        dataset_name=config.eval_dataset_name,
        allowed_splits=["train"],
        allowed_episode_ids=[args.episode],
        stride=args.stride,
        center_target_only=True,
        jitter=False,
    )
    base = instantiate(ds_config, _convert_="all")
    matches = [i for i, name in enumerate(base._episode_id) if name == args.episode]
    if len(matches) != 1:
        raise ValueError(f"Expected one episode {args.episode}, got {matches}")
    ep_index = matches[0]
    participant = base._episode_subject[ep_index]
    if args.episode == "episode_000020" and participant != "wmh":
        raise ValueError(f"episode_000020 participant mismatch: {participant}")
    origin, stop = int(base._episode_start_idx[ep_index]), int(
        base._episode_end_idx[ep_index]
    )
    manifest = json.loads((root / "manifest.json").read_text())
    names = manifest["generated_joint_angles_semantics"]
    assert names == [joint.name.lower() for joint in JOINTS]
    train_id = base._splits.index("train")
    split = open_field(root, manifest, "frame_split_id")[origin:stop]
    starts, phases, selection = select_windows(
        split,
        open_field(root, manifest, "generated_label_valid")[origin:stop, 1],
        open_field(root, manifest, "emg_right_filtered_paper")[origin:stop],
        open_field(root, manifest, "generated_joint_angles_right")[origin:stop],
        train_id,
        base.window_length,
        args.stride,
        args.calibration_seconds,
        args.evaluation_seconds,
    )
    centers = starts + base.window_length // 2 + origin
    global_starts = starts + origin
    subjects = np.unique(
        open_field(root, manifest, "subject_id")[
            global_starts[0] : global_starts[-1] + base.window_length
        ]
    ).tolist()
    assert len(subjects) == 1
    assert np.all(
        open_field(root, manifest, "episode_index")[
            global_starts[0] : global_starts[-1] + base.window_length
        ]
        == ep_index
    )
    assert np.all(open_field(root, manifest, "dataset_source_id")[centers] == 0)
    selection.update(
        participant=participant,
        episode=args.episode,
        metadata_subject_id=int(base._episode_subject_id[ep_index]),
        row_subject_ids=subjects,
        episode_global=[origin, stop],
        split_runs_local=runs(split),
    )
    table = []
    for start, center, phase in zip(starts, centers, phases):
        table.append(
            dict(
                phase=phase,
                participant=participant,
                episode=args.episode,
                split="train",
                split_id=train_id,
                local_start=int(start),
                local_end=int(start + base.window_length),
                local_center=int(center - origin),
                global_start=int(start + origin),
                global_end=int(start + origin + base.window_length),
                global_center=int(center),
            )
        )
    with (run / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for phase in ("calibration", "evaluation"):
        ix = phases == phase
        selection[phase] = {
            "windows": int(ix.sum()),
            "support_local": [
                int(starts[ix].min()),
                int((starts[ix] + base.window_length).max()),
            ],
            "support_global": [
                int(global_starts[ix].min()),
                int((global_starts[ix] + base.window_length).max()),
            ],
            "center_interval_global": [
                int(centers[ix].min()),
                int(
                    centers[ix].min()
                    + round(selection[phase + "_seconds"] * EMG_SAMPLE_RATE)
                ),
            ],
            "center_interval_local": [
                int((centers[ix] - origin).min()),
                int(
                    (centers[ix] - origin).min()
                    + round(selection[phase + "_seconds"] * EMG_SAMPLE_RATE)
                ),
            ],
        }
    norm = Path(base.norm_stats_path)
    settings = {
        "experiment": "one-participant offline preview reference-label smoke test",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "run_directory": str(run),
        "data_root": str(root),
        "selection": selection,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "workflow_sha256": sha256(Path(__file__)),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "numpy",
                "hydra-core",
                "pytorch-lightning",
                "scipy",
                "lmdb",
                "decord",
                "Pillow",
            )
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "normalization": {
            "path": str(norm),
            "sha256": sha256(norm),
            "dataset_key": "egoemg__filtered_paper_right",
            "mean": np.asarray(base._emg_mean).tolist(),
            "std": np.asarray(base._emg_std).tolist(),
            "epsilon": 1e-6,
        },
        "manifest_sha256": sha256(root / "manifest.json"),
        "metadata_sha256": sha256(root / "metadata.npz"),
        "preprocessing": {
            "field": "emg_right_filtered_paper",
            "channels": list(range(8)),
            "layout": "target_hand",
            "channel_transforms": "none; stored order 0..7, transpose T,C to C,T",
            "filter": "released whole-recording FFT mean subtraction, 20Hz high-pass, 850Hz low-pass, 50/100Hz notches; noncausal",
            "augmentation": "none",
            "sample_rate_hz": EMG_SAMPLE_RATE,
            "sample_rate_evidence": "egoemg.constants.EMG_SAMPLE_RATE and filter_emg_into_new_columns.FS; timestamps are not uniformly spaced",
            "window_rows": base.window_length,
            "stride_rows": args.stride,
        },
        "calibrated_joint_indices": list(range(20)),
        "joint_names": names + ["wrist_pitch", "wrist_yaw"],
        "target_units": "finger values already radians; wrist stored degrees converted to radians by loader",
        "mask": "generated_label_valid[:,1] broadcast across fingers, finite reference per joint; wrist additionally mocap_right_wrist_angles_valid",
        "metric": "absolute angular difference, no wrapping/clipping; aggregate mean over valid entries; degrees=radians*180/pi",
        "support_separation_verified": True,
        "hydra_output": str(run / "hydra"),
        "logger_output": str(run / "logger"),
        "execution": "compose only; no Hydra entrypoint, Lightning Trainer or logger instantiated; inference FP32 on one GPU",
        "model_config": OmegaConf.to_container(config.module, resolve=True),
        "dataset_config": OmegaConf.to_container(ds_config, resolve=True),
        "limitations": [
            "One participant on train rows; checkpoint exposure is not excluded",
            "Offline noncausal released filtering; disjoint supports guarantee sample-selection isolation only",
            "Overlapping windows within a phase; this center pilot differs from pooled benchmark",
            "Reference labels supervise calibration; no RGB teacher inference or RGB efficacy claim",
        ],
    }
    settings["git_dirty"] = bool(settings["git_status"])
    write_json(run / "settings.json", settings)
    OmegaConf.save(config, run / "resolved_config.yaml", resolve=True)
    log.info("Resolved settings: %s", json.dumps(settings, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference unavailable; settings/windows saved. Re-run this command with CUDA available."
        )
    model = instantiate(config.module)
    # Ignore old checkpoint import paths and training init assets; load all model keys strictly.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("model."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval().to("cuda:0")
    assert not any(p.requires_grad for p in model.parameters())
    settings.update(
        device=torch.cuda.get_device_name(0),
        cuda_version=torch.version.cuda,
        checkpoint_load="strict, all model keys matched",
        parameters=sum(p.numel() for p in model.parameters()),
        alignment={
            "left_context": model.left_context,
            "right_context": model.right_context,
            "resampled_length": base.window_length
            - model.left_context
            - model.right_context,
            "center_resampled_index": base.window_length // 2 - model.left_context,
            "method": "model.align_predictions(raw, window-left-right), linear align_corners=False; select input-center-minus-left-context",
        },
    )
    loader = DataLoader(
        SelectedSupports(base, ep_index, global_starts),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions, references, labels = [], [], []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            raw = model({"emg": batch["emg"].to("cuda:0")})
            assert raw.ndim == 3 and raw.shape[1] == 22 and torch.isfinite(raw).all()
            settings["alignment"]["raw_timeline_length"] = raw.shape[-1]
            predictions.append(
                aligned_center(model, raw, base.window_length).float().cpu().numpy()
            )
            references.append(batch["joint_angles"].numpy()[..., 0])
            labels.append(batch["label_valid_mask"].numpy()[:, 0])
            if i % 25 == 0:
                log.info("Inference batch %d/%d", i + 1, len(loader))
    prediction, reference = np.concatenate(predictions), np.concatenate(references)
    mask = np.repeat(np.concatenate(labels)[:, None], 22, axis=1) & np.isfinite(
        reference
    )
    wrist_valid = open_field(root, manifest, "mocap_right_wrist_angles_valid")[centers]
    mask[:, 20:] &= wrist_valid[:, None]
    assert np.isfinite(prediction).all() and mask[:, :20].all()
    cal, ev = phases == "calibration", phases == "evaluation"
    offsets, counts = fit_offsets(
        prediction[cal, :20], reference[cal, :20], mask[cal, :20], args.regularization
    )
    correction = np.r_[offsets, np.zeros(2)]
    corrected = apply_offsets(prediction, correction)
    metrics = {
        "evaluation_windows": int(ev.sum()),
        "calibration_windows": int(cal.sum()),
        "regularization": args.regularization,
        "aggregate": "mean over valid evaluation entries",
    }
    per_joint = []
    for label, indices in [("fingers", slice(0, 20)), ("wrist", slice(20, 22))]:
        baseline = angular_mae(
            prediction[ev, indices], reference[ev, indices], mask[ev, indices]
        )
        calibrated = angular_mae(
            corrected[ev, indices], reference[ev, indices], mask[ev, indices]
        )
        metrics[label] = {
            "baseline": baseline,
            "corrected": calibrated,
            "change_rad": (
                calibrated["mae_rad"] - baseline["mae_rad"]
                if baseline["mae_rad"] is not None
                else None
            ),
            "change_deg": (
                calibrated["mae_deg"] - baseline["mae_deg"]
                if baseline["mae_deg"] is not None
                else None
            ),
        }
    for j, name in enumerate(settings["joint_names"]):
        before = angular_mae(prediction[ev, j], reference[ev, j], mask[ev, j])
        after = angular_mae(corrected[ev, j], reference[ev, j], mask[ev, j])
        per_joint.append(
            dict(
                index=j,
                joint=name,
                calibrated=j < 20,
                calibration_valid=int(mask[cal, j].sum()),
                evaluation_valid=before["valid_entries"],
                offset_rad=float(correction[j]),
                offset_deg=float(np.rad2deg(correction[j])),
                baseline_rad=before["mae_rad"],
                baseline_deg=before["mae_deg"],
                corrected_rad=after["mae_rad"],
                corrected_deg=after["mae_deg"],
                change_rad=(
                    after["mae_rad"] - before["mae_rad"]
                    if before["mae_rad"] is not None
                    else None
                ),
                change_deg=(
                    after["mae_deg"] - before["mae_deg"]
                    if before["mae_deg"] is not None
                    else None
                ),
            )
        )
    write_json(
        run / "offsets.json",
        {
            "regularization": args.regularization,
            "fit_phase": "calibration only",
            "joint_indices": list(range(20)),
            "joint_names": names,
            "offsets_rad": offsets.tolist(),
            "valid_counts": counts.tolist(),
        },
    )
    write_json(run / "metrics.json", metrics)
    with (run / "per_joint.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_joint[0]))
        writer.writeheader()
        writer.writerows(per_joint)
    np.savez_compressed(
        run / "center_predictions.npz",
        predictions=prediction,
        references=reference,
        masks=mask,
        corrected_predictions=corrected,
        center_global=centers,
        center_local=centers - origin,
        phases=phases,
        offsets_rad=correction,
    )
    write_json(run / "settings.json", settings)
    rgb = rgb_readiness(
        root,
        manifest,
        base,
        args.episode,
        ep_index,
        int(centers[0]),
        run,
        args.rgb_checkpoint,
    )
    write_json(run / "rgb_readiness.json", rgb)
    finger = metrics["fingers"]
    summary = (
        f"# One-participant offline preview smoke test\n\n"
        f"Completed: {args.episode}, {participant}, right hand, train block {selection['train_block_local']}.\n"
        f"Calibration: {selection['calibration']}; evaluation: {selection['evaluation']}.\n"
        f"Complete supports inside recording/train; max calibration end <= min evaluation start verified.\n\n"
        f"Finger MAE: {finger['baseline']['mae_rad']:.6f} rad ({finger['baseline']['mae_deg']:.4f} deg) baseline; "
        f"{finger['corrected']['mae_rad']:.6f} rad ({finger['corrected']['mae_deg']:.4f} deg) corrected.\n"
        f"Change corrected - baseline: {finger['change_rad']:+.6f} rad ({finger['change_deg']:+.4f} deg); "
        f"{finger['baseline']['valid_entries']} valid evaluation finger entries.\n"
        f"Calibration coverage per finger: {counts.tolist()}; lambda fixed at 1.0; wrist correction zero.\n"
        f"Strict checkpoint loading; frozen eval mode; finite predictions; resampled center scoring.\n\n"
        f"RGB readiness: {rgb['crop_status']}; teacher checkpoint exists: {rgb['checkpoint_exists']}.\n\n"
        + "\n".join(f"- {limit}" for limit in settings["limitations"])
        + "\n\n"
        + "Before RGB calibration: "
        + "; ".join(rgb["remaining_checks"])
        + ".\n"
    )
    (run / "summary.md").write_text(summary)
    log.info(summary)


if __name__ == "__main__":
    main()
