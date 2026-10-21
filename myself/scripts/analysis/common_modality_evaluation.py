#!/usr/bin/env python3
"""Released modalities on identical centers, with fusion branch diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from myself.scripts.analysis.preview_calibration_smoke import (
    aligned_center,
    open_field,
    sha256,
)
from myself.scripts.analysis.single_subject_calibration_suite import (
    angular_mae,
    csv_write,
    json_write,
    npz_write,
    rgb_reason,
    strict_model,
)


SOURCE = (
    REPO
    / "myself/outputs/single_subject_calibration_suite/2026-10-05_wmh_right_suite_full_02"
)
ROOT = Path("/mnt/d/egoemg/data/dataset_egoemg_preview/data/memmap_data")


def configuration(experiment):
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        return compose(config_name="base", overrides=[f"experiment={experiment}"])


def make_dataset(fusion=False):
    base = EgoEmgMemmapDataset(
        memmap_dir=ROOT,
        window_length=7790 if fusion else 12000,
        stride=200,
        allowed_episode_ids=["episode_000020"],
        allowed_splits=["train", "user", "gesture", "both"],
        modalities=["emg", "joint_angles", "labels"],
        target_hand="right",
        emg_field_preference="filtered_paper",
        emg_layout="emg2pose_interpolate16" if fusion else "target_hand",
        emg2pose_channel_indices=[10, 12, 0, 1, 2, 4, 5, 6] if fusion else None,
        channel_interpolate=False,
        norm_mode="per-dataset",
        norm_stats_path=str(
            REPO
            / "assets"
            / (
                "per_dataset_norm_stats.json"
                if fusion
                else "per_dataset_norm_stats_repro_filtered_paper_alias.json"
            )
        ),
        center_target_only=True,
    )
    # Preview contains episodes 8/20/28; the original constructor probes 0.
    # Use the existing read-only per-episode reader for the actual chosen ID.
    base.per_episode_crops_dir = str(ROOT.parent / "pre-crop_webcam_videoframes")
    base._pec_pid = os.getpid()
    base._pec_lmdb_cache = {}
    base._pec_lmdb_max_open = 1
    return base


def model_inference(model, inputs, times, key):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    out = model(inputs)
    torch.cuda.synchronize()
    times[key] = times.get(key, 0.0) + time.perf_counter() - begin
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(REPO / "myself/outputs"):
        raise ValueError("Output must be inside myself/outputs")
    run.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(run / "run.log"), logging.StreamHandler()],
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    shutil.copyfile(__file__, run / "executed_workflow.py")
    source = np.load(SOURCE / "manifest.npz")
    # Verify historical artifact integrity before taking only its selection.
    for phase in ("manifest", "emg"):
        marker = json.loads((SOURCE / f"{phase}.complete.json").read_text())
        for name, digest in marker["files"].items():
            assert sha256(SOURCE / name) == digest, (phase, name)
    e = np.flatnonzero(source["pools"] == "E")
    centers = source["centers"][e]
    manifest = json.loads((ROOT / "manifest.json").read_text())
    field = lambda name: open_field(ROOT, manifest, name)
    frame = field("image_head_frame_index")[centers]
    stale = field("image_head_stale")[centers]
    delta = field("image_head_delta_ms")[centers]
    reasons = np.array(
        [
            rgb_reason(int(f), bool(s), float(d), 50.0)
            for f, s, d in zip(frame, stale, delta)
        ],
        dtype="U40",
    )
    base8, base16 = make_dataset(), make_dataset(True)
    ep = base8._episode_id.index("episode_000020")
    cfg8 = configuration("emgformer/egoemg_emgformer_middle")
    cfgv = configuration("fusion/vision_resnet18")
    cfgv.module.pretrained = False
    cfgf = configuration("fusion/fusion_rn18_s_center_16ch_wl7790")
    cfgf.module.vision_pretrained_checkpoint = str(
        REPO / "checkpoints/vision_resnet18.ckpt"
    )
    models = {}
    for key, cfg, filename in [
        ("emg", cfg8, "egoemg_emgformer_middle"),
        ("vision", cfgv, "vision_resnet18"),
        ("fusion", cfgf, "fusion_resnet18_emgfusion_center"),
    ]:
        models[key], hparams = strict_model(
            cfg.module, REPO / f"checkpoints/{filename}.ckpt"
        )
        OmegaConf.save(cfg.module, run / f"{key}_model.yaml", resolve=True)
        json_write(run / f"{key}_checkpoint_hparams.json", hparams)
    keys = [
        "emg",
        "vision",
        "fusion",
        "fusion_vision_branch",
        "fusion_zero_emg",
        "fusion_reverse_time",
    ]
    outputs = {k: np.full((len(e), 22), np.nan, np.float32) for k in keys}
    timings = {}
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(e), 8):
            indices = []
            images = []
            emg8 = []
            emg16 = []
            for i in range(offset, min(offset + 8, len(e))):
                if reasons[i] != "eligible":
                    continue
                try:
                    rgb = base8._read_episode_crops(
                        "episode_000020", np.array([frame[i]]), "R"
                    )
                except (OSError, ValueError, TypeError):
                    reasons[i] = "decode"
                    continue
                if rgb is None:
                    reasons[i] = "missing_crop"
                    continue
                if rgb.shape != (1, 256, 256, 3) or rgb.dtype != np.uint8:
                    reasons[i] = "preprocessing"
                    continue
                patch = (
                    rgb[0].transpose(2, 0, 1).astype(np.float32)
                    - base8.vision_mean[:, None, None]
                ) / base8.vision_std[:, None, None]
                if not np.isfinite(patch).all():
                    reasons[i] = "preprocessing"
                    continue
                c = int(centers[i])
                x8 = base8._getitem_center_supervised_emg(ep, c - 6000, c + 6000)
                x16 = base16._getitem_center_supervised_emg(ep, c - 3895, c + 3895)
                np.testing.assert_allclose(
                    x8["joint_angles"][..., 0], source["reference"][e[i]], atol=1e-6
                )
                np.testing.assert_array_equal(x8["joint_angles"], x16["joint_angles"])
                assert np.isfinite(x8["emg"]).all() and np.isfinite(x16["emg"]).all()
                indices.append(i)
                images.append(patch)
                emg8.append(x8["emg"])
                emg16.append(x16["emg"])
            if not indices:
                continue
            img = torch.tensor(np.stack(images), device="cuda:0")
            x8 = torch.tensor(np.stack(emg8), device="cuda:0")
            x16 = torch.tensor(np.stack(emg16), device="cuda:0")
            pred8 = model_inference(models["emg"], {"emg": x8}, timings, "emg")
            outputs["emg"][indices] = (
                aligned_center(models["emg"], pred8, 12000).cpu().numpy()
            )
            outputs["vision"][indices] = (
                model_inference(
                    models["vision"], {"vision_img": img}, timings, "vision"
                )[..., 0]
                .cpu()
                .numpy()
            )
            fused = model_inference(
                models["fusion"], {"emg": x16, "vision_img": img}, timings, "fusion"
            )
            outputs["fusion"][indices] = fused[..., 0].cpu().numpy()
            outputs["fusion_vision_branch"][indices] = (
                (fused - models["fusion"]._last_delta)[..., 0].cpu().numpy()
            )
            for key, signal in [
                ("fusion_zero_emg", torch.zeros_like(x16)),
                ("fusion_reverse_time", x16.flip(-1)),
            ]:
                outputs[key][indices] = (
                    model_inference(
                        models["fusion"],
                        {"emg": signal, "vision_img": img},
                        timings,
                        key,
                    )[..., 0]
                    .cpu()
                    .numpy()
                )
            if offset % 400 == 0:
                logging.info(
                    "Common centers %d/%d elapsed %.1fs",
                    offset + len(indices),
                    len(e),
                    time.perf_counter() - begin,
                )
    keep = reasons == "eligible"
    assert all(np.isfinite(pred[keep]).all() for pred in outputs.values())
    old = np.load(SOURCE / "emg_cache.npz")
    error = float(np.max(np.abs(outputs["emg"][keep] - old["prediction"][e[keep]])))
    np.testing.assert_allclose(
        outputs["emg"][keep], old["prediction"][e[keep]], atol=1e-5, rtol=1e-4
    )
    csv_write(
        run / "rgb_coverage.csv",
        [
            dict(center=int(c), frame=int(f), delta_ms=float(d), reason=str(r))
            for c, f, d, r in zip(centers, frame, delta, reasons)
        ],
    )
    rows = []
    for key, pred in outputs.items():
        for grouping in ("all", "split", "gesture"):
            labels = (
                ["all"]
                if grouping == "all"
                else sorted(np.unique(source[grouping][e[keep]]).tolist())
            )
            for label in labels:
                ix = keep.copy()
                if grouping != "all":
                    ix &= source[grouping][e] == label
                s = angular_mae(
                    pred[ix, :20],
                    source["reference"][e[ix], :20],
                    source["mask"][e[ix], :20],
                )
                rows.append(
                    dict(
                        condition=key,
                        grouping=grouping,
                        label=label,
                        centers=int(ix.sum()),
                        mae_deg=s["mae_deg"],
                    )
                )
    csv_write(run / "metrics.csv", rows)
    npz_write(
        run / "predictions.npz",
        ids=source["ids"][e[keep]],
        centers=centers[keep],
        reference=source["reference"][e[keep]],
        mask=source["mask"][e[keep]],
        split=source["split"][e[keep]],
        gesture=source["gesture"][e[keep]],
        **{k: v[keep] for k, v in outputs.items()},
    )
    paths = [
        Path(__file__),
        ROOT / "manifest.json",
        ROOT / "metadata.npz",
        SOURCE / "manifest.npz",
        SOURCE / "emg_cache.npz",
        REPO / "egoemg/datasets/egoemg_memmap_dataset.py",
        REPO / "assets/per_dataset_norm_stats.json",
        REPO / "assets/per_dataset_norm_stats_repro_filtered_paper_alias.json",
    ]
    paths += list((REPO / "checkpoints").glob("*resnet18*.ckpt")) + [
        REPO / "checkpoints/egoemg_emgformer_middle.ckpt"
    ]
    paths += sorted((REPO / "egoemg/models").rglob("*.py"))
    json_write(
        run / "verification.json",
        dict(
            original_evaluation_centers=len(e),
            common_centers=int(keep.sum()),
            exclusions={
                r: int((reasons == r).sum())
                for r in np.unique(reasons)
                if r != "eligible"
            },
            fresh_emg_max_discrepancy_from_archived_predictions=error,
            strict_load_all_models=True,
            identical_references=True,
            precision="FP32; matmul and cuDNN TF32 disabled",
            peak_gpu_allocated_bytes_with_three_resident_models=torch.cuda.max_memory_allocated(),
            inference_seconds=timings,
            wall_seconds=time.perf_counter() - begin,
            batch_size=8,
            parameter_counts={
                key: sum(p.numel() for p in model.parameters())
                for key, model in models.items()
            },
            inference_timing_convention="Synchronized forward calls; excludes input I/O, normalization, host-to-device transfer, alignment and output transfer; includes first call; all three models resident. Not a deployment benchmark.",
            input_sha256={str(p): sha256(p) for p in paths},
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
        ),
    )
    print(json.dumps([r for r in rows if r["grouping"] == "all"], indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
