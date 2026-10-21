#!/usr/bin/env python3
"""Independent frozen-model checks, trained EMG ablation, isolated GPU budget."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch.utils.data import DataLoader

from myself.scripts.analysis.common_modality_evaluation import (
    SOURCE,
    configuration,
    make_dataset,
)
from myself.scripts.analysis.single_subject_calibration_suite import (
    SEEDS,
    csv_write,
    json_write,
    npz_write,
    strict_model,
)
from myself.scripts.analysis.visual_guided_pool_pilot import (
    DirectCenters,
    PoolingBranch,
    extract,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    args = p.parse_args()
    pilot = args.pilot.resolve()
    if not pilot.is_relative_to(REPO / "myself/outputs"):
        raise ValueError(pilot)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = np.load(SOURCE / "manifest.npz")
    test = m["pools"] == "E"
    ds = DirectCenters(make_dataset(True), m["centers"][test], m["reference"][test])
    cfg = configuration("fusion/fusion_rn18_s_center_16ch_wl7790")
    cfg.module.vision_pretrained_checkpoint = str(
        REPO / "checkpoints/vision_resnet18.ckpt"
    )
    model, _ = strict_model(
        cfg.module, REPO / "checkpoints/fusion_resnet18_emgfusion_center.ckpt"
    )
    branches = {}
    for kind in ("original", "guided"):
        for seed in SEEDS:
            branch = PoolingBranch(model, kind == "guided").cuda()
            branch.load_state_dict(
                torch.load(
                    pilot / f"{kind}_seed{seed}.pt",
                    map_location="cuda:0",
                    weights_only=True,
                ),
                strict=True,
            )
            branch.requires_grad_(False).eval()
            branches[f"{kind}_seed{seed}"] = branch
    normal = {k: [] for k in branches}
    zero = {k: [] for k in branches}
    scales = []
    probe = None
    with torch.inference_mode():
        for x, img, y in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            x, img = x.cuda(), img.cuda()
            if probe is None:
                probe = (x[:8].clone(), img[:8].clone())
            scales.extend(x.std(dim=-1).cpu().numpy())
            features = extract(model, x, img)
            absent = (
                model.decoder(model.featurizer(torch.zeros_like(x))),
                features[1],
                features[2],
            )
            for key, branch in branches.items():
                normal[key].append(branch(*features).cpu().numpy())
                zero[key].append(branch(*absent).cpu().numpy())
    z = np.load(pilot / "predictions.npz")
    rows = []
    max_error = 0.0
    arrays = {}
    for key in branches:
        a, b = np.concatenate(normal[key]), np.concatenate(zero[key])
        max_error = max(max_error, float(abs(a - z[key]).max()))
        np.testing.assert_allclose(a, z[key], atol=1e-5, rtol=1e-4)
        ref = z["reference"][:, :20].astype(np.float64)
        rows.append(
            dict(
                condition=key,
                normal_mae_deg=float(np.rad2deg(abs(a[:, :20] - ref).mean())),
                zero_emg_mae_deg=float(np.rad2deg(abs(b[:, :20] - ref).mean())),
                mean_prediction_change_deg=float(
                    np.rad2deg(abs(a[:, :20] - b[:, :20]).mean())
                ),
            )
        )
        arrays[key + "_zero_emg"] = b
    csv_write(pilot / "emg_dependence.csv", rows)
    npz_write(pilot / "zero_emg_predictions.npz", ids=z["ids"], **arrays)
    del branches, normal, zero, model, features, absent, x, img, y, branch
    gc.collect()
    torch.cuda.empty_cache()
    # Each measurement has exactly one model resident. Inputs are already on GPU.
    budget = []
    for key, experiment, filename, signal in [
        ("vision", "fusion/vision_resnet18", "vision_resnet18", None),
        (
            "fusion",
            "fusion/fusion_rn18_s_center_16ch_wl7790",
            "fusion_resnet18_emgfusion_center",
            probe[0],
        ),
        ("emg", "emgformer/egoemg_emgformer_middle", "egoemg_emgformer_middle", None),
    ]:
        cfg = configuration(experiment)
        if key == "vision":
            cfg.module.pretrained = False
        if key == "fusion":
            cfg.module.vision_pretrained_checkpoint = str(
                REPO / "checkpoints/vision_resnet18.ckpt"
            )
        model, _ = strict_model(cfg.module, REPO / f"checkpoints/{filename}.ckpt")
        if key == "emg":
            base8 = make_dataset()
            ep = base8._episode_id.index("episode_000020")
            centers = (
                z["centers"][:8]
                if "centers" in z.files
                else m["centers"][test][ds.indices[:8]]
            )
            signal = torch.tensor(
                np.stack(
                    [
                        base8._getitem_center_supervised_emg(
                            ep, int(c) - 6000, int(c) + 6000
                        )["emg"]
                        for c in centers
                    ]
                ),
                device="cuda:0",
            )
        inputs = (
            {"vision_img": probe[1]}
            if key == "vision"
            else (
                {"emg": signal}
                if key == "emg"
                else {"emg": signal, "vision_img": probe[1]}
            )
        )
        with torch.inference_mode():
            for _ in range(10):
                model(inputs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            measurements = []
            for _ in range(50):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                model(inputs)
                torch.cuda.synchronize()
                measurements.append((time.perf_counter() - begin) * 1000)
        budget.append(
            dict(
                model=key,
                batch_size=8,
                repeats=50,
                warmup=10,
                median_batch_ms=float(np.median(measurements)),
                p95_batch_ms=float(np.quantile(measurements, 0.95)),
                median_amortized_ms_per_sample=float(np.median(measurements) / 8),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                parameters=sum(p.numel() for p in model.parameters()),
            )
        )
        del model, inputs, signal
        gc.collect()
        torch.cuda.empty_cache()
    json_write(
        pilot / "isolated_budget.json",
        dict(
            gpu=torch.cuda.get_device_name(),
            measurements=budget,
            convention="One model resident; FP32 TF32 off; synchronized forward only; B=8; excludes I/O and transfers; outputs unused; same small batch repeated; no concurrent experiment process",
        ),
    )
    json_write(
        pilot / "independent_verification.json",
        dict(
            max_reloaded_prediction_error=max_error,
            normalized_16ch_per_window_std_quantiles=np.quantile(
                np.asarray(scales), [0, 0.5, 0.95, 1]
            ),
            all_zero_predictions_finite=all(
                np.isfinite(v).all() for v in arrays.values()
            ),
        ),
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
