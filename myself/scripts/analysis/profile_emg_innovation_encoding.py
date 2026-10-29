"""Actual-data encoding batch sweep; no input/feature files written."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("EGOEMG_NO_PRETRAINED_DOWNLOAD", "1")
import numpy as np
import torch

from myself.scripts.analysis.effective_emg_contribution import (
    Centers,
    configure_precision,
    make_base,
)
from myself.scripts.analysis.emg_innovation_suite import Bank, load_assets
from myself.scripts.analysis.single_subject_calibration_suite import json_write


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--selection-run", type=Path, required=True)
parser.add_argument(
    "--encoding-batches", type=int, nargs="+", default=[16, 64, 128, 256]
)
parser.add_argument(
    "--training-batches", type=int, nargs="+", default=[32, 64, 128, 256, 512]
)
cli = parser.parse_args()
run = cli.run_dir.resolve()
assert run.is_relative_to(ROOT / "myself/outputs")
run.mkdir(parents=True, exist_ok=False)
args = SimpleNamespace(
    memmap_dir=Path("/mnt/d/egoemg/data/dataset_egoemg_preview/data/memmap_data"),
    crops_dir=Path(
        "/mnt/d/egoemg/data/dataset_egoemg_preview/data/pre-crop_webcam_videoframes"
    ),
    checkpoint=ROOT / "checkpoints/fusion_resnet18_emgfusion_center.ckpt",
    vision_checkpoint=ROOT / "checkpoints/vision_resnet18.ckpt",
)
args.norm_stats = ROOT / "assets/per_dataset_norm_stats.json"
configure_precision()
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
fusion, vision, _ = load_assets(args, run)
ds = Centers(
    make_base(args, "episode_000020"),
    "episode_000020",
    dict(np.load(cli.selection_run / "T_fit_selection.npz")),
)
bank = Bank.__new__(Bank)
bank.ds = ds
bank.fusion = fusion
bank.vision = vision
bank.reverse = False
baseline = bank.compute(np.arange(16))
rows = []
for batch in cli.encoding_batches:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            v = bank.compute(np.arange(batch))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        errors = [float((x[:16] - y).abs().max()) for x, y in zip(v, baseline)]
        relative = [
            float((x[:16] - y).norm() / y.norm().clamp_min(1e-12))
            for x, y in zip(v, baseline)
        ]
        assert max(relative) < 1e-4 and errors[-1] < 1e-5, (errors, relative)
        row = dict(
            batch=batch,
            seconds_median=float(np.median(times)),
            samples_per_second=batch / float(np.median(times)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            max_abs_feature_errors=errors,
            relative_l2_feature_errors=relative,
            numerical_comparison_pass=True,
        )
        rows.append(row)
        print(row, flush=True)
        del v
    except torch.cuda.OutOfMemoryError as e:
        rows.append(dict(batch=batch, status="OOM", message=str(e)))
        torch.cuda.empty_cache()
        break
json_write(
    run / "profile.json",
    dict(
        measurements=rows,
        includes_real_reads_and_backbones=True,
        FP32=True,
        TF32=False,
        no_features_written=True,
        reference_batch=16,
    ),
)
__import__("shutil").copyfile(__file__, run / "executed_profile.py")

refdata = np.load(cli.selection_run / "reference.npz")
ref = torch.from_numpy(refdata["encoded_reference"]).cuda()
bank = Bank(ds, fusion, vision, ref, batch=16)
from myself.scripts.analysis.emg_innovation_suite import init_head, loss_masked, targets


headrows = []
for fam in ("R1", "R4"):
    for batch in cli.training_batches:
        h = init_head(fam, 42, 256)
        opt = torch.optim.Adam(
            [p for p in h.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
        )
        ix = np.arange(batch)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for repeat in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter()
            vals = bank.get(ix)
            y, m, d, dm = targets(ds, ix)
            opt.zero_grad(set_to_none=True)
            pred, aux = h(*vals, ref, auxiliary=True)
            loss = loss_masked(pred, y, m)
            if aux is not None:
                loss = (
                    loss
                    + loss_masked(aux[:, 0], y, m)
                    + (
                        loss_masked(aux[:, 1], d[:, 0], dm[:, 0])
                        + loss_masked(aux[:, 2], d[:, 1], dm[:, 1])
                    )
                    / 2
                )
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if repeat >= 2:
                times.append(time.perf_counter() - start)
        row = dict(
            family=fam,
            batch=batch,
            training_step_seconds_median=float(np.median(times)),
            samples_per_second=batch / float(np.median(times)),
            estimated_epoch_seconds=float(np.median(times))
            * ((len(ds) + batch - 1) // batch),
            steps_per_epoch=(len(ds) + batch - 1) // batch,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        headrows.append(row)
        print(row, flush=True)
        del h, opt, vals, y, m, d, dm, pred, aux, loss
json_write(
    run / "head_profile.json",
    dict(
        measurements=headrows,
        device=torch.cuda.get_device_name(),
        includes_batch_H2D=True,
        FP32=True,
        TF32=False,
        selection="hardware throughput only; no E error computed",
    ),
)
