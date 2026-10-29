#!/usr/bin/env python3
"""Supplementary bounded input audit and real read+backbone+head timing."""
from __future__ import annotations

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
from torch.nn import functional as F

from myself.scripts.analysis.effective_emg_contribution import (
    Centers,
    configure_precision,
    make_base,
)
from myself.scripts.analysis.emg_innovation_suite import (
    init_head,
    load_assets,
    load_cell,
)
from myself.scripts.analysis.preview_calibration_smoke import open_field, sha256
from myself.scripts.analysis.single_subject_calibration_suite import (
    csv_write,
    json_write,
    npz_write,
)
from myself.scripts.analysis.visual_guided_pool_pilot import frozen_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--time-pipeline", action="store_true")
    p.add_argument("--hash-payloads", action="store_true")
    a = p.parse_args()
    run = a.run_dir.resolve()
    receipt = run / "supplement"
    receipt.mkdir(exist_ok=True)
    identity = json.loads((run / "identity.json").read_text())
    protocol = identity["protocol"]
    args = SimpleNamespace(
        memmap_dir=Path(protocol["memmap"]),
        crops_dir=Path(protocol["crops"]),
        checkpoint=ROOT / "checkpoints/fusion_resnet18_emgfusion_center.ckpt",
        vision_checkpoint=ROOT / "checkpoints/vision_resnet18.ckpt",
        norm_stats=ROOT / "assets/per_dataset_norm_stats.json",
    )
    configure_precision()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    base = make_base(args, "episode_000020")
    ds = Centers(base, "episode_000020", dict(np.load(run / "T_fit_selection.npz")))
    ix = np.unique(
        [
            int(np.argmin(np.abs(ds.s["centers"] - t)))
            for t in np.linspace(ds.s["centers"][0], ds.s["centers"][-1], 24)
        ]
    )
    fusion, vision, hp = load_assets(args, receipt)
    before = (frozen_digest(fusion), frozen_digest(vision))
    refdata = np.load(run / "reference.npz")
    ref = torch.from_numpy(refdata["encoded_reference"]).cuda()
    channels = []
    stages = []
    mf = json.loads((args.memmap_dir / "manifest.json").read_text())
    raw = open_field(args.memmap_dir, mf, "emg_right_filtered_paper")
    xs = np.stack([ds.emg(int(i)) for i in ix])
    center = ds.s["centers"][ix]
    rr = np.stack([raw[c - 3895 : c + 3895] for c in center])
    for name, array in [("raw_filtered_paper", rr), ("normalized_sparse16", xs)]:
        axis = 2 if name == "raw_filtered_paper" else 1
        for channel in range(array.shape[axis]):
            v = array[:, :, channel] if axis == 2 else array[:, channel, :]
            channels.append(
                dict(
                    stage=name,
                    channel=channel,
                    live=(
                        channel in base._emg2pose_channel_indices if axis == 1 else True
                    ),
                    mean=float(v.mean()),
                    std=float(v.std()),
                    minimum=float(v.min()),
                    p01=float(np.quantile(v, 0.01)),
                    median=float(np.median(v)),
                    p99=float(np.quantile(v, 0.99)),
                    maximum=float(v.max()),
                )
            )
    with torch.no_grad():
        for a0 in range(0, len(ix), 8):
            x = torch.from_numpy(xs[a0 : a0 + 8]).cuda()
            normal = fusion.decoder(fusion.featurizer(x))
            for name, sig in [
                ("correct", x),
                (
                    "constant",
                    torch.from_numpy(refdata["normalized_input"])
                    .cuda()[None]
                    .expand_as(x),
                ),
                ("reverse", x.flip(-1)),
            ]:
                val = fusion.decoder(fusion.featurizer(sig))
                stages.append(
                    dict(
                        condition=name,
                        batch=a0 // 8,
                        shape=list(val.shape),
                        mean=float(val.mean()),
                        std=float(val.std()),
                        mean_abs=float(val.abs().mean()),
                        mean_abs_change=float((val - normal).abs().mean()),
                        relative_l2=float(
                            (val - normal).norm() / normal.norm().clamp_min(1e-12)
                        ),
                    )
                )
    csv_write(receipt / "input_channels.csv", channels)
    json_write(
        receipt / "feature_sensitivity.json",
        dict(
            selection="nearest eligible T_fit centers to 24 uniform nominal row-time targets; no errors used",
            ids=ds.s["ids"][ix].tolist(),
            stages=stages,
            token_position="146 decoder index positions, exact physical receptive field not claimed; original complete window, no added padding or dropped tokens",
            input_sha256=__import__("hashlib").sha256(xs.tobytes()).hexdigest(),
            input_count=len(ix),
        ),
    )
    # Check every saved auxiliary endpoint directly against the designated memmaps.
    refs = open_field(args.memmap_dir, mf, "generated_joint_angles_right")
    valid = open_field(args.memmap_dir, mf, "generated_label_valid")[:, 1]
    splits = open_field(args.memmap_dir, mf, "frame_split_id")
    endpoint_rows = []
    for path in [
        run / "T_fit_selection.npz",
        run / "T_val_selection.npz",
        *[
            run / ep / "enriched_E.npz"
            for ep in ("episode_000020", "episode_000008", "episode_000028")
        ],
    ]:
        d = np.load(path)
        c = d["centers"]
        y = d["reference"][:, :20]
        for j, offset in enumerate((-200, 200)):
            t = y - refs[c + offset] if offset < 0 else refs[c + offset] - y
            mask = d["delta_mask"][:, j]
            np.testing.assert_allclose(
                d["delta"][:, j][mask], t[mask], atol=1e-7, rtol=1e-6
            )
            assert np.all(valid[c + offset][mask.any(1)]) and np.all(
                splits[c + offset][mask.any(1)] == d["split"][mask.any(1)]
            )
            assert np.all(
                (c + offset >= d["support_start"]) & (c + offset < d["support_stop"])
            )
            endpoint_rows.append(
                dict(
                    pool=str(path.relative_to(run)),
                    offset_rows=offset,
                    valid_angles=int(mask.sum()),
                    masked_angles=int(mask.size - mask.sum()),
                    checked_against_designated_memmap=True,
                )
            )
    csv_write(receipt / "endpoint_verification.csv", endpoint_rows)
    frozen_used = sum(
        p.numel()
        for module in (fusion.featurizer, fusion.decoder, vision.backbone, vision.head)
        for p in module.parameters()
    )
    json_write(
        receipt / "parameter_accounting.json",
        dict(
            frozen_used_parameters=frozen_used,
            loaded_release_fusion_parameters=sum(
                p.numel() for p in fusion.parameters()
            ),
            loaded_independent_vision_parameters=sum(
                p.numel() for p in vision.parameters()
            ),
            unused_new_modules="Common shape/state is stored for initialization checks. Disabled auxiliary and R0 pooling modules excluded from trainable/inference counts.",
            inference_definition="frozen used backbone+anchor parameters plus active fusion parameters, auxiliary heads excluded",
            training_definition="same frozen used parameters plus active trainable fusion and auxiliary parameters",
            joint_mapping_verified_against="egoemg.constants.JOINTS and input manifest",
            dropout="0 in every newly trained module; all frozen modules eval",
        ),
    )
    if a.time_pipeline:
        state = json.loads((run / "state.json").read_text())
        lock = json.loads((run / "selection_lock.json").read_text())
        rows = []
        for fam in sorted({"R0", lock["selected_family"]}):
            h = load_cell(run, state, "fit", fam, "B1", 42, ref.shape[-1])
            count = int(protocol["batch"])
            indices = np.linspace(0, len(ds) - 1, count, dtype=int)

            # Warm CUDA and OS caches, then include fresh dataset reads and full
            # backbone computation each repeat. Not cold-storage throughput.
            def pipeline():
                samples = [ds[int(i)] for i in indices]
                x = torch.from_numpy(np.stack([v[0] for v in samples])).cuda()
                im = torch.from_numpy(np.stack([v[1] for v in samples])).cuda()
                with torch.no_grad():
                    e = fusion.decoder(fusion.featurizer(x)).transpose(1, 2)
                    fm = vision.backbone(im)
                    gl = vision.avgpool(fm).flatten(1)
                    an = vision.head(gl)[:, :20]
                    sp = F.adaptive_avg_pool2d(fm, (4, 4)).flatten(2).transpose(1, 2)
                    return h(e, sp, gl, an, ref)[0]

            pipeline()
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                start = time.perf_counter()
                pp = pipeline()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            rows.append(
                dict(
                    family=fam,
                    batch=count,
                    device=torch.cuda.get_device_name(),
                    pipeline_seconds_mean=float(np.mean(times)),
                    pipeline_seconds_median=float(np.median(times)),
                    samples_per_second=count / float(np.mean(times)),
                    cuda_synchronized=True,
                    includes_lmdb_and_memmap_reads=True,
                    includes_frozen_backbones_and_head=True,
                    storage_cache="OS cache warmed; inputs read directly each repeat",
                    reference_encoder="fixed reference feature reused, adapter/H reference recomputed",
                )
            )
        csv_write(receipt / "pipeline_timing.csv", rows)
    assert before == (frozen_digest(fusion), frozen_digest(vision))
    if a.hash_payloads:
        consumed = [
            "emg_right_filtered_paper",
            "generated_joint_angles_right",
            "generated_label_valid",
            "frame_split_id",
            "episode_index",
            "subject_id",
            "label_gesture_class",
            "label_gesture_active",
            "image_head_frame_index",
            "image_head_stale",
            "image_head_delta_ms",
            "mocap_right_wrist_pitch",
            "mocap_right_wrist_yaw",
            "mocap_right_wrist_angles_valid",
        ]
        files = [args.memmap_dir / mf["fields"][k]["filename"] for k in consumed]
        files += [args.crops_dir / "manifest.json"] + [
            args.crops_dir / f"{ep}.lmdb/data.mdb"
            for ep in ("episode_000020", "episode_000008", "episode_000028")
        ]
        hashes = []
        for file in files:
            stat = file.stat()
            hashes.append(
                dict(
                    path=str(file),
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=sha256(file),
                )
            )
            print("hashed", file, flush=True)
        json_write(receipt / "input_payload_hashes.json", hashes)
    shutil = __import__("shutil")
    shutil.copyfile(__file__, receipt / "executed_input_audit.py")
    json_write(
        receipt / "receipt.json",
        dict(
            script_sha256=sha256(Path(__file__)),
            command=sys.argv,
            frozen_unchanged=True,
            endpoint_checks=len(endpoint_rows),
            selected_input_windows=len(ix),
            bulk_input_or_feature_files_written=False,
            created_at=__import__("datetime")
            .datetime.now(__import__("zoneinfo").ZoneInfo("Asia/Seoul"))
            .isoformat(),
        ),
    )


if __name__ == "__main__":
    main()
