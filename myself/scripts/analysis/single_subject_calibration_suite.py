#!/usr/bin/env python3
"""Resumable offline within-participant calibration using released preview assets."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from egoemg.constants import EMG_SAMPLE_RATE, JOINTS
from myself.scripts.analysis.preview_calibration_smoke import (
    SelectedSupports,
    aligned_center,
    angular_mae,
    fit_offsets,
    open_field,
    runs,
    sha256,
)
from myself.scripts.analysis.run_directories import create_run_directory


OUTPUT_PARENT = REPO / "myself/outputs/single_subject_calibration_suite"
VERSION = 1
SEEDS = (42, 43, 44)
LOG = logging.getLogger(__name__)


def json_write(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=lambda v: (
                v.item()
                if isinstance(v, np.generic)
                else (
                    v.tolist()
                    if isinstance(v, np.ndarray)
                    else OmegaConf.to_container(v, resolve=True)
                )
            ),
        )
        + "\n"
    )
    tmp.replace(path)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def npz_write(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def union_intervals(intervals: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([int(a), int(b)])
    return merged


def assert_support_separation(
    starts: np.ndarray, pools: np.ndarray, window: int, split: np.ndarray, train_id: int
) -> None:
    assert len(starts) == len(np.unique(starts))
    assert np.all(np.diff(starts) > 0)
    for a, pool in zip(starts, pools):
        assert 0 <= a and a + window <= len(split)
        assert np.all(split[a : a + window] == split[a]), "Support crosses split"
        assert (split[a] == train_id) == (pool == "T_all")
    train = union_intervals([(a, a + window) for a in starts[pools == "T_all"]])
    ev = union_intervals([(a, a + window) for a in starts[pools == "E"]])
    i = j = 0
    while i < len(train) and j < len(ev):
        a, b = train[i]
        c, d = ev[j]
        assert max(a, c) >= min(b, d), "Fitting/evaluation supports intersect"
        if b <= d:
            i += 1
        else:
            j += 1


def enumerate_windows(
    split: np.ndarray,
    valid: np.ndarray,
    emg: np.ndarray,
    reference: np.ndarray,
    split_names: dict[int, str],
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Check center labels only; finite EMG prefix sums test complete support."""
    if window <= 0 or stride <= 0:
        raise ValueError("Positive window and stride required")
    bad = np.r_[0, np.cumsum(~np.isfinite(emg).all(axis=1), dtype=np.int64)]
    selected, pools, intervals = [], [], []
    for interval, (a, b, sid) in enumerate(runs(split)):
        name = split_names[int(sid)]
        if name not in ("train", "user", "gesture", "both"):
            continue
        candidates = np.arange(a, b - window + 1, stride, dtype=np.int64)
        reasons = Counter()
        eligible = []
        for start in candidates:
            center = start + window // 2
            reason = None
            if bad[start + window] != bad[start]:
                reason = "nonfinite_input"
            elif not valid[center]:
                reason = "invalid_center_supervision"
            elif not np.isfinite(reference[center]).all():
                reason = "nonfinite_center_supervision"
            if reason:
                reasons[reason] += 1
            else:
                eligible.append(int(start))
        pool = "T_all" if name == "train" else "E"
        selected.extend(eligible)
        pools.extend([pool] * len(eligible))
        intervals.append(
            dict(
                interval=interval,
                split=name,
                split_id=int(sid),
                local_start=a,
                local_end=b,
                rows=b - a,
                candidate_windows=len(candidates),
                eligible_windows=len(eligible),
                insufficient_length=b - a < window,
                exclusions=dict(reasons),
            )
        )
    starts = np.asarray(selected, dtype=np.int64)
    pools = np.asarray(pools, dtype="U5")
    train_id = next(i for i, name in split_names.items() if name == "train")
    assert_support_separation(starts, pools, window, split, train_id)
    return starts, pools, intervals


def coverage(starts: np.ndarray, window: int, stride: int) -> dict:
    support = union_intervals([(a, a + window) for a in starts])
    centers = union_intervals(
        [(a + window // 2, a + window // 2 + stride) for a in starts]
    )
    support_rows = sum(b - a for a, b in support)
    return dict(
        centers=len(starts),
        support_union_rows=support_rows,
        support_union_seconds=support_rows / EMG_SAMPLE_RATE,
        nominal_center_seconds=sum(b - a for a, b in centers) / EMG_SAMPLE_RATE,
        support_unions=support,
        overlapping_supports=len(starts) * window > support_rows,
    )


def rgb_reason(
    frame: int,
    stale: bool,
    delta: float,
    max_delta: float,
    crop_status: str = "ok",
    finite: bool = True,
) -> str:
    """Eligibility depends on mapping/quality, never reference or prediction error."""
    if frame < 0:
        return "mapping"
    if stale:
        return "stale"
    if not np.isfinite(delta):
        return "nonfinite_delta"
    if abs(delta) > max_delta:
        return "delta_tolerance"
    if crop_status != "ok":
        return crop_status
    return "eligible" if finite else "nonfinite_prediction"


def fit_affine(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    scales, offsets = np.ones(pred.shape[1]), np.zeros(pred.shape[1])
    counts = mask.sum(axis=0)
    variance, fallbacks = np.zeros(pred.shape[1]), []
    for j in range(pred.shape[1]):
        ix = mask[:, j]
        if not ix.any():
            fallbacks.append("empty_identity")
            continue
        p, t = pred[ix, j].astype(np.float64), target[ix, j].astype(np.float64)
        assert np.isfinite(p).all() and np.isfinite(t).all()
        variance[j] = p.var()
        if variance[j] < 1e-6:
            offsets[j] = (t - p).mean() / 2
            fallbacks.append("low_variation_offset")
        else:
            x = np.column_stack([p, np.ones(len(p))])
            q = np.linalg.solve(
                x.T @ x / len(p) + np.diag([0.01, 1.0]), x.T @ (t - p) / len(p)
            )
            scales[j], offsets[j] = 1 + q[0], q[1]
            fallbacks.append("none")
    return dict(
        scales=scales,
        offsets=offsets,
        counts=counts,
        variance=variance,
        fallbacks=fallbacks,
    )


def apply_fingers(baseline: np.ndarray, fingers: np.ndarray) -> np.ndarray:
    out = baseline.astype(np.float64).copy()
    out[:, :20] = fingers
    np.testing.assert_array_equal(out[:, 20:], baseline[:, 20:])
    return out


def train_head(
    features: np.ndarray,
    targets: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    seed: int,
    epochs: int = 50,
    batch_size: int = 256,
    device: str = "cuda:0",
) -> tuple[dict, dict]:
    """Only finger rows exist in the optimizer; wrist rows cannot be updated."""
    torch.manual_seed(seed)
    begin = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    w = torch.nn.Parameter(torch.tensor(w0[:20], device=device))
    b = torch.nn.Parameter(torch.tensor(b0[:20], device=device))
    initial_w, initial_b = w.detach().clone(), b.detach().clone()
    opt = torch.optim.Adam([w, b], lr=1e-3, weight_decay=0)
    x, y = torch.from_numpy(features), torch.from_numpy(targets.astype(np.float32))
    assert (
        len(x) == len(y) and np.isfinite(features).all() and np.isfinite(targets).all()
    )
    losses = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        total, mse_total = 0.0, 0.0
        for ix in order.split(batch_size):
            if not len(ix):
                continue
            xb, yb = x[ix].to(device), y[ix].to(device)
            opt.zero_grad(set_to_none=True)
            mse = F.mse_loss(F.linear(xb, w, b), yb)
            loss = (
                mse
                + 1e-3 * (w - initial_w).square().mean()
                + 1e-3 * (b - initial_b).square().mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite head loss seed={seed} epoch={epoch+1}"
                )
            loss.backward()
            opt.step()
            total += loss.item() * len(ix)
            mse_total += mse.item() * len(ix)
        losses.append(
            dict(
                epoch=epoch + 1,
                objective=total / len(x) if len(x) else 0,
                mse=mse_total / len(x) if len(x) else 0,
            )
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result = dict(weight=w.detach().cpu().numpy(), bias=b.detach().cpu().numpy())
    assert all(np.isfinite(v).all() for v in result.values())
    stats = dict(
        seed=seed,
        epochs=epochs,
        losses=losses,
        fitting_seconds=time.perf_counter() - begin,
        peak_gpu_bytes=(
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        ),
        updated_parameters=w.numel() + b.numel(),
        weight_rms_deviation=float((w - initial_w).square().mean().sqrt()),
        bias_rms_deviation=float((b - initial_b).square().mean().sqrt()),
        finite=True,
        supervision_centers=len(x),
    )
    return result, stats


def configure(args: argparse.Namespace, run: Path) -> tuple[Any, Any, int, Path, dict]:
    root = args.data_root.resolve()
    if not (root / "manifest.json").is_file():
        root = root / "data/memmap_data"
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(root)
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        cfg = compose(
            config_name="base",
            overrides=[
                "experiment=emgformer/egoemg_emgformer_middle",
                "train=false",
                "eval=true",
                "num_workers=0",
                f"egoemg_unified_memmap_dir={root}",
                "trainer.devices=[0]",
                f"checkpoint={args.checkpoint}",
                f"batch_size={args.emg_batch_size}",
                f"hydra.run.dir={run/'hydra'}",
                f"logger.save_dir={run/'logger'}",
            ],
        )
    cfg.datamodule.per_dataset_norm_stats_path = str(
        REPO / "assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
    )
    ds = OmegaConf.create(OmegaConf.to_container(cfg.dataset.val[1], resolve=True))
    ds.update(
        dataset_name="egoemg",
        allowed_splits=["train", "user", "gesture", "both"],
        allowed_episode_ids=[args.episode],
        stride=args.stride,
        center_target_only=True,
        jitter=False,
    )
    base = instantiate(ds, _convert_="all")
    index = base._episode_id.index(args.episode)
    assert base._episode_subject[index] == "wmh"
    assert base.window_length == 12000 and base.target_hand == "right"
    assert (
        base.emg_layout == "target_hand"
        and base.emg_field_preference == "filtered_paper"
    )
    assert base.transform is None and not base.channel_interpolate
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["generated_joint_angles_semantics"] == [
        j.name.lower() for j in JOINTS
    ]
    OmegaConf.save(cfg, run / "resolved_emg_config.yaml", resolve=True)
    OmegaConf.save(ds, run / "resolved_dataset_config.yaml", resolve=True)
    return cfg, base, index, root, manifest


def identity(
    args: argparse.Namespace, run: Path, cfg: Any, base: Any, root: Path
) -> dict:
    small = [
        root / "manifest.json",
        root / "metadata.npz",
        Path(base.norm_stats_path),
        Path(__file__),
        REPO / "myself/scripts/analysis/preview_calibration_smoke.py",
        REPO / "myself/scripts/analysis/run_directories.py",
    ]
    small += sorted((REPO / "config").rglob("*.yaml"))
    small += sorted((REPO / "egoemg/models").rglob("*.py"))
    small += [REPO / "egoemg/datasets/egoemg_memmap_dataset.py"]
    for p in (
        root / "gesture_classes.json",
        root.parent / "pre-crop_webcam_videoframes/manifest.json",
    ):
        if p.is_file():
            small.append(p)
    return dict(
        version=VERSION,
        small_file_sha256={str(p): sha256(p) for p in small},
        checkpoint_sha256=sha256(args.checkpoint),
        teacher_sha256=sha256(args.rgb_checkpoint),
        recording=args.episode,
        participant="wmh",
        hand="right",
        data_root=str(root),
        stride=args.stride,
        tiny_limit=args.limit_per_pool,
        window=base.window_length,
        max_rgb_delta_ms=args.max_rgb_delta_ms,
        emg_batch=args.emg_batch_size,
        rgb_batch=args.rgb_batch_size,
        feature_batch=args.feature_batch_size,
        emg_module=OmegaConf.to_container(cfg.module, resolve=True),
        field="emg_right_filtered_paper",
        normalization=dict(
            path=str(base.norm_stats_path),
            key="egoemg__filtered_paper_right",
            mean=base._emg_mean.tolist(),
            std=base._emg_std.tolist(),
            epsilon=1e-6,
        ),
        alignment="full-sequence linear align_corners=False then input-center-minus-left-context",
        finger_joints=[j.name.lower() for j in JOINTS],
        units="radians; wrist source degrees converted",
        head="actual last head.net Linear, BTC flattened inputs; only reshape after layer",
        seeds=list(SEEDS),
        offset_lambda=1.0,
        affine_lambda_a=0.01,
        affine_lambda_b=1.0,
        affine_variance_floor=1e-6,
        head_lr=0.001,
        head_regularization=0.001,
        head_epochs=50,
    )


def mark(run: Path, stage: str, ident: dict, files: list[str]) -> None:
    json_write(
        run / f"{stage}.complete.json",
        dict(identity=ident, files={f: sha256(run / f) for f in files}),
    )


def completed(run: Path, stage: str, ident: dict) -> bool:
    marker = run / f"{stage}.complete.json"
    if not marker.exists():
        return False
    value = json.loads(marker.read_text())
    if value["identity"] != ident:
        raise ValueError(f"Incompatible {stage} cache")
    for f, digest in value["files"].items():
        if not (run / f).is_file() or sha256(run / f) != digest:
            raise ValueError(f"Incomplete/corrupt {stage} cache: {f}")
    return True


def manifest_stage(
    args: argparse.Namespace,
    run: Path,
    base: Any,
    ep: int,
    root: Path,
    manifest: dict,
    ident: dict,
) -> None:
    origin, stop = int(base._episode_start_idx[ep]), int(base._episode_end_idx[ep])
    field = lambda name: open_field(root, manifest, name)
    md = np.load(root / "metadata.npz")
    names = {
        int(i): s.decode() for i, s in zip(md["splits_split_id"], md["splits_split"])
    }
    assert [names[i] for i in sorted(names)] == manifest["frame_split_labels"]
    split = field("frame_split_id")[origin:stop]
    starts, pools, intervals = enumerate_windows(
        split,
        field("generated_label_valid")[origin:stop, 1],
        field("emg_right_filtered_paper")[origin:stop],
        field("generated_joint_angles_right")[origin:stop],
        names,
        base.window_length,
        args.stride,
    )
    if args.limit_per_pool:
        keep = np.zeros(len(starts), bool)
        for p in ("T_all", "E"):
            keep[np.flatnonzero(pools == p)[: args.limit_per_pool]] = True
        starts, pools = starts[keep], pools[keep]
    centers = starts + base.window_length // 2 + origin
    gestures = np.asarray(field("label_gesture_class")[centers])
    active = np.asarray(field("label_gesture_active")[centers], bool)
    ref = np.column_stack(
        [
            field("generated_joint_angles_right")[centers],
            np.deg2rad(field("mocap_right_wrist_pitch")[centers]),
            np.deg2rad(field("mocap_right_wrist_yaw")[centers]),
        ]
    ).astype(np.float32)
    masks = np.isfinite(ref)
    masks[:, 20:] &= field("mocap_right_wrist_angles_valid")[centers, None]
    assert masks[:, :20].all()
    assert np.all(field("episode_index")[origin:stop] == ep)
    subjects = np.unique(field("subject_id")[origin:stop]).tolist()
    assert len(subjects) == 1 and np.all(field("dataset_source_id")[origin:stop] == 0)
    vocab_path = root / "gesture_classes.json"
    vocab = json.loads(vocab_path.read_text()) if vocab_path.is_file() else {}
    classes = vocab.get("classes", {})
    gesture_name = np.asarray(
        [
            classes.get(str(int(g)), {}).get(
                "name", "inactive" if g == -1 else "unknown"
            )
            for g in gestures
        ]
    )
    interval_id = np.asarray(
        [
            next(
                v["interval"]
                for v in intervals
                if v["local_start"] <= a < v["local_end"]
            )
            for a in starts
        ]
    )
    rows = [
        dict(
            id=f"{args.episode}:R:{int(c)}",
            pool=str(p),
            participant="wmh",
            episode=args.episode,
            interval=int(i),
            split=names[int(split[a])],
            split_id=int(split[a]),
            local_start=int(a),
            local_end=int(a + base.window_length),
            local_center=int(c - origin),
            global_start=int(a + origin),
            global_end=int(a + origin + base.window_length),
            global_center=int(c),
            gesture_id=int(g),
            gesture_name=str(n),
            gesture_active=bool(ac),
        )
        for a, c, p, i, g, n, ac in zip(
            starts, centers, pools, interval_id, gestures, gesture_name, active
        )
    ]
    csv_write(run / "windows.csv", rows)
    npz_write(
        run / "manifest.npz",
        starts=starts + origin,
        centers=centers,
        pools=pools,
        reference=ref,
        mask=masks,
        gesture=gestures,
        active=active,
        gesture_name=gesture_name,
        split=np.asarray([r["split"] for r in rows]),
        interval=interval_id,
        ids=np.asarray([r["id"] for r in rows]),
    )
    summary = dict(
        intervals=intervals,
        pool_coverage={
            p: coverage(starts[pools == p], base.window_length, args.stride)
            for p in ("T_all", "E")
        },
        episode_global=[origin, stop],
        participant="wmh",
        metadata_subject_id=int(base._episode_subject_id[ep]),
        row_subject_ids=subjects,
        splits=names,
        nominal_sample_rate_hz=EMG_SAMPLE_RATE,
        duration_convention="Support union length/2000; nominal center union of [center,center+stride)/2000. Overlapping supports are not summed; timestamps need not be uniform.",
        gesture_vocabulary=vocab,
        support_intersection_checked=True,
        exclusion_convention="First failing reason: input, center validity, center finiteness; intervals shorter than window separately reported.",
        tiny_machinery_check=bool(args.limit_per_pool),
    )
    json_write(run / "coverage.json", summary)
    gesture_rows = []
    for p in ("T_all", "E"):
        for g, ac in sorted(
            set(zip(gestures[pools == p].tolist(), active[pools == p].tolist()))
        ):
            ix = (pools == p) & (gestures == g) & (active == ac)
            gesture_rows.append(
                dict(
                    pool=p,
                    gesture_id=g,
                    active=ac,
                    name=str(gesture_name[ix][0]),
                    centers=int(ix.sum()),
                )
            )
    csv_write(run / "gesture_coverage.csv", gesture_rows)
    mark(
        run,
        "manifest",
        ident,
        ["manifest.npz", "windows.csv", "coverage.json", "gesture_coverage.csv"],
    )
    LOG.info("Manifest: %s", {p: int((pools == p).sum()) for p in ("T_all", "E")})
    if not (pools == "T_all").any() or not (pools == "E").any():
        raise RuntimeError("Protocol blocker: empty T_all or E; see coverage.json")


def strict_model(config: Any, path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model = instantiate(config)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to("cuda:0")
    return model, ckpt.get("hyper_parameters", {})


def emg_stage(
    args: argparse.Namespace, run: Path, cfg: Any, base: Any, ep: int, ident: dict
) -> None:
    m = np.load(run / "manifest.npz")
    begin = time.perf_counter()
    model, hparams = strict_model(cfg.module, args.checkpoint)
    linears = [
        (n, v) for n, v in model.head.named_modules() if isinstance(v, torch.nn.Linear)
    ]
    name, layer = linears[-1]
    assert layer is model.head.net[-1] and layer.out_features == 22
    captured = []
    hook = layer.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])
    )
    loader = DataLoader(
        SelectedSupports(base, ep, m["starts"]),
        batch_size=args.emg_batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions, features = [], []
    max_error = 0.0
    max_relative = 0.0
    raw_length = None
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            captured.clear()
            raw = model({"emg": batch["emg"].to("cuda:0")})
            assert len(captured) == 1 and raw.ndim == 3 and raw.shape[1] == 22
            b, _, t = raw.shape
            raw_length = t
            fc = captured[0].reshape(b, t, layer.in_features).transpose(1, 2)
            center = aligned_center(model, fc, base.window_length)
            original = aligned_center(model, raw, base.window_length)
            cached = F.linear(center, layer.weight, layer.bias)
            error = (cached - original).abs()
            max_error = max(max_error, error.max().item())
            max_relative = max(
                max_relative, (error / (original.abs() + 1e-12)).max().item()
            )
            torch.testing.assert_close(cached, original, atol=1e-5, rtol=1e-4)
            assert torch.isfinite(center).all() and torch.isfinite(original).all()
            np.testing.assert_allclose(
                batch["joint_angles"].numpy()[..., 0],
                m["reference"][i * args.emg_batch_size : i * args.emg_batch_size + b],
                atol=1e-6,
            )
            features.append(center.cpu().numpy())
            predictions.append(original.cpu().numpy())
            if i % 50 == 0:
                LOG.info(
                    "EMG batch %d/%d elapsed %.1fs",
                    i + 1,
                    len(loader),
                    time.perf_counter() - begin,
                )
    hook.remove()
    torch.cuda.synchronize()
    npz_write(
        run / "emg_cache.npz",
        ids=m["ids"],
        features=np.concatenate(features),
        prediction=np.concatenate(predictions),
        weight=layer.weight.detach().cpu().numpy(),
        bias=layer.bias.detach().cpu().numpy(),
    )
    evidence = dict(
        layer_name="head." + name,
        feature_dimension=layer.in_features,
        raw_timeline_length=raw_length,
        layer_input_layout="(B*T,F); preceding frozen MLP included",
        post_layer="reshape/transpose, then linear temporal interpolation",
        left_context=model.left_context,
        right_context=model.right_context,
        aligned_length=base.window_length - model.left_context - model.right_context,
        center_index=base.window_length // 2 - model.left_context,
        atol=1e-5,
        rtol=1e-4,
        maximum_absolute_discrepancy=max_error,
        maximum_relative_discrepancy=max_relative,
        parameters=sum(p.numel() for p in model.parameters()),
        strict_state=True,
        frozen_eval=True,
        checkpoint_hparams=hparams,
        obsolete_import_handling="Local configuration architecture; strict model.* state; checkpoint historical emg2pose import strings not instantiated",
        extraction_seconds=time.perf_counter() - begin,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    json_write(run / "emg_evidence.json", evidence)
    del model, layer, captured, raw, fc, center, original, cached, batch
    gc.collect()
    torch.cuda.empty_cache()
    mark(run, "emg", ident, ["emg_cache.npz", "emg_evidence.json"])


def teacher_stage(
    args: argparse.Namespace,
    run: Path,
    base: Any,
    root: Path,
    manifest: dict,
    ident: dict,
) -> None:
    from PIL import Image, ImageDraw

    m = np.load(run / "manifest.npz")
    train = np.flatnonzero(m["pools"] == "T_all")
    for k in ("image_head_frame_index", "image_head_stale", "image_head_delta_ms"):
        if k not in manifest["fields"]:
            raise RuntimeError(f"RGB readiness blocker: missing {k}")
    centers = m["centers"][train]
    frame = np.asarray(open_field(root, manifest, "image_head_frame_index")[centers])
    stale = np.asarray(open_field(root, manifest, "image_head_stale")[centers])
    delta = np.asarray(open_field(root, manifest, "image_head_delta_ms")[centers])
    reasons = np.asarray(
        [
            rgb_reason(int(f), bool(s), float(d), args.max_rgb_delta_ms)
            for f, s, d in zip(frame, stale, delta)
        ],
        dtype="U40",
    )
    base.per_episode_crops_dir = str(root.parent / "pre-crop_webcam_videoframes")
    base._pec_pid = os.getpid()
    base._pec_lmdb_cache = {}
    base._pec_lmdb_max_open = 1
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        cfg = compose(
            config_name="base",
            overrides=[
                "experiment=fusion/vision_resnet18",
                f"hydra.run.dir={run/'hydra'}",
                f"logger.save_dir={run/'logger'}",
            ],
        )
    cfg.module.pretrained = False
    OmegaConf.save(cfg, run / "resolved_teacher_config.yaml", resolve=True)
    begin = time.perf_counter()
    model, hparams = strict_model(cfg.module, args.rgb_checkpoint)
    trained = hparams["module_conf"]
    for key in ("out_channels", "backbone_type", "head_hidden", "head_dropout"):
        assert trained[key] == cfg.module[key]
    torch.cuda.reset_peak_memory_stats()
    unique = np.unique(frame[reasons == "eligible"])
    outputs = {}
    statuses = {}
    contact = []
    buffer = []
    buffer_frames = []

    def infer() -> None:
        if not buffer:
            return
        with torch.inference_mode():
            pred = (
                model({"vision_img": torch.tensor(np.stack(buffer), device="cuda:0")})[
                    ..., 0
                ]
                .cpu()
                .numpy()
            )
        for f, p in zip(buffer_frames, pred):
            outputs[int(f)] = p
            statuses[int(f)] = (
                "ok" if np.isfinite(p[:20]).all() else "nonfinite_prediction"
            )
        buffer.clear()
        buffer_frames.clear()

    for i, f in enumerate(unique):
        status = "ok"
        try:
            images = base._read_episode_crops(args.episode, np.asarray([f]), "R")
            if images is None:
                status = "missing_crop"
        except (OSError, ValueError, TypeError) as error:
            status = "decode"
            LOG.warning("Crop %s decode: %s", f, error)
        if status == "ok":
            try:
                rgb = images[0]
                if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                    raise ValueError(f"Unexpected crop {rgb.shape}/{rgb.dtype}")
                patch = (
                    rgb.transpose(2, 0, 1).astype(np.float32)
                    - base.vision_mean[:, None, None]
                ) / base.vision_std[:, None, None]
                if not np.isfinite(patch).all():
                    raise ValueError("Nonfinite preprocessing")
                buffer.append(patch)
                buffer_frames.append(int(f))
                if len(contact) < 12:
                    contact.append((int(f), Image.fromarray(rgb).resize((128, 128))))
            except (ValueError, TypeError) as error:
                status = "preprocessing"
                LOG.warning("Crop %s preprocess: %s", f, error)
        statuses[int(f)] = status
        if len(buffer) == args.rgb_batch_size:
            infer()
        if i % 200 == 0:
            LOG.info("RGB crop %d/%d", i + 1, len(unique))
    infer()
    torch.cuda.synchronize()
    target = np.full((len(train), 22), np.nan, dtype=np.float32)
    for i, f in enumerate(frame):
        if reasons[i] == "eligible":
            status = statuses[int(f)]
            if status == "ok":
                target[i] = outputs[int(f)]
            else:
                reasons[i] = status
    eligible = reasons == "eligible"
    ix = train[eligible]
    npz_write(
        run / "teacher_cache.npz",
        train_ids=m["ids"][train],
        train_indices=train,
        eligible_indices=ix,
        eligible_ids=m["ids"][ix],
        frame=frame,
        stale=stale,
        delta_ms=delta,
        reason=reasons,
        target=target,
        unique_frames=unique,
        unique_prediction=(
            np.stack([outputs.get(int(f), np.full(22, np.nan)) for f in unique])
            if len(unique)
            else np.empty((0, 22))
        ),
    )
    sheet = Image.new("RGB", (4 * 128, 3 * 152), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (f, img) in enumerate(contact):
        x, y = i % 4 * 128, i // 4 * 152
        sheet.paste(img, (x, y))
        draw.text((x, y + 130), f"frame {f} R", fill="black")
    sheet.save(run / "crop_contact_sheet.png")
    interval_rows = []
    for v in json.loads((run / "coverage.json").read_text())["intervals"]:
        if v["split"] == "train":
            sel = m["interval"][train] == v["interval"]
            interval_rows.append(
                dict(
                    interval=v["interval"],
                    local_start=v["local_start"],
                    local_end=v["local_end"],
                    T_all=int(sel.sum()),
                    T_rgb=int((sel & eligible).sum()),
                    unique_rgb_frames=len(np.unique(frame[sel & eligible])),
                    exclusions=dict(Counter(reasons[sel & ~eligible])),
                )
            )
    csv_write(run / "rgb_interval_coverage.csv", interval_rows)
    finite_delta = delta[np.isfinite(delta)]
    diagnostic = dict(
        T_rgb_coverage=coverage(m["starts"][ix], base.window_length, args.stride),
        eligible_centers=int(eligible.sum()),
        all_train_centers=len(train),
        eligible_unique_frames=len(np.unique(frame[eligible])),
        inferred_unique_frames=len(outputs),
        exclusions=dict(Counter(reasons[~eligible])),
        stale_count=int(stale.sum()),
        delta_ms_quantiles=(
            np.quantile(finite_delta, [0, 0.25, 0.5, 0.75, 1]).tolist()
            if len(finite_delta)
            else []
        ),
        absolute_delta_ms_quantiles=(
            np.quantile(abs(finite_delta), [0, 0.25, 0.5, 0.75, 1]).tolist()
            if len(finite_delta)
            else []
        ),
        nonfinite_delta_count=int((~np.isfinite(delta)).sum()),
        intervals=interval_rows,
        teacher_finger_mae=angular_mae(
            target[eligible, :20], m["reference"][ix, :20], m["mask"][ix, :20]
        ),
        construction_seconds=time.perf_counter() - begin,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        preprocessing=dict(
            rgb=True,
            right_hand_key="%08d_R",
            orientation="stored R crop unchanged, no mirroring",
            size=[256, 256],
            mean=base.vision_mean.tolist(),
            std=base.vision_std.tolist(),
            resize="none; require released stored 256x256",
        ),
        checkpoint_hparams=hparams,
        strict_state=True,
        pretrained_initialization=False,
        target_units="20 finger + 2 wrist radians, same joint order",
        max_rgb_delta_ms=args.max_rgb_delta_ms,
        tolerance_note="Predeclared first-pass threshold, not a demonstrated synchronization guarantee",
        label_independent_eligibility=True,
    )
    json_write(run / "teacher_diagnostics.json", diagnostic)
    quality = []
    for j, n in enumerate(ident["finger_joints"]):
        quality.append(
            dict(
                kind="joint",
                key=n,
                **angular_mae(
                    target[eligible, j : j + 1],
                    m["reference"][ix, j : j + 1],
                    m["mask"][ix, j : j + 1],
                ),
            )
        )
    for g, ac in sorted(set(zip(m["gesture"][ix].tolist(), m["active"][ix].tolist()))):
        select = (m["gesture"][ix] == g) & (m["active"][ix] == ac)
        quality.append(
            dict(
                kind="gesture",
                key=f"{g}:active={ac}",
                **angular_mae(
                    target[eligible, :20][select],
                    m["reference"][ix, :20][select],
                    m["mask"][ix, :20][select],
                ),
            )
        )
    csv_write(run / "teacher_quality.csv", quality)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mark(
        run,
        "teacher",
        ident,
        [
            "teacher_cache.npz",
            "teacher_diagnostics.json",
            "teacher_quality.csv",
            "rgb_interval_coverage.csv",
            "crop_contact_sheet.png",
            "resolved_teacher_config.yaml",
        ],
    )
    LOG.info(
        "RGB coverage: %d/%d centers, %d unique frames",
        eligible.sum(),
        len(train),
        len(np.unique(frame[eligible])),
    )


def fitting_stage(args: argparse.Namespace, run: Path, ident: dict) -> None:
    m = np.load(run / "manifest.npz")
    c = np.load(run / "emg_cache.npz")
    np.testing.assert_array_equal(m["ids"], c["ids"])
    # Before any update, every compact original-weight center must reproduce frozen output.
    with torch.inference_mode():
        p = F.linear(
            torch.from_numpy(c["features"]),
            torch.from_numpy(c["weight"]),
            torch.from_numpy(c["bias"]),
        ).numpy()
    np.testing.assert_allclose(p, c["prediction"], atol=1e-5, rtol=1e-4)
    train = np.flatnonzero(m["pools"] == "T_all")
    specifications = [
        ("M1", "offset", "reference", "T_all", train, m["reference"][train, :20]),
        ("M2", "affine", "reference", "T_all", train, m["reference"][train, :20]),
        ("M3", "head", "reference", "T_all", train, m["reference"][train, :20]),
    ]
    blocked = {}
    teacher = None
    if completed(run, "teacher", ident):
        teacher = np.load(run / "teacher_cache.npz")
        np.testing.assert_array_equal(teacher["train_ids"], m["ids"][train])
        rgb = teacher["eligible_indices"]
        if len(rgb):
            eligible = teacher["reason"] == "eligible"
            for cid, method in (("M4", "offset"), ("M5", "affine"), ("M6", "head")):
                specifications.append(
                    (cid, method, "RGB", "T_rgb", rgb, teacher["target"][eligible, :20])
                )
            if not np.array_equal(train, rgb):
                for cid, method in (("R1", "offset"), ("R2", "affine"), ("R3", "head")):
                    specifications.append(
                        (
                            cid,
                            method,
                            "reference",
                            "T_rgb",
                            rgb,
                            m["reference"][rgb, :20],
                        )
                    )
        else:
            blocked = {
                key: "Empty T_rgb" for key in ("M4", "M5", "M6", "R1", "R2", "R3")
            }
    else:
        blocked = {
            key: "Teacher stage blocked; see blockers.json"
            for key in ("M4", "M5", "M6", "R1", "R2", "R3")
        }
    directory = run / "parameters"
    directory.mkdir(exist_ok=True)
    conditions = []
    for cid, method, supervision, pool, ix, target in specifications:
        for seed in SEEDS if method == "head" else (None,):
            key = cid + (f"_seed{seed}" if seed is not None else "")
            path = directory / f"{key}.npz"
            log_path = directory / f"{key}.json"
            if path.exists() and log_path.exists():
                previous = json.loads(log_path.read_text())
                if previous.get("identity") != ident or previous.get(
                    "parameter_sha256"
                ) != sha256(path):
                    raise ValueError(f"Incompatible/incomplete fitted parameters {key}")
                conditions.append(previous)
                continue
            begin = time.perf_counter()
            if method == "offset":
                b, counts = fit_offsets(
                    c["prediction"][ix, :20], target, m["mask"][ix, :20], 1.0
                )
                params = dict(offsets=b, counts=counts)
                stats = dict(updated_parameters=20, peak_gpu_bytes=0)
            elif method == "affine":
                result = fit_affine(
                    c["prediction"][ix, :20], target, m["mask"][ix, :20]
                )
                params = {k: np.asarray(v) for k, v in result.items()}
                stats = dict(updated_parameters=40, peak_gpu_bytes=0)
            else:
                params, stats = train_head(
                    c["features"][ix],
                    target,
                    c["weight"],
                    c["bias"],
                    seed,
                    batch_size=args.feature_batch_size,
                )
            stats.pop("seed", None)
            stats.setdefault("fitting_seconds", time.perf_counter() - begin)
            npz_write(
                path,
                **params,
                initial_weight=c["weight"][:20],
                initial_bias=c["bias"][:20],
                fit_ids=m["ids"][ix],
            )
            fit_prediction = evaluated_predictions(
                c["prediction"][ix], c["features"][ix], method, params
            )
            stats["fit_target_mae"] = angular_mae(
                fit_prediction[:, :20], target, m["mask"][ix, :20]
            )
            stats["fit_reference_mae"] = angular_mae(
                fit_prediction[:, :20], m["reference"][ix, :20], m["mask"][ix, :20]
            )
            info = dict(
                id=cid,
                key=key,
                method=method,
                supervision=supervision,
                pool=pool,
                seed=seed,
                fitting_centers=len(ix),
                unique_rgb_frames=(
                    len(np.unique(teacher["frame"][teacher["reason"] == "eligible"]))
                    if pool == "T_rgb"
                    else None
                ),
                active_gestures=len(np.unique(m["gesture"][ix][m["active"][ix]])),
                total_model_parameters=json.loads(
                    (run / "emg_evidence.json").read_text()
                )["parameters"],
                identity=ident,
                parameter_sha256=sha256(path),
                **stats,
            )
            json_write(log_path, info)
            conditions.append(info)
            LOG.info("Fitted %s %s/%s (%d centers)", key, supervision, pool, len(ix))
    json_write(
        run / "conditions.json",
        dict(
            conditions=conditions,
            blocked=blocked,
            matched_reuses_full_reference=teacher is not None
            and np.array_equal(train, teacher["eligible_indices"]),
        ),
    )
    files = ["conditions.json"] + [
        str(p.relative_to(run))
        for p in directory.iterdir()
        if p.suffix in (".npz", ".json")
    ]
    mark(run, "fit", ident, files)


def evaluated_predictions(
    baseline: np.ndarray, features: np.ndarray, method: str, parameters: dict
) -> np.ndarray:
    if method == "offset":
        fingers = baseline[:, :20] + parameters["offsets"]
    elif method == "affine":
        fingers = baseline[:, :20] * parameters["scales"] + parameters["offsets"]
    elif method == "head":
        # CPU FP32 same compact center convention; no feature normalization.
        with torch.inference_mode():
            fingers = F.linear(
                torch.from_numpy(features),
                torch.from_numpy(parameters["weight"]),
                torch.from_numpy(parameters["bias"]),
            ).numpy()
    else:
        raise ValueError(method)
    return apply_fingers(baseline, fingers)


def summarize_seeds(rows: list[dict]) -> list[dict]:
    summary = []
    for cid in dict.fromkeys(r["id"] for r in rows):
        group = [r for r in rows if r["id"] == cid]
        row = group[0].copy()
        if row["method"] == "head":
            assert sorted(r["seed"] for r in group) == list(SEEDS)
            row["key"] = cid + "_mean"
            row["seed"] = "42/43/44"
            for name in (
                "mae_rad",
                "mae_deg",
                "delta_deg",
                "delta_rad",
                "fitting_seconds",
                "peak_gpu_bytes",
            ):
                row[name] = float(np.mean([r[name] for r in group]))
            row["std_deg"] = float(np.std([r["mae_deg"] for r in group], ddof=1))
        else:
            row["std_deg"] = 0.0
        summary.append(row)
    return summary


def evaluation_stage(run: Path, ident: dict) -> None:
    m = np.load(run / "manifest.npz")
    c = np.load(run / "emg_cache.npz")
    np.testing.assert_array_equal(m["ids"], c["ids"])
    ev = np.flatnonzero(m["pools"] == "E")
    ref = m["reference"][ev]
    mask = m["mask"][ev]
    baseline = c["prediction"][ev]
    conditions = json.loads((run / "conditions.json").read_text())
    base_info = dict(
        id="M0",
        key="M0",
        method="baseline",
        supervision="none",
        pool="none",
        seed=None,
        fitting_centers=0,
        unique_rgb_frames=None,
        active_gestures=0,
        updated_parameters=0,
        fitting_seconds=0.0,
        peak_gpu_bytes=0,
    )
    base_mae = angular_mae(baseline[:, :20], ref[:, :20], mask[:, :20])
    wrist = angular_mae(baseline[:, 20:], ref[:, 20:], mask[:, 20:])
    preds = {"M0": baseline}
    tables = []
    details = []
    for info in [base_info] + conditions["conditions"]:
        key = info["key"]
        if key != "M0":
            params = np.load(run / f"parameters/{key}.npz")
            preds[key] = evaluated_predictions(
                baseline, c["features"][ev], info["method"], params
            )
        p = preds[key]
        np.testing.assert_array_equal(p[:, 20:], baseline[:, 20:])
        assert angular_mae(p[:, 20:], ref[:, 20:], mask[:, 20:]) == wrist
        score = angular_mae(p[:, :20], ref[:, :20], mask[:, :20])
        row = {k: info[k] for k in base_info}
        row.update(
            score,
            delta_rad=score["mae_rad"] - base_mae["mae_rad"],
            delta_deg=score["mae_deg"] - base_mae["mae_deg"],
            wrist_mae_deg=wrist["mae_deg"],
            evaluation_centers=len(ev),
        )
        tables.append(row)
        for split in np.unique(m["split"][ev]):
            ix = m["split"][ev] == split
            details.append(
                dict(
                    key=key,
                    kind="split",
                    group=str(split),
                    centers=int(ix.sum()),
                    **angular_mae(p[ix, :20], ref[ix, :20], mask[ix, :20]),
                )
            )
        for g, ac in sorted(
            set(zip(m["gesture"][ev].tolist(), m["active"][ev].tolist()))
        ):
            ix = (m["gesture"][ev] == g) & (m["active"][ev] == ac)
            details.append(
                dict(
                    key=key,
                    kind="gesture",
                    group=f"{g}:{m['gesture_name'][ev][ix][0]}:active={ac}",
                    centers=int(ix.sum()),
                    **angular_mae(p[ix, :20], ref[ix, :20], mask[ix, :20]),
                )
            )
        for j, name in enumerate(ident["finger_joints"]):
            details.append(
                dict(
                    key=key,
                    kind="joint",
                    group=name,
                    centers=len(ev),
                    **angular_mae(
                        p[:, j : j + 1], ref[:, j : j + 1], mask[:, j : j + 1]
                    ),
                )
            )
    npz_write(
        run / "evaluation_predictions.npz",
        ids=m["ids"][ev],
        centers=m["centers"][ev],
        reference=ref,
        mask=mask,
        split=m["split"][ev],
        gesture=m["gesture"][ev],
        active=m["active"][ev],
        **preds,
    )
    csv_write(run / "per_seed_metrics.csv", tables)
    csv_write(run / "detailed_metrics.csv", details)
    summary = summarize_seeds(tables)
    csv_write(
        run / "primary_comparison.csv", [r for r in summary if r["id"].startswith("M")]
    )
    matched = [
        r for r in summary if r["id"] in ("M0", "M4", "M5", "M6", "R1", "R2", "R3")
    ]
    if conditions["matched_reuses_full_reference"]:
        matched = summary
    csv_write(run / "matched_comparison.csv", matched)
    variation = []
    rgb = (
        np.load(run / "teacher_cache.npz")["eligible_indices"]
        if (run / "teacher_cache.npz").exists()
        else np.array([], int)
    )
    for pool, ix in (
        ("T_all", np.flatnonzero(m["pools"] == "T_all")),
        ("E", ev),
        ("T_rgb", rgb),
    ):
        for j, name in enumerate(ident["finger_joints"]):
            values = m["reference"][ix, j]
            variation.append(
                dict(
                    pool=pool,
                    joint=name,
                    n=len(values),
                    variance_rad2=(
                        float(values.astype(float).var()) if len(values) else None
                    ),
                    std_deg=(
                        float(np.rad2deg(values.astype(float).std()))
                        if len(values)
                        else None
                    ),
                    minimum_deg=(
                        float(np.rad2deg(values.min())) if len(values) else None
                    ),
                    maximum_deg=(
                        float(np.rad2deg(values.max())) if len(values) else None
                    ),
                )
            )
    csv_write(run / "reference_variation.csv", variation)
    json_write(
        run / "metrics.json",
        dict(
            primary=[r for r in summary if r["id"].startswith("M")],
            matched=matched,
            all_seeds=tables,
            wrist=wrist,
            blocked=conditions["blocked"],
            head_std_convention="Sample standard deviation across seeds, ddof=1; not population uncertainty",
            aggregate="mean absolute difference over all valid finger entries; radians/degrees; no wrapping, clipping or smoothing",
            gesture_macro="Not used for primary metric; detailed metrics include active and inactive raw IDs separately",
            evaluation_ids_identical=True,
            wrist_predictions_identical=True,
        ),
    )
    mark(
        run,
        "evaluate",
        ident,
        [
            "evaluation_predictions.npz",
            "per_seed_metrics.csv",
            "detailed_metrics.csv",
            "primary_comparison.csv",
            "matched_comparison.csv",
            "reference_variation.csv",
            "metrics.json",
        ],
    )


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Condition | Supervision / pool | Seed | MAE (deg) | Δ baseline (deg) | Fit centers | Parameters | Fit seconds | Peak GPU MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        value = f"{r['mae_deg']:.6f}"
        if r.get("std_deg"):
            value += f" ± {r['std_deg']:.6f}"
        lines.append(
            f"| {r['key']} | {r['supervision']} / {r['pool']} | {r['seed']} | {value} | {r['delta_deg']:+.6f} | {r['fitting_centers']} | {r['updated_parameters']} | {r['fitting_seconds']:.3f} | {r['peak_gpu_bytes']/2**20:.2f} |"
        )
    return "\n".join(lines) + "\n"


def contiguous_trace_indices(
    candidates: np.ndarray, centers: np.ndarray, gestures: np.ndarray, stride: int
) -> np.ndarray:
    """First chronological gesture occurrence, stopping at any sampling gap."""
    if not len(candidates):
        return np.array([], dtype=np.int64)
    selected = [int(candidates[0])]
    for candidate in candidates[1:]:
        previous = selected[-1]
        if (
            centers[candidate] - centers[previous] != stride
            or gestures[candidate] != gestures[previous]
        ):
            break
        selected.append(int(candidate))
        if len(selected) == 80:
            break
    return np.asarray(selected, dtype=np.int64)


def reporting_stage(run: Path, ident: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads((run / "metrics.json").read_text())
    cov = json.loads((run / "coverage.json").read_text())
    emg = json.loads((run / "emg_evidence.json").read_text())
    teacher = (
        json.loads((run / "teacher_diagnostics.json").read_text())
        if (run / "teacher_diagnostics.json").exists()
        else None
    )
    (run / "primary_comparison.md").write_text(markdown_table(metrics["primary"]))
    (run / "matched_comparison.md").write_text(markdown_table(metrics["matched"]))
    (run / "all_seeds.md").write_text(markdown_table(metrics["all_seeds"]))
    with (run / "detailed_metrics.csv").open() as f:
        detail = list(csv.DictReader(f))
    keys = [r["key"] for r in metrics["primary"] if r["key"] != "M0"]

    def values(kind: str, key: str) -> tuple[list[str], np.ndarray]:
        rows = [r for r in detail if r["kind"] == kind and r["key"] == key]
        return [r["group"] for r in rows], np.asarray(
            [float(r["mae_deg"]) for r in rows]
        )

    def seed_values(kind: str, key: str) -> np.ndarray:
        if key.endswith("_mean"):
            return np.mean(
                [
                    values(kind, key.removesuffix("_mean") + f"_seed{s}")[1]
                    for s in SEEDS
                ],
                axis=0,
            )
        return values(kind, key)[1]

    for kind in ("joint", "gesture"):
        groups, baseline = values(kind, "M0")
        fig, ax = plt.subplots(figsize=(13, 5))
        for key in keys:
            y = seed_values(kind, key)
            ax.plot(
                np.arange(len(groups)),
                y - baseline if kind == "joint" else y,
                label=key,
                marker=".",
                lw=1,
            )
        if kind == "gesture":
            ax.plot(np.arange(len(groups)), baseline, label="M0", color="black")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(np.arange(len(groups)), groups, rotation=90, fontsize=7)
        ax.set_ylabel(
            "MAE change (degrees)" if kind == "joint" else "Finger MAE (degrees)"
        )
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(run / f"per_{kind}.png", dpi=170)
        plt.close(fig)
    p = np.load(run / "evaluation_predictions.npz")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    trace_records = []
    # First contiguous active-gesture occurrence per split; never bridge unavailable time.
    for ax, split in zip(axes, ("user", "gesture", "both")):
        candidates = np.flatnonzero((p["split"] == split) & p["active"])
        if not len(candidates):
            candidates = np.flatnonzero(p["split"] == split)
        if not len(candidates):
            ax.set_title(f"{split}: no eligible centers")
            continue
        gesture = p["gesture"][candidates[0]]
        ix = contiguous_trace_indices(
            candidates, p["centers"], p["gesture"], ident["stride"]
        )
        trace_records.append(
            dict(
                split=split,
                gesture_id=int(gesture),
                joint="index_mcp_fe",
                ids=p["ids"][ix].tolist(),
            )
        )
        t = (p["centers"][ix] - p["centers"][ix[0]]) / EMG_SAMPLE_RATE
        ax.plot(t, np.rad2deg(p["reference"][ix, 5]), label="reference", color="black")
        ax.plot(t, np.rad2deg(p["M0"][ix, 5]), label="M0")
        for key in keys:
            y = (
                np.mean(
                    [p[key.removesuffix("_mean") + f"_seed{s}"][ix, 5] for s in SEEDS],
                    axis=0,
                )
                if key.endswith("_mean")
                else p[key][ix, 5]
            )
            ax.plot(t, np.rad2deg(y), label=key, alpha=0.7, lw=0.8)
        ax.set_title(f"{split}, raw gesture {gesture}, index_mcp_fe")
        ax.set_ylabel("degrees")
        ax.set_xlabel("nominal seconds from first selected center")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "time_traces.png", dpi=150)
    plt.close(fig)
    json_write(run / "trace_identifiers.json", trace_records)
    report = """# Offline single-participant calibration suite

All eligible data in preview recording episode_000020, participant wmh, right hand.
Fixed evaluation supports from user/gesture/both splits; EMG-only evaluation.
Fitting and evaluation input support unions are disjoint despite interleaved intervals.
Overlapping windows within pools are allowed and are not independent replications.

"""
    report += (
        markdown_table(metrics["primary"])
        + "\n## Every head seed\n\n"
        + markdown_table([r for r in metrics["all_seeds"] if r["method"] == "head"])
    )
    report += "\n## Matched supervision\n\n" + markdown_table(metrics["matched"])
    report += "\n## Coverage and evidence\n\n"
    for pool, v in cov["pool_coverage"].items():
        report += f"{pool}: {v['centers']} centers, {v['nominal_center_seconds']:.3f} nominal center seconds, {v['support_union_seconds']:.3f} support union seconds.\n\n"
    report += f"Wrist MAE: {metrics['wrist']['mae_deg']:.6f} degrees, identical across every condition.\n\n"
    report += f"EMG extraction: {emg['extraction_seconds']:.3f}s; peak allocated GPU {emg['peak_gpu_bytes']/2**20:.2f} MiB. Original head/cache maximum absolute discrepancy {emg['maximum_absolute_discrepancy']:.3g} radians, tolerance atol=1e-5 rtol=1e-4.\n\n"
    if teacher:
        report += f"RGB: {teacher['eligible_centers']}/{teacher['all_train_centers']} centers; {teacher['eligible_unique_frames']} unique frames. Teacher MAE {teacher['teacher_finger_mae']['mae_deg']} degrees. Construction {teacher['construction_seconds']:.3f}s; peak allocated GPU {teacher['peak_gpu_bytes']/2**20:.2f} MiB.\n\n"
    report += "GPU peaks are torch.cuda.max_memory_allocated after reset per stage/seed, including resident model/parameters and transient tensors; allocator reserved memory, driver allocations, CPU caches and other processes are excluded. Shared extraction includes model construction/read/load; per-method fitting includes optimization or analytic solve, without shared extraction.\n\n"
    report += "Full interval coverage: coverage.json and rgb_interval_coverage.csv. Raw IDs, active flags and vocabulary names: gesture_coverage.csv and rgb_gesture_coverage.csv. Reference ranges/variance: reference_variation.csv. All scores can be recomputed from evaluation_predictions.npz without data/models. Train residuals/teacher quality are diagnostics and did not choose parameters. Head final epoch 50 for each seed; sample seed SD uses ddof=1 and is not a population confidence interval.\n\n"
    report += "Released whole-recording FFT filtering is noncausal; train intervals may occur after evaluated intervals. Checkpoint exposure of this participant remains unresolved. This is an offline within-participant engineering study in one preview recording; it does not establish unseen-user generalization, short-calibration retention, causal deployment, or REACT reproduction.\n\n"
    report += "Blocked conditions: " + json.dumps(metrics["blocked"]) + "\n"
    (run / "report.md").write_text(report)
    if teacher:
        m = np.load(run / "manifest.npz")
        tc = np.load(run / "teacher_cache.npz")
        ix = tc["eligible_indices"]
        rows = []
        for g, ac in sorted(
            set(zip(m["gesture"][ix].tolist(), m["active"][ix].tolist()))
        ):
            select = (m["gesture"][ix] == g) & (m["active"][ix] == ac)
            rows.append(
                dict(
                    gesture_id=g,
                    active=ac,
                    name=str(m["gesture_name"][ix][select][0]),
                    T_rgb=int(select.sum()),
                    unique_frames=len(
                        np.unique(tc["frame"][tc["reason"] == "eligible"][select])
                    ),
                )
            )
        csv_write(run / "rgb_gesture_coverage.csv", rows)
    coverage_lines = [
        "# Interval and gesture coverage",
        "",
        "All positions are local half-open row intervals. T_all and T_rgb are fitting pools; E is fixed evaluation.",
        "",
        "| Interval | Split | Rows [a,b) | Candidate | Eligible | RGB | Exclusions |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    rgb_intervals = {v["interval"]: v for v in teacher["intervals"]} if teacher else {}
    for v in cov["intervals"]:
        rgb_value = rgb_intervals.get(v["interval"], {}).get("T_rgb", "—")
        reason = (
            "insufficient length"
            if v["insufficient_length"]
            else json.dumps(v["exclusions"])
        )
        coverage_lines.append(
            f"| {v['interval']} | {v['split']} | [{v['local_start']},{v['local_end']}) | {v['candidate_windows']} | {v['eligible_windows']} | {rgb_value} | {reason} |"
        )
    with (run / "gesture_coverage.csv").open() as stream:
        gesture_rows = list(csv.DictReader(stream))
    rgb_rows = []
    if teacher:
        with (run / "rgb_gesture_coverage.csv").open() as stream:
            rgb_rows = list(csv.DictReader(stream))
    lookup = {(v["pool"], v["gesture_id"], v["active"]): v for v in gesture_rows}
    rgb_lookup = {(v["gesture_id"], v["active"]): v for v in rgb_rows}
    coverage_lines += [
        "",
        "| Raw ID | Name | Active | T_all | T_rgb | E |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for g, active in sorted(
        {(v["gesture_id"], v["active"]) for v in gesture_rows},
        key=lambda v: (int(v[0]), v[1]),
    ):
        training = lookup.get(("T_all", g, active), {})
        evaluation = lookup.get(("E", g, active), {})
        name = (training or evaluation)["name"]
        coverage_lines.append(
            f"| {g} | {name} | {active} | {training.get('centers', 0)} | {rgb_lookup.get((g, active), {}).get('T_rgb', 0)} | {evaluation.get('centers', 0)} |"
        )
    coverage_text = "\n".join(coverage_lines) + "\n"
    (run / "coverage.md").write_text(coverage_text)
    with (run / "report.md").open("a") as stream:
        stream.write("\n" + coverage_text)
    mark(
        run,
        "report",
        ident,
        [
            "report.md",
            "primary_comparison.md",
            "matched_comparison.md",
            "all_seeds.md",
            "per_joint.png",
            "per_gesture.png",
            "time_traces.png",
            "trace_identifiers.json",
            "coverage.md",
        ]
        + (["rgb_gesture_coverage.csv"] if teacher else []),
    )


def main() -> None:
    invocation_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/d/egoemg/data/dataset_egoemg_preview"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "checkpoints/egoemg_emgformer_middle.ckpt",
    )
    parser.add_argument(
        "--rgb-checkpoint", type=Path, default=REPO / "checkpoints/vision_resnet18.ckpt"
    )
    parser.add_argument("--episode", default="episode_000020")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--phase",
        choices=("all", "manifest", "emg", "teacher", "fit", "evaluate", "report"),
        default="all",
    )
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-rgb-delta-ms", type=float, default=50.0)
    parser.add_argument("--emg-batch-size", type=int, default=2)
    parser.add_argument("--rgb-batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument(
        "--limit-per-pool",
        type=int,
        default=0,
        help="Separate tiny machinery check only; 0 means complete pools",
    )
    args = parser.parse_args()
    if (
        args.episode != "episode_000020"
        or min(
            args.stride,
            args.emg_batch_size,
            args.rgb_batch_size,
            args.feature_batch_size,
        )
        <= 0
        or args.limit_per_pool < 0
    ):
        parser.error(
            "Require episode_000020, positive strides/batches and nonnegative tiny limit"
        )
    if not np.isfinite(args.max_rgb_delta_ms) or args.max_rgb_delta_ms < 0:
        parser.error("Require finite nonnegative RGB delta")
    run = (
        args.run_dir.resolve()
        if args.run_dir
        else create_run_directory(
            OUTPUT_PARENT,
            "wmh",
            "right",
            "suite_tiny" if args.limit_per_pool else "suite_full",
        )
    )
    if (
        not run.is_relative_to(OUTPUT_PARENT.resolve())
        or run == OUTPUT_PARENT.resolve()
    ):
        raise ValueError(f"Run directory must be beneath {OUTPUT_PARENT}")
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run / "run.log"),
        ],
        format="%(asctime)s %(message)s",
        force=True,
    )
    LOG.info("Run directory: %s", run)
    os.environ["EGOEMG_ROOT"] = str(REPO)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCH_HOME"] = str(REPO / "myself/.cache/torch")
    os.environ["MPLCONFIGDIR"] = str(REPO / "myself/.cache/matplotlib")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, base, ep, root, manifest = configure(args, run)
    ident = identity(args, run, cfg, base, root)
    if (run / "settings.json").exists():
        if json.loads((run / "settings.json").read_text())["identity"] != ident:
            raise ValueError("Incompatible run identity: create a new run directory")
    else:
        settings = dict(
            identity=ident,
            run_dir=str(run),
            interpreter=sys.executable,
            python=sys.version,
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, cwd=REPO
            ).strip(),
            git_branch=subprocess.check_output(
                ["git", "branch", "--show-current"], text=True, cwd=REPO
            ).strip(),
            git_status=subprocess.check_output(
                ["git", "status", "--short"], text=True, cwd=REPO
            ),
            packages={
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "numpy",
                    "hydra-core",
                    "pytorch-lightning",
                    "scipy",
                    "lmdb",
                    "Pillow",
                    "torchvision",
                    "matplotlib",
                )
            },
            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            cuda=torch.version.cuda,
            command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            precision="FP32; TF32 disabled",
            normalization_no_estimation=True,
            augmentation="none",
            hydra_output=str(run / "hydra"),
            logger_output=str(run / "logger"),
            execution="Hydra compose only; no trainer/logger instantiated",
        )
        json_write(run / "settings.json", settings)
    with (run / "commands.jsonl").open("a") as f:
        f.write(
            json.dumps([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
            + "\n"
        )
    stages = ["manifest", "emg", "teacher", "fit", "evaluate", "report"]
    functions = {
        "manifest": lambda: manifest_stage(args, run, base, ep, root, manifest, ident),
        "emg": lambda: emg_stage(args, run, cfg, base, ep, ident),
        "teacher": lambda: teacher_stage(args, run, base, root, manifest, ident),
        "fit": lambda: fitting_stage(args, run, ident),
        "evaluate": lambda: evaluation_stage(run, ident),
        "report": lambda: reporting_stage(run, ident),
    }
    for stage in stages if args.phase == "all" else [args.phase]:
        if completed(run, stage, ident):
            LOG.info("Reusing completed %s", stage)
            continue
        needs = {
            "emg": ["manifest"],
            "teacher": ["manifest"],
            "fit": ["emg"],
            "evaluate": ["fit"],
            "report": ["evaluate"],
        }.get(stage, [])
        for need in needs:
            if not completed(run, need, ident):
                raise RuntimeError(f"Run --phase {need} first")
        try:
            functions[stage]()
        except Exception as error:
            blockers = (
                json.loads((run / "blockers.json").read_text())
                if (run / "blockers.json").exists()
                else {}
            )
            blockers[stage] = dict(
                error=f"{type(error).__name__}: {error}",
                remaining_command=[
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-dir",
                    str(run),
                    "--phase",
                    stage,
                ],
            )
            json_write(run / "blockers.json", blockers)
            if stage == "teacher" and args.phase == "all":
                LOG.exception(
                    "Teacher blocked; completing independent reference conditions"
                )
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise
    with (run / "execution_times.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                dict(
                    phase=args.phase,
                    invocation_seconds=time.perf_counter() - invocation_started,
                    command=sys.argv,
                )
            )
            + "\n"
        )
    LOG.info("Completed requested phases: %s", run)


if __name__ == "__main__":
    main()
