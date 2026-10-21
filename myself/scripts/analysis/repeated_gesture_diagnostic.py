#!/usr/bin/env python3
"""First-to-second execution calibration; inputs stay on the designated disk."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from myself.scripts.analysis.preview_calibration_smoke import open_field, runs, sha256
from myself.scripts.analysis.single_subject_calibration_suite import (
    SEEDS,
    angular_mae,
    csv_write,
    emg_stage,
    evaluated_predictions,
    fit_affine,
    fit_offsets,
    json_write,
    npz_write,
    train_head,
)


def repetition_pairs(gesture, active, origin=0):
    """A split-label change never creates a new execution of a gesture."""
    segments = defaultdict(list)
    for a, b, g in runs(np.where(active, gesture, -1)):
        if g >= 0:
            segments[int(g)].append((a + origin, b + origin))
    return {g: spans[:2] for g, spans in segments.items() if len(spans) >= 2}


def select_pairs(gesture, active, valid, reference, emg, window=12000, stride=200):
    pairs = repetition_pairs(gesture, active)
    bad = np.r_[0, np.cumsum(~np.isfinite(emg).all(axis=1), dtype=np.int64)]
    selected, audit = [], []
    for g, spans in sorted(pairs.items()):
        groups = []
        record = dict(gesture=g, first=list(spans[0]), second=list(spans[1]))
        for phase, (a, b) in zip(("fit", "later"), spans):
            candidates = np.arange(a, b - window + 1, stride, dtype=np.int64)
            c = candidates + window // 2
            ok = (
                (bad[candidates + window] == bad[candidates])
                & valid[c]
                & np.isfinite(reference[c]).all(axis=1)
            )
            groups.append(candidates[ok])
            record[phase + "_candidates"] = len(candidates)
            record[phase + "_eligible"] = int(ok.sum())
        record["included"] = all(len(x) for x in groups)
        audit.append(record)
        if record["included"]:
            for phase, starts in zip(("fit", "later"), groups):
                selected.extend((int(s), phase, g) for s in starts)
    selected.sort()
    if not selected:
        raise ValueError("No eligible independent first/second execution pairs")
    starts = np.array([r[0] for r in selected], dtype=np.int64)
    phases = np.array([r[1] for r in selected])
    gestures = np.array([r[2] for r in selected])
    # Stronger than per-gesture separation: all fitting precedes all evaluation.
    if starts[phases == "fit"].max() + window > starts[phases == "later"].min():
        raise ValueError("Pooled protocol requires global chronological separation")
    return starts, phases, gestures, audit


def score_rows(predictions, reference, phases, gestures, scope="pooled"):
    result = []
    for key, pred in predictions.items():
        for phase in ("fit", "later"):
            for g in [None] + sorted(np.unique(gestures).tolist()):
                ix = phases == phase
                if g is not None:
                    ix &= gestures == g
                score = angular_mae(
                    pred[ix, :20], reference[ix, :20], np.isfinite(reference[ix, :20])
                )
                result.append(
                    dict(
                        scope=scope,
                        condition=key,
                        phase=phase,
                        gesture="all" if g is None else int(g),
                        centers=int(ix.sum()),
                        mae_deg=score["mae_deg"],
                    )
                )
    return result


def fit_and_evaluate(run, base, args):
    m, cache = np.load(run / "manifest.npz"), np.load(run / "emg_cache.npz")
    np.testing.assert_array_equal(m["ids"], cache["ids"])
    fit = m["phase"] == "fit"
    reference, baseline = m["reference"], cache["prediction"]
    predictions = {"frozen": baseline}
    params_dir = run / "parameters"
    params_dir.mkdir()
    stats = {}
    offset, _ = fit_offsets(
        baseline[fit, :20], reference[fit, :20], m["mask"][fit, :20], 1.0
    )
    predictions["offset"] = evaluated_predictions(
        baseline, cache["features"], "offset", dict(offsets=offset)
    )
    affine = fit_affine(baseline[fit, :20], reference[fit, :20], m["mask"][fit, :20])
    predictions["affine"] = evaluated_predictions(
        baseline, cache["features"], "affine", affine
    )
    json_write(params_dir / "offset.json", dict(offsets=offset))
    json_write(params_dir / "affine.json", affine)
    for seed in SEEDS:
        params, evidence = train_head(
            cache["features"][fit],
            reference[fit, :20],
            cache["weight"],
            cache["bias"],
            seed,
        )
        key = f"head_seed{seed}"
        predictions[key] = evaluated_predictions(
            baseline, cache["features"], "head", params
        )
        npz_write(params_dir / f"{key}.npz", **params)
        stats[key] = evidence
    rows = score_rows(predictions, reference, m["phase"], m["gesture"])
    csv_write(run / "metrics.csv", rows)
    npz_write(run / "predictions.npz", ids=m["ids"], reference=reference, **predictions)
    json_write(run / "head_training.json", stats)
    summaries = []
    for phase in ("fit", "later"):
        scores = {
            key: [r for r in rows if r["condition"] == key and r["phase"] == phase]
            for key in predictions
        }
        frozen = scores["frozen"]
        frozen_micro = frozen[0]["mae_deg"]
        frozen_gestures = np.array([r["mae_deg"] for r in frozen[1:]])
        for name, keys in [
            ("frozen", ["frozen"]),
            ("offset", ["offset"]),
            ("affine", ["affine"]),
            ("head", [f"head_seed{s}" for s in SEEDS]),
        ]:
            micro = np.array([scores[k][0]["mae_deg"] for k in keys])
            by_gesture = np.array([[r["mae_deg"] for r in scores[k][1:]] for k in keys])
            delta = by_gesture.mean(axis=0) - frozen_gestures
            # Descriptive paired bootstrap over gesture classes, not windows.
            rng = np.random.default_rng(20261014)
            boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(
                axis=1
            )
            summaries.append(
                dict(
                    phase=phase,
                    method=name,
                    micro_deg=float(micro.mean()),
                    micro_seed_sd_deg=(
                        float(micro.std(ddof=1)) if len(keys) > 1 else 0.0
                    ),
                    delta_micro_deg=float(micro.mean() - frozen_micro),
                    macro_deg=float(by_gesture.mean()),
                    delta_macro_deg=float(delta.mean()),
                    improved_gestures=int((delta < 0).sum()),
                    gestures=len(delta),
                    macro_delta_bootstrap_low=float(np.quantile(boot, 0.025)),
                    macro_delta_bootstrap_high=float(np.quantile(boot, 0.975)),
                )
            )
    csv_write(run / "summary.csv", summaries)
    # Independent recomputation with float64 arithmetic and no metrics helper.
    checks = {}
    for key, pred in predictions.items():
        np.testing.assert_array_equal(pred[:, 20:], baseline[:, 20:])
        assert np.isfinite(pred).all()
        for phase in ("fit", "later"):
            ix = m["phase"] == phase
            actual = float(
                np.rad2deg(
                    np.abs(pred[ix, :20].astype(np.float64) - reference[ix, :20]).mean()
                )
            )
            saved = next(
                r["mae_deg"]
                for r in rows
                if r["condition"] == key
                and r["phase"] == phase
                and r["gesture"] == "all"
            )
            assert abs(actual - saved) < 1e-5
            checks[f"{key}_{phase}_mae_deg"] = actual
    json_write(
        run / "verification.json",
        dict(
            recomputed=checks,
            wrist_unchanged=True,
            finite=True,
            global_chronological_support_separation=True,
            actual_last_linear_equivalence=json.loads(
                (run / "emg_evidence.json").read_text()
            )["maximum_absolute_discrepancy"],
        ),
    )
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="episode_000020")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_root = Path("/mnt/d/egoemg/data/dataset_egoemg_preview/data/memmap_data")
    args.checkpoint = REPO / "checkpoints/egoemg_emgformer_middle.ckpt"
    args.emg_batch_size = 2
    args.stride = 200
    run = args.run_dir.resolve()
    if not run.is_relative_to(REPO / "myself/outputs"):
        raise ValueError("Outputs must remain under myself/outputs")
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
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        cfg = compose(
            config_name="base",
            overrides=["experiment=emgformer/egoemg_emgformer_middle"],
        )
    base = EgoEmgMemmapDataset(
        memmap_dir=args.data_root,
        window_length=12000,
        stride=200,
        allowed_episode_ids=[args.episode],
        allowed_splits=["train", "user", "gesture", "both"],
        modalities=["emg", "joint_angles", "labels"],
        target_hand="right",
        emg_field_preference="filtered_paper",
        emg_layout="target_hand",
        channel_interpolate=False,
        norm_mode="per-dataset",
        norm_stats_path=str(
            REPO / "assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
        ),
        center_target_only=True,
    )
    ep = base._episode_id.index(args.episode)
    origin, stop = int(base._episode_start_idx[ep]), int(base._episode_end_idx[ep])
    root = args.data_root
    manifest = json.loads((root / "manifest.json").read_text())
    field = lambda key: open_field(root, manifest, key)
    starts, phases, gestures, audit = select_pairs(
        field("label_gesture_class")[origin:stop],
        field("label_gesture_active")[origin:stop],
        field("generated_label_valid")[origin:stop, 1],
        field("generated_joint_angles_right")[origin:stop],
        field("emg_right_filtered_paper")[origin:stop],
    )
    starts += origin
    centers = starts + 6000
    ref = np.column_stack(
        [
            field("generated_joint_angles_right")[centers],
            np.deg2rad(field("mocap_right_wrist_pitch")[centers]),
            np.deg2rad(field("mocap_right_wrist_yaw")[centers]),
        ]
    ).astype(np.float32)
    mask = np.isfinite(ref)
    mask[:, 20:] &= field("mocap_right_wrist_angles_valid")[centers, None]
    npz_write(
        run / "manifest.npz",
        starts=starts,
        centers=centers,
        phase=phases,
        gesture=gestures,
        reference=ref,
        mask=mask,
        ids=np.array([f"{args.episode}:R:{c}" for c in centers]),
        official_split=field("frame_split_id")[centers],
    )
    json_write(run / "segments.json", audit)
    input_paths = [
        Path(__file__),
        REPO / "myself/scripts/analysis/single_subject_calibration_suite.py",
        REPO / "myself/scripts/analysis/preview_calibration_smoke.py",
        root / "manifest.json",
        root / "metadata.npz",
        root / "gesture_classes.json",
        Path(base.norm_stats_path),
        args.checkpoint,
    ]
    input_paths += sorted((REPO / "egoemg/models").rglob("*.py"))
    input_paths += [REPO / "egoemg/datasets/egoemg_memmap_dataset.py"]
    ident = dict(
        episode=args.episode,
        participant=base._episode_subject[ep],
        input_sha256={str(p): sha256(p) for p in input_paths},
        window=12000,
        stride=200,
        seeds=list(SEEDS),
        head_epochs=50,
        head_lr=0.001,
        precision="FP32; matmul and cuDNN TF32 disabled",
        regularization=0.001,
        feature_batch=256,
        offset_lambda=1.0,
        affine_lambda_a=0.01,
        affine_lambda_b=1.0,
        normalization_mean=base._emg_mean,
        normalization_std=base._emg_std,
        official_split_counts={
            phase: dict(
                Counter(map(int, field("frame_split_id")[centers[phases == phase]]))
            )
            for phase in ("fit", "later")
        },
        protocol="First two maximal active same-class runs; complete supports inside each run; all fitting precedes evaluation; official splits are metadata, NOT this adaptation split; no test selection or early stopping.",
    )
    json_write(run / "settings.json", ident)
    OmegaConf.save(cfg.module, run / "model_config.yaml", resolve=True)
    logging.info("Selected %s gestures: %s", len(set(gestures)), Counter(phases))
    emg_stage(args, run, cfg, base, ep, ident)
    fit_and_evaluate(run, base, args)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
