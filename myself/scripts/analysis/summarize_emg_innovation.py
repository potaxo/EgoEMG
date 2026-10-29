#!/usr/bin/env python3
"""Independent float64 prediction audit, research figures and curated export.

Does not import Torch, models, loaders or the training metric implementation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FAMILIES = [f"R{i}" for i in range(7)]
DIFF = {"R2", "R3", "R4"}
CAPACITY = {"R2": "R1", "R3": "R6", "R4": "R5"}
EPISODES = ["episode_000020", "episode_000008", "episode_000028"]
PEOPLE = dict(zip(EPISODES, ["wmh", "zbk", "wsj"]))
DEG = 180 / np.pi


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def weighted(pred, target, mask):
    e = np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64))
    m = np.asarray(mask, bool)
    if not np.isfinite(e[m]).all() or not m.sum():
        raise ValueError("Nonfinite prediction or empty scoring mask")
    return float(np.sum(e[m], dtype=np.float64) / np.sum(m) * DEG)


def stats(values):
    values = np.asarray(values, np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def parse_key(key):
    match = re.fullmatch(r"(fit|all)_(R[0-6])_(B[0-2])_s(42|43|44)", key)
    if match is None:
        raise ValueError(key)
    phase, family, cond, seed = match.groups()
    return phase, family, cond, int(seed)


def primary(cond):
    return dict(B0="constant", B1="normal", B2="mismatch_101")[cond]


def collect(run, state):
    main, interventions, groups, pairrows, auxiliary = [], [], [], [], []
    arrays = {}
    for phase_folder in sorted((run / "predictions").iterdir()):
        phase = phase_folder.name
        for ep_folder in sorted(phase_folder.iterdir()):
            ep = ep_folder.name
            canonical = dict(np.load(run / ep / "enriched_E.npz"))
            anchor_data = np.load(run / "predictions/fit" / ep / "fit_R2_B0_s42.npz")
            anchor_mask = canonical["mask"][:, :20]
            anchor_error = np.sum(
                np.abs(
                    anchor_data["constant"].astype(np.float64)
                    - canonical["reference"][:, :20]
                )
                * anchor_mask,
                axis=1,
            ) / anchor_mask.sum(1)
            visual_error_bins = np.searchsorted(
                np.quantile(anchor_error, [1 / 3, 2 / 3]), anchor_error
            )
            for path in sorted(ep_folder.glob("*.npz")):
                phase, fam, cond, seed = parse_key(path.stem)
                with np.load(path) as data:
                    for k in (
                        "centers",
                        "reference",
                        "mask",
                        "frame",
                        "ids",
                        "execution",
                        "split",
                    ):
                        np.testing.assert_array_equal(data[k], canonical[k])
                    y, mask = data["reference"][:, :20], data["mask"][:, :20]
                    pred = data[primary(cond)]
                    error = weighted(pred, y, mask)
                    cell = state["cells"][path.stem]
                    row = dict(
                        phase=phase,
                        episode=ep,
                        participant=PEOPLE[ep],
                        family=fam,
                        training_condition=cond,
                        inference_condition=primary(cond),
                        seed=seed,
                        centers=len(y),
                        valid_angles=int(mask.sum()),
                        mae_deg=error,
                        analytical_anchor=cond == "B0" and fam in DIFF,
                    )
                    main.append(row)
                    arrays[(phase, ep, fam, cond, seed)] = pred.copy()
                    recorded = state["formal_evaluations"][str(path.relative_to(run))]
                    assert abs(error - recorded["mae_deg"]) < 1e-12
                    if cond in ("B1", "B2"):
                        normal = data["normal"]
                        for k in data.files:
                            if k not in (
                                "normal",
                                "constant",
                                "reverse",
                                "mismatch_101",
                                "mismatch_102",
                                "mismatch_103",
                            ):
                                continue
                            p = data[k]
                            em = weighted(p, y, mask)
                            interventions.append(
                                dict(
                                    phase=phase,
                                    episode=ep,
                                    participant=PEOPLE[ep],
                                    family=fam,
                                    training_condition=cond,
                                    seed=seed,
                                    inference_condition=k,
                                    mae_deg=em,
                                    delta_mae_vs_correct_deg=em
                                    - weighted(normal, y, mask),
                                    prediction_abs_change_deg=weighted(p, normal, mask),
                                    valid_angles=int(mask.sum()),
                                )
                            )
                    if "auxiliary_normal" in data:
                        aux = data["auxiliary_normal"]
                        for name, t, m, pp in (
                            ("pose", y, mask, aux[:, 0]),
                            (
                                "minus",
                                data["delta"][:, 0],
                                data["delta_mask"][:, 0],
                                aux[:, 1],
                            ),
                            (
                                "plus",
                                data["delta"][:, 1],
                                data["delta_mask"][:, 1],
                                aux[:, 2],
                            ),
                        ):
                            if name != "pose" and fam not in ("R4", "R5"):
                                continue
                            if np.any(m):
                                auxiliary.append(
                                    dict(
                                        phase=phase,
                                        episode=ep,
                                        family=fam,
                                        condition=cond,
                                        seed=seed,
                                        target=name,
                                        mae_deg=weighted(pp, t, m),
                                        valid_angles=int(m.sum()),
                                    )
                                )
                    # Main-condition groups retain valid-angle weighting throughout.
                    err = np.abs(pred.astype(np.float64) - y.astype(np.float64)) * DEG
                    for grouping in (
                        "split",
                        "execution",
                        "block",
                        "dynamic_bin",
                        "transition",
                        "visual_error_bin",
                    ):
                        labels = (
                            visual_error_bins
                            if grouping == "visual_error_bin"
                            else data[grouping]
                        )
                        reconciled_sum, reconciled_n = 0.0, 0
                        for value in np.unique(labels):
                            ix = labels == value
                            n = int(mask[ix].sum())
                            if not n:
                                continue
                            numerator = float(err[ix][mask[ix]].sum())
                            reconciled_sum += numerator
                            reconciled_n += n
                            groups.append(
                                dict(
                                    phase=phase,
                                    episode=ep,
                                    family=fam,
                                    condition=cond,
                                    seed=seed,
                                    grouping=grouping,
                                    group=str(value),
                                    centers=int(ix.sum()),
                                    valid_angles=n,
                                    mae_deg=numerator / n,
                                )
                            )
                        assert reconciled_n == mask.sum()
                        assert abs(reconciled_sum / reconciled_n - error) < 1e-10
                    for grouping, width in (("joint", 1), ("finger", 4)):
                        for a in range(0, 20, width):
                            m = mask[:, a : a + width]
                            n = int(m.sum())
                            groups.append(
                                dict(
                                    phase=phase,
                                    episode=ep,
                                    family=fam,
                                    condition=cond,
                                    seed=seed,
                                    grouping=grouping,
                                    group=str(a // width),
                                    centers=len(y),
                                    valid_angles=n,
                                    mae_deg=float(err[:, a : a + width][m].sum() / n),
                                )
                            )
    return main, interventions, groups, auxiliary, arrays


def effects(main):
    index = {
        (r["phase"], r["episode"], r["family"], r["training_condition"], r["seed"]): r
        for r in main
    }
    out = []
    for r in main:
        if r["training_condition"] != "B1":
            continue
        phase, ep, f, s = r["phase"], r["episode"], r["family"], r["seed"]
        lookup = lambda fam, c: index[(phase, ep, fam, c, s)]["mae_deg"]
        cap = CAPACITY.get(f, f)
        # T_all may contain the chosen differential family but not its ordinary
        # paired capacity family. Leave this comparison missing, never substitute.
        capval = lookup(cap, "B0") if (phase, ep, cap, "B0", s) in index else None
        out.append(
            dict(
                phase=phase,
                episode=ep,
                participant=PEOPLE[ep],
                family=f,
                seed=s,
                G_method_deg=lookup("R0", "B1") - r["mae_deg"],
                G_EMG_deg=lookup(f, "B0") - r["mae_deg"],
                B2_minus_B1_deg=lookup(f, "B2") - r["mae_deg"],
                capacity_family=cap,
                ordinary_capacity_B0_mae_deg=capval,
                gain_vs_ordinary_capacity_B0_deg=(
                    capval - r["mae_deg"] if capval is not None else None
                ),
                B0_meaning=(
                    "analytical fixed independent vision anchor"
                    if f in DIFF
                    else "trained constant-input visual capacity control"
                ),
            )
        )
    return out


def summaries(main, effect):
    rows = []
    for phase, ep, f, c in sorted(
        set(
            (r["phase"], r["episode"], r["family"], r["training_condition"])
            for r in main
        )
    ):
        selected = [
            r
            for r in main
            if (r["phase"], r["episode"], r["family"], r["training_condition"])
            == (phase, ep, f, c)
        ]
        assert sorted(r["seed"] for r in selected) == [42, 43, 44]
        mean, sd = stats([r["mae_deg"] for r in selected])
        rows.append(
            dict(
                phase=phase,
                episode=ep,
                participant=PEOPLE[ep],
                family=f,
                condition=c,
                centers=selected[0]["centers"],
                valid_angles_per_seed=selected[0]["valid_angles"],
                mean_mae_deg=mean,
                sample_seed_sd_deg=sd,
                n_seeds=3,
                analytical_anchor=selected[0]["analytical_anchor"],
            )
        )
    erows = []
    for phase, ep, f in sorted(
        set((r["phase"], r["episode"], r["family"]) for r in effect)
    ):
        chosen = [
            r
            for r in effect
            if (r["phase"], r["episode"], r["family"]) == (phase, ep, f)
        ]
        row = dict(phase=phase, episode=ep, participant=PEOPLE[ep], family=f)
        for field in (
            "G_method_deg",
            "G_EMG_deg",
            "B2_minus_B1_deg",
            "gain_vs_ordinary_capacity_B0_deg",
        ):
            values = [r[field] for r in chosen if r[field] is not None]
            mean, sd = stats(values) if values else (None, None)
            row[field + "_mean"] = mean
            row[field + "_sample_seed_sd"] = sd
        erows.append(row)
    return rows, erows


def probe_tables(run):
    rows = []
    for path in sorted((run / "probes").glob("*.npz")):
        target, condition = path.stem.rsplit("_", 1)
        d = np.load(path)
        p, y, m = d["prediction"], d["target"], d["target_mask"]
        for grouping in ("overall", "transition", "dynamic_bin"):
            labels = np.zeros(len(p), int) if grouping == "overall" else d[grouping]
            for value in np.unique(labels):
                ix = labels == value
                mask = m[ix]
                if not mask.any():
                    continue
                pm, med = weighted(p[ix], y[ix], mask), weighted(
                    np.broadcast_to(d["median"], p[ix].shape), y[ix], mask
                )
                zero = weighted(np.zeros_like(p[ix]), y[ix], mask)
                rows.append(
                    dict(
                        target=target,
                        condition=condition,
                        seed=42,
                        grouping=grouping,
                        group=str(value),
                        centers=int(ix.sum()),
                        valid_angles=int(mask.sum()),
                        mae_deg=pm,
                        fit_median_baseline_mae_deg=med,
                        gain_vs_fit_median_deg=med - pm,
                        zero_baseline_mae_deg=zero,
                        gain_vs_zero_deg=zero - pm,
                        exploratory=True,
                    )
                )
    return rows


def paired_groups(run, arrays):
    rows = []
    interval = []
    for phase, ep, f, c, s in arrays:
        if c != "B1":
            continue
        d = dict(np.load(run / ep / "enriched_E.npz"))
        d["dynamic_bin"] = np.searchsorted(
            json.loads((run / "dynamic_rule.json").read_text())[
                "amplitude_tertiles_rad"
            ],
            d["amplitude"],
        )
        y = d["reference"][:, :20]
        m = d["mask"][:, :20]
        p = arrays[(phase, ep, f, "B1", s)]
        anchor = arrays[("fit", ep, "R2", "B0", 42)]
        anchor_error = np.sum(
            np.abs(anchor.astype(np.float64) - y) * m, axis=1
        ) / m.sum(1)
        d["visual_error_bin"] = np.searchsorted(
            np.quantile(anchor_error, [1 / 3, 2 / 3]), anchor_error
        )
        refs = dict(
            G_method=arrays[(phase, ep, "R0", "B1", s)],
            G_EMG=arrays[(phase, ep, f, "B0", s)],
        )
        for kind, ref in refs.items():
            diff = (
                np.abs(ref.astype(np.float64) - y) - np.abs(p.astype(np.float64) - y)
            ) * DEG
            for grouping in (
                "execution",
                "dynamic_bin",
                "transition",
                "split",
                "visual_error_bin",
            ):
                for val in np.unique(d[grouping]):
                    ix = d[grouping] == val
                    mask = m[ix]
                    n = int(mask.sum())
                    if n:
                        rows.append(
                            dict(
                                phase=phase,
                                episode=ep,
                                family=f,
                                seed=s,
                                metric=kind,
                                grouping=grouping,
                                group=str(val),
                                centers=int(ix.sum()),
                                valid_angles=n,
                                gain_deg=float(diff[ix][mask].sum() / n),
                            )
                        )
    rng = np.random.default_rng(20261029)
    for phase, ep, f, metric in sorted(
        set((r["phase"], r["episode"], r["family"], r["metric"]) for r in rows)
    ):
        rr = [
            r
            for r in rows
            if (r["phase"], r["episode"], r["family"], r["metric"], r["grouping"])
            == (phase, ep, f, metric, "execution")
        ]
        ids = sorted(set(r["group"] for r in rr))
        values = np.array(
            [np.mean([r["gain_deg"] for r in rr if r["group"] == i]) for i in ids]
        )
        boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
        interval.append(
            dict(
                phase=phase,
                episode=ep,
                family=f,
                metric=metric,
                execution_groups=len(values),
                execution_equal_weight_mean_gain_deg=float(values.mean()),
                positive_execution_groups=int((values > 0).sum()),
                descriptive_bootstrap_p025_deg=float(np.quantile(boot, 0.025)),
                descriptive_bootstrap_p975_deg=float(np.quantile(boot, 0.975)),
                interpretation="descriptive execution resampling; overlapping supports can couple adjacent executions; not participant uncertainty",
            )
        )
    return rows, interval


def verify_structure(run, state, arrays):
    identity = json.loads((run / "identity.json").read_text())
    checked = []
    for name, h in identity["files_sha256"].items():
        assert sha(Path(name)) == h, name
        source = (
            run / "executed_sources" / Path(name).relative_to(ROOT)
            if Path(name).is_relative_to(ROOT)
            else None
        )
        if source is not None and source.exists():
            assert sha(source) == h
        checked.append(name)
    lock = json.loads((run / "selection_lock.json").read_text())
    saved_hash = lock.pop("content_sha256")
    assert (
        hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
        == saved_hash
    )
    assert lock["identity"] == identity["identity"] == state["identity"]
    assert set(lock["final_retraining"]["families"]) == {"R0", lock["selected_family"]}
    cells = state["cells"]
    fit = [r for r in cells.values() if r["phase"] == "fit"]
    assert len(fit) == 63 and all(r["status"] == "complete" for r in cells.values())
    analytical = [r for r in cells.values() if r["planned_kind"] == "analytical_anchor"]
    assert sum(r["phase"] == "fit" for r in analytical) == 9
    for row in analytical:
        assert row["actual_optimization_epochs"] == 0 and not row["auxiliary_trained"]
    for key, c in cells.items():
        if c["planned_kind"] == "optimization":
            assert (
                c["actual_optimization_epochs"] == 30
                and c["epoch"] == 30
                and c["reload_exact"]
            )
            assert sha(run / "cells" / key / "last.pt") == c["checkpoint_sha256"]
        if c["phase"] == "all":
            peer = cells[key.replace("all_", "fit_", 1)]
            assert c["initialization_sha256"] == peer["initialization_sha256"]
    # Independently recompute all validation selection values from predictions.
    verrors = {}
    for fam in FAMILIES:
        scores = []
        for seed in (42, 43, 44):
            d = np.load(run / "cells" / f"fit_{fam}_B1_s{seed}" / "validation.npz")
            value = weighted(d["prediction"], d["reference"][:, :20], d["mask"][:, :20])
            assert (
                abs(value - cells[f"fit_{fam}_B1_s{seed}"]["validation_mae_deg"])
                < 1e-12
            )
            scores.append(value)
        verrors[fam] = float(np.mean(scores))
    minimum = min(verrors.values())
    candidates = [r for r in lock["scores"] if verrors[r["family"]] <= minimum + 0.001]
    chosen = min(
        candidates,
        key=lambda r: (
            r["measured_batch_seconds"],
            r["trainable_parameters"],
            r["family"],
        ),
    )["family"]
    assert chosen == lock["selected_family"]
    for rel, receipt in state["formal_evaluations"].items():
        assert receipt["scored_at"] > lock["generated_at"]
        assert receipt["selection_lock_sha256"] == sha(run / "selection_lock.json")
        assert receipt["prediction_sha256"] == sha(run / rel)
    assert state["frozen_before"] == state["frozen_after"]
    # Exact reference anchor is identical in every analytical cell and seed.
    for phase, ep, f, c, seed in arrays:
        if c == "B0" and f in DIFF:
            baseline = arrays[(phase, ep, f, "B0", 42)]
            np.testing.assert_array_equal(arrays[(phase, ep, f, c, seed)], baseline)
    fit = np.load(run / "T_fit_selection.npz")
    val = np.load(run / "T_val_selection.npz")
    assert not (np.abs(fit["centers"][:, None] - val["centers"][None]) < 7790).any()
    assert not set(fit["group_id"]) & set(val["group_id"])
    donors_checked = 0
    for phase in ("fit", "all"):
        s = fit if phase == "fit" else np.load(run / EPISODES[0] / "enriched_T_all.npz")
        for path in (run / "orders").glob(f"{phase}_donors_*.npz"):
            mp = np.load(path)["donor"]
            assert np.array_equal(np.sort(mp), np.arange(len(s["centers"])))
            assert np.all(s["split"] == s["split"][mp])
            assert np.all(np.abs(s["centers"] - s["centers"][mp]) >= 7790)
            donors_checked += 1
        for path in (run / "orders").glob(f"{phase}_s*.npz"):
            order = np.load(path)["order"]
            assert np.array_equal(np.sort(order), np.arange(len(s["centers"])))
    for ep in EPISODES:
        s = np.load(run / ep / "enriched_E.npz")
        if len(s["centers"]) < 32:
            continue
        mpfile = np.load(run / ep / "fixed_E_donors.npz")
        for ms in mpfile.files:
            mp = mpfile[ms]
            assert np.array_equal(np.sort(mp), np.arange(len(mp)))
            assert np.all(s["split"] == s["split"][mp])
            assert np.all(np.abs(s["centers"] - s["centers"][mp]) >= 7790)
            donors_checked += 1
    return dict(
        status="verified",
        independent_numpy_float64=True,
        training_helpers_imported=False,
        source_and_asset_files_checked=len(checked),
        checkpoint_hashes_verified=True,
        selection_rule_recomputed=True,
        selection_precedes_all_formal_E_scores=True,
        T_all_original_initialization=True,
        frozen_state_unchanged=True,
        donor_mappings_verified=donors_checked,
        primary_fit_planned=63,
        primary_fit_optimized=sum(
            r["planned_kind"] == "optimization" for r in fit_cells(state)
        ),
        primary_fit_analytical=9,
        total_cells=len(cells),
        total_optimized=sum(
            r["planned_kind"] == "optimization" for r in cells.values()
        ),
        total_analytical=len(analytical),
        failed_cells=sum(r["status"] == "failed" for r in cells.values()),
        missing_cells=sum(r["status"] != "complete" for r in cells.values()),
        weighted_group_reconciliation=True,
        execution_group_intervals="descriptive only",
    )


def fit_cells(state):
    return [r for r in state["cells"].values() if r["phase"] == "fit"]


def figures(
    out,
    main,
    summary,
    effect_summary,
    interventions,
    probes,
    pairs,
    arrays,
    chosen,
    run,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    colors = {"B0": "#7a8797", "B1": "#117a75", "B2": "#d18434"}
    # 1. Current-round primary wmh matrix: points show seeds, bars show mean.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, c in enumerate(("B0", "B1", "B2")):
        rows = [
            next(
                r
                for r in summary
                if (r["phase"], r["episode"], r["family"], r["condition"])
                == ("fit", EPISODES[0], f, c)
            )
            for f in FAMILIES
        ]
        x = np.arange(7) + (j - 1) * 0.24
        ax.errorbar(
            x,
            [r["mean_mae_deg"] for r in rows],
            yerr=[r["sample_seed_sd_deg"] for r in rows],
            fmt="o",
            capsize=3,
            color=colors[c],
            label=c,
        )
        for i, f in enumerate(FAMILIES):
            values = [
                r["mae_deg"]
                for r in main
                if (r["phase"], r["episode"], r["family"], r["training_condition"])
                == ("fit", EPISODES[0], f, c)
            ]
            ax.scatter(
                x[i] + np.linspace(-0.025, 0.025, 3),
                values,
                s=14,
                facecolors="none",
                edgecolors=colors[c],
                alpha=0.8,
            )
    ax.set(
        xticks=np.arange(7),
        xticklabels=FAMILIES,
        ylabel="MAE (degrees; lower is better)",
        title="wmh development pool: mean ± sample seed SD, with all seed points",
    )
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "R2/R3/R4 B0 are analytical vision anchors; other B0 cells are trained constant-input controls.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(figdir / "01_main_matrix.png")
    plt.close(fig)
    # 2. Three separate claims, preserving their signs and capacity limitations.
    fields = [
        ("G_method_deg_mean", "Method gain vs R0 B1"),
        ("G_EMG_deg_mean", "Within-family B0 − B1"),
        ("B2_minus_B1_deg_mean", "B2 − B1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    for ax, (field, title) in zip(axes, fields):
        rr = [
            next(
                r
                for r in effect_summary
                if (r["phase"], r["episode"], r["family"]) == ("fit", EPISODES[0], f)
            )
            for f in FAMILIES
        ]
        ax.axvline(0, color="#777", lw=0.8)
        ax.errorbar(
            [r[field] for r in rr],
            np.arange(7),
            xerr=[r[field.replace("_mean", "_sample_seed_sd")] for r in rr],
            fmt="o",
            capsize=3,
            color="#117a75",
        )
        ax.set(
            yticks=np.arange(7),
            yticklabels=FAMILIES,
            xlabel="Gain (degrees)",
            title=title,
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.text(
        0.5,
        0.01,
        "Difference B0 has less trainable visual capacity. Paired ordinary B0 controls: R2→R1, R3→R6, R4→R5 (see tables).",
        ha="center",
        fontsize=9,
    )
    fig.suptitle("wmh development pool: paired gains, mean ± sample seed SD")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(figdir / "02_method_and_emg_gains.png")
    plt.close(fig)
    # 3. Each probe is compared with its OWN fit-only baseline.
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, t, title in zip(
        axes,
        ("pose", "minus", "plus", "visual_residual"),
        (
            "Absolute pose",
            "Past local change",
            "Subsequent local change\n(offline decoding)",
            "Visual residual",
        ),
    ):
        rr = [r for r in probes if r["target"] == t and r["grouping"] == "overall"]
        rr.sort(key=lambda r: r["condition"])
        x = np.arange(len(rr))
        ax.bar(
            x,
            [r["gain_vs_fit_median_deg"] for r in rr],
            color=[colors[r["condition"]] for r in rr],
            width=0.55,
            label="vs fit median",
        )
        if t in ("minus", "plus"):
            ax.scatter(
                x,
                [r["gain_vs_zero_deg"] for r in rr],
                marker="x",
                s=45,
                color="#222",
                label="vs zero change",
            )
        ax.axhline(0, color="#777", lw=0.7)
        ax.set(
            xticks=x,
            xticklabels=[r["condition"] for r in rr],
            title=title,
            ylabel="Baseline MAE − probe MAE (degrees)",
        )
        ax.grid(axis="y", alpha=0.2)
        if t == "minus":
            ax.legend(fontsize=8)
    fig.suptitle(
        "wmh, exploratory seed 42; T_fit fitting and T_val diagnosis; target-specific baselines"
    )
    fig.tight_layout()
    fig.savefig(figdir / "03_probes.png")
    plt.close(fig)
    # 4. Paired group gains plus exact sample coverage.
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    for col, grouping, labels in (
        (0, "dynamic_bin", ["Low", "Middle", "High"]),
        (1, "transition", ["Steady", "Transition"]),
    ):
        upper, lower = axes[0, col], axes[1, col]
        values = ["0", "1", "2"] if col == 0 else ["False", "True"]
        for f in FAMILIES:
            rr = [
                r
                for r in pairs
                if (r["phase"], r["episode"], r["family"], r["metric"], r["grouping"])
                == ("fit", EPISODES[0], f, "G_method", grouping)
            ]
            means = [
                (
                    np.mean([r["gain_deg"] for r in rr if r["group"] == v])
                    if any(r["group"] == v for r in rr)
                    else np.nan
                )
                for v in values
            ]
            upper.plot(np.arange(len(values)), means, marker="o", label=f, lw=1.2)
        coverage = [
            next((r["centers"] for r in rr if r["group"] == v), 0) for v in values
        ]
        lower.bar(np.arange(len(values)), coverage, color="#b7c8c4")
        lower.set(xticks=np.arange(len(values)), xticklabels=labels, ylabel="Centers")
        upper.axhline(0, color="#777", lw=0.8)
        upper.set(
            xticks=np.arange(len(values)),
            xticklabels=labels,
            ylabel="R0 B1 MAE − model B1 MAE (degrees)",
            title=(
                "T_fit-defined amplitude tertiles"
                if col == 0
                else "Metadata-defined transition / steady"
            ),
        )
        upper.grid(axis="y", alpha=0.2)
        for i, n in enumerate(coverage):
            lower.text(i, n, str(n), ha="center", va="bottom", fontsize=9)
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.suptitle("wmh development pool: group gains and center coverage")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "04_dynamic_groups.png")
    plt.close(fig)
    # 5. Performance and sensitivity are distinct panels for every B1 family.
    conditions = ["constant", "mismatch_101", "mismatch_102", "mismatch_103", "reverse"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for col, ep in enumerate(EPISODES):
        for row, field, title in (
            (0, "delta_mae_vs_correct_deg", "MAE change"),
            (1, "prediction_abs_change_deg", "Prediction change"),
        ):
            vals = np.array(
                [
                    [
                        np.mean(
                            [
                                r[field]
                                for r in interventions
                                if (
                                    r["phase"],
                                    r["episode"],
                                    r["family"],
                                    r["training_condition"],
                                    r["inference_condition"],
                                )
                                == ("fit", ep, f, "B1", c)
                            ]
                        )
                        for c in conditions
                    ]
                    for f in FAMILIES
                ]
            )
            ax = axes[row, col]
            limit = max(float(np.abs(vals).max()), 0.001)
            im = ax.imshow(
                vals,
                aspect="auto",
                cmap="RdBu_r" if row == 0 else "viridis",
                vmin=-limit if row == 0 else 0,
                vmax=limit,
            )
            ax.set(
                yticks=np.arange(7),
                yticklabels=FAMILIES,
                xticks=np.arange(5),
                xticklabels=["Const", "M101", "M102", "M103", "Reverse"],
                title=f"{PEOPLE[ep]}: {title} (degrees)",
            )
            for i in range(7):
                for j in range(5):
                    ax.text(
                        j,
                        i,
                        f"{vals[i,j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=(
                            "white"
                            if np.dot(
                                np.asarray(im.cmap(im.norm(vals[i, j])))[:3],
                                [0.2126, 0.7152, 0.0722],
                            )
                            < 0.48
                            else "#111"
                        ),
                    )
            fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle(
        "Same B1 checkpoint and recipients; mismatch whole windows; reverse before frozen encoder"
    )
    fig.text(
        0.5,
        0.01,
        "Per-recording color scales; numerical values are in degrees.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(figdir / "05_interventions.png")
    plt.close(fig)
    # 6. Frozen review and T_all are displayed separately.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, ep in enumerate(EPISODES):
        ax = axes[0, col]
        rr = [
            next(
                r
                for r in effect_summary
                if (r["phase"], r["episode"], r["family"]) == ("fit", ep, f)
            )
            for f in FAMILIES
        ]
        ax.errorbar(
            np.arange(7),
            [r["G_method_deg_mean"] for r in rr],
            yerr=[r["G_method_deg_sample_seed_sd"] for r in rr],
            fmt="o",
            capsize=2,
            color="#117a75",
        )
        ax.axhline(0, color="#777", lw=0.8)
        ax.set(
            xticks=np.arange(7),
            xticklabels=FAMILIES,
            ylabel="Method gain vs R0 B1 (degrees)",
            title=f"{PEOPLE[ep]}: locked T_fit models",
        )
        ax = axes[1, col]
        families = sorted({"R0", chosen})
        for j, c in enumerate(("B0", "B1", "B2")):
            rr = [
                next(
                    r
                    for r in summary
                    if (r["phase"], r["episode"], r["family"], r["condition"])
                    == ("all", ep, f, c)
                )
                for f in families
            ]
            ax.errorbar(
                np.arange(len(families)) + (j - 1) * 0.16,
                [r["mean_mae_deg"] for r in rr],
                yerr=[r["sample_seed_sd_deg"] for r in rr],
                fmt="o",
                capsize=3,
                color=colors[c],
                label=c,
            )
        ax.set(
            xticks=np.arange(len(families)),
            xticklabels=families,
            ylabel="MAE (degrees)",
            title=f"{PEOPLE[ep]}: T_all retraining",
        )
        if col == 0:
            ax.legend(ncol=3, fontsize=8)
    fig.text(
        0.5,
        0.01,
        "wmh development and zbk/wsj fixed review recordings; checkpoint participant exposure unresolved. T_all is not independent validation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(figdir / "06_recording_review_and_retrain.png")
    plt.close(fig)
    # Fixed post hoc selection: largest/smallest execution-level method gain,
    # predetermined joint 5. Both outcomes are shown, regardless of sign.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    selection = []
    for col, ep in enumerate(EPISODES):
        rr = [
            r
            for r in pairs
            if (r["phase"], r["episode"], r["family"], r["metric"], r["grouping"])
            == ("fit", ep, chosen, "G_method", "execution")
        ]
        scores = {
            g: np.mean([r["gain_deg"] for r in rr if r["group"] == g])
            for g in sorted(set(r["group"] for r in rr))
        }
        ordered = sorted(scores, key=lambda g: (scores[g], int(g)))
        d = np.load(run / ep / "enriched_E.npz")
        pred = np.mean([arrays[("fit", ep, chosen, "B1", s)] for s in (42, 43, 44)], 0)
        base = np.mean([arrays[("fit", ep, "R0", "B1", s)] for s in (42, 43, 44)], 0)
        for row, g in enumerate((ordered[-1], ordered[0])):
            ix = d["execution"] == int(g)
            t = (d["centers"][ix] - d["centers"][ix][0]) / 2000
            ax = axes[row, col]
            ax.plot(t, d["reference"][ix, 5] * DEG, color="#222", label="Reference")
            ax.plot(t, base[ix, 5] * DEG, color="#d18434", label="R0")
            ax.plot(t, pred[ix, 5] * DEG, color="#117a75", label=chosen)
            ax.set(
                title=f'{PEOPLE[ep]}: {"largest" if row==0 else "smallest"} gain {scores[g]:+.3f}°',
                xlabel="Nominal time within execution (s)",
                ylabel="Index MCP flexion (degrees)",
            )
            selection.append(
                dict(
                    episode=ep,
                    execution=int(g),
                    selection="largest" if row == 0 else "smallest",
                    method_gain_deg=float(scores[g]),
                    joint=5,
                    plot_prediction="seed mean for illustration only; primary MAE is mean of per-seed MAEs",
                )
            )
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "07_fixed_rule_trajectories.png")
    plt.close(fig)
    write_json(
        out / "trajectory_selection.json",
        dict(
            rule="largest and smallest execution-level mean-seed G_method; tie by execution ID; joint 5 fixed; no trajectory smoothing",
            selected=selection,
        ),
    )


def report(out, run, state, summary, effect_summary, probes, verification, intervals):
    lock = json.loads((run / "selection_lock.json").read_text())
    chosen = lock["selected_family"]
    protocol = state["protocol"]
    identity = json.loads((run / "identity.json").read_text())
    lines = [
        "# EMG information, finger queries and dynamic supervision",
        "",
        "Offline Preview right-hand study; 20 finger angles. All MAEs are in degrees (lower is better). Primary results use the final epoch and the mean of three separately scored seeds; error bars are sample seed SD.",
        "",
    ]
    for name, caption in [
        ("01_main_matrix", "Current-round controlled matrix on wmh development data."),
        (
            "02_method_and_emg_gains",
            "Method improvement, within-family EMG gain and wrong-window training are separate comparisons.",
        ),
        (
            "03_probes",
            "Exploratory feature decodability relative to each target’s own baseline.",
        ),
        ("04_dynamic_groups", "Dynamic groups and their exact center coverage."),
        (
            "05_interventions",
            "Performance changes and prediction sensitivity under the same B1 checkpoint.",
        ),
        (
            "06_recording_review_and_retrain",
            "Locked review recordings and T_all retraining, shown separately.",
        ),
        (
            "07_fixed_rule_trajectories",
            "Largest and smallest execution-level method gains across all 20 angles; fixed joint 5 is shown. Seed-mean trajectories are illustrative, not the primary scoring estimator.",
        ),
    ]:
        lines += [f"![{caption}](figures/{name}.png)", caption, ""]
        if name == "01_main_matrix":

            def get_mean(ep, family, condition="B1", phase="fit"):
                return next(
                    r["mean_mae_deg"]
                    for r in summary
                    if (r["phase"], r["episode"], r["family"], r["condition"])
                    == (phase, ep, family, condition)
                )

            anchor_validation = np.load(run / "cells/fit_R2_B0_s42/validation.npz")
            anchor_validation_mae = weighted(
                anchor_validation["prediction"],
                anchor_validation["reference"][:, :20],
                anchor_validation["mask"][:, :20],
            )
            val_mean = next(
                r["mean_validation_mae_deg"]
                for r in lock["scores"]
                if r["family"] == chosen
            )
            ints = list(csv.DictReader((out / "interventions.csv").open()))
            wrong_r6 = np.mean(
                [
                    float(r["delta_mae_vs_correct_deg"])
                    for r in ints
                    if r["phase"] == "fit"
                    and r["episode"] == EPISODES[0]
                    and r["family"] == "R6"
                    and r["training_condition"] == "B1"
                    and r["inference_condition"].startswith("mismatch_")
                ]
            )
            wsj_r3_mismatch = np.mean(
                [
                    float(r["delta_mae_vs_correct_deg"])
                    for r in ints
                    if r["phase"] == "fit"
                    and r["episode"] == EPISODES[2]
                    and r["family"] == "R3"
                    and r["training_condition"] == "B1"
                    and r["inference_condition"].startswith("mismatch_")
                ]
            )
            lines += [
                "## Findings",
                "",
                f"**Selection is not evidence of improvement.** The rule selected {chosen} at T_val {val_mean:.6f}°, while the fixed visual anchor scored {anchor_validation_mae:.6f}°. The selected family was retained for all later comparisons regardless of its E performance.",
                "",
                f"On wmh, R6 B1 scored {get_mean(EPISODES[0],'R6'):.6f}° versus R0 B1 {get_mean(EPISODES[0],'R0'):.6f}° (method gain {get_mean(EPISODES[0],'R0')-get_mean(EPISODES[0],'R6'):+.6f}°). Its same-checkpoint mismatch intervention changed MAE by {wrong_r6:+.6f}° on average, so this small training-group improvement is not supported as a correct-EMG-specific benefit.",
                "",
                "The exploratory correct-EMG probes did not beat their own fit-median baselines on T_val; change probes also failed to beat zero change. This is a negative result for these frozen features, readers and budget, not a claim that EMG lacks task information. Sensitivity, decodability and useful conditional increment remain distinct.",
                "",
                f"T_all did not rescue the selected R2 on wmh: R0 B1={get_mean(EPISODES[0],'R0',phase='all'):.6f}°, R2 B1={get_mean(EPISODES[0],'R2',phase='all'):.6f}°. R2 retained lower MAE than R0 on the two review recordings, but its same-family anchor and mismatch controls are needed before attributing that difference to correct EMG.",
                "",
                f"**A limited positive conditional result appears on wsj.** R3 improves the vision anchor from {get_mean(EPISODES[2],'R3','B0'):.6f}° to {get_mean(EPISODES[2],'R3'):.6f}°. Mismatching the same R3 checkpoint worsens MAE by {wsj_r3_mismatch:+.6f}° on average, but most of the anchor-relative gain survives mismatch. This pool contains only 216 centers from three executions, all steady under the predefined rule. It supports further replication of pose-supervised reference differences, not a broad transfer or transition claim.",
                "",
            ]
    lines += [
        "## Main evidence",
        "",
        f"T_val selected **{chosen}** before every formal E error score. The complete selection rule and all seed scores are in [selection_lock.json](selection_lock.json). The E_dev pool is wmh; zbk and wsj are fixed review recordings excluded from fitting and method selection. These are not an unseen-participant benchmark.",
        "",
    ]
    lookup = {
        (r["phase"], r["episode"], r["family"], r["condition"]): r for r in summary
    }
    previous_phase = None
    for e in sorted(
        effect_summary, key=lambda r: (r["phase"] != "fit", r["episode"], r["family"])
    ):
        if e["phase"] != previous_phase:
            lines += [
                "",
                (
                    "### Controlled T_fit models"
                    if e["phase"] == "fit"
                    else "### Fresh T_all retraining"
                ),
                "",
                "| Phase | Recording | Family | B0 mean ± SD | B1 mean ± SD | B2 mean ± SD | G_method | G_EMG | B2−B1 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            previous_phase = e["phase"]
        vals = [
            lookup[(e["phase"], e["episode"], e["family"], c)]
            for c in ("B0", "B1", "B2")
        ]
        fmt = lambda r: f"{r['mean_mae_deg']:.6f} ± {r['sample_seed_sd_deg']:.6f}"
        lines += [
            f"| {e['phase']} | {e['participant']} | {e['family']} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {e['G_method_deg_mean']:+.6f} | {e['G_EMG_deg_mean']:+.6f} | {e['B2_minus_B1_deg_mean']:+.6f} |"
        ]
    lines += [
        "",
        "**B0 interpretation:** R2/R3/R4 use the fixed independent vision anchor by algebra. R0/R1/R5/R6 B0 are optimized constant-input visual-capacity controls. For reference-difference models, [effects_by_seed.csv](effects_by_seed.csv) also reports gains against R1/R6/R5 B0 respectively. Winning against an anchor alone does not establish an EMG-specific benefit.",
        "",
        "**Comparison definitions:** G_method = current-round R0_B1 MAE − family B1 MAE. G_EMG = same-family B0 − B1. B2 uses mismatched training and fixed mismatch-101 inference; correct-input B2 scores and all three inference maps are retained in [interventions.csv](interventions.csv). No historical model score is substituted for R0.",
        "",
        "## Feature information and supervision",
        "",
        "| Target | Input | T_val MAE | Fit-median baseline | Gain vs median | Gain vs zero |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in probes:
        if r["grouping"] == "overall":
            lines.append(
                f"| {r['target']} | {r['condition']} | {r['mae_deg']:.6f} | {r['fit_median_baseline_mae_deg']:.6f} | {r['gain_vs_fit_median_deg']:+.6f} | {r['gain_vs_zero_deg']:+.6f} |"
            )
    lines += [
        "",
        "All probes use the same ordered temporal reader and seed 42, fit only on T_fit. Each target has its own fit-only joint-median baseline; change targets also have a zero-change baseline. Raw MAEs across targets are not measures of relative information content. P3 is **offline local-change decoding**, because the EMG input includes future context. Visual-residual probes compare constant, correct and mismatched EMG under the same visual condition. Independent decodability is not equivalent to a conditional visual increment.",
        "",
        "Paired supervision contrasts are R2→R3 and R1→R6 (pose), R3→R4 and R6→R5 (additional dynamics). Reference subtraction contrasts are R1↔R2, R6↔R3 and R5↔R4. [group_gains.csv](group_gains.csv) reports all directions, including worsening, rather than selecting favorable joints or transitions.",
        "",
        "## Fixed protocol and coverage",
        "",
        f"Execution checkout: `{identity['git_branch']}` at `{identity['git_head']}`. Execution identity: `{identity['identity']}`. The requested branch name was not forced; no checkout, reset, pull, commit or push was performed.",
        "",
        f"Adam lr=1e-4, weight decay=1e-4, {protocol['epochs']} epochs, effective batch **{protocol['batch']}**, seeds 42/43/44, FP32 and TF32 off, last epoch primary. The user explicitly authorized the hardware-driven batch adjustment before E scoring. Encoding batch is {protocol.get('encoding_batch',16)}. The superseded batch-32 run remains preserved and contributes no results here.",
        "",
        "Whole continuous gesture/active execution IDs are created across the recording independently of original split boundaries. Seed 20261028 allocates approximately 20% of the **train execution groups** to T_val; support-boundary purging changes the retained center ratio. T_fit=3,071, T_val=518, purged=585, T_all=4,174. The retained fit/validation split is about 85.6/14.4% by centers, not exactly 80/20. E centers: wmh 3,507; zbk 6,966; wsj 216. Counts were reconstructed/revalidated, not hard-coded for selection. [coverage.csv](coverage.csv) records valid auxiliary coverage and original exclusions.",
        "",
        "Actual EMG support is [center−3895, center+3895), within a single recording and original split, with finite full EMG input. Historical 12,000-row qualification is retained. Auxiliary labels at ±200 nominal 2-kHz rows are checked independently and masked per target; missing dynamics never remove a main-pose score. The selected crop has valid decoding, non-stale mapping and |alignment delta|≤50 ms; its mapped center is within the much wider EMG support. All primary conditions share centers, images, labels and masks.",
        "",
        "Donors form same-pool, same-recording/right-hand/original-split bijections with nonoverlapping complete supports, preferring different executions. Training maps change per epoch, inference maps use seeds 101/102/103. Donor labels are never used. Input reversal occurs before the frozen encoder. [verification.json](verification.json) verifies saved orders/maps and the selection/score chronology.",
        "",
        "The released fusion supplies its own strictly loaded EMG featurizer/decoder, never a substituted EMGFormer-M. The shared visual anchor is strictly loaded independent vision_resnet18.ckpt. The unmodified fusion wrapper and independent map→original pooling/head were verified separately. New residual heads start at the independent visual anchor, not the historical full-fusion output.",
        "",
        "All families share a 128-wide adapter, 5 finger queries, 4 attention heads, one interaction block, 4×4 visual tokens with 2D positions, global visual/anchor conditioning, ordered 146×256 EMG decoder features, and a zero-initialized per-finger output. R0 adds an attention pool reducing 146 EMG tokens to one; R1–R6 read all 146. R0 has 8,321 additional active pooling parameters. Decoder positions are ordered indices, not a verified physical receptive-field map. No new multiscale/lag/gate module was added.",
        "",
        "R1/R2 share every tensor shape and initialization; only H(q,E_ref) subtraction changes. Reference physical zero goes through the actual sparse layout, scalar normalization and frozen encoder. Reference features before the adapter may be reused; the current adapter/H is always recomputed with gradients in both branches. New dropout is zero. The cancellation, mixed q/E interaction, zero-init first-output gradient, subsequent adapter/query gradients, auxiliary-only adapter gradient, frozen state/BN and checkpoint reload were tested.",
        "",
        "The local normalization audit found no definite implementation error: filtered_paper currently falls back to legacy egoemg scalar statistics, as historically validated. Padded slots are normalized with the same scalar. Original training-statistics bytes/hash remain unavailable; this study fixes the existing protocol and does not select a scale by E error.",
        "",
        "R3/R6 supervise EMG-only pose; R4/R5 also supervise past/subsequent changes. All enabled coefficients are 1 and use masked per-angle L1 in radians. Auxiliary heads read the adapter used by fusion and never read vision. The final prediction is absolute pose; no dynamics integration is performed. Analytical B0 cells do not claim trained auxiliary performance.",
        "",
        "T_all retraining begins from the original seed-specific initialization, not the selected T_fit checkpoint. Former T_val is included in fitting; no new checkpoint selection is applied. Its comparison answers performance after using all eligible wmh train data and is not a second independent validation.",
        "",
        "## Execution, uncertainty and reproducibility",
        "",
        f"Primary planned cells: {verification['primary_fit_planned']}; actual optimization: {verification['primary_fit_optimized']}; analytical reuse: 9. Including T_all: {verification['total_cells']} logical cells, {verification['total_optimized']} optimization cells, {verification['total_analytical']} analytical cells, {verification['failed_cells']} failed, {verification['missing_cells']} missing.",
        "",
        "All primary metrics were independently recomputed from saved predictions with NumPy float64, without importing the training metric or model. Split, execution, block, joint, finger, transition and dynamic-bin weighted sums reconcile with overall valid-angle-weighted MAE. Seed means are not ensemble MAEs. Full predictions/checkpoints/logs remain in ignored outputs.",
        "",
        "Execution-level paired distributions and descriptive bootstrap intervals are in [execution_intervals.csv](execution_intervals.csv). These are descriptive recording-level variation, not confidence intervals for new participants. Adjacent executions can share contextual windows, and 30-second center blocks do not guarantee disjoint support. Overlapping windows are not counted as independent experimental repeats.",
        "",
        "Resource accounting is in [resources.csv](resources.csv) and [profile.json](profile.json). Cached-head timings and actual read+frozen-backbone+head timings are distinguished; pipeline timing uses warmed OS caches and repeated direct input reads. Batch sizes, CUDA synchronization, device and included components are recorded. Auxiliary heads are excluded from inference parameter counts.",
        "",
        "Historical full-fusion matching scores (different anchor, architecture and 10-epoch protocol) are background only: wmh B0/B1/B2=6.214894/6.208619/6.209106°, zbk=5.644711/5.662749/5.662682°. Their differences from this suite cannot be assigned to one new fusion module.",
        "",
        "Limitations: one fitted recording, small wsj review coverage, offline future-context input, correlated windows, unresolved checkpoint participant exposure and original statistics provenance. Negative outcomes do not imply near-zero conditional mutual information. Reference subtraction enforces zero at E_ref but can still use signal presence to enable visual correction; it does not remove all visual shortcuts.",
        "",
        "The most useful next experiment must distinguish visual-capacity correction from correct-window information and confirm the observed patterns in genuinely new recordings. Attention, gradients and feature sensitivity establish pathways, not useful causal contribution.",
        "",
        "Reproduction commands are in the [workflow](../../docs/emg_innovation_suite.md). Snapshot export identity and checkout commit are not the original execution identity. All changes are local; no commit or push was made.",
    ]
    contrasts = list(csv.DictReader((out / "supervision_contrasts.csv").open()))
    lines += [
        "",
        "## Paired architecture and supervision contrasts",
        "",
        "Positive values mean the target B1 family has lower MAE. These are descriptive fixed contrasts; no new family was selected from E.",
        "",
        "| Contrast | Question | wmh gain | zbk gain | wsj gain |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for source, target in [
        ("R0", "R1"),
        ("R1", "R2"),
        ("R6", "R3"),
        ("R5", "R4"),
        ("R2", "R3"),
        ("R1", "R6"),
        ("R3", "R4"),
        ("R6", "R5"),
    ]:
        rows = [
            next(
                r
                for r in contrasts
                if r["source"] == source
                and r["target"] == target
                and r["episode"] == ep
                and r["grouping"] == "overall"
            )
            for ep in EPISODES
        ]
        lines.append(
            f"| {source}→{target} | {rows[0]['question']} | "
            + " | ".join(f"{float(r['mean_gain_deg']):+.6f}" for r in rows)
            + " |"
        )
    lines += [
        "",
        "### Additional dynamic supervision on transition centers",
        "",
        "Transition is fixed from metadata, not visual error. Small coverage and contextual overlap limit interpretation. [supervision_contrasts.csv](supervision_contrasts.csv) also retains steady and amplitude groups.",
        "",
        "| Contrast | wmh gain | zbk gain | wsj gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source, target in [("R3", "R4"), ("R6", "R5")]:
        values = []
        for ep in EPISODES:
            rr = [
                r
                for r in contrasts
                if r["source"] == source
                and r["target"] == target
                and r["episode"] == ep
                and r["grouping"] == "transition"
                and r["group"] == "True"
            ]
            values.append(
                f"{float(rr[0]['mean_gain_deg']):+.6f}" if rr else "not covered"
            )
        lines.append(f"| {source}→{target} | " + " | ".join(values) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def export_snapshot(run, out, snapshot):
    snapshot.mkdir(parents=True, exist_ok=False)
    selected = [
        "report.md",
        "main_tables.json",
        "main_fit.csv",
        "main_T_all.csv",
        "validation_metrics.csv",
        "validation_effects.csv",
        "supervision_contrasts.csv",
        "main_by_seed.csv",
        "main_summary.csv",
        "effects_by_seed.csv",
        "effect_summary.csv",
        "probe_metrics.csv",
        "coverage.csv",
        "groups.csv",
        "group_gains.csv",
        "execution_intervals.csv",
        "interventions.csv",
        "auxiliary_metrics.csv",
        "resources.csv",
        "matrix_status.csv",
        "verification.json",
        "trajectory_selection.json",
        "visual_error_bins.json",
    ]
    records = []
    for name in selected:
        src = out / name
        dst = snapshot / name
        shutil.copyfile(src, dst)
        records.append(
            dict(file=name, source=str(src.relative_to(ROOT)), sha256=sha(src))
        )
    for name in (
        "selection_lock.json",
        "profile.json",
        "partition_rule.json",
        "dynamic_rule.json",
        "input_audit.json",
        "reference_audit.json",
        "joint_mapping.json",
    ):
        src = run / name
        shutil.copyfile(src, snapshot / name)
        records.append(
            dict(file=name, source=str(src.relative_to(ROOT)), sha256=sha(src))
        )
    for src in sorted((out / "figures").glob("*.png")):
        dst = snapshot / "figures" / src.name
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dst)
        records.append(
            dict(
                file=str(dst.relative_to(snapshot)),
                source=str(src.relative_to(ROOT)),
                sha256=sha(src),
            )
        )
    if (run / "supplement").exists():
        for name in (
            "input_channels.csv",
            "feature_sensitivity.json",
            "endpoint_verification.csv",
            "parameter_accounting.json",
            "pipeline_timing.csv",
            "input_payload_hashes.json",
            "receipt.json",
            "hardware_batch_profile.json",
            "test_receipt.json",
            "environment.json",
        ):
            src = run / "supplement" / name
            if src.exists():
                dst = snapshot / "input_and_resources" / name
                dst.parent.mkdir(exist_ok=True)
                shutil.copyfile(src, dst)
                records.append(
                    dict(
                        file=str(dst.relative_to(snapshot)),
                        source=str(src.relative_to(ROOT)),
                        sha256=sha(src),
                    )
                )
    identity = json.loads((run / "identity.json").read_text())
    readme = f"""# EMG innovation evidence snapshot\n\n[English report](report.md) · [Personal research](../../README.md) · [Workflow](../../docs/emg_innovation_suite.md)\n\nCompleted offline Preview right-hand study, fitted only on wmh train. zbk/wsj\nare fixed review recordings, not unknown-participant benchmarks. The selected\nfamily was locked before formal evaluation errors. All tables/figures here are\nnewly executed results, not archival tests. Batch128 was explicitly authorized\nby the user after hardware profiling; the original 30-epoch/lr protocol remains.\n\nValid execution: `{run.relative_to(ROOT)}`.\nExecution checkout: `{identity['git_head']}` on `{identity['git_branch']}`.\nExecution identity: `{identity['identity']}`.\n\nThe interrupted full_01 and smoke_01 failure are preserved separately and are\nnot combined with this result. Supplied checkpoints and inputs were read-only.\nBatch features were never written to disk. Full predictions, optimizer states,\nweights, orders, logs and source copies remain in ignored outputs and are\nintentionally omitted. No full logs, input copies or per-window features are\nincluded in this snapshot.\n\nThe manifest maps every exported evidence file to its repository-relative\nsource and SHA-256. Export date and current checkout commit are distinct from\nexecution identity. The sources are local until separately committed/pushed;\nthis task did neither. Prediction-level reproduction needs the original data,\ncheckpoints and full output directory; compact tables and figures are readable\non GitHub independently of outputs. See workflow for smoke/full/resume/verify/export.\n"""
    (snapshot / "README.md").write_text(readme)
    write_json(
        snapshot / "snapshot_manifest.json",
        dict(
            exported_at=dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            export_checkout_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            execution_source=str(run.relative_to(ROOT)),
            execution_identity=identity["identity"],
            execution_checkout_commit=identity["git_head"],
            files=records,
        ),
    )
    verify_snapshot(snapshot)


def verify_snapshot(snapshot):
    manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
    links = 0
    for r in manifest["files"]:
        assert sha(snapshot / r["file"]) == sha(ROOT / r["source"]) == r["sha256"]
        assert (
            subprocess.run(
                ["git", "check-ignore", "-q", str(snapshot / r["file"])], cwd=ROOT
            ).returncode
            == 1
        )
    for path in snapshot.rglob("*.md"):
        for url in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if re.match(r"\w+://", url) or url.startswith("#"):
                continue
            # The workflow link is relative to the snapshot; the report source is
            # also at equivalent depth under outputs/analysis, but only snapshot
            # link validity is a public delivery requirement.
            assert (path.parent / url.split("#")[0]).resolve().exists(), (path, url)
            links += 1
    assert (
        subprocess.run(
            [
                "git",
                "check-ignore",
                "-q",
                str(ROOT / manifest["execution_source"] / "run.log"),
            ],
            cwd=ROOT,
        ).returncode
        == 0
    )
    return dict(
        files=len(manifest["files"]),
        links=links,
        source_hashes_match=True,
        outputs_ignored=True,
        snapshot_trackable=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["summarize", "verify", "export"])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--snapshot", type=Path)
    a = p.parse_args()
    run = a.run_dir.resolve()
    assert (
        run / "training_complete.json"
    ).exists(), "Training/scoring suite is not complete"
    state = json.loads((run / "state.json").read_text())
    assert state["status"] == "training_and_scoring_complete"
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    mainrows, intervention, groups, auxiliary, arrays = collect(run, state)
    effects_ = effects(mainrows)
    summary, esummary = summaries(mainrows, effects_)
    probes = probe_tables(run)
    pairs, intervals = paired_groups(run, arrays)
    verification = verify_structure(run, state, arrays)
    coverage = []
    for ep in EPISODES:
        info = json.loads((run / ep / "selection_audit.json").read_text())
        for pool in ("T_all", "E"):
            d = np.load(run / ep / f"enriched_{pool}.npz")
            dm = d["delta_mask"]
            coverage.append(
                dict(
                    episode=ep,
                    participant=PEOPLE[ep],
                    pool=pool,
                    centers=len(d["centers"]),
                    valid_pose_angles=int(d["mask"][:, :20].sum()),
                    valid_delta_minus_angles=int(dm[:, 0].sum()),
                    valid_delta_plus_angles=int(dm[:, 1].sum()),
                    transition_centers=int(d["transition"].sum()),
                    original_candidates=info["candidates"],
                    exclusions_json=json.dumps(info["exclusions"], sort_keys=True),
                    used_for_fitting=ep == EPISODES[0] and pool == "T_all",
                )
            )
    resources = []
    for key, c in state["cells"].items():
        resources.append(
            dict(
                unit=key,
                phase=c["phase"],
                family=c["family"],
                condition=c["condition"],
                seed=c["seed"],
                kind=c["planned_kind"],
                training_seconds=c.get("training_seconds", 0),
                peak_allocated_bytes=c.get("peak_allocated_bytes", 0),
                peak_reserved_bytes=c.get("peak_reserved_bytes", 0),
                trainable_parameters=c["parameters"]["trainable"],
                inference_fusion_parameters=c["parameters"]["inference"],
                stored_new_parameters=c["parameters"]["stored"],
                cached_head_batch_seconds=c.get("cached_head_batch_seconds"),
                timing_batch=state["protocol"]["batch"],
            )
        )
    accounting_path = run / "supplement/parameter_accounting.json"
    frozen_parameters = (
        json.loads(accounting_path.read_text())["frozen_used_parameters"]
        if accounting_path.exists()
        else None
    )
    for row in resources:
        row["frozen_used_parameters"] = frozen_parameters
        row["total_effective_training_parameters"] = (
            frozen_parameters + row["trainable_parameters"]
            if frozen_parameters is not None
            else None
        )
        row["total_inference_parameters"] = (
            frozen_parameters + row["inference_fusion_parameters"]
            if frozen_parameters is not None
            else None
        )
        row["emg_decoder_tokens"] = 146
        row["emg_tokens_read_by_queries"] = 1 if row["family"] == "R0" else 146
        row["residual_read_branches"] = 2 if row["family"] in DIFF else 1
        row["attention_score_elements_per_sample_per_branch"] = (
            4 * 5 * row["emg_tokens_read_by_queries"]
        )
    matrices = [
        dict(
            unit=k,
            phase=c["phase"],
            family=c["family"],
            condition=c["condition"],
            seed=c["seed"],
            planned_kind=c["planned_kind"],
            status=c["status"],
            optimized_epochs=c["actual_optimization_epochs"],
            auxiliary_trained=c["auxiliary_trained"],
            initialization_sha256=c["initialization_sha256"],
        )
        for k, c in state["cells"].items()
    ]
    visual_definitions = []
    for ep in EPISODES:
        ds = np.load(run / ep / "enriched_E.npz")
        mask = ds["mask"][:, :20]
        error = np.sum(
            np.abs(
                arrays[("fit", ep, "R2", "B0", 42)].astype(np.float64)
                - ds["reference"][:, :20]
            )
            * mask,
            axis=1,
        ) / mask.sum(1)
        visual_definitions.append(
            dict(
                episode=ep,
                thresholds_deg=(np.quantile(error, [1 / 3, 2 / 3]) * DEG).tolist(),
                rule="per-recording frozen-vision-error tertiles, post hoc explanation only",
                used_for_training_gating_or_selection=False,
            )
        )
    write_json(out / "visual_error_bins.json", visual_definitions)
    tables = {
        "main_by_seed.csv": mainrows,
        "main_summary.csv": summary,
        "effects_by_seed.csv": effects_,
        "effect_summary.csv": esummary,
        "interventions.csv": intervention,
        "groups.csv": groups,
        "auxiliary_metrics.csv": auxiliary,
        "probe_metrics.csv": probes,
        "group_gains.csv": pairs,
        "execution_intervals.csv": intervals,
        "coverage.csv": coverage,
        "resources.csv": resources,
        "matrix_status.csv": matrices,
    }
    validation_rows = []
    for key, cell in state["cells"].items():
        if cell["phase"] != "fit":
            continue
        data = np.load(run / "cells" / key / "validation.npz")
        value = weighted(
            data["prediction"], data["reference"][:, :20], data["mask"][:, :20]
        )
        assert abs(value - cell["validation_mae_deg"]) < 1e-12
        validation_rows.append(
            dict(
                phase="fit",
                episode=EPISODES[0],
                participant="wmh",
                family=cell["family"],
                training_condition=cell["condition"],
                inference_condition=primary(cell["condition"]),
                seed=cell["seed"],
                centers=len(data["centers"]),
                valid_angles=int(data["mask"][:, :20].sum()),
                mae_deg=value,
                analytical_anchor=cell["condition"] == "B0" and cell["family"] in DIFF,
            )
        )
    contrasts = []
    for ep in EPISODES:
        for source, target, question in [
            ("R0", "R1", "pooled_vs_direct"),
            ("R1", "R2", "difference_no_aux"),
            ("R6", "R3", "difference_pose_aux"),
            ("R5", "R4", "difference_dynamic_aux"),
            ("R2", "R3", "pose_aux_difference"),
            ("R1", "R6", "pose_aux_ordinary"),
            ("R3", "R4", "extra_dynamics_difference"),
            ("R6", "R5", "extra_dynamics_ordinary"),
        ]:
            for grouping in ("overall", "transition", "dynamic_bin"):
                labels = (
                    ["all"]
                    if grouping == "overall"
                    else sorted(
                        set(
                            r["group"]
                            for r in groups
                            if r["grouping"] == grouping and r["episode"] == ep
                        )
                    )
                )
                for label in labels:
                    gains = []
                    for seed in (42, 43, 44):
                        if grouping == "overall":
                            get = lambda f: next(
                                r["mae_deg"]
                                for r in mainrows
                                if (
                                    r["phase"],
                                    r["episode"],
                                    r["family"],
                                    r["training_condition"],
                                    r["seed"],
                                )
                                == ("fit", ep, f, "B1", seed)
                            )
                        else:
                            get = lambda f: next(
                                r["mae_deg"]
                                for r in groups
                                if (
                                    r["phase"],
                                    r["episode"],
                                    r["family"],
                                    r["condition"],
                                    r["seed"],
                                    r["grouping"],
                                    r["group"],
                                )
                                == ("fit", ep, f, "B1", seed, grouping, label)
                            )
                        gains.append(get(source) - get(target))
                    mean, sd = stats(gains)
                    contrasts.append(
                        dict(
                            episode=ep,
                            participant=PEOPLE[ep],
                            question=question,
                            source=source,
                            target=target,
                            grouping=grouping,
                            group=label,
                            mean_gain_deg=mean,
                            sample_seed_sd_deg=sd,
                            positive_seeds=sum(g > 0 for g in gains),
                        )
                    )
    tables["validation_metrics.csv"] = validation_rows
    tables["validation_effects.csv"] = effects(validation_rows)
    tables["supervision_contrasts.csv"] = contrasts
    tables["main_fit.csv"] = [r for r in mainrows if r["phase"] == "fit"]
    tables["main_T_all.csv"] = [r for r in mainrows if r["phase"] == "all"]
    for name, rows in tables.items():
        write_csv(out / name, rows)
    write_json(
        out / "main_tables.json", dict(main=mainrows, summary=summary, effects=esummary)
    )
    verification.update(
        main_prediction_files=len(state["formal_evaluations"]),
        main_metric_rows=len(mainrows),
        intervention_rows=len(intervention),
        group_rows=len(groups),
        probe_metric_rows=len(probes),
        verified_at=(
            json.loads((out / "verification.json").read_text())["verified_at"]
            if a.mode == "verify" and (out / "verification.json").exists()
            else dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        ),
        summarizer_sha256=sha(Path(__file__)),
    )
    if (run / "supplement/receipt.json").exists():
        verification["supplement"] = json.loads(
            (run / "supplement/receipt.json").read_text()
        )
        payload = (
            json.loads((run / "supplement/input_payload_hashes.json").read_text())
            if (run / "supplement/input_payload_hashes.json").exists()
            else []
        )
        for r in payload:
            stat = Path(r["path"]).stat()
            assert stat.st_size == r["bytes"] and stat.st_mtime_ns == r["mtime_ns"]
            assert sha(Path(r["path"])) == r["sha256"], r["path"]
        verification["input_payload_full_sha256_recorded"] = len(payload)
        test_receipt_path = run / "supplement/test_receipt.json"
        if test_receipt_path.exists():
            test_receipt = json.loads(test_receipt_path.read_text())
            assert (
                sha(ROOT / test_receipt["output_path"]) == test_receipt["output_sha256"]
            )
            assert (
                sha(ROOT / "myself/egoemg/tests/test_emg_innovation_suite.py")
                == test_receipt["test_source_sha256"]
            )
            verification["related_tests_passed"] = test_receipt["passed"]
        smoke = ROOT / "myself/outputs/2026-10-29_emg_innovation_smoke_03"
        if smoke.exists():
            smoke_receipt = json.loads((smoke / "complete.json").read_text())
            assert smoke_receipt["status"] == "smoke_complete"
            assert sha(smoke / "smoke_checks.json") == smoke_receipt["checks_sha256"]
            verification["real_data_smoke_structure_condition_cases"] = 21
            verification["real_data_smoke_receipt_sha256"] = sha(
                smoke / "complete.json"
            )
    write_json(out / "verification.json", verification)
    chosen = json.loads((run / "selection_lock.json").read_text())["selected_family"]
    if a.mode != "verify":
        figures(
            out,
            mainrows,
            summary,
            esummary,
            intervention,
            probes,
            pairs,
            arrays,
            chosen,
            run,
        )
        report(out, run, state, summary, esummary, probes, verification, intervals)
    if a.mode == "export":
        assert a.snapshot is not None
        export_snapshot(run, out, a.snapshot.resolve())
    if a.mode == "verify" and a.snapshot:
        print(json.dumps(verify_snapshot(a.snapshot.resolve()), indent=2))
    shutil.copyfile(__file__, run / "executed_independent_summarizer.py")
    write_json(
        run / "verified_complete.json",
        dict(
            training_complete_sha256=sha(run / "training_complete.json"),
            verification_sha256=sha(out / "verification.json"),
            status="verified_complete",
            completed_at=dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        ),
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
