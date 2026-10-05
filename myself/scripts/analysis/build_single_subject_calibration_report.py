#!/usr/bin/env python3
"""Build a detailed, evidence-backed Word work report from saved suite artifacts.

Figures use the existing conda environment. DOCX uses the bundled Windows Python
runtime. Dataset reads are limited to two small EMG previews and 25 crop previews.
Original experiment artifacts and model/dataset bytes are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True
import numpy as np


REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN = (
    REPO
    / "myself/outputs/single_subject_calibration_suite/2026-10-05_wmh_right_suite_full_02"
)
REPORT_PARENT = REPO / "myself/outputs/single_subject_calibration_suite_word_report"
SEEDS = [42, 43, 44]
COLORS = {
    "M0": "#334155",
    "M1": "#0284c7",
    "M2": "#16a34a",
    "M3": "#dc2626",
    "M4": "#0891b2",
    "M5": "#7c3aed",
    "M6": "#ea580c",
}
NAMES = {
    "M0": "Frozen baseline",
    "M1": "Reference offset",
    "M2": "Reference affine",
    "M3": "Reference head",
    "M4": "RGB offset",
    "M5": "RGB affine",
    "M6": "RGB head",
}
JOINT_GROUPS = (
    ["Thumb"] * 4 + ["Index"] * 4 + ["Middle"] * 4 + ["Ring"] * 4 + ["Pinky"] * 4
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources(run):
    return dict(
        settings=read_json(run / "settings.json"),
        coverage=read_json(run / "coverage.json"),
        emg=read_json(run / "emg_evidence.json"),
        teacher=read_json(run / "teacher_diagnostics.json"),
        metrics=read_json(run / "metrics.json"),
        conditions=read_json(run / "conditions.json"),
        verification=read_json(run / "verification.json"),
        recalculation=read_json(run / "verification_recalculation.json"),
        gesture_rows=read_csv(run / "gesture_coverage.csv"),
        rgb_rows=read_csv(run / "rgb_gesture_coverage.csv"),
        variation=read_csv(run / "reference_variation.csv"),
        detailed=read_csv(run / "detailed_metrics.csv"),
        quality=read_csv(run / "teacher_quality.csv"),
        manifest=np.load(run / "manifest.npz"),
        cache=np.load(run / "emg_cache.npz"),
        teacher_cache=np.load(run / "teacher_cache.npz"),
        evaluation=np.load(run / "evaluation_predictions.npz"),
    )


def condition_prediction(evaluation, cid):
    if cid in ("M3", "M6"):
        return np.mean([evaluation[f"{cid}_seed{s}"] for s in SEEDS], axis=0)
    return evaluation[cid]


def build_figures(run, out):
    os.environ["MPLCONFIGDIR"] = str(REPO / "myself/.cache/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image, ImageDraw, ImageFont

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "savefig.dpi": 190,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    s = load_sources(run)
    m, c, tc, ev = (s[k] for k in ("manifest", "cache", "teacher_cache", "evaluation"))
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(fig, name, title, explanation, source):
        fig.tight_layout()
        path = figdir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        figures.append(
            dict(
                id=name,
                file=f"figures/{name}.png",
                title=title,
                explanation=explanation,
                source=source,
            )
        )

    def diagram(name, title, boxes, explanation, source):
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.set(xlim=(0, 10), ylim=(0, 3))
        ax.axis("off")
        for i, (label, color) in enumerate(boxes):
            x = 0.2 + i * 9.6 / len(boxes)
            width = 9.1 / len(boxes)
            ax.add_patch(
                FancyBboxPatch(
                    (x, 0.75),
                    width,
                    1.4,
                    boxstyle="round,pad=0.07",
                    facecolor=color,
                    edgecolor="#64748b",
                )
            )
            ax.text(x + width / 2, 1.45, label, ha="center", va="center", fontsize=10)
            if i < len(boxes) - 1:
                ax.annotate(
                    "",
                    xy=(x + width + 0.35, 1.45),
                    xytext=(x + width + 0.06, 1.45),
                    arrowprops=dict(arrowstyle="->", lw=1.4),
                )
        save(fig, name, title, explanation, source)

    diagram(
        "workflow",
        "The complete offline experiment",
        [
            ("Direct dataset\nreads on D:", "#dbeafe"),
            ("Frozen EMG\nfeatures / angles", "#dbeafe"),
            ("Reference or\ncrop-teacher labels", "#ede9fe"),
            ("Independent\nfinger adaptation", "#dcfce7"),
            ("Fixed E\nEMG-only scores", "#ffedd5"),
        ],
        "The dataset remains on the designated data disk. Training references and RGB teacher predictions are separate target sources. Each method starts from the same released initialization; adaptation is not stacked. Only the fixed evaluation reference labels determine the reported final MAE.",
        "settings.json; suite implementation; manifest.npz",
    )
    diagram(
        "architecture",
        "Where the permitted update occurs",
        [
            ("8-channel EMG\n12,000 rows", "#dbeafe"),
            ("Frozen TDS +\nTransformer", "#dbeafe"),
            ("Frozen head MLP\n512 features", "#dbeafe"),
            ("head.net.3\n22 outputs", "#dcfce7"),
            ("Linear alignment\ncenter prediction", "#ffedd5"),
        ],
        "The final layer has 512 input features and 22 angle rows. Head adaptation updates only rows 0–19 and their biases (10,260 parameters). Wrist rows 20–21 and all earlier modules remain frozen. Context is 510 rows on the left and zero on the right; the raw 230-step output is aligned to 11,490 rows before selecting center index 5,490.",
        "emg_evidence.json; emg_cache.npz",
    )
    fig, ax = plt.subplots(figsize=(9, 2.6))
    split_colors = {
        "train": "#16a34a",
        "user": "#0284c7",
        "gesture": "#7c3aed",
        "both": "#ea580c",
    }
    for v in s["coverage"]["intervals"]:
        ax.broken_barh(
            [(v["local_start"] / 2000, (v["local_end"] - v["local_start"]) / 2000)],
            (0.2, 0.6),
            facecolors=split_colors[v["split"]],
        )
        if v["local_end"] - v["local_start"] > 35000:
            ax.text(
                (v["local_start"] + v["local_end"]) / 4000,
                0.5,
                str(v["interval"]),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    for split, color in split_colors.items():
        ax.plot([], [], color=color, lw=8, label=split)
    ax.set(ylim=(0, 1), yticks=[], xlabel="Nominal recording time (rows / 2,000 Hz)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    save(
        fig,
        "split_timeline",
        "All 32 constant-split intervals",
        "Train and evaluation intervals are interleaved. Later train intervals can follow earlier evaluated intervals. Colors use authoritative row split IDs, not episode-level placeholder zeros. Numeric annotations identify sufficiently wide intervals; all intervals appear in the full table.",
        "coverage.json",
    )
    fig, ax = plt.subplots(figsize=(9, 3.8))
    intervals = s["coverage"]["intervals"]
    x = np.arange(len(intervals))
    candidate = [v["candidate_windows"] for v in intervals]
    eligible = [v["eligible_windows"] for v in intervals]
    ax.bar(x, candidate, color="#cbd5e1", label="Candidates")
    ax.bar(
        x,
        eligible,
        color=[split_colors[v["split"]] for v in intervals],
        width=0.65,
        label="Eligible",
    )
    ax.set(
        xticks=x,
        xticklabels=[v["interval"] for v in intervals],
        xlabel="Interval ID",
        ylabel="Window count",
    )
    ax.legend()
    save(
        fig,
        "interval_counts",
        "Candidate and eligible windows by interval",
        "Only interval 25 excludes candidates: five invalid center supervision labels. Interval 31 contains 111 rows, shorter than the required 12,000-row input window, and therefore contributes zero windows. Every other candidate is eligible.",
        "coverage.json",
    )
    fig, ax = plt.subplots(figsize=(9, 3.7))
    for pool, y in [("T_all", 1), ("E", 0)]:
        ix = m["pools"] == pool
        for a, b in s["coverage"]["pool_coverage"][pool]["support_unions"]:
            ax.broken_barh(
                [(a / 2000, (b - a) / 2000)],
                (y + 0.05, 0.7),
                facecolors="#16a34a" if pool == "T_all" else "#0284c7",
            )
    ax.set(
        yticks=[0.4, 1.4],
        yticklabels=["E supports", "T_all supports"],
        xlabel="Nominal recording time (seconds)",
    )
    save(
        fig,
        "support_unions",
        "Fitting and evaluation support unions",
        "The displayed unions count covered input rows once. Fitting and evaluation supports do not intersect. Overlap within a pool is allowed, so summing 6-second window durations would substantially overstate coverage. T_all covers 476.9 nominal seconds and E covers 489.3.",
        "coverage.json",
    )
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set(xlim=(0, 16000), ylim=(0, 4))
    ax.axvspan(0, 14000, color="#dcfce7", alpha=0.5)
    ax.axvspan(14000, 16000, color="#ede9fe", alpha=0.8)
    for i, a in enumerate([0, 200, 400]):
        ax.plot([a, a + 12000], [3 - i * 0.8] * 2, lw=5, color="#16a34a")
        ax.scatter([a + 6000], [3 - i * 0.8], color="#ea580c", s=40, zorder=3)
    ax.text(
        7000, 0.25, "Complete support must stay inside one split interval", ha="center"
    )
    ax.set(yticks=[], xlabel="Illustrative local row index")
    ax.text(14100, 3.6, "next split", fontsize=9)
    save(
        fig,
        "window_protocol",
        "Input support and center-only supervision",
        "This schematic uses the real 12,000-row length and 200-row stride. Orange points are supervised centers at start + 6,000. Center finger labels must be valid and finite; labels elsewhere in the context do not supervise the run. Every input EMG value must be finite, and a window crossing a split boundary is excluded. The interval lengths here are illustrative.",
        "enumerate_windows() and fixed settings",
    )
    for pool, title in [
        ("T_all", "Fitting action coverage"),
        ("E", "Evaluation action coverage"),
    ]:
        rows = [v for v in s["gesture_rows"] if v["pool"] == pool]
        fig, ax = plt.subplots(figsize=(9, 6.8 if pool == "E" else 5.5))
        labels = [f"{v['gesture_id']}: {v['name']}" for v in rows]
        counts = [int(v["centers"]) for v in rows]
        ax.barh(labels, counts, color="#16a34a" if pool == "T_all" else "#0284c7")
        ax.invert_yaxis()
        ax.set_xlabel("Eligible centers")
        ax.tick_params(axis="y", labelsize=8)
        save(
            fig,
            f"gesture_coverage_{pool}",
            title,
            "Counts preserve raw IDs and the supplied vocabulary. The inactive ID −1 is shown separately. T_all has 25 active gesture IDs and four inactive centers; E has 34 active gesture IDs and 68 inactive centers. The active fitting and evaluation gesture sets are disjoint in this recording. Coverage therefore does not mean that the fitting pool contains every evaluation action or angle range.",
            "gesture_coverage.csv; supplied gesture_classes.json",
        )
    means = np.asarray(s["settings"]["identity"]["normalization"]["mean"])
    stds = np.asarray(s["settings"]["identity"]["normalization"]["std"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(np.arange(8), means, color="#0284c7")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[0].set(title="Stored means", xlabel="Stored channel", ylabel="Asset value")
    axes[1].bar(np.arange(8), stds, color="#16a34a")
    axes[1].set(
        title="Stored standard deviations",
        xlabel="Stored channel",
        ylabel="Asset value",
    )
    save(
        fig,
        "normalization",
        "Exact right-hand normalization asset",
        "The eight channels remain in stored order. Normalization is (EMG − mean)/(std + 1e−6), using egoemg__filtered_paper_right in the smoke workflow alias asset. The means are near zero; the right panel shows channel-specific scales. No statistics are estimated from fitting or evaluation rows.",
        "settings.json identity.normalization",
    )
    # Small inspection previews only; no raw window cache is written.
    data_root = Path(s["settings"]["identity"]["data_root"])
    data_manifest = read_json(data_root / "manifest.json")
    spec = data_manifest["fields"]["emg_right_filtered_paper"]
    mm = np.memmap(
        data_root / spec["filename"],
        mode="r",
        dtype=spec["dtype"],
        shape=tuple(spec["shape"]),
    )
    preview_records = []
    for pool in ["T_all", "E"]:
        i = int(np.flatnonzero(m["pools"] == pool)[0])
        a = int(m["starts"][i])
        raw = np.asarray(mm[a : a + 12000])
        norm = (raw - means) / (stds + 1e-6)
        fig, axes = plt.subplots(4, 2, figsize=(9, 6.2), sharex=True)
        for ch, ax in enumerate(axes.flat):
            ax.plot(
                np.arange(0, 12000, 20) / 2000, norm[::20, ch], lw=0.45, color="#0284c7"
            )
            ax.axvline(3, color="#ea580c", ls="--", lw=0.8)
            ax.set_title(f"Channel {ch}", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("Seconds from window start")
        save(
            fig,
            f"emg_input_{pool}",
            f"Actual normalized EMG input example: {pool}",
            f"The first eligible {pool} window is {m['ids'][i]}, with support [{a},{a+12000}). Only every 20th row is plotted for readability; model inference used all 12,000 rows and all eight channels. The dashed line marks the supervised input center at three nominal seconds. Vertical values are normalized amplitudes, not joint angles.",
            "Direct read of emg_right_filtered_paper; saved manifest ID and normalization",
        )
        preview_records.append(
            dict(
                pool=pool,
                id=str(m["ids"][i]),
                global_start=a,
                global_end=a + 12000,
                plot_decimation=20,
            )
        )
    joints = s["settings"]["identity"]["finger_joints"]
    jx = np.arange(20)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for pool, color in [("T_all", "#16a34a"), ("E", "#0284c7")]:
        rows = [v for v in s["variation"] if v["pool"] == pool]
        lo = np.asarray([float(v["minimum_deg"]) for v in rows])
        hi = np.asarray([float(v["maximum_deg"]) for v in rows])
        sd = np.asarray([float(v["std_deg"]) for v in rows])
        axes[0].plot(jx, sd, "o-", label=pool, color=color)
        axes[1].vlines(
            jx + (-0.12 if pool == "T_all" else 0.12),
            lo,
            hi,
            color=color,
            lw=3,
            label=pool,
        )
    axes[0].set_ylabel("Reference SD (degrees)")
    axes[0].legend()
    axes[1].set_ylabel("Reference range (degrees)")
    axes[1].set_xticks(jx, [f"J{j:02d}" for j in jx])
    axes[1].set_xlabel("Joint IDs; complete joint names are in the table")
    save(
        fig,
        "reference_variation",
        "Per-joint reference variation and range",
        "Top: population standard deviation of sampled reference angles within each pool. Bottom: minimum-to-maximum reference angle range. These describe sampled movement, not confidence intervals. T_all and E need not cover the same joint ranges; T_rgb equals T_all.",
        "reference_variation.csv",
    )
    rows = [v for v in s["quality"] if v["kind"] == "joint"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(jx, [float(v["mae_deg"]) for v in rows], color="#7c3aed")
    ax.set(
        xticks=jx,
        xticklabels=[f"J{j:02d}" for j in jx],
        ylabel="Teacher MAE (degrees)",
        xlabel="Finger joint",
    )
    save(
        fig,
        "teacher_joint",
        "RGB teacher quality by finger joint",
        "Each bar compares crop-only teacher output with the reference label at the eligible fitting center. Aggregate teacher MAE is 1.760795°. Reference labels are diagnostic ground truth here; RGB fitting uses the teacher outputs themselves. These diagnostics did not filter samples or tune parameters.",
        "teacher_quality.csv",
    )
    rows = [v for v in s["quality"] if v["kind"] == "gesture"]
    name_map = {int(v["gesture_id"]): v["name"] for v in s["gesture_rows"]}
    labels = [
        f"{v['key'].split(':')[0]}: {name_map[int(v['key'].split(':')[0])]}"
        for v in rows
    ]
    fig, ax = plt.subplots(figsize=(9, 5.7))
    ax.barh(labels, [float(v["mae_deg"]) for v in rows], color="#7c3aed")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Teacher MAE (degrees)")
    save(
        fig,
        "teacher_gesture",
        "RGB teacher quality by training action",
        "The teacher is scored only on eligible fitting centers. Gesture bars are center/joint-weighted MAEs within each raw gesture and active status, rather than equal-weight action scores. There is no teacher inference on E and no implication that teacher quality on training actions transfers to evaluation actions.",
        "teacher_quality.csv",
    )
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.hist(tc["delta_ms"], bins=np.arange(-2, 55, 2), color="#7c3aed")
    ax.axvline(50, color="#dc2626", ls="--", label="Predeclared 50 ms limit")
    ax.set(xlabel="Mapped head-image delta (ms)", ylabel="Training centers")
    ax.legend()
    save(
        fig,
        "rgb_timing",
        "Observed RGB mapping deltas",
        "All 4,174 fitting centers are nonstale, have decodable right-hand crops, and pass the 50 ms tolerance. Observed delta quantiles are 0/4/8/12/32 ms at 0/25/50/75/100 percent. A passing tolerance is a first-pass availability check, not a demonstrated synchronization guarantee.",
        "teacher_cache.npz; teacher_diagnostics.json",
    )
    ti = tc["eligible_indices"]
    tg = tc["target"][:, :20]
    ref = m["reference"][ti, :20]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, j in zip(axes.flat, [0, 5, 9, 17]):
        ax.scatter(
            np.rad2deg(ref[::5, j]),
            np.rad2deg(tg[::5, j]),
            s=3,
            alpha=0.3,
            color="#7c3aed",
        )
        lo, hi = np.rad2deg(ref[:, j]).min(), np.rad2deg(ref[:, j]).max()
        ax.plot([lo, hi], [lo, hi], color="#334155", lw=1)
        ax.set(
            title=joints[j], xlabel="Reference (degrees)", ylabel="Teacher (degrees)"
        )
    save(
        fig,
        "teacher_scatter",
        "Crop teacher versus fitting reference examples",
        "Each panel displays every fifth eligible center in manifest order for readability. The diagonal is perfect agreement, and point deviations show teacher/reference error. Four preselected joints span thumb, index, middle and pinky; they were not selected for favorable agreement. All centers contributed to the reported MAE.",
        "teacher_cache.npz; manifest.npz",
    )
    # Diverse real-crop gallery: first eligible occurrence of every active train gesture.
    sys.path.insert(0, str(REPO))
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

    reader = object.__new__(EgoEmgMemmapDataset)
    reader.per_episode_crops_dir = str(data_root.parent / "pre-crop_webcam_videoframes")
    reader._pec_pid = os.getpid()
    reader._pec_lmdb_cache = {}
    reader._pec_lmdb_max_open = 1
    gallery = Image.new("RGB", (1000, 5 * 230), "white")
    draw = ImageDraw.Draw(gallery)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14)
    crop_records = []
    for k, g in enumerate(
        sorted(np.unique(m["gesture"][ti][m["active"][ti]]).tolist())
    ):
        i = int(np.flatnonzero((m["gesture"][ti] == g) & m["active"][ti])[0])
        frame = int(tc["frame"][i])
        rgb = reader._read_episode_crops("episode_000020", np.asarray([frame]), "R")
        if rgb is None:
            raise RuntimeError(f"Expected saved eligible crop frame {frame}")
        image = Image.fromarray(rgb[0])
        x, y = (k % 5) * 200, (k // 5) * 230
        gallery.paste(image.resize((200, 200)), (x, y))
        draw.text(
            (x + 3, y + 202),
            f"{g}: {m['gesture_name'][ti][i]}",
            font=font,
            fill="black",
        )
        draw.text((x + 3, y + 217), f"frame {frame} R", font=font, fill="#475569")
        crop_records.append(
            dict(
                gesture_id=g,
                name=str(m["gesture_name"][ti][i]),
                frame=frame,
                center_id=str(m["ids"][ti][i]),
            )
        )
    gallery.save(figdir / "diverse_crops.png")
    figures.append(
        dict(
            id="diverse_crops",
            file="figures/diverse_crops.png",
            title="Actual right-hand crops across all 25 active fitting gestures",
            explanation="One crop is selected deterministically: the first eligible fitting center for each active raw gesture ID. The crops preserve stored right-hand orientation and RGB colors; thumbnails are resized only for the document gallery, not for teacher inference. These are small inspection previews read directly from the original LMDB, without copying the database or rerunning the teacher.",
            source="Per-episode LMDB; teacher_cache.npz frame mapping; gallery_identifiers.json",
        )
    )
    for pool in ("reference", "RGB"):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for cid in ["M1", "M2"] if pool == "reference" else ["M4", "M5"]:
            a = np.load(run / f"parameters/{cid}.npz")
            ax.plot(
                jx, np.rad2deg(a["offsets"]), "o-", label=NAMES[cid], color=COLORS[cid]
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(
            xticks=jx,
            xticklabels=[f"J{j:02d}" for j in jx],
            ylabel="Fitted offset (degrees)",
        )
        ax.legend()
        save(
            fig,
            f"offsets_{pool}",
            f"Fitted finger offsets: {pool} supervision",
            "Offset fitting uses lambda_b = 1, so each offset is half the mean target-minus-baseline residual. Affine intercepts are shown separately because their joint slope/intercept fit differs. Offsets are learned on fitting centers only and are applied unchanged to E. Wrist offsets are always zero.",
            "parameters/*.npz",
        )
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for cid in ["M2", "M5"]:
        a = np.load(run / f"parameters/{cid}.npz")
        ax.plot(jx, a["scales"], "o-", label=NAMES[cid], color=COLORS[cid])
    ax.axhline(1, color="gray", lw=0.7)
    ax.set(xticks=jx, xticklabels=[f"J{j:02d}" for j in jx], ylabel="Affine scale")
    ax.legend()
    save(
        fig,
        "affine_scales",
        "Fitted diagonal affine scales",
        "Identity is scale 1 and offset 0. The float64 convex solve uses lambda_a = 0.01 and lambda_b = 1.0. All 20 joints in both conditions have sufficient prediction variation; no low-variation fallback was used. Slopes are not clipped or revised based on evaluation MAE.",
        "parameters/M2.npz; parameters/M5.npz",
    )
    for cid in ["M3", "M6"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for seed in SEEDS:
            log = read_json(run / f"parameters/{cid}_seed{seed}.json")
            loss = log["losses"]
            epochs = [v["epoch"] for v in loss]
            axes[0].plot(epochs, [v["objective"] for v in loss], label=str(seed))
            axes[1].plot(epochs, [v["mse"] for v in loss], label=str(seed))
        axes[0].set(
            title="Regularized objective",
            xlabel="Epoch",
            ylabel="Loss (radians squared)",
        )
        axes[1].set(title="Finger MSE", xlabel="Epoch", ylabel="MSE (radians squared)")
        axes[0].legend(title="Seed")
        save(
            fig,
            f"loss_{cid}",
            f"Head training history: {NAMES[cid]}",
            "Every seed starts from released finger rows and trains exactly 50 epochs. Adam uses lr = 1e−3 and weight_decay = 0. The objective adds 1e−3 times mean squared weight deviation and 1e−3 times mean squared bias deviation. These are training histories; no evaluation score selected an epoch or learning rate.",
            "parameters/*_seed*.json losses",
        )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    labels = []
    wd = []
    bd = []
    for cid in ["M3", "M6"]:
        for seed in SEEDS:
            log = read_json(run / f"parameters/{cid}_seed{seed}.json")
            labels.append(f"{cid}/{seed}")
            wd.append(log["weight_rms_deviation"])
            bd.append(log["bias_rms_deviation"])
    axes[0].bar(labels, wd, color=["#dc2626"] * 3 + ["#ea580c"] * 3)
    axes[1].bar(labels, bd, color=["#dc2626"] * 3 + ["#ea580c"] * 3)
    axes[0].set_ylabel("Weight RMS deviation")
    axes[1].set_ylabel("Bias RMS deviation")
    [ax.tick_params(axis="x", rotation=45) for ax in axes]
    save(
        fig,
        "head_deviation",
        "How far each head moved from released parameters",
        "The bars are RMS deviations restricted to the updated finger rows/biases. They quantify parameter movement, not statistical uncertainty. The optimizer contains no wrist rows or backbone parameters. The full model has 6,617,382 parameters; each head condition updates 10,260 (about 0.155%).",
        "parameters/*_seed*.json",
    )
    rows = s["metrics"]["primary"]
    ids = [v["id"] for v in rows]
    scores = [v["mae_deg"] for v in rows]
    errors = [v["std_deg"] for v in rows]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.bar(ids, scores, color=[COLORS[v] for v in ids], yerr=errors, capsize=4)
    ax.set(ylabel="Evaluation finger MAE (degrees)", ylim=(0, 18))
    [
        ax.text(i, y + 0.2, f"{y:.4f}", ha="center", fontsize=9)
        for i, y in enumerate(scores)
    ]
    save(
        fig,
        "primary_mae",
        "Primary seven-condition comparison",
        "All conditions score the same 3,654 E centers and 73,080 finger entries. The primary metric is mean absolute angular difference without wrapping, clipping or smoothing. Head bars use the mean of each seed’s MAE, with sample SD across seeds; this is not the MAE of an averaged predictor or a population confidence interval.",
        "metrics.json primary",
    )
    fig, ax = plt.subplots(figsize=(9, 3.8))
    delta = [v["delta_deg"] for v in rows[1:]]
    ax.barh(
        ids[1:],
        delta,
        color=[COLORS[v] for v in ids[1:]],
        xerr=[v["std_deg"] for v in rows[1:]],
        capsize=3,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Adapted MAE − baseline MAE (degrees)")
    ax.invert_yaxis()
    save(
        fig,
        "baseline_delta",
        "Small analytic changes and worse head results",
        "Negative values improve MAE; positive values worsen it. RGB affine has the lowest observed MAE, 15.332166°, only 0.017060° below baseline. Reference head worsens by 0.288818° on average and RGB head by 0.135257°. These small analytic differences are descriptive, not evidence of population-level improvement.",
        "metrics.json primary",
    )
    fig, ax = plt.subplots(figsize=(9, 3.8))
    seed_rows = [v for v in s["metrics"]["all_seeds"] if v["method"] == "head"]
    x = np.arange(3)
    for cid, offset in [("M3", -0.15), ("M6", 0.15)]:
        values = [
            next(
                v["mae_deg"] for v in seed_rows if v["id"] == cid and v["seed"] == seed
            )
            for seed in SEEDS
        ]
        ax.scatter(x + offset, values, color=COLORS[cid], s=65, label=NAMES[cid])
        ax.plot(x + offset, values, color=COLORS[cid], alpha=0.5)
    ax.axhline(scores[0], color=COLORS["M0"], ls="--", label="Frozen baseline")
    ax.set(
        xticks=x, xticklabels=SEEDS, ylabel="Evaluation MAE (degrees)", xlabel="Seed"
    )
    ax.legend()
    save(
        fig,
        "seed_results",
        "Every head seed versus baseline",
        "All six final-epoch head runs are shown. No best seed was retained. Every reference-head and RGB-head seed is worse than the frozen baseline on this E pool, despite optimization on the fitting population. Seed variability is not an independent-participant estimate.",
        "metrics.json all_seeds",
    )

    def detail_values(kind, cid, groups):
        keys = [f"{cid}_seed{seed}" for seed in SEEDS] if cid in ["M3", "M6"] else [cid]
        return np.mean(
            [
                [
                    float(
                        next(
                            v["mae_deg"]
                            for v in s["detailed"]
                            if v["key"] == key and v["kind"] == kind and v["group"] == g
                        )
                    )
                    for g in groups
                ]
                for key in keys
            ],
            axis=0,
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    groups = ["user", "gesture", "both"]
    x = np.arange(3)
    for i, cid in enumerate(ids):
        ax.bar(
            x + (i - 3) * 0.1,
            detail_values("split", cid, groups),
            width=0.095,
            label=cid,
            color=COLORS[cid],
        )
    ax.set(
        xticks=x,
        xticklabels=["user (2,853)", "gesture (516)", "both (285)"],
        ylabel="Finger MAE (degrees)",
    )
    ax.legend(ncol=7, fontsize=8)
    save(
        fig,
        "split_results",
        "Results by evaluation split",
        "Each split uses exactly the same centers across methods. The pooled result weights splits by their valid finger entries: user dominates because it supplies most centers. Head split values are means of seed-specific split MAEs. Split labels follow row metadata; they are not a proof of unseen-user generalization for the released checkpoint.",
        "detailed_metrics.csv",
    )
    joint_matrix = np.stack([detail_values("joint", cid, joints) for cid in ids])
    fig, ax = plt.subplots(figsize=(9, 3.7))
    im = ax.imshow(joint_matrix, aspect="auto", cmap="magma")
    ax.set(
        yticks=np.arange(7),
        yticklabels=ids,
        xticks=jx,
        xticklabels=[f"J{j:02d}" for j in jx],
    )
    fig.colorbar(im, ax=ax, label="MAE (degrees)")
    save(
        fig,
        "joint_heatmap",
        "Absolute MAE across methods and finger joints",
        "The heatmap separates absolute error magnitude from baseline-relative changes. Columns use the J00–J19 joint mapping. Head rows are means of per-seed joint MAEs; their predictions are not averaged before computing this table. Wrist errors are excluded from these finger metrics.",
        "detailed_metrics.csv joint rows",
    )
    gesture_groups = [
        v["group"] for v in s["detailed"] if v["kind"] == "gesture" and v["key"] == "M0"
    ]
    for part in range(2):
        gg = gesture_groups[part * 18 : (part + 1) * 18]
        mat = np.stack([detail_values("gesture", cid, gg) for cid in ids])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set(
            yticks=np.arange(7),
            yticklabels=ids,
            xticks=np.arange(len(gg)),
            xticklabels=[g.rsplit(":active=", 1)[0] for g in gg],
        )
        ax.tick_params(axis="x", rotation=65, labelsize=8)
        fig.colorbar(im, ax=ax, label="MAE (degrees)")
        save(
            fig,
            f"gesture_heatmap_{part+1}",
            f"Evaluation action error matrix, panel {part+1}",
            "Rows are the same seven methods and columns are the recorded evaluation actions. Inactive centers remain a separate group. A color difference describes this recording only; unequal gesture sample counts and overlapping windows prevent treating columns as independent experiment replications.",
            "detailed_metrics.csv gesture rows",
        )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ref = ev["reference"][:, :20]
    for cid in ["M0", "M2", "M3", "M5", "M6"]:
        # Head distribution here averages per-center seed errors, not predictions.
        ee = (
            np.mean(
                [
                    np.abs(ev[f"{cid}_seed{seed}"][:, :20] - ref).mean(axis=1)
                    for seed in SEEDS
                ],
                axis=0,
            )
            if cid in ["M3", "M6"]
            else np.abs(ev[cid][:, :20] - ref).mean(axis=1)
        )
        values = np.sort(np.rad2deg(ee))
        axes[0].plot(
            values, np.linspace(0, 1, len(values)), label=cid, color=COLORS[cid]
        )
        axes[1].boxplot(
            np.rad2deg(ee),
            positions=[["M0", "M2", "M3", "M5", "M6"].index(cid)],
            widths=0.5,
            showfliers=False,
        )
    axes[0].set(
        xlabel="Per-center mean finger error (degrees)",
        ylabel="Empirical cumulative fraction",
    )
    axes[0].legend()
    axes[1].set(
        xticks=np.arange(5),
        xticklabels=["M0", "M2", "M3", "M5", "M6"],
        ylabel="Per-center error (degrees)",
    )
    save(
        fig,
        "error_distribution",
        "Distribution of per-center evaluation errors",
        "Left: empirical CDF of per-center mean finger error. Right: median/interquartile boxes with outliers hidden for readability. All centers still contribute to MAE. Head values average seed-specific per-center errors. These distributions show error heterogeneity and are not independent-sample confidence analyses.",
        "evaluation_predictions.npz",
    )
    trace_records = read_json(run / "trace_identifiers.json")
    trace_used = []
    for trace in trace_records:
        ix = np.flatnonzero(np.isin(ev["ids"], trace["ids"]))
        t = (ev["centers"][ix] - ev["centers"][ix[0]]) / 2000
        fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
        for ax, j in zip(axes, [0, 9, 17]):
            ax.plot(
                t,
                np.rad2deg(ev["reference"][ix, j]),
                label="Reference",
                color="black",
                lw=1.8,
            )
            for cid in ids:
                ax.plot(
                    t,
                    np.rad2deg(condition_prediction(ev, cid)[ix, j]),
                    label=cid,
                    color=COLORS[cid],
                    lw=1 if cid in ["M0", "M3", "M6"] else 0.6,
                    alpha=0.9,
                )
            ax.set_ylabel("Degrees")
            ax.set_title(joints[j], fontsize=10)
        axes[0].legend(ncol=4, fontsize=8)
        axes[-1].set_xlabel("Nominal seconds from first selected center")
        save(
            fig,
            f'traces_{trace["split"]}',
            f'Additional angle traces: {trace["split"]} split, gesture {trace["gesture_id"]}',
            "The same deterministic contiguous occurrence used in the original trace is shown at three additional preselected joints: thumb CMC flexion, middle MCP flexion and pinky MCP flexion. Trace selection is chronological and does not depend on improvement. Head lines average predictions across seeds for display only; headline MAEs remain averages of separate seed scores. Lines never bridge unavailable time.",
            "evaluation_predictions.npz; trace_identifiers.json",
        )
        trace_used.append(trace)
    for cid in ["M3", "M6"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        log = [read_json(run / f"parameters/{cid}_seed{seed}.json") for seed in SEEDS]
        for ax, key, title in [
            (axes[0], "fit_target_mae", "Error against fitting target"),
            (axes[1], "fit_reference_mae", "Error against fitting reference"),
        ]:
            ax.bar(
                [str(seed) for seed in SEEDS],
                [v[key]["mae_deg"] for v in log],
                color=COLORS[cid],
            )
            ax.set(title=title, xlabel="Seed", ylabel="MAE (degrees)")
        save(
            fig,
            f"fit_residual_{cid}",
            f"Head fitting residual diagnostics: {NAMES[cid]}",
            "The left panel uses each condition’s designated target; RGB fitting targets are teacher angles. The right panel uses reference labels strictly as post-fit diagnostics. These are fitting-pool scores, not evaluation performance or tuning signals. Lower fitting residuals do not establish better evaluation MAE.",
            "parameters/*_seed*.json fit_target_mae and fit_reference_mae",
        )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    rt = [s["emg"]["extraction_seconds"], s["teacher"]["construction_seconds"]] + [
        np.mean(
            [
                v["fitting_seconds"]
                for v in s["conditions"]["conditions"]
                if v["id"] == cid
            ]
        )
        for cid in ["M3", "M6"]
    ]
    mem = [
        s["emg"]["peak_gpu_bytes"] / 2**20,
        s["teacher"]["peak_gpu_bytes"] / 2**20,
    ] + [
        np.mean(
            [
                v["peak_gpu_bytes"]
                for v in s["conditions"]["conditions"]
                if v["id"] == cid
            ]
        )
        / 2**20
        for cid in ["M3", "M6"]
    ]
    labels = ["EMG cache", "RGB cache", "M3 / seed", "M6 / seed"]
    axes[0].bar(labels, rt, color=["#0284c7", "#7c3aed", "#dc2626", "#ea580c"])
    axes[1].bar(labels, mem, color=["#0284c7", "#7c3aed", "#dc2626", "#ea580c"])
    axes[0].set_ylabel("Seconds")
    axes[1].set_ylabel("Peak allocated GPU MiB")
    [ax.tick_params(axis="x", rotation=25) for ax in axes]
    save(
        fig,
        "runtime_memory",
        "Shared inference cost and per-seed fitting cost",
        "EMG and RGB cache construction are shared costs. Head bars are mean fitting cost per seed, not totals for all seeds. GPU peaks use torch.cuda.max_memory_allocated after each stage/seed reset and exclude allocator-reserved/driver memory, CPU caches and other processes. Warm operating-system caches may affect elapsed times; these are observed run costs, not hardware benchmarks.",
        "emg_evidence.json; teacher_diagnostics.json; per-condition logs",
    )
    for name, title, explanation in [
        (
            "per_joint",
            "Original per-joint MAE-change figure",
            "This is the original completed-run figure. Each value is adapted minus baseline MAE for one finger joint. Head lines average the three per-seed joint scores. Larger positive changes in several index, ring and pinky joints explain the head aggregate degradation.",
        ),
        (
            "per_gesture",
            "Original per-gesture MAE overview",
            "This original figure preserves the full raw-ID/name/active grouping. The enlarged action heatmaps elsewhere make the many labels easier to inspect. The inactive −1 group is not assigned an invented gesture name.",
        ),
        (
            "time_traces",
            "Original deterministic index-MCP traces",
            "The original figure shows index_mcp_fe at the first available active gesture occurrence in each evaluation split. The final version stops at gaps and uses only contiguous centers. Head curves average predictions for display; final MAEs are per-seed metrics.",
        ),
        (
            "crop_contact_sheet",
            "Original twelve-crop readiness contact sheet",
            "These twelve actual right-hand crops are the earliest eligible training frames, before any error-based selection. They verify crop decoding and stored orientation. They do not represent every training gesture; the additional 25-gesture gallery does.",
        ),
    ]:
        shutil.copyfile(run / f"{name}.png", figdir / f"original_{name}.png")
        figures.append(
            dict(
                id=f"original_{name}",
                file=f"figures/original_{name}.png",
                title=title,
                explanation=explanation,
                source=f"{run.name}/{name}.png",
            )
        )
    (out / "figure_manifest.json").write_text(json.dumps(figures, indent=2) + "\n")
    (out / "gallery_identifiers.json").write_text(
        json.dumps(crop_records, indent=2) + "\n"
    )
    (out / "input_preview_identifiers.json").write_text(
        json.dumps(preview_records, indent=2) + "\n"
    )
    (out / "source_identity.json").write_text(
        json.dumps(
            dict(
                source_run=str(run),
                source_files={
                    p.name: digest(p)
                    for p in run.iterdir()
                    if p.is_file() and p.suffix in [".json", ".npz", ".csv", ".png"]
                },
                builder_sha256=digest(Path(__file__)),
                created_jst=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                figures=len(figures),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Generated {len(figures)} figures in {out}", flush=True)


def build_docx(run, out):
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
    from PIL import Image

    s = load_sources(run)
    m, c, tc, ev = (s[k] for k in ("manifest", "cache", "teacher_cache", "evaluation"))
    figures = read_json(out / "figure_manifest.json")
    figure_lookup = {v["id"]: v for v in figures}
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.32)
    styles = doc.styles
    for name in ["Normal", "Body Text", "Caption", "Table Grid"]:
        styles[name].font.name = "Calibri"
        styles[name].font.size = Pt(10)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Normal"].paragraph_format.line_spacing = 1.12
    for name, size in [
        ("Title", 29),
        ("Subtitle", 15),
        ("Heading 1", 19),
        ("Heading 2", 13),
        ("Heading 3", 11),
    ]:
        styles[name].font.name = "Calibri"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor.from_string("16324F")
    styles["Caption"].font.size = Pt(9)
    styles["Caption"].font.color.rgb = RGBColor.from_string("334155")
    header = section.header.paragraphs[0]
    header.text = "EgoEMG  |  Single-participant offline calibration"
    header.style = "Caption"
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Preview recording study  •  Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.title = (
        "Single-participant Offline Calibration: Detailed Work Report"
    )
    doc.core_properties.subject = (
        "EgoEMG wmh episode_000020 right-hand calibration suite"
    )
    doc.core_properties.author = "EgoEMG Research"
    doc.core_properties.keywords = (
        "EMGFormer; RGB teacher; offline calibration; reproducibility; wmh"
    )
    table_number = 0
    figure_number = 0
    inserted_figures = []
    tables = []

    def para(text, style=None):
        return doc.add_paragraph(text, style)

    def heading(text, level=1, page=False):
        if page:
            doc.add_page_break()
        doc.add_heading(text, level)

    def bullet(text):
        para(text, "List Bullet")

    def table(title, headers, rows, explanation, source, font_size=8.5):
        nonlocal table_number
        table_number += 1
        p = para(f"Table {table_number}. {title}", "Caption")
        p.paragraph_format.keep_with_next = True
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = True
        for cell, label in zip(t.rows[0].cells, headers):
            cell.text = str(label)
        repeat = OxmlElement("w:tblHeader")
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for values in rows:
            cells = t.add_row().cells
            for cell, val in zip(cells, values):
                cell.text = "—" if val is None else str(val)
        for ri, row in enumerate(t.rows):
            pr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            pr.append(cant)
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p0 in cell.paragraphs:
                    p0.paragraph_format.space_after = Pt(2)
                    p0.paragraph_format.space_before = Pt(2)
                    for rr in p0.runs:
                        rr.font.size = Pt(font_size)
                        if ri == 0:
                            rr.bold = True
                            rr.font.color.rgb = RGBColor(255, 255, 255)
                if ri == 0 or ri % 2 == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "16324F" if ri == 0 else "F0F5FA")
                    cell._tc.get_or_add_tcPr().append(shade)
        para(explanation)
        p = para("Source: " + source, "Caption")
        p.paragraph_format.space_after = Pt(10)
        tables.append(
            dict(
                number=table_number,
                title=title,
                rows=len(rows),
                columns=len(headers),
                source=source,
            )
        )

    def figure(fid):
        nonlocal figure_number
        v = figure_lookup[fid]
        path = out / v["file"]
        w, h = Image.open(path).size
        width = min(6.8, 6.7 * w / h)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(width))
        figure_number += 1
        p = para(f'Figure {figure_number}. {v["title"]}', "Caption")
        p.paragraph_format.keep_with_next = True
        p = para(v["explanation"])
        p.paragraph_format.keep_with_next = True
        para("Source: " + v["source"], "Caption")
        inserted_figures.append(dict(number=figure_number, **v))

    def formula(text, explanation):
        p = para(text)
        p.paragraph_format.left_indent = Inches(0.15)
        for rr in p.runs:
            rr.font.name = "Consolas"
            rr.font.size = Pt(9)
        para(explanation)

    def fnum(x, n=6):
        return f"{float(x):.{n}f}"

    def score_values(kind, cid, group):
        keys = [f"{cid}_seed{seed}" for seed in SEEDS] if cid in ["M3", "M6"] else [cid]
        return np.mean(
            [
                float(
                    next(
                        v["mae_deg"]
                        for v in s["detailed"]
                        if v["kind"] == kind and v["key"] == key and v["group"] == group
                    )
                )
                for key in keys
            ]
        )

    ids = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    primary = s["metrics"]["primary"]
    joints = s["settings"]["identity"]["finger_joints"]
    source_run_posix = str(run)
    root_posix = "/home/potaxo/research/egoemg/EgoEMG"

    doc.add_heading("Single-participant\noffline calibration", 0)
    doc.add_paragraph(
        "Detailed implementation, execution, and evidence report", "Subtitle"
    )
    para("EgoEMG preview • wmh • episode_000020 • right hand")
    para(f"Completed experiment: 5 October 2026 (Asia/Tokyo). Source run: {run.name}.")
    para(
        "This document explains what went into the experiment, what was implemented and run, what came out, and what the measured results support. It includes the full important tables, original experiment figures, and additional plots generated from saved evidence. All numerical outcomes come from the completed suite; no model was refitted for this report."
    )
    table(
        "At a glance",
        ["Item", "Observed result"],
        [
            ["Primary conditions", "All seven completed; no blocked conditions"],
            ["Fitting population", "4,174 train centers; 25 active gestures"],
            [
                "RGB availability",
                "4,174 / 4,174 eligible centers and unique crop frames",
            ],
            ["Evaluation population", "3,654 fixed centers; 34 active gestures"],
            ["Frozen baseline finger MAE", "15.349226°"],
            ["Lowest observed finger MAE", "M5 RGB affine: 15.332166° (−0.017060°)"],
            [
                "Head adaptation outcome",
                "Both supervision variants worsened MAE at every seed",
            ],
            [
                "Teacher quality on fitting centers",
                "1.760795° finger MAE against reference",
            ],
            ["Wrist MAE", "12.704364°, identical in every condition"],
            [
                "Verification",
                "44 focused/relevant tests passed; cached scores independently recalculated",
            ],
        ],
        "This is a descriptive within-participant preview-recording engineering study. The lowest score is an outcome, not a hyperparameter-selection rule or an established population improvement.",
        "metrics.json; coverage.json; teacher_diagnostics.json; verification.json",
    )
    heading("Contents", page=True)
    for text in [
        "1. Goal, scope and completed work",
        "2. Inputs, environment and model interfaces",
        "3. Frozen data protocol and population coverage",
        "4. RGB teacher and supervision quality",
        "5. Adaptation methods and learned parameters",
        "6. Evaluation results and interpretation",
        "7. Runtime, verification and engineering controls",
        "8. Outputs and reproducibility",
        "9. Limitations and practical conclusions",
        "Appendix A. Complete interval and gesture coverage",
        "Appendix B. Complete joint, action and reference-range tables",
        "Appendix C. Provenance, commands and artifact inventory",
    ]:
        para(text)
    para(
        "Figures and tables appear next to their discussion. The appendices retain the complete coverage and numerical detail so that a reader can inspect the work without reopening the dataset or loading either model."
    )

    heading("1. Goal, scope and completed work", page=True)
    para(
        "The task extended an earlier 15-second reference-supervised offset smoke test into a complete offline comparison over every eligible train interval of one participant in one preview recording. The earlier short segment could contain only one gesture; this suite therefore removed a short-duration calibration budget and used all available fitting data under a frozen validity protocol."
    )
    para(
        "The comparison contains a frozen EMGFormer-M baseline and three independent finger adaptation methods, each fitted either to reference labels or to RGB-teacher predictions. Evaluation predictions are EMG-only. All seven conditions use exactly the same held-out row-split evaluation population from this recording."
    )
    table(
        "Condition definitions",
        ["ID", "Method", "Supervision", "Fitting pool"],
        [[v["id"], NAMES[v["id"]], v["supervision"], v["pool"]] for v in primary],
        "Reference conditions use the dataset finger angles. RGB conditions use crop-only teacher finger angles. None of the adaptations are stacked or warm-started from another adaptation. The M0 baseline does not fit parameters.",
        "metrics.json primary; conditions.json",
    )
    figure("workflow")
    table(
        "Completed work and boundaries",
        ["Item", "What was done / why it matters"],
        [
            [
                "Inspection",
                "Confirmed branch, interpreter, GPU, package layout, row splits, participant identity, configurations, checkpoints and normalization.",
            ],
            [
                "Implementation",
                "Added a separate resumable suite, focused tests and usage documentation under myself/.",
            ],
            [
                "Protocol",
                "Built a deterministic complete-support manifest before fitting, including unusable intervals and exclusion reasons.",
            ],
            [
                "Shared caches",
                "Extracted frozen EMG predictions and actual last-linear input features; separately ran the crop teacher.",
            ],
            [
                "Fitting",
                "Solved offsets/affine fits and trained finger-only final rows for seeds 42, 43 and 44 at fixed epoch 50.",
            ],
            [
                "Evaluation",
                "Computed pooled, split, gesture, joint and wrist metrics with fixed identifiers and masks.",
            ],
            [
                "Reporting",
                "Saved readable tables, raw recalculation data, parameter/log files, diagnostics, original figures and this detailed Word report.",
            ],
            [
                "Outside scope",
                "No FiLM, REACT reproduction, backbone fine-tuning, online/causal processing, raw-image detector, extra participants, MANO conversion or unified-data repair.",
            ],
        ],
        "The scope keeps the comparison focused on simple output calibration and permitted final-layer updates. Better MAE was not a completion requirement; unfavorable results were retained.",
        "Original task specification; suite source; completed phase markers",
    )
    para(
        "Historical context: the earlier smoke test reported 2.7825° baseline and 2.8613° after offset correction on 300 evaluation centers. It used a different, short within-train selection. Those numbers are not inserted into the new E comparison and cannot be directly compared with the 15.349226° baseline here. The original smoke workflow and archived artifacts were preserved."
    )

    heading("2. Inputs, environment and model interfaces", page=True)
    table(
        "External and repository inputs",
        ["Input", "Actual path or value", "Purpose"],
        [
            ["Checkout", root_posix, "Original code, configurations and assets"],
            [
                "Branch / Git revision",
                s["settings"]["git_branch"] + " / " + s["settings"]["git_revision"],
                "Reproducible source identity; no commit/push/branch change",
            ],
            [
                "Interpreter",
                s["settings"]["interpreter"],
                "Existing egoemg environment; not recreated",
            ],
            [
                "Preview package",
                "/mnt/d/egoemg/data/dataset_egoemg_preview",
                "Designated D: data inputs",
            ],
            [
                "Memmap root",
                s["settings"]["identity"]["data_root"],
                "Manifest, metadata and row-level arrays",
            ],
            [
                "Crop root",
                "/mnt/d/egoemg/data/dataset_egoemg_preview/data/pre-crop_webcam_videoframes",
                "Original per-episode LMDB crops",
            ],
            [
                "Video root",
                "/mnt/d/egoemg/data/dataset_egoemg_preview/data/webcam_videos",
                "Located from package; raw video inference not needed",
            ],
            [
                "EMG checkpoint",
                root_posix + "/checkpoints/egoemg_emgformer_middle.ckpt",
                "Frozen released EMGFormer-M",
            ],
            [
                "Teacher checkpoint",
                root_posix + "/checkpoints/vision_resnet18.ckpt",
                "Frozen released ResNet18 crop teacher",
            ],
            [
                "Normalization",
                root_posix
                + "/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json",
                "Right-hand key egoemg__filtered_paper_right",
            ],
            [
                "Source output run",
                source_run_posix,
                "Immutable evidence used by this document",
            ],
        ],
        "Dataset inputs are read directly from the designated data disk. No large array, raw window cache, video or LMDB copy was staged into the repository. New workflow and output artifacts reside under myself/. Metadata and checkpoint hashes are retained in settings.json.",
        "settings.json; package manifest; repository AGENTS.md",
        font_size=8,
    )
    table(
        "Runtime environment",
        ["Item", "Recorded value"],
        [
            ["GPU", s["settings"]["device"]],
            [
                "CUDA / precision",
                str(s["settings"]["cuda"]) + " / " + s["settings"]["precision"],
            ],
            ["Python", s["settings"]["python"].splitlines()[0]],
            [
                "Git working tree",
                "Dirty only with the three newly added personal suite files at run start",
            ],
            *[[name, version] for name, version in s["settings"]["packages"].items()],
        ],
        "Version values are runtime evidence from the completed suite, rather than expected values copied from the task prompt. The environment used one GPU; EMG and teacher models had separate inference lifetimes.",
        "settings.json",
    )
    table(
        "Recording and model input contract",
        ["Item", "Recorded convention"],
        [
            ["Participant / hand / recording", "wmh / right / episode_000020"],
            [
                "Participant resolution",
                "By recording name and episode_subject; metadata subject ID "
                + str(s["coverage"]["metadata_subject_id"])
                + "; row subject IDs "
                + str(s["coverage"]["row_subject_ids"]),
            ],
            [
                "EMG input",
                "emg_right_filtered_paper; eight stored right-hand channels in existing order",
            ],
            [
                "Nominal sample rate",
                "2,000 rows per second; timestamp spacing need not be uniform",
            ],
            [
                "Window / stride",
                "12,000 rows (6 nominal seconds) / 200 rows (10 Hz nominal centers)",
            ],
            [
                "Loader layout",
                "T,C stored input → normalized C,T model input; batch B,8,12000",
            ],
            [
                "Augmentation",
                "No jitter, rotation, interpolation of channels or feature/label augmentation",
            ],
            [
                "Supervision",
                "Finger labels at input center only; wrist validity separate",
            ],
            [
                "Angle units",
                "Finger labels already radians; stored wrist degrees converted to radians",
            ],
            [
                "Joint layout",
                "Outputs 0–19 finger angles, output 20 wrist pitch, output 21 wrist yaw",
            ],
        ],
        "The eight input channels are signals, whereas the 22 outputs are joint/wrist angles. The nominal sample rate defines row-based durations only; the report does not assert uniform measured timestamps. Subject integer schemes need not match, so the name/recording link is authoritative.",
        "settings.json; coverage.json; resolved_dataset_config.yaml",
    )
    table(
        "Normalization values",
        ["Stored channel", "Mean", "Standard deviation"],
        [
            [i, f"{a:.9g}", f"{b:.7f}"]
            for i, (a, b) in enumerate(
                zip(
                    s["settings"]["identity"]["normalization"]["mean"],
                    s["settings"]["identity"]["normalization"]["std"],
                )
            )
        ],
        "Each channel is normalized independently with the supplied right-hand asset. These values were not estimated from T_all, T_rgb or E. The near-zero means and unequal standard deviations must not be replaced with unified-training keys by convenience.",
        "settings.json identity.normalization",
    )
    figure("normalization")
    figure("emg_input_T_all")
    figure("emg_input_E")
    figure("architecture")
    table(
        "Frozen feature and alignment evidence",
        ["Item", "Value / explanation"],
        [
            [
                "Strict state load",
                "All active model parameters matched; no missing trained parameters accepted",
            ],
            [
                "Historical import paths",
                "Local equivalent architecture instantiated; old emg2pose strings were not imported",
            ],
            ["Final angle layer", s["emg"]["layer_name"]],
            ["Feature layout", s["emg"]["layer_input_layout"]],
            ["Feature dimension", s["emg"]["feature_dimension"]],
            ["Total model parameters", s["emg"]["parameters"]],
            [
                "Frozen modules",
                "eval() + inference mode for encoder, temporal decoder and preceding head MLP",
            ],
            ["Post-linear operations", s["emg"]["post_layer"]],
            [
                "Contexts / raw timeline",
                f"left {s['emg']['left_context']}, right {s['emg']['right_context']}; raw {s['emg']['raw_timeline_length']} steps",
            ],
            [
                "Aligned length / center",
                f"{s['emg']['aligned_length']} / {s['emg']['center_index']}",
            ],
            ["Equivalence tolerance", "atol 1e−5 and rtol 1e−4 in FP32"],
            [
                "Observed maximum difference",
                f"{s['emg']['maximum_absolute_discrepancy']:.9g} radians",
            ],
            [
                "Cached content",
                "Center identifiers, frozen predictions, compact actual last-linear features, released final weight/bias",
            ],
        ],
        "Linear temporal interpolation commutes with the final affine layer here because only reshape/transpose and linear interpolation follow it. This equivalence was checked against the original full aligned forward prediction for every extracted batch before fitting. The feature cache includes the preceding nonlinear MLP operations, rather than a guessed decoder output.",
        "emg_evidence.json; emg_cache.npz",
    )
    formula(
        "p_center = Linear( aligned_center(actual_final_layer_inputs), W0, b0 )",
        "The expression describes the verified compact cache. The original path instead applies the layer across the raw timeline, linearly aligns the full output sequence, and then selects the input center. Their maximum recorded absolute discrepancy is far below the fixed tolerance.",
    )
    table(
        "Output joint mapping",
        ["ID", "Finger / output", "Joint name", "Units"],
        [
            [f"J{j:02d}", JOINT_GROUPS[j], name, "radians"]
            for j, name in enumerate(joints)
        ]
        + [
            ["20", "Wrist", "wrist_pitch", "radians"],
            ["21", "Wrist", "wrist_yaw", "radians"],
        ],
        "J00–J19 are compact labels used in plots and wide numerical tables. FE denotes flexion/extension, AA abduction/adduction; CMC, MCP, PIP, DIP and IP identify the anatomical joint named by the dataset. The supplied manifest joint order is retained.",
        "manifest generated_joint_angles_semantics; settings.json",
    )

    heading("3. Frozen data protocol and population coverage", page=True)
    para(
        "The manifest reads row-level frame_split_id and resolves IDs from metadata: train=0, user=1, gesture=2, both=3. Episode-level preview zeros are placeholders. For each maximal constant-split interval [a,b), starts are a, a+200, a+400, … while start+12000≤b. Centers are start+6000. Local and global start/end/center positions and a stable recording/hand/center identifier are saved."
    )
    figure("window_protocol")
    figure("split_timeline")
    table(
        "Eligibility and exclusion rules",
        ["Item", "Required check", "Why it is needed"],
        [
            [
                "Support containment",
                "Full window remains inside recording and one constant row split",
                "Prevents context-row sharing across fitting and E",
            ],
            [
                "Input finiteness",
                "Every EMG value in the complete window is finite",
                "Avoids invalid model inputs",
            ],
            [
                "Finger center validity",
                "generated_label_valid right-hand flag is true",
                "Provides valid center supervision",
            ],
            [
                "Finger center finiteness",
                "All 20 center reference values finite",
                "Supports equal-weight complete-joint fitting/scoring",
            ],
            [
                "Wrist handling",
                "Separate wrist angle validity and finite checks",
                "Wrist failures do not incorrectly redefine finger supervision",
            ],
            [
                "Pool separation",
                "Direct support-union intersection check",
                "Interleaved time does not require chronological ordering",
            ],
            [
                "Selection independence",
                "No baseline error or teacher/reference agreement filter",
                "Keeps population selection independent of results",
            ],
        ],
        "Only center labels supervise this experiment. Context finger labels are not required. Exclusion counts use the first failed condition in the order input finiteness, center validity, center finiteness; insufficient interval length is reported separately. Overlap within one pool is allowed.",
        "suite enumerate_windows(); coverage.json",
    )
    table(
        "Pool sizes and duration coverage",
        [
            "Pool",
            "Centers",
            "Active IDs",
            "Inactive centers",
            "Center coverage (s)",
            "Support union (s)",
        ],
        [
            ["T_all", 4174, 25, 4, "417.4", "476.9"],
            ["T_rgb", 4174, 25, 4, "417.4", "476.9"],
            ["E", 3654, 34, 68, "365.4", "489.3"],
        ],
        "Nominal center coverage is the union of [center,center+stride) divided by 2,000. Support coverage is the union of [start,end) divided by 2,000. These durations are row-based, do not double-count overlapping windows, and do not represent independently observed replication time.",
        "coverage.json; teacher_diagnostics.json T_rgb_coverage",
    )
    table(
        "Interval-level availability",
        [
            "Split",
            "Intervals",
            "Candidate centers",
            "Eligible centers",
            "Observed exclusions",
        ],
        [
            [
                "train",
                11,
                4179,
                4174,
                "5 invalid centers in interval 25; interval 31 too short",
            ],
            ["user", 8, 2853, 2853, "None"],
            ["gesture", 8, 516, 516, "None"],
            ["both", 5, 285, 285, "None"],
        ],
        "All 32 row-split intervals are reported. Ten train intervals contain usable windows; the final train fragment has only 111 rows. Every eligible E center remains in the evaluation pool regardless of RGB availability.",
        "coverage.json intervals; manifest.npz",
    )
    figure("interval_counts")
    figure("support_unions")
    figure("gesture_coverage_T_all")
    figure("gesture_coverage_E")
    figure("reference_variation")
    para(
        "The fitting and evaluation active gesture sets are disjoint in this preview recording: 25 fitting IDs and 34 evaluation IDs. The supplied vocabulary maps all selected active IDs; the inactive −1 entries remain explicitly inactive. The study therefore tests transfer of these fitted output changes across the recording’s selected actions as well as across time. This observation does not resolve whether the released checkpoint previously saw the participant or these actions."
    )

    heading("4. RGB teacher and supervision quality", page=True)
    para(
        "The teacher is the released vision-only ResNet18 configuration verified against checkpoint architecture metadata and active state shapes. It takes only a crop; neither EMG nor reference angles enter its prediction path. The checkpoint supplies the complete backbone/head state, so constructor ImageNet initialization downloads are disabled and the active state loads strictly."
    )
    table(
        "Teacher input/output contract",
        ["Item", "Verified convention"],
        [
            [
                "Architecture",
                "ResNet18 → global average pooling → 512-unit MLP → 22 angles",
            ],
            ["Input", "B,3,256,256 right-hand crop; RGB order"],
            ["Crop key", "Eight-digit mapped image_head_frame_index followed by _R"],
            ["Orientation", "Stored right-hand crop unchanged; no mirroring"],
            ["Resize", "None during teacher inference; stored 256×256 shape required"],
            [
                "Normalization mean (0–255 scale)",
                str(s["teacher"]["preprocessing"]["mean"]),
            ],
            [
                "Normalization standard deviation (0–255 scale)",
                str(s["teacher"]["preprocessing"]["std"]),
            ],
            [
                "Outputs",
                "20 finger + 2 wrist angles directly, in the shared radians/joint representation",
            ],
            ["Inference", "FP32; eval() and inference mode; batch size 8"],
            [
                "Unique input reuse",
                "Each unique episode/frame/hand crop read/inferred once during the teacher phase",
            ],
            [
                "Preview loader handling",
                "Existing per-episode reader reused; generic episode_000000.lmdb constructor check bypassed locally",
            ],
        ],
        "The preview contains episode_000020.lmdb, not the generic constructor’s sample episode_000000.lmdb. Reusing the existing per-episode reader resolves that packaging mismatch without copying crops, inventing black replacements or changing global loader semantics. Wrist teacher outputs are cached but are not used to adapt wrist predictions.",
        "resolved_teacher_config.yaml; teacher_diagnostics.json; suite source",
    )
    figure("original_crop_contact_sheet")
    figure("diverse_crops")
    table(
        "RGB availability criteria and observations",
        ["Criterion / reason", "Rule", "Excluded centers"],
        [
            ["Mapping", "Frame index is nonnegative", 0],
            ["Stale", "image_head_stale is false", 0],
            ["Nonfinite delta", "image_head_delta_ms finite", 0],
            ["Timing threshold", "Absolute delta ≤ 50 ms", 0],
            ["Missing crop", "Right-hand LMDB key exists", 0],
            ["Decode", "Stored crop decodes", 0],
            ["Preprocessing", "RGB 256×256 input normalizes successfully", 0],
            ["Nonfinite prediction", "All 20 teacher finger angles finite", 0],
        ],
        "All 4,174 T_all centers pass every predeclared criterion and map to 4,174 unique frames. T_rgb therefore equals T_all. Missing required fields would have blocked RGB readiness rather than authorizing an omitted check. Reference agreement is not an eligibility criterion.",
        "teacher_diagnostics.json; teacher_cache.npz",
    )
    figure("rgb_timing")
    figure("teacher_joint")
    figure("teacher_gesture")
    figure("teacher_scatter")
    para(
        "Teacher aggregate finger MAE is 0.030731674 radians = 1.760795215 degrees over 83,480 fitting entries. This is an on-fitting-pool diagnostic. It does not replace the EMG evaluation metric, validate synchronization beyond the chosen tolerance, or establish teacher quality on the different E actions. There are no E teacher predictions in this workflow."
    )

    heading("5. Adaptation methods and learned parameters", page=True)
    para(
        "All methods adapt only the 20 finger outputs and start independently from the released baseline. Finger targets and predictions remain in radians; every valid center/joint has equal weight. Wrist outputs are copied from the frozen baseline. The fixed E pool is excluded from parameter selection, normalization estimation, early stopping and RGB eligibility."
    )
    table(
        "Predeclared hyperparameters",
        ["Method", "Fixed settings", "Behavior"],
        [
            ["Offset", "lambda_b=1.0", "20 offsets; deterministic analytic fit"],
            [
                "Affine",
                "lambda_a=0.01; lambda_b=1.0; variance floor=1e−6 rad²",
                "20 independent float64 2×2 solves; 40 parameters",
            ],
            [
                "Head optimizer",
                "Adam; lr=1e−3; weight_decay=0",
                "Only finger-row weights and biases in optimizer",
            ],
            [
                "Head schedule",
                "50 fixed epochs; feature batch=256; seeds 42/43/44",
                "Fresh released initialization per seed; fitting samples shuffled",
            ],
            [
                "Head loss",
                "Finger MSE + 1e−3 mean squared weight deviation + 1e−3 mean squared bias deviation",
                "Regularization centered at released finger parameters",
            ],
            [
                "Selection",
                "Final epoch 50",
                "No best-epoch/seed selection or E-based tuning",
            ],
        ],
        "Reference and RGB variants share the same algorithm and hyperparameters. They differ only in the supplied fitting target. Compact adapted parameters are saved rather than complete duplicated model checkpoints.",
        "settings.json identity; per-condition logs",
    )
    heading("5.1 Per-joint regularized offset", 2)
    formula(
        "b_j = mean(target_j − prediction_j) / (1 + lambda_b);  lambda_b = 1",
        "The correction is half the mean residual because the fixed ridge penalty shrinks the offset toward zero. It shifts each output independently without changing its slope. Empty supervision returns zero offset with an explicit zero observation count; the completed run has 4,174 observations for every finger joint.",
    )
    figure("offsets_reference")
    figure("offsets_RGB")
    heading("5.2 Per-joint diagonal affine calibration", 2)
    formula(
        "p′_j = a_j p_j + b_j\nObjective = mean((a_j p_j + b_j − target_j)^2) + 0.01(a_j−1)^2 + 1.0 b_j^2",
        "Identity initialization is a=1,b=0. For X=[prediction,1], residual r=target−prediction and q=[a−1,b], the solver uses (XᵀX/n + diag(0.01,1.0))q = Xᵀr/n in float64. If fitting prediction variance is below 1e−6 rad², the joint keeps scale 1 and uses the regularized offset fallback. Empty supervision returns identity. Neither happened in the completed affine fits.",
    )
    figure("affine_scales")
    a_ref = np.load(run / "parameters/M2.npz")
    a_rgb = np.load(run / "parameters/M5.npz")
    b_ref = np.load(run / "parameters/M1.npz")["offsets"]
    b_rgb = np.load(run / "parameters/M4.npz")["offsets"]
    table(
        "All analytic fitted parameters",
        ["Joint", "M1 b (°)", "M2 a", "M2 b (°)", "M4 b (°)", "M5 a", "M5 b (°)"],
        [
            [
                f"J{j:02d}",
                fnum(np.rad2deg(b_ref[j]), 4),
                fnum(a_ref["scales"][j], 5),
                fnum(np.rad2deg(a_ref["offsets"][j]), 4),
                fnum(np.rad2deg(b_rgb[j]), 4),
                fnum(a_rgb["scales"][j], 5),
                fnum(np.rad2deg(a_rgb["offsets"][j]), 4),
            ]
            for j in range(20)
        ],
        "All fits have 4,174 observations per joint. Scales are dimensionless; offsets are shown in degrees for readability but saved/applied in radians. All affine fallback flags are none. Reference scales range from 0.935693 to 1.028333; RGB scales from 0.927438 to 1.010039.",
        "parameters/M1.npz, M2.npz, M4.npz, M5.npz",
        font_size=8,
    )
    heading("5.3 Last-output-linear finger-row fine-tuning", 2)
    para(
        "The actual final layer is head.net.3, with weight shape 22×512 and bias shape 22. The optimizer owns an independent 20×512 weight parameter and 20 bias values, initialized from released rows 0–19. Frozen wrist rows cannot receive an effective update because they are absent from the optimizer. Encoder, decoder and preceding head layers remain frozen and in evaluation mode; optimization consumes the CPU center-feature cache in small batches."
    )
    figure("loss_M3")
    figure("loss_M6")
    figure("head_deviation")
    figure("fit_residual_M3")
    figure("fit_residual_M6")
    head_logs = [v for v in s["conditions"]["conditions"] if v["method"] == "head"]
    table(
        "Every head fit and its final deviations",
        [
            "Run",
            "Fit target MAE (°)",
            "Fit reference MAE (°)",
            "Weight RMS Δ",
            "Bias RMS Δ",
            "Fit seconds",
        ],
        [
            [
                v["key"],
                fnum(v["fit_target_mae"]["mae_deg"], 5),
                fnum(v["fit_reference_mae"]["mae_deg"], 5),
                fnum(v["weight_rms_deviation"], 7),
                fnum(v["bias_rms_deviation"], 7),
                fnum(v["fitting_seconds"], 3),
            ]
            for v in head_logs
        ],
        "These are fitting diagnostics. M3’s target is reference; M6’s target is teacher. Training and final parameters passed finite checks, and every fit updated exactly 10,260 parameters for 50 epochs. Parameter deviation is measured from released finger rows; it does not describe wrist or backbone changes.",
        "conditions.json; parameters/*_seed*.json",
        font_size=8,
    )

    heading("6. Evaluation results and interpretation", page=True)
    para(
        "The primary metric is absolute angular MAE over all valid evaluation finger entries, with no wrapping, clipping, smoothing or post-hoc gesture balancing. All seven conditions score the same 3,654 centers and 73,080 finger entries using the same reference labels and masks. Radians are converted to degrees by multiplying by 180/π."
    )
    table(
        "Primary seven-condition numerical results",
        [
            "ID",
            "Supervision / pool",
            "MAE (rad)",
            "MAE (°)",
            "Δ baseline (°)",
            "Updated",
        ],
        [
            [
                v["id"],
                v["supervision"] + " / " + v["pool"],
                fnum(v["mae_rad"], 9),
                fnum(v["mae_deg"], 6)
                + (f" ± {v['std_deg']:.6f}" if v["method"] == "head" else ""),
                f"{v['delta_deg']:+.6f}",
                v["updated_parameters"],
            ]
            for v in primary
        ],
        "Delta is adapted minus baseline; negative values improve MAE. Head rows report the mean of the three independent final-epoch seed scores and sample SD (ddof=1). They do not use the best seed. The aggregate is center/joint weighted, not a gesture macro-average.",
        "metrics.json primary",
        font_size=8,
    )
    figure("primary_mae")
    figure("baseline_delta")
    table(
        "Every final-epoch head seed",
        ["Run", "Seed", "MAE (rad)", "MAE (°)", "Δ baseline (°)"],
        [
            [
                v["id"],
                v["seed"],
                fnum(v["mae_rad"], 9),
                fnum(v["mae_deg"]),
                f"{v['delta_deg']:+.6f}",
            ]
            for v in s["metrics"]["all_seeds"]
            if v["method"] == "head"
        ],
        "M3 seeds 42/43/44 average 15.638044° with SD 0.021868°. M6 seeds average 15.484483° with SD 0.013651°. Every head seed is worse than baseline. These seed differences are not participant-level confidence intervals, and overlapping windows/joints are not independent experiment replications.",
        "metrics.json all_seeds; per_seed_metrics.csv",
    )
    figure("seed_results")
    table(
        "Matched-supervision comparison",
        ["Method", "Reference fitting pool", "RGB fitting pool", "Additional refit?"],
        [
            ["Offset", "T_rgb = T_all", "T_rgb", "No; M1 reused beside M4"],
            ["Affine", "T_rgb = T_all", "T_rgb", "No; M2 reused beside M5"],
            [
                "Last-linear head",
                "T_rgb = T_all",
                "T_rgb",
                "No; every M3 seed reused beside every M6 seed",
            ],
        ],
        "Matched coverage is a control for target-source comparisons, not a hyperparameter search. Because every train center is RGB-eligible, the matched comparison is exactly the full primary reference/RGB comparison. No separate reduced-reference fits are needed.",
        "conditions.json matched_reuses_full_reference; matched_comparison.csv",
    )
    table(
        "Per-split finger MAE in degrees",
        ["Split", "Centers", *ids],
        [
            [
                group,
                int((ev["split"] == group).sum()),
                *[fnum(score_values("split", cid, group), 4) for cid in ids],
            ]
            for group in ["user", "gesture", "both"]
        ],
        "Head values are means of per-seed split MAEs. User supplies 2,853/3,654 centers and therefore dominates the pooled entry-weighted metric. The labels are dataset row split categories; they do not resolve released-checkpoint exposure of this participant.",
        "detailed_metrics.csv",
        font_size=7.8,
    )
    figure("split_results")
    figure("joint_heatmap")
    figure("original_per_joint")
    figure("gesture_heatmap_1")
    figure("gesture_heatmap_2")
    figure("original_per_gesture")
    figure("error_distribution")
    table(
        "Wrist invariance",
        ["Item", "Recorded value"],
        [
            ["Wrist valid entries", s["metrics"]["wrist"]["valid_entries"]],
            ["Wrist MAE (radians)", fnum(s["metrics"]["wrist"]["mae_rad"], 9)],
            ["Wrist MAE (degrees)", fnum(s["metrics"]["wrist"]["mae_deg"])],
            ["Parameter handling", "No wrist row/bias optimization"],
            [
                "Prediction/metric checks",
                "All saved wrist predictions exactly equal baseline; identical metric precision",
            ],
        ],
        "Wrist scores are separate from the primary finger aggregate. Their exact invariance verifies that the experiment implements the agreed finger-only adaptation scope.",
        "metrics.json wrist; verification_recalculation.json",
    )
    figure("original_time_traces")
    figure("traces_user")
    figure("traces_gesture")
    figure("traces_both")
    para(
        "The measured result is limited improvement from the analytic calibrations and worse performance from last-layer adaptation. The lowest observed score is M5, but its 0.017060° change is very small relative to the 15.349226° baseline. Head fitting can reduce fitting error while transferring poorly to the different evaluation action population. The evidence is consistent with that behavior here; it does not prove a general causal explanation or justify changing the predeclared hyperparameters after inspecting E."
    )

    heading("7. Runtime, verification and engineering controls", page=True)
    timings = [
        json.loads(line)
        for line in (run / "execution_times.jsonl").read_text().splitlines()
    ]
    table(
        "Execution time and GPU allocation",
        ["Stage", "Observed seconds", "Peak allocated MiB", "Convention"],
        [
            [
                "Complete original invocation",
                fnum(timings[0]["invocation_seconds"], 3),
                "—",
                "Includes configuration, manifest, inference, fits, scoring and reporting",
            ],
            [
                "Shared EMG extraction",
                fnum(s["emg"]["extraction_seconds"], 3),
                fnum(s["emg"]["peak_gpu_bytes"] / 2**20, 2),
                "One shared cache for T_all + E",
            ],
            [
                "Shared teacher construction",
                fnum(s["teacher"]["construction_seconds"], 3),
                fnum(s["teacher"]["peak_gpu_bytes"] / 2**20, 2),
                "One shared crop inference cache",
            ],
            *[
                [
                    v["key"],
                    fnum(v["fitting_seconds"], 3),
                    fnum(v["peak_gpu_bytes"] / 2**20, 2),
                    "Per-condition fit; excludes shared cache construction",
                ]
                for v in s["conditions"]["conditions"]
            ],
        ],
        "GPU peaks are PyTorch allocated tensor bytes after stage/seed reset. They include resident model/parameters and transients but exclude allocator-reserved memory, driver allocations, CPU caches and other processes. Observed time can be affected by warm OS caches; it is not a universal performance benchmark.",
        "execution_times.jsonl; emg_evidence.json; teacher_diagnostics.json; condition logs",
        font_size=8,
    )
    figure("runtime_memory")
    table(
        "Verification evidence",
        ["Check", "Outcome", "Meaning"],
        [
            [
                "Focused + relevant existing tests",
                "44 passed; 15 existing dependency warnings",
                "Protocol, synthetic fits, cached alignment, designated targets, wrist invariance and seed reporting checked",
            ],
            [
                "Formatting",
                "Black, isort and git diff checks passed",
                "New source is locally reviewable",
            ],
            [
                "Tiny machinery execution",
                "Teacher, EMG features, all methods/seeds, metrics and plots completed",
                "Validated execution plumbing before full pools",
            ],
            [
                "Original-head equivalence",
                "Maximum absolute discrepancy 3.58×10−7 radians",
                "Compact features reproduce the frozen aligned prediction",
            ],
            [
                "Saved MAE recalculation",
                "All 11 scored rows recalculated from saved arrays",
                "No model/dataset reload needed for scores",
            ],
            [
                "Identifiers and masks",
                "Identical E IDs and reference masks in every condition",
                "RGB coverage does not shrink E",
            ],
            [
                "Wrist invariance",
                "Exact saved-prediction and metric equality",
                "Only finger outputs adapted",
            ],
            [
                "Resume",
                "All six completed stages reused",
                "No GPU extraction repeated on compatible resume",
            ],
            [
                "Cache integrity",
                "Version/provenance identity and completion-file hashes checked",
                "Incompatible or incomplete caches rejected",
            ],
            [
                "Plot repair",
                "Long-gap trace connections removed; all numerical predictions unchanged",
                "Presentation correction did not refit or alter the experiment",
            ],
            [
                "Preservation",
                "No dataset/checkpoint copy; no commit/push/branch change",
                "Original smoke artifacts and user work preserved",
            ],
        ],
        "The verification record distinguishes synthetic tests, tiny plumbing checks, real-data execution and independent output recalculation. The plots were inspected after rendering; the Word report adds explanatory presentation without modifying experiment artifacts.",
        "verification.json; verification_recalculation.json; phase completion markers",
        font_size=8,
    )

    heading("8. Outputs and reproducibility", page=True)
    para(
        "The experiment output directory is "
        + source_run_posix
        + ". This document and its additional figures live in a distinct personal report directory. The original completed-run files remain unchanged, including their provenance and completion hashes."
    )
    table(
        "Input/output arrays available without model reload",
        ["Artifact", "Array / shape", "Units / explanation"],
        [
            [
                "manifest.npz",
                "7,828 ordered IDs; starts/centers/pools/splits/gestures; reference and mask 7,828×22",
                "Reference radians; identifiers link every fitting/evaluation center",
            ],
            [
                "emg_cache.npz",
                "features 7,828×512; frozen predictions 7,828×22; original weights 22×512 and bias 22",
                "Features are actual final-layer inputs; prediction radians",
            ],
            [
                "teacher_cache.npz",
                "4,174 center targets ×22; unique frame predictions ×22; eligibility fields",
                "Teacher radians kept separate from references",
            ],
            [
                "evaluation_predictions.npz",
                "3,654 IDs; reference/mask ×22; 11 baseline/analytic/seed predictions ×22",
                "Complete fixed-E score inputs and outputs",
            ],
            [
                "parameters/*.npz",
                "Offsets, affine values or 20×512 finger weights +20 bias; fitting IDs",
                "Compact adaptations and released initialization; no full checkpoint copies",
            ],
            [
                "parameters/*.json",
                "50 epoch losses, seed, deviations, fit metrics/time/memory and identity",
                "Per-condition training evidence",
            ],
        ],
        "There are seven primary methods but 11 saved scoring rows because each of the two head methods has three separate seeds. The output arrays preserve CPU values and IDs; the dataset is not embedded in them. Evaluation scores can be recalculated from reference/mask/prediction arrays alone.",
        "manifest.npz; emg_cache.npz; teacher_cache.npz; evaluation_predictions.npz",
    )
    first = int(np.flatnonzero(m["pools"] == "E")[0])
    ei = int(np.flatnonzero(ev["ids"] == m["ids"][first])[0])
    table(
        "Concrete EMG-only evaluation input/output example",
        ["Item", "Actual value"],
        [
            ["Center ID", str(ev["ids"][ei])],
            ["Split / gesture", str(ev["split"][ei]) + " / " + str(ev["gesture"][ei])],
            ["Global support", f"[{m['starts'][first]}, {m['starts'][first]+12000})"],
            ["Global center", int(ev["centers"][ei])],
            ["Model input", "Normalized 8×12,000 right-hand EMG; no crop supplied"],
            ["Feature output", "512-dimensional compact frozen-head input"],
            [
                "Angle output",
                "22 angles; baseline and every adapted/seed prediction cached",
            ],
            [
                "Finger/wrist masks",
                str(ev["mask"][ei, :20].all())
                + " / "
                + str(ev["mask"][ei, 20:].tolist()),
            ],
        ],
        "This is the first eligible E center in deterministic manifest order, not a case selected by error or improvement. Its input signal is illustrated earlier. The following table gives its concrete finger reference/prediction values.",
        "manifest.npz; evaluation_predictions.npz",
    )
    table(
        "Concrete center angle outputs, in degrees",
        ["Joint", "Reference", *ids],
        [
            [
                f"J{j:02d}",
                fnum(np.rad2deg(ev["reference"][ei, j]), 3),
                *[
                    fnum(np.rad2deg(condition_prediction(ev, cid)[ei, j]), 3)
                    for cid in ids
                ],
            ]
            for j in range(20)
        ],
        "This example table averages head predictions across seeds only to keep one readable value per method. It is not used to compute the headline mean of per-seed MAEs. Every analytic fit and head seed remains separately available in the saved outputs. Reference is evaluation ground truth; RGB predictions are not evaluation inputs.",
        "evaluation_predictions.npz",
        font_size=7.8,
    )
    table(
        "Output artifact groups",
        ["Files", "What a reader can inspect"],
        [
            [
                "settings.json; resolved_*_config.yaml",
                "Arguments, environment, hashes, preprocessing and exact architecture/data configurations",
            ],
            [
                "windows.csv; manifest.npz",
                "All local/global supports, stable IDs, split and gesture labels, masks and references",
            ],
            [
                "coverage.json/.md; gesture_coverage.csv; rgb_*_coverage.csv",
                "Complete interval/action coverage, zero-window intervals and exclusions",
            ],
            [
                "emg_evidence.json; emg_cache.npz",
                "Strict load, actual layer, context/alignment equivalence, frozen predictions/features",
            ],
            [
                "teacher_cache.npz; teacher_diagnostics.json; teacher_quality.csv",
                "Crop mapping, teacher targets, timing/availability and diagnostic quality",
            ],
            [
                "parameters/; conditions.json",
                "Compact independent adaptation values, every head seed, losses and finite checks",
            ],
            [
                "metrics.json; *_comparison.csv/.md; detailed_metrics.csv; per_seed_metrics.csv",
                "Aggregate/matched and split/gesture/joint scores, all seeds and deltas",
            ],
            [
                "evaluation_predictions.npz; reference_variation.csv",
                "Recalculation arrays and reference variation/ranges",
            ],
            [
                "crop_contact_sheet.png; per_joint.png; per_gesture.png; time_traces.png",
                "Original deterministic visual diagnostics and comparisons",
            ],
            [
                "commands.jsonl; execution_times.jsonl; verification*.json; *.complete.json",
                "Exact invocation history, run times, verification and cache completion/integrity",
            ],
            [
                "This report directory",
                "DOCX, added figures, figure/gallery/input identifier manifests, builder and source hashes",
            ],
        ],
        "The artifact inventory ties each output to its purpose. Exact filesystem paths are recorded in the source provenance appendix. The report does not claim that every cache byte is a measurement independent of the source model.",
        "Completed run file inventory",
    )

    heading("9. Limitations and practical conclusions", page=True)
    for text in [
        "One participant, one right hand and one preview recording: this is not an evaluation on the complete EgoEMG dataset.",
        "Released whole-recording FFT filtering is noncausal. The split-contained window manifest prevents shared input rows across pools, but it does not convert the filtered input into a causal signal.",
        "Train intervals are interleaved with and may follow evaluated intervals. The design is offline within-participant adaptation, not online calibration followed by future-only deployment.",
        "Checkpoint exposure of wmh remains unresolved. Row split names are not sufficient to call the result unseen-user generalization.",
        "The active fitting and evaluation gestures differ. All eligible train intervals do not necessarily cover every gesture or joint range in E.",
        "Windows overlap heavily within each pool. Centers and the 20 finger joints are not independent experiment replications; seed SD is not a population confidence interval.",
        "The 50 ms RGB rule is a predeclared availability tolerance, not a demonstrated synchronization guarantee. Teacher quality is measured on fitting centers only.",
        "This batch used fixed first-comparison regularization and optimizer settings. Evaluation results did not select parameters, epochs, slopes or the best seed.",
        "The small analytic MAE changes should not be overstated. No population significance analysis or new hyperparameter search was performed.",
        "FiLM, REACT, MANO conversion, raw-image detection, backbone updates and additional participants remain outside this work.",
    ]:
        bullet(text)
    para(
        "The work is complete even though head adaptation worsened the primary metric. The useful deliverable is the reproducible implementation and evidence: simple output corrections barely change E, while fitting a more flexible final layer can move several joints in an unfavorable direction on this recording’s different action population. Any future experiment should declare its protocol and settings independently; these E scores should not be reused to portray this fixed batch as an optimized calibration method."
    )

    heading("Appendix A. Complete interval and gesture coverage", page=True)
    rg = {v["interval"]: v for v in s["teacher"]["intervals"]}
    table(
        "All constant-split intervals",
        ["ID", "Split", "Local [start,end)", "Candidates", "Eligible", "RGB", "Reason"],
        [
            [
                v["interval"],
                v["split"],
                f"[{v['local_start']},{v['local_end']})",
                v["candidate_windows"],
                v["eligible_windows"],
                rg.get(v["interval"], {}).get("T_rgb", "—"),
                (
                    "too short"
                    if v["insufficient_length"]
                    else ("5 invalid centers" if v["exclusions"] else "none")
                ),
            ]
            for v in s["coverage"]["intervals"]
        ],
        "Positions are half-open local row ranges in episode_000020. Train rows feed T_all/T_rgb; user, gesture and both rows feed E. RGB is intentionally not required or inferred on E. The complete 32-interval table includes the final zero-window train fragment.",
        "coverage.json; rgb_interval_coverage.csv",
        font_size=8,
    )
    lookup = {
        (v["pool"], int(v["gesture_id"]), v["active"]): v for v in s["gesture_rows"]
    }
    rgb_lookup = {(int(v["gesture_id"]), v["active"]): v for v in s["rgb_rows"]}
    groups = sorted({(int(v["gesture_id"]), v["active"]) for v in s["gesture_rows"]})
    grows = []
    for g, active in groups:
        tr = lookup.get(("T_all", g, active), {})
        ee = lookup.get(("E", g, active), {})
        rr = rgb_lookup.get((g, active), {})
        grows.append(
            [
                g,
                (tr or ee)["name"],
                active,
                tr.get("centers", 0),
                rr.get("T_rgb", 0),
                ee.get("centers", 0),
            ]
        )
    table(
        "Every selected gesture and inactive group",
        ["Raw ID", "Vocabulary name", "Active", "T_all", "T_rgb", "E"],
        grows,
        "Raw IDs are preserved alongside actual supplied names and active flags. ID −1 is inactive and has no invented action name. Zero counts are explicit; training does not cover the evaluation action IDs in this recording. Every RGB fitting count equals its full-reference counterpart.",
        "gesture_coverage.csv; rgb_gesture_coverage.csv; supplied vocabulary",
        font_size=8,
    )

    heading("Appendix B. Complete joint, action and reference-range tables", page=True)
    table(
        "Per-joint evaluation finger MAE, degrees",
        ["Joint", *ids],
        [
            [f"J{j:02d}", *[fnum(score_values("joint", cid, name), 4) for cid in ids]]
            for j, name in enumerate(joints)
        ],
        "Rows follow the complete joint mapping in Section 2. Every joint has the same valid E center count. Head values average separate per-seed joint scores. Baseline-relative changes can be computed by subtracting the M0 column.",
        "detailed_metrics.csv",
        font_size=8,
    )
    groups = [
        v["group"] for v in s["detailed"] if v["kind"] == "gesture" and v["key"] == "M0"
    ]
    table(
        "Per-gesture evaluation finger MAE, degrees",
        ["Raw ID / name", *ids],
        [
            [
                g.rsplit(":active=", 1)[0],
                *[fnum(score_values("gesture", cid, g), 3) for cid in ids],
            ]
            for g in groups
        ],
        "Every observed E group is retained, including inactive −1. Within a row, MAE is a mean over that group’s valid finger entries. The primary aggregate is not the unweighted mean of these gesture rows. Head values are means of seed-specific group metrics.",
        "detailed_metrics.csv",
        font_size=7.7,
    )
    table(
        "Teacher per-joint fitting MAE",
        ["Joint", "MAE (radians)", "MAE (degrees)"],
        [
            [f"J{j:02d}", v["mae_rad"], v["mae_deg"]]
            for j, v in enumerate([v for v in s["quality"] if v["kind"] == "joint"])
        ],
        "These teacher scores compare crop output with fitting references only. The cached teacher angles, not these errors, supply RGB fitting targets. No disagreement threshold filtered T_rgb.",
        "teacher_quality.csv",
    )
    table(
        "Teacher per-gesture fitting MAE",
        ["Raw ID / active status", "MAE (radians)", "MAE (degrees)"],
        [
            [v["key"], v["mae_rad"], v["mae_deg"]]
            for v in s["quality"]
            if v["kind"] == "gesture"
        ],
        "The raw IDs map to names in Appendix A. Every active training gesture and the four inactive training centers are represented. E teacher inference was not required.",
        "teacher_quality.csv",
    )
    varrows = []
    for j, name in enumerate(joints):
        tr = next(
            v for v in s["variation"] if v["pool"] == "T_all" and v["joint"] == name
        )
        ee = next(v for v in s["variation"] if v["pool"] == "E" and v["joint"] == name)
        varrows.append(
            [
                f"J{j:02d}",
                fnum(tr["std_deg"], 3),
                f"{float(tr['minimum_deg']):.2f} to {float(tr['maximum_deg']):.2f}",
                fnum(ee["std_deg"], 3),
                f"{float(ee['minimum_deg']):.2f} to {float(ee['maximum_deg']):.2f}",
            ]
        )
    table(
        "Reference angle standard deviations and ranges",
        ["Joint", "T_all SD (°)", "T_all range (°)", "E SD (°)", "E range (°)"],
        varrows,
        "Reference SD/ranges describe sampled reference-angle variation. They are not adaptation accuracy, error bars or confidence intervals. T_rgb equals T_all, so its values coincide rather than needing duplicate rows. Population variance in radians squared is also saved in reference_variation.csv.",
        "reference_variation.csv",
        font_size=8,
    )

    heading("Appendix C. Provenance, commands and artifact inventory", page=True)
    table(
        "Critical file identities",
        ["Item", "SHA-256"],
        [
            ["Released EMG checkpoint", s["settings"]["identity"]["checkpoint_sha256"]],
            [
                "Released teacher checkpoint",
                s["settings"]["identity"]["teacher_sha256"],
            ],
            [
                "Suite workflow",
                s["settings"]["identity"]["small_file_sha256"][
                    root_posix
                    + "/myself/scripts/analysis/single_subject_calibration_suite.py"
                ],
            ],
            [
                "Dataset manifest",
                s["settings"]["identity"]["small_file_sha256"][
                    s["settings"]["identity"]["data_root"] + "/manifest.json"
                ],
            ],
            [
                "Dataset metadata",
                s["settings"]["identity"]["small_file_sha256"][
                    s["settings"]["identity"]["data_root"] + "/metadata.npz"
                ],
            ],
            [
                "Normalization asset",
                s["settings"]["identity"]["small_file_sha256"][
                    s["settings"]["identity"]["normalization"]["path"]
                ],
            ],
        ],
        "Hashes identify supplied checkpoints and small provenance/configuration assets. Large dataset arrays were not hashed. The separate report source_identity.json records the saved evidence hashes and report-builder identity.",
        "settings.json identity; report source_identity.json",
        font_size=7.8,
    )
    heading("Exact execution and verification commands", 2)
    for label, key in [
        ("Complete source run", None),
        ("Focused and relevant tests", "pytest_command"),
        ("Independent saved-output verification", "saved_output_verification_command"),
        ("Compatible resume", "resume_command"),
    ]:
        para(label + ":")
        argv = s["settings"]["command"] if key is None else s["verification"][key]
        formula(
            " ".join(argv),
            "Environment: PYTHONDONTWRITEBYTECODE=1; repository working directory "
            + root_posix
            + ".",
        )
    para(
        "The tiny machinery check additionally used --limit-per-pool 4 in a separate run directory. Formatting commands were python -m black --check, python -m isort --profile black --check-only for the two new suite Python files, and git diff --check. The full argv records and runtime records are preserved in commands.jsonl and execution_times.jsonl."
    )
    table(
        "Artifact sizes in the completed experiment",
        ["Relative path", "Size (KiB)", "Purpose"],
        [
            [p.name, f"{p.stat().st_size/1024:.1f}", "Saved experiment evidence"]
            for p in sorted(run.iterdir())
            if p.is_file()
        ]
        + [
            [
                "parameters/ (all files)",
                f'{sum(p.stat().st_size for p in (run/"parameters").iterdir())/1024:.1f}',
                "Compact independent adaptations and logs",
            ]
        ],
        "Sizes describe the original run files, not input dataset size. This inventory helps locate the reusable evidence and confirms that output caches are compact derived artifacts rather than bulk dataset copies.",
        "Actual completed-run file metadata",
        font_size=8,
    )
    heading("Figure register", 2)
    table(
        "Every report figure and its evidence source",
        ["Figure", "Title", "Source"],
        [[v["number"], v["title"], v["source"]] for v in inserted_figures],
        "Every embedded figure is backed by a saved result, a deterministic small input preview or an explicitly labelled schematic. Additional plots were generated for this report without refitting a method or running teacher/model inference.",
        "figure_manifest.json; gallery_identifiers.json; input_preview_identifiers.json",
        font_size=8,
    )
    para(
        "No conditions remain blocked and no remaining experiment commands are required. The finished work remains local and reviewable. No report was uploaded or shared to an external service."
    )
    output = out / "single_subject_calibration_detailed_work_report.docx"
    doc.save(output)
    (out / "document_manifest.json").write_text(
        json.dumps(
            dict(
                document=output.name,
                figures=inserted_figures,
                tables=tables,
                source_run=source_run_posix,
                paragraphs=len(doc.paragraphs),
                image_count=len(doc.inline_shapes),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"Saved {output}; {figure_number} figures; {table_number} tables; {len(doc.paragraphs)} paragraphs",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["figures", "docx"], required=True)
    args = parser.parse_args()
    run = args.source_run.resolve()
    out = args.output_dir.resolve()
    if not out.is_relative_to(REPORT_PARENT.resolve()):
        raise ValueError("Report outputs must remain beneath " + str(REPORT_PARENT))
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "figures":
        build_figures(run, out)
    else:
        build_docx(run, out)


if __name__ == "__main__":
    main()
