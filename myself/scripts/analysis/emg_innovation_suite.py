#!/usr/bin/env python3
"""Resumable fixed EMG innovation matrix. Never write bulk inputs/features to disk."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("EGOEMG_NO_PRETRAINED_DOWNLOAD", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "myself/.cache/matplotlib"))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from myself.egoemg.models.emg_innovation import (
    DIFFERENCE,
    DYNAMIC_AUX,
    FAMILIES,
    POSE_AUX,
    InnovationHead,
    TemporalProbe,
)
from myself.scripts.analysis.common_modality_evaluation import configuration
from myself.scripts.analysis.effective_emg_contribution import (
    HALF,
    WINDOW,
    Centers,
    audit,
    configure_precision,
    donor_mapping,
    prepare,
)
from myself.scripts.analysis.preview_calibration_smoke import open_field, sha256
from myself.scripts.analysis.single_subject_calibration_suite import (
    csv_write,
    json_write,
    npz_write,
    strict_model,
)
from myself.scripts.analysis.visual_guided_pool_pilot import (
    PoolingBranch,
    frozen_digest,
)


TRAIN_BATCH = 128
ENCODING_BATCH = 128
SEEDS = (42, 43, 44)
MAPS = (101, 102, 103)
EPISODES = ("episode_000020", "episode_000008", "episode_000028")


def now():
    return dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".pending")
    json_write(tmp, value)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def mae(pred, target, mask):
    return float(
        np.abs(pred.astype(np.float64) - target.astype(np.float64))[mask].mean()
        * 180
        / np.pi
    )


def loss_masked(pred, target, mask):
    return (
        (pred - target).abs().masked_select(mask).mean()
        if mask.any()
        else pred.sum() * 0
    )


def support_partition(s, origin, stop, execution_rows):
    """Assign full executions, then purge every input touching another assignment."""
    ids = np.unique(execution_rows[s["centers"] - origin])
    rng = np.random.default_rng(20261028)
    shuffled = rng.permutation(ids)
    val_ids = shuffled[: max(1, round(len(ids) * 0.2))]
    assignment = np.isin(execution_rows, val_ids).astype(np.int8)
    # Purge support across either assignment, not merely the center's group.
    prefix = np.r_[0, np.cumsum(assignment, dtype=np.int64)]
    a = s["centers"] - HALF - origin
    b = s["centers"] + HALF - origin
    total = prefix[b] - prefix[a]
    fit, val = total == 0, total == WINDOW
    labels = np.where(fit, "T_fit", np.where(val, "T_val", "purged_partition_support"))
    return labels, val_ids


def enrich(args, ds, run, partition=None):
    mf = json.loads((args.memmap_dir / "manifest.json").read_text())
    field = lambda k: open_field(args.memmap_dir, mf, k)
    s, base = ds.s, ds.base
    ep = ds.ep
    lo, hi = int(base._episode_start_idx[ep]), int(base._episode_end_idx[ep])
    g, active = (
        field("label_gesture_class")[lo:hi],
        field("label_gesture_active")[lo:hi],
    )
    erows = np.r_[0, np.cumsum((g[1:] != g[:-1]) | (active[1:] != active[:-1]))]
    assert np.array_equal(s["execution"], erows[s["centers"] - lo])
    if partition is None:
        partition = np.full(len(ds), "E", dtype="U32")
    s["partition"] = partition
    s["support_start"] = s["centers"] - HALF
    s["support_stop"] = s["centers"] + HALF
    s["group_id"] = np.array([f"{ds.episode}:execution:{e}" for e in s["execution"]])
    y = s["reference"][:, :20]
    valid = field("generated_label_valid")[:, 1]
    ref = field("generated_joint_angles_right")
    split = field("frame_split_id")
    target, masks = [], []
    for offset in (-200, 200):
        c = s["centers"] + offset
        in_episode = (c >= lo) & (c < hi)
        bounded = np.clip(c, lo, hi - 1)
        m = (
            in_episode[:, None]
            & valid[bounded, None]
            & np.isfinite(ref[bounded])
            & s["mask"][:, :20]
        )
        m &= (split[bounded] == s["split"])[:, None]
        # Endpoint is inside actual input support. For fit/val, require the row's
        # execution assignment to match the retained support partition as well.
        m &= ((bounded >= s["support_start"]) & (bounded < s["support_stop"]))[:, None]
        t = y - ref[bounded] if offset < 0 else ref[bounded] - y
        target.append(np.where(m, t, 0).astype(np.float32))
        masks.append(m)
    s["delta"] = np.stack(target, 1)
    s["delta_mask"] = np.stack(masks, 1)
    s["active"] = field("label_gesture_active")[s["centers"]]
    # Transition definition uses training METADATA: any gesture/active boundary
    # within +/-100 ms. Amplitude bins additionally derive only from T_fit.
    changes = np.r_[False, (g[1:] != g[:-1]) | (active[1:] != active[:-1])]
    pref = np.r_[0, np.cumsum(changes, dtype=np.int64)]
    a = np.maximum(s["centers"] - lo - 200, 0)
    b = np.minimum(s["centers"] - lo + 201, hi - lo)
    s["transition"] = pref[b] > pref[a]
    s["amplitude"] = np.divide(
        (np.abs(s["delta"]) * s["delta_mask"]).sum((1, 2)),
        s["delta_mask"].sum((1, 2)),
        out=np.zeros(len(ds)),
        where=s["delta_mask"].sum((1, 2)) > 0,
    ).astype(np.float32)
    csv_write(
        run
        / ds.episode
        / f"{str(partition[0]) if len(partition) and len(np.unique(partition)) == 1 else 'T_all'}_support.csv",
        [
            dict(
                center=int(c),
                group_id=str(gid),
                original_split=int(sid),
                partition=str(p),
                support_start=int(a),
                support_stop_exclusive=int(b),
                frame=int(f),
                delta_minus_valid=int(dm[0].sum()),
                delta_plus_valid=int(dm[1].sum()),
                transition=bool(tr),
            )
            for c, gid, sid, p, a, b, f, dm, tr in zip(
                s["centers"],
                s["group_id"],
                s["split"],
                partition,
                s["support_start"],
                s["support_stop"],
                s["frame"],
                s["delta_mask"],
                s["transition"],
            )
        ],
    )
    return ds


def subset(ds, indices):
    return Centers(ds.base, ds.episode, {k: v[indices] for k, v in ds.s.items()})


def mappings(ds):
    return {
        str(seed): donor_mapping(
            ds.s["centers"], ds.s["split"], ds.s["execution"], seed
        )
        for seed in MAPS
    }


def prepare_all(args, run, resume):
    # Existing prepare is reused only for eligibility; never for prediction or fitting.
    sets, audits = {}, {}
    for ep in EPISODES:
        history = args.history if ep == EPISODES[0] else None
        if resume and (run / ep / "enriched_T_all.npz").exists():
            from myself.scripts.analysis.effective_emg_contribution import make_base

            base = make_base(args, ep)
            sets[ep] = {
                p: Centers(base, ep, dict(np.load(run / ep / f"enriched_{p}.npz")))
                for p in ("T_all", "E")
            }
            audits[ep] = json.loads((run / ep / "selection_audit.json").read_text())
            continue
        raw, info = prepare(args, ep, history, run)
        train, ev = raw["T_all"][0], raw["E"][0]
        if ep == EPISODES[0]:
            mf = json.loads((args.memmap_dir / "manifest.json").read_text())
            lo, hi = info["episode_bounds"]
            g = open_field(args.memmap_dir, mf, "label_gesture_class")[lo:hi]
            a = open_field(args.memmap_dir, mf, "label_gesture_active")[lo:hi]
            ex = np.r_[0, np.cumsum((g[1:] != g[:-1]) | (a[1:] != a[:-1]))]
            labels, valid = support_partition(train.s, lo, hi, ex)
            enrich(args, train, run, labels)
            json_write(
                run / "partition_rule.json",
                dict(
                    seed=20261028,
                    rule="whole recording-level continuous gesture/active executions assigned 80/20; purge input support crossing assignments",
                    validation_execution_ids=valid.tolist(),
                    counts={p: int((labels == p).sum()) for p in np.unique(labels)},
                    grouping_independent_of_original_split=True,
                    input_rows=WINDOW,
                    delta_offsets=[-200, 200],
                ),
            )
        else:
            # Review training rows are never used for fitting or selection.
            enrich(
                args,
                train,
                run,
                np.full(len(train), "review_train_unused", dtype="U32"),
            )
        enrich(args, ev, run)
        sets[ep], audits[ep] = {"T_all": train, "E": ev}, info
        for p, ds in sets[ep].items():
            npz_write(run / ep / f"enriched_{p}.npz", **ds.s)
    all_ds = sets[EPISODES[0]]["T_all"]
    partitions = {}
    for p in ("T_fit", "T_val"):
        ix = np.flatnonzero(all_ds.s["partition"] == p)
        ds = subset(all_ds, ix)
        if not len(ds):
            raise ValueError(f"Empty purged {p}")
        maps = mappings(ds)
        npz_write(run / f"{p}_selection.npz", **ds.s)
        npz_write(run / f"{p}_donors.npz", **maps)
        partitions[p] = ds
    a, b = partitions["T_fit"].s, partitions["T_val"].s
    assert not (np.abs(a["centers"][:, None] - b["centers"][None]) < WINDOW).any()
    # Execution identities never cross fit/val even if split metadata cuts them.
    assert not set(a["group_id"]) & set(b["group_id"])
    thresholds = np.quantile(a["amplitude"], [1 / 3, 2 / 3]).tolist()
    json_write(
        run / "dynamic_rule.json",
        dict(
            amplitude_tertiles_rad=thresholds,
            source="T_fit only",
            transition="gesture/active metadata changes within +/-200 rows",
            nominal_rate_hz=2000,
        ),
    )
    for ds in [
        *partitions.values(),
        *[v[p] for v in sets.values() for p in ("T_all", "E")],
    ]:
        ds.s["dynamic_bin"] = np.searchsorted(thresholds, ds.s["amplitude"]).astype(
            np.int8
        )
    return sets, partitions, audits


def load_assets(args, run):
    cfg = configuration("fusion/fusion_rn18_s_center_16ch_wl7790")
    cfg.module.vision_pretrained_checkpoint = str(args.vision_checkpoint.resolve())
    fusion, hp = strict_model(cfg.module, args.checkpoint)
    cfgv = configuration("fusion/vision_resnet18")
    cfgv.module.pretrained = False
    vision, vhp = strict_model(cfgv.module, args.vision_checkpoint)
    OmegaConf.save(cfg.module, run / "fusion_model.yaml", resolve=True)
    OmegaConf.save(cfgv.module, run / "vision_model.yaml", resolve=True)
    json_write(run / "vision_checkpoint_hparams.json", vhp)
    return fusion, vision, hp


class Bank:
    def __init__(
        self, ds, fusion, vision, reference, batch=ENCODING_BATCH, reverse=False
    ):
        self.ds, self.fusion, self.vision = ds, fusion, vision
        self.reference, self.batch, self.reverse = reference, batch, reverse
        self.data = None
        start = time.perf_counter()
        available = (
            int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        budget = min(4 * 1024**3, int(available * 0.32))
        n = len(ds)
        if not n:
            raise ValueError(f"Empty bank {ds.episode}")
        first = self.compute(np.arange(min(batch, n)))
        required = sum(v[0].numel() * v.element_size() * n for v in first)
        self.info = dict(
            episode=ds.episode,
            centers=n,
            budget_bytes=budget,
            required_bytes=required,
            available_bytes=available,
            feature_shapes=[list(v.shape[1:]) for v in first],
            cached=required <= budget,
            reverse_before_encoder=reverse,
        )
        if required <= budget:
            arrays = [torch.empty((n, *v.shape[1:]), dtype=v.dtype) for v in first]
            for a, v in zip(arrays, first):
                a[: len(v)] = v
            for start_idx in range(len(first[0]), n, batch):
                values = self.compute(np.arange(start_idx, min(start_idx + batch, n)))
                for a, v in zip(arrays, values):
                    a[start_idx : start_idx + len(v)] = v
                if start_idx % (batch * 32) == 0:
                    logging.info(
                        "encoding %s reverse=%s %d/%d",
                        ds.episode,
                        reverse,
                        start_idx,
                        n,
                    )
            self.data = arrays
        torch.cuda.synchronize()
        self.info["pipeline_read_backbones_seconds"] = time.perf_counter() - start
        self.info["encoding_batch"] = batch
        self.info["read_included"] = True
        logging.info("Bank %s", self.info)

    def compute(self, indices):
        items = [self.ds[int(i)] for i in indices]
        x = torch.from_numpy(np.stack([v[0] for v in items])).cuda()
        img = torch.from_numpy(np.stack([v[1] for v in items])).cuda()
        if self.reverse:
            x = x.flip(-1)
        # no_grad rather than inference_mode: trainable layers need saved inputs.
        with torch.no_grad():
            e = self.fusion.decoder(self.fusion.featurizer(x)).transpose(1, 2)
            fmap = self.vision.backbone(img)
            glob = self.vision.avgpool(fmap).flatten(1)
            anchor = self.vision.head(glob)[:, :20]
            spatial = F.adaptive_avg_pool2d(fmap, (4, 4)).flatten(2).transpose(1, 2)
        return [v.cpu() for v in (e, spatial, glob, anchor)]

    def get(self, indices, donors=None, constant=False):
        indices = np.asarray(indices)
        if self.data is None:
            vals = self.compute(indices)
        else:
            vals = [v[indices] for v in self.data]
        vals = [v.cuda() for v in vals]
        if constant:
            vals[0] = self.reference.expand(len(indices), -1, -1)
        elif donors is not None:
            if self.data is None:
                vals[0] = self.compute(np.asarray(donors))[0].cuda()
            else:
                vals[0] = self.data[0][np.asarray(donors)].cuda()
        return vals


def targets(ds, ix):
    y = torch.from_numpy(ds.s["reference"][ix, :20]).cuda()
    m = torch.from_numpy(ds.s["mask"][ix, :20]).cuda()
    d = torch.from_numpy(ds.s["delta"][ix]).cuda()
    dm = torch.from_numpy(ds.s["delta_mask"][ix]).cuda()
    return y, m, d, dm


def predict(head, bank, condition="B1", mapping=None, batch=TRAIN_BATCH, aux=False):
    head.eval()
    out, auxs = [], []
    with torch.no_grad():
        for a in range(0, len(bank.ds), batch):
            ix = np.arange(a, min(a + batch, len(bank.ds)))
            values = bank.get(
                ix, mapping[ix] if mapping is not None else None, condition == "B0"
            )
            p, ap = head(*values, bank.reference, auxiliary=aux)
            out.append(p.cpu().numpy())
            if ap is not None:
                auxs.append(ap.cpu().numpy())
    return np.concatenate(out), np.concatenate(auxs) if auxs else None


def bench(head, bank):
    ix = np.arange(min(TRAIN_BATCH, len(bank.ds)))
    values = bank.get(ix)
    head.eval()
    with torch.no_grad():
        for _ in range(5):
            head(*values, bank.reference)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(40):
            head(*values, bank.reference)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / 40


def init_head(family, seed, emg_dim):
    seed_all(seed)
    return InnovationHead(family, emg_dim).cuda()


def smoke_checks(fusion, vision, bank, run):
    ix = np.arange(min(4, len(bank.ds)))
    values = bank.get(ix)
    checks = []
    e, sp, gl, anchor = values
    first = [bank.ds[int(i)] for i in ix]
    x = torch.from_numpy(np.stack([v[0] for v in first])).cuda()
    im = torch.from_numpy(np.stack([v[1] for v in first])).cuda()
    with torch.no_grad():
        released = fusion({"emg": x, "vision_img": im})[..., 0]
        wrapper = PoolingBranch(fusion).cuda().eval()
        wrapped = wrapper(
            fusion.decoder(fusion.featurizer(x)),
            fusion._extract_vision_features(im),
            fusion.head_vision(fusion._extract_vision_features(im)),
        )
        torch.testing.assert_close(wrapped, released, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            anchor, vision({"vision_img": im})[:, :20, 0], atol=1e-6, rtol=1e-5
        )
    checks.append(
        dict(
            released_forward_max_rad=float((wrapped - released).abs().max()),
            independent_vision_map_forward_max_rad=float(
                (anchor - vision({"vision_img": im})[:, :20, 0]).abs().max()
            ),
        )
    )
    initial_states = {}
    for fam in FAMILIES:
        h = init_head(fam, 42, e.shape[-1])
        initial_states[fam] = {
            k: v.detach().cpu().clone() for k, v in h.state_dict().items()
        }
        opt = torch.optim.Adam(
            [p for p in h.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
        )
        for condition in ("B0", "B1", "B2"):
            h.load_state_dict(initial_states[fam], strict=True)
            ee = (
                bank.reference.expand_as(e)
                if condition == "B0"
                else (
                    bank.get(mappings(bank.ds)["101"][ix])[0]
                    if condition == "B2"
                    else e
                )
            )
            with torch.no_grad():
                pp = h(ee, sp, gl, anchor, bank.reference)[0]
                torch.testing.assert_close(pp, anchor, atol=0, rtol=0)
            grads = []
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                pp, ap = h(ee, sp, gl, anchor, bank.reference, auxiliary=True)
                y, m, d, dm = targets(bank.ds, ix)
                loss = loss_masked(pp, y, m)
                aux_loss = loss * 0
                if ap is not None:
                    aux_loss = loss_masked(ap[:, 0], y, m)
                    if fam in DYNAMIC_AUX:
                        aux_loss = (
                            aux_loss
                            + (
                                loss_masked(ap[:, 1], d[:, 0], dm[:, 0])
                                + loss_masked(ap[:, 2], d[:, 1], dm[:, 1])
                            )
                            / 2
                        )
                (loss + aux_loss).backward()
                grads.append(
                    {
                        n: float(
                            sum(
                                (
                                    p.grad.square().sum()
                                    for p in mod.parameters()
                                    if p.grad is not None
                                ),
                                torch.tensor(0.0, device="cuda"),
                            ).sqrt()
                        )
                        for n, mod in (
                            ("adapter", h.adapter),
                            ("output", h.output),
                            ("vision_attention", h.vision_attention),
                        )
                    }
                )
                opt.step()
            if fam in DIFFERENCE and condition == "B0":
                torch.testing.assert_close(pp, anchor, atol=0, rtol=0)
            else:
                assert grads[0]["output"] > 0
                if condition != "B0":
                    assert (
                        grads[-1]["adapter"] > 0 and grads[-1]["vision_attention"] > 0
                    )
            if ap is not None:
                opt.zero_grad(set_to_none=True)
                _, aa = h(ee, sp, gl, anchor, bank.reference, auxiliary=True)
                loss_masked(aa[:, 0], y, m).backward()
                assert (
                    sum(
                        float(p.grad.abs().sum())
                        for p in h.adapter.parameters()
                        if p.grad is not None
                    )
                    > 0
                )
            q = h.condition(sp, gl, anchor)
            z = h.adapted(e)
            with torch.no_grad():
                sensitivity = float(
                    (h.read(q, z) - h.read(q + torch.randn_like(q) * 0.1, z))
                    .abs()
                    .mean()
                )
                # Mixed q/E interaction must survive reference subtraction before zero head.
                mixed = float(
                    (
                        (
                            h.read(q, z)
                            - h.read(q, h.adapted(bank.reference.expand_as(e)))
                        )
                        - (
                            h.read(q.flip(1), z)
                            - h.read(q.flip(1), h.adapted(bank.reference.expand_as(e)))
                        )
                    )
                    .abs()
                    .mean()
                )
            assert sensitivity > 0
            if fam != "R0":
                assert mixed > 0
            path = run / f"smoke_{fam}_{condition}.pt"
            torch.save(h.state_dict(), path)
            re = init_head(fam, 42, e.shape[-1])
            re.load_state_dict(torch.load(path, weights_only=True), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(
                    h(ee, sp, gl, anchor, bank.reference)[0],
                    re(ee, sp, gl, anchor, bank.reference)[0],
                    atol=0,
                    rtol=0,
                )
            checks.append(
                dict(
                    family=fam,
                    condition=condition,
                    gradients=grads,
                    query_readout_sensitivity=sensitivity,
                    mixed_query_emg_readout=mixed,
                    reload_exact=True,
                )
            )
    # All initialized tensors are equal; flags/family alone change behavior.
    for fam in FAMILIES[1:]:
        for k, v in initial_states["R0"].items():
            assert torch.equal(v, initial_states[fam][k]), (fam, k)
    json_write(run / "smoke_checks.json", checks)
    return checks


def unit_key(phase, family, condition, seed):
    return f"{phase}_{family}_{condition}_s{seed}"


def planned_cells(phase, families=FAMILIES):
    return {
        unit_key(phase, f, c, s): dict(
            phase=phase,
            family=f,
            condition=c,
            seed=s,
            status="pending",
            planned_kind=(
                "analytical_anchor" if f in DIFFERENCE and c == "B0" else "optimization"
            ),
        )
        for f in families
        for c in ("B0", "B1", "B2")
        for s in SEEDS
    }


def checkpoint_save(path, value):
    pending = path.with_suffix(".pending")
    torch.save(value, pending)
    pending.replace(path)


def train_cell(
    args, family, condition, seed, bank, valbank, phase, run, identity, state
):
    key = unit_key(phase, family, condition, seed)
    cell = state["cells"][key]
    folder = run / "cells" / key
    folder.mkdir(parents=True, exist_ok=True)
    h = init_head(family, seed, bank.reference.shape[-1])
    initial = frozen_digest(h)
    ident = digest_json(
        dict(
            run=identity,
            key=key,
            centers=sha256(
                run
                / (
                    "T_fit_selection.npz"
                    if phase == "fit"
                    else f"{EPISODES[0]}/enriched_T_all.npz"
                )
            ),
            initial=initial,
        )
    )
    cell.update(identity=ident, initialization_sha256=initial)
    analytical = condition == "B0" and family in DIFFERENCE
    path = folder / "last.pt"
    if cell["status"] == "complete":
        if analytical:
            return h
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["identity"] == ident and saved["epoch"] == args.epochs
        h.load_state_dict(saved["model"], strict=True)
        assert sha256(path) == cell["checkpoint_sha256"]
        return h
    if analytical:
        cell.update(
            status="complete",
            actual_optimization_epochs=0,
            auxiliary_trained=False,
            reason="H(q,E_ref)-H(q,E_ref)=0; anchor prediction analytically reused",
            parameters=h.parameter_counts(),
            completed_at=now(),
        )
        if valbank is not None:
            p = (
                valbank.data[3].numpy()
                if valbank.data is not None
                else np.concatenate(
                    [
                        valbank.compute(
                            np.arange(a, min(a + TRAIN_BATCH, len(valbank.ds)))
                        )[3].numpy()
                        for a in range(0, len(valbank.ds), TRAIN_BATCH)
                    ]
                )
            )
            npz_write(folder / "validation.npz", **valbank.ds.s, prediction=p)
            cell["validation_mae_deg"] = mae(
                p, valbank.ds.s["reference"][:, :20], valbank.ds.s["mask"][:, :20]
            )
        save_json(run / "state.json", state)
        return h
    opt = torch.optim.Adam(
        [p for p in h.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    start_epoch, curves = 0, []
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["identity"] == ident, "Resume identity mismatch"
        h.load_state_dict(saved["model"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        for state_value in opt.state.values():
            for k, v in state_value.items():
                if isinstance(v, torch.Tensor):
                    state_value[k] = v.cuda()
        start_epoch, curves = saved["epoch"], saved["curves"]
        restore_rng(saved["rng"])
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cell.update(
        status="running", resumed_epoch=start_epoch, parameters=h.parameter_counts()
    )
    save_json(run / "state.json", state)
    n = len(bank.ds)
    for epoch in range(start_epoch, args.epochs):
        estart = time.perf_counter()
        order_seed = seed * 10000 + epoch
        order = np.random.default_rng(order_seed).permutation(n)
        mapping_seed = 900000 + seed * 1000 + epoch
        mp = (
            (
                np.load(
                    run / "orders" / f"{phase}_donors_s{seed}_e{epoch + 1:02d}.npz"
                )["donor"]
                if (
                    run / "orders" / f"{phase}_donors_s{seed}_e{epoch + 1:02d}.npz"
                ).exists()
                else donor_mapping(
                    bank.ds.s["centers"],
                    bank.ds.s["split"],
                    bank.ds.s["execution"],
                    mapping_seed,
                )
            )
            if condition == "B2"
            else None
        )
        # One identity-checked order/map per phase/seed/epoch shared by all families.
        order_path = run / "orders" / f"{phase}_s{seed}_e{epoch + 1:02d}.npz"
        order_path.parent.mkdir(exist_ok=True)
        if order_path.exists():
            old = np.load(order_path)
            assert np.array_equal(old["order"], order)
        else:
            npz_write(order_path, order=order, order_seed=np.array(order_seed))
        if mp is not None:
            map_path = run / "orders" / f"{phase}_donors_s{seed}_e{epoch + 1:02d}.npz"
            if map_path.exists():
                assert np.array_equal(np.load(map_path)["donor"], mp)
            else:
                npz_write(map_path, donor=mp, mapping_seed=np.array(mapping_seed))
        h.train()
        sums = np.zeros(4, np.float64)
        grad_sum = 0.0
        for a in range(0, n, TRAIN_BATCH):
            ix = order[a : a + TRAIN_BATCH]
            vals = bank.get(ix, mp[ix] if mp is not None else None, condition == "B0")
            y, mask, delta, dm = targets(bank.ds, ix)
            opt.zero_grad(set_to_none=True)
            p, aux = h(*vals, bank.reference, auxiliary=True)
            fused = loss_masked(p, y, mask)
            pose, minus, plus = fused * 0, fused * 0, fused * 0
            if family in POSE_AUX:
                pose = loss_masked(aux[:, 0], y, mask)
            if family in DYNAMIC_AUX:
                minus = loss_masked(aux[:, 1], delta[:, 0], dm[:, 0])
                plus = loss_masked(aux[:, 2], delta[:, 1], dm[:, 1])
            loss = fused + pose + (minus + plus) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError(key)
            loss.backward()
            grad_sum += float(
                sum(
                    (
                        p.grad.square().sum()
                        for p in h.adapter.parameters()
                        if p.grad is not None
                    ),
                    torch.tensor(0.0, device="cuda"),
                ).sqrt()
            )
            opt.step()
            sums += np.array(
                [float(v.detach()) for v in (fused, pose, minus, plus)]
            ) * len(ix)
        torch.cuda.synchronize()
        row = dict(
            epoch=epoch + 1,
            fused_l1_rad=sums[0] / n,
            emg_pose_l1_rad=sums[1] / n,
            delta_minus_l1_rad=sums[2] / n,
            delta_plus_l1_rad=sums[3] / n,
            adapter_gradient_l2_mean=grad_sum / ((n + TRAIN_BATCH - 1) // TRAIN_BATCH),
            epoch_seconds=time.perf_counter() - estart,
        )
        curves.append(row)
        checkpoint_save(
            path,
            dict(
                identity=ident,
                model=h.state_dict(),
                optimizer=opt.state_dict(),
                epoch=epoch + 1,
                step=(epoch + 1) * ((n + TRAIN_BATCH - 1) // TRAIN_BATCH),
                rng=rng_state(),
                curves=curves,
                order_seed=order_seed,
                mapping_seed=mapping_seed,
                last_order=order,
                original_initialization_sha256=initial,
            ),
        )
        csv_write(folder / "curve.csv", curves)
        cell.update(epoch=epoch + 1, latest_loss=row)
        save_json(run / "state.json", state)
        logging.info(
            "%s epoch=%d/%d fused=%.6f rad aux=%.6f/%.6f/%.6f seconds=%.2f",
            key,
            epoch + 1,
            args.epochs,
            *sums / n,
            row["epoch_seconds"],
        )
    h.eval()
    reload = init_head(family, seed, bank.reference.shape[-1])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    reload.load_state_dict(saved["model"], strict=True)
    vals = bank.get(np.arange(min(8, n)), constant=condition == "B0")
    with torch.no_grad():
        torch.testing.assert_close(
            h(*vals, bank.reference)[0],
            reload(*vals, bank.reference)[0],
            rtol=0,
            atol=0,
        )
    if valbank is not None:
        mp = mappings(valbank.ds)["101"] if condition == "B2" else None
        p, ap = predict(h, valbank, condition, mp, aux=True)
        vals_to_save = dict(prediction=p)
        if ap is not None:
            vals_to_save["auxiliary"] = ap
        npz_write(folder / "validation.npz", **valbank.ds.s, **vals_to_save)
        cell["validation_mae_deg"] = mae(
            p, valbank.ds.s["reference"][:, :20], valbank.ds.s["mask"][:, :20]
        )
        if condition == "B2":
            correct, _ = predict(h, valbank)
            npz_write(
                folder / "validation_correct.npz", **valbank.ds.s, prediction=correct
            )
        cell["cached_head_batch_seconds"] = bench(h, valbank)
    cell.update(
        status="complete",
        actual_optimization_epochs=args.epochs,
        auxiliary_trained=family in POSE_AUX,
        checkpoint_sha256=sha256(path),
        reload_exact=True,
        training_seconds=sum(r["epoch_seconds"] for r in curves),
        current_session_seconds=time.perf_counter() - begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        completed_at=now(),
    )
    save_json(run / "state.json", state)
    return h


def run_probes(args, fit, val, run, identity):
    folder = run / "probes"
    folder.mkdir(exist_ok=True)
    names = [
        (target, c) for target in ("pose", "minus", "plus") for c in ("B1", "B2")
    ] + [("visual_residual", c) for c in ("B0", "B1", "B2")]
    rows = []
    fit_maps, val_maps = mappings(fit.ds), mappings(val.ds)
    for target, condition in names:
        path = folder / f"{target}_{condition}.npz"
        if path.exists():
            continue
        seed_all(42)
        h = TemporalProbe(
            fit.reference.shape[-1], visual=target == "visual_residual"
        ).cuda()
        opt = torch.optim.Adam(h.parameters(), lr=1e-4, weight_decay=1e-4)

        def get_target(bank, ix, anchor):
            y, mask, delta, dm = targets(bank.ds, ix)
            if target == "minus":
                return delta[:, 0], dm[:, 0]
            if target == "plus":
                return delta[:, 1], dm[:, 1]
            return (y - anchor if target == "visual_residual" else y), mask

        # Fit-only joint medians; residual uses the same frozen visual anchor.
        ys, ms = [], []
        for a in range(0, len(fit.ds), TRAIN_BATCH):
            ix = np.arange(a, min(a + TRAIN_BATCH, len(fit.ds)))
            vals = fit.get(ix)
            yy, mm = get_target(fit, ix, vals[3])
            ys.append(yy.cpu().numpy())
            ms.append(mm.cpu().numpy())
        yy, mm = np.concatenate(ys), np.concatenate(ms)
        median = np.nanmedian(np.where(mm, yy, np.nan), 0).astype(np.float32)
        history = []
        for epoch in range(30 if not args.smoke else 1):
            order = np.random.default_rng(420000 + epoch).permutation(len(fit.ds))
            mp = (
                donor_mapping(
                    fit.ds.s["centers"],
                    fit.ds.s["split"],
                    fit.ds.s["execution"],
                    942000 + epoch,
                )
                if condition == "B2"
                else None
            )
            for a in range(0, len(order), TRAIN_BATCH):
                ix = order[a : a + TRAIN_BATCH]
                e, sp, gl, anchor = fit.get(
                    ix, mp[ix] if mp is not None else None, condition == "B0"
                )
                y, mask = get_target(fit, ix, anchor)
                opt.zero_grad(set_to_none=True)
                pred = h(e, gl, anchor)
                loss = loss_masked(pred, y, mask)
                loss.backward()
                opt.step()
            history.append(
                dict(epoch=epoch + 1, last_batch_loss_rad=float(loss.detach()))
            )
            if (epoch + 1) % 5 == 0:
                logging.info("probe %s %s epoch=%d", target, condition, epoch + 1)
        preds, targets_, masks_ = [], [], []
        with torch.no_grad():
            for a in range(0, len(val.ds), TRAIN_BATCH):
                ix = np.arange(a, min(a + TRAIN_BATCH, len(val.ds)))
                e, sp, gl, anchor = val.get(
                    ix,
                    val_maps["101"][ix] if condition == "B2" else None,
                    condition == "B0",
                )
                y, mask = get_target(val, ix, anchor)
                preds.append(h(e, gl, anchor).cpu().numpy())
                targets_.append(y.cpu().numpy())
                masks_.append(mask.cpu().numpy())
        npz_write(
            path,
            prediction=np.concatenate(preds),
            target=np.concatenate(targets_),
            target_mask=np.concatenate(masks_),
            median=median,
            **val.ds.s,
        )
        checkpoint_save(
            folder / f"{target}_{condition}.pt",
            dict(
                model=h.state_dict(),
                identity=identity,
                epoch=30 if not args.smoke else 1,
                exploratory=True,
                seed=42,
            ),
        )
        csv_write(folder / f"{target}_{condition}_curve.csv", history)
        logging.info("Probe complete %s %s", target, condition)
    # The independent summarizer computes baselines and grouped probe metrics.


def lock_selection(run, state, identity):
    path = run / "selection_lock.json"
    if path.exists():
        locked = json.loads(path.read_text())
        assert locked["identity"] == identity
        return locked["selected_family"]
    scores = []
    for family in FAMILIES:
        cells = [state["cells"][unit_key("fit", family, "B1", seed)] for seed in SEEDS]
        assert all(c["status"] == "complete" for c in cells)
        scores.append(
            dict(
                family=family,
                mean_validation_mae_deg=float(
                    np.mean([c["validation_mae_deg"] for c in cells])
                ),
                seed_validation_mae_deg=[c["validation_mae_deg"] for c in cells],
                measured_batch_seconds=float(
                    np.median([c["cached_head_batch_seconds"] for c in cells])
                ),
                trainable_parameters=cells[0]["parameters"]["trainable"],
            )
        )
    minimum = min(r["mean_validation_mae_deg"] for r in scores)
    eligible = [r for r in scores if r["mean_validation_mae_deg"] <= minimum + 0.001]
    chosen = min(
        eligible,
        key=lambda r: (
            r["measured_batch_seconds"],
            r["trainable_parameters"],
            r["family"],
        ),
    )["family"]
    locked = dict(
        selected_family=chosen,
        identity=identity,
        generated_at=now(),
        scores=scores,
        rule="three seeds equal weight after per-seed valid-angle-weighted T_val MAE; within .001 deg of minimum choose measured inference seconds, parameters, then ID",
        final_retraining=dict(
            families=sorted({chosen, "R0"}),
            conditions=["B0", "B1", "B2"],
            seeds=list(SEEDS),
            epochs=30,
            initialization="original seed initialization, not T_fit checkpoint",
            primary="last epoch",
            T_val_merged_into_T_all=True,
        ),
        inputs=state["protocol"],
        supporting_files={
            str(p.relative_to(run)): sha256(p)
            for p in [
                run / "T_fit_selection.npz",
                run / "T_val_selection.npz",
                run / "T_fit_donors.npz",
                run / "T_val_donors.npz",
                run / "dynamic_rule.json",
                run / "partition_rule.json",
            ]
        },
    )
    locked["content_sha256"] = digest_json(locked)
    save_json(path, locked)
    logging.info("SELECTION LOCKED %s T_val means %s", chosen, scores)
    return chosen


def load_cell(run, state, phase, family, condition, seed, dim):
    h = init_head(family, seed, dim)
    cell = state["cells"][unit_key(phase, family, condition, seed)]
    assert cell["status"] == "complete"
    if not (family in DIFFERENCE and condition == "B0"):
        path = run / "cells" / unit_key(phase, family, condition, seed) / "last.pt"
        assert sha256(path) == cell["checkpoint_sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["identity"] == cell["identity"]
        h.load_state_dict(saved["model"], strict=True)
    return h.eval()


def score_evaluation(
    args, sets, fusion, vision, reference, state, phase, families, run, identity
):
    lock_path = run / "selection_lock.json"
    assert lock_path.exists()
    lock = json.loads(lock_path.read_text())
    assert lock["identity"] == identity
    for ep in EPISODES:
        ds = sets[ep]["E"]
        if len(ds) < 32:
            state.setdefault("review_omissions", {})[ep] = dict(
                reason="fewer than 32 legal evaluation centers", centers=len(ds)
            )
            save_json(run / "state.json", state)
            continue
        folder = run / "predictions" / phase / ep
        folder.mkdir(parents=True, exist_ok=True)
        needed = [
            unit_key(phase, f, c, s)
            for f in families
            for c in ("B0", "B1", "B2")
            for s in SEEDS
        ]
        if all((folder / f"{key}.npz").exists() for key in needed):
            continue
        bank = Bank(ds, fusion, vision, reference)
        reverse = Bank(ds, fusion, vision, reference, reverse=True)
        state.setdefault("bank_resources", []).extend(
            [
                dict(phase=phase, pool="E", **bank.info),
                dict(phase=phase, pool="E", **reverse.info),
            ]
        )
        maps = mappings(ds)
        npz_write(run / ep / "fixed_E_donors.npz", **maps)
        for family in families:
            for condition in ("B0", "B1", "B2"):
                for seed in SEEDS:
                    key = unit_key(phase, family, condition, seed)
                    path = folder / f"{key}.npz"
                    if path.exists():
                        continue
                    h = load_cell(
                        run, state, phase, family, condition, seed, reference.shape[-1]
                    )
                    outputs = {}
                    if condition == "B0":
                        outputs["constant"] = predict(h, bank, "B0")[0]
                    else:
                        outputs["normal"], aux = predict(h, bank, aux=True)
                        if aux is not None:
                            outputs["auxiliary_normal"] = aux
                        for ms in MAPS:
                            outputs[f"mismatch_{ms}"] = predict(
                                h, bank, "B2", maps[str(ms)]
                            )[0]
                        if condition == "B1":
                            outputs["constant"] = predict(h, bank, "B0")[0]
                            outputs["reverse"] = predict(h, reverse)[0]
                    pred_key = (
                        "constant"
                        if condition == "B0"
                        else "normal" if condition == "B1" else "mismatch_101"
                    )
                    npz_write(path, **ds.s, **outputs)
                    state.setdefault("formal_evaluations", {})[
                        str(path.relative_to(run))
                    ] = dict(
                        scored_at=now(),
                        selection_lock_sha256=sha256(lock_path),
                        prediction_sha256=sha256(path),
                        primary=pred_key,
                        mae_deg=mae(
                            outputs[pred_key],
                            ds.s["reference"][:, :20],
                            ds.s["mask"][:, :20],
                        ),
                        valid_angles=int(ds.s["mask"][:, :20].sum()),
                    )
                    save_json(run / "state.json", state)
                    logging.info(
                        "E %s %s MAE=%.6f",
                        ep,
                        key,
                        state["formal_evaluations"][str(path.relative_to(run))][
                            "mae_deg"
                        ],
                    )
                    del h
        del bank, reverse
        torch.cuda.empty_cache()


def execution_identity(args, run, resume):
    sources = [Path(__file__), ROOT / "myself/egoemg/models/emg_innovation.py"]
    sources += [
        ROOT / f"myself/scripts/analysis/{name}.py"
        for name in (
            "effective_emg_contribution",
            "common_modality_evaluation",
            "preview_calibration_smoke",
            "single_subject_calibration_suite",
            "visual_guided_pool_pilot",
            "run_directories",
        )
    ]
    sources += sorted((ROOT / "egoemg").rglob("*.py")) + sorted(
        (ROOT / "config").rglob("*.yaml")
    )
    assets = [
        args.checkpoint,
        args.vision_checkpoint,
        args.norm_stats,
        args.memmap_dir / "manifest.json",
        args.memmap_dir / "metadata.npz",
    ]
    hashes = {str(p.resolve()): sha256(p) for p in sources + assets}
    protocol = dict(
        families=list(FAMILIES),
        seeds=list(SEEDS),
        epochs=args.epochs,
        batch=TRAIN_BATCH,
        encoding_batch=ENCODING_BATCH,
        Adam=dict(lr=1e-4, weight_decay=1e-4),
        precision="deterministic FP32, TF32 disabled",
        width=128,
        heads=4,
        queries=5,
        vision_tokens=16,
        dropout=0,
        primary="last epoch",
        lambda_pose=1,
        lambda_delta=1,
        split_seed=20261028,
        mapping_seeds=list(MAPS),
        smoke=args.smoke,
        feature_storage="bounded CPU RAM only; otherwise stream and recompute",
        memmap=str(args.memmap_dir.resolve()),
        crops=str(args.crops_dir.resolve()),
        selection_before_E=True,
    )
    identity = digest_json(dict(protocol=protocol, hashes=hashes))
    if resume:
        old = json.loads((run / "identity.json").read_text())
        assert (
            old["identity"] == identity
        ), "Execution code/config/assets changed: preserve this run and start a new one"
        return identity, protocol
    source_folder = run / "executed_sources"
    for p in sources:
        dst = source_folder / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    (run / "dirty.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    )
    json_write(
        run / "identity.json",
        dict(
            identity=identity,
            protocol=protocol,
            files_sha256=hashes,
            created_at=now(),
            command=sys.argv,
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            git_branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=ROOT, text=True
            ).strip(),
            git_status=subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
            python=sys.version,
            executable=sys.executable,
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            input_mount=subprocess.check_output(
                ["findmnt", "-T", str(args.memmap_dir)], text=True
            ),
        ),
    )
    return identity, protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["smoke", "full", "resume"])
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument(
        "--memmap-dir",
        type=Path,
        default=Path("/mnt/d/egoemg/data/dataset_egoemg_preview/data/memmap_data"),
    )
    p.add_argument(
        "--crops-dir",
        type=Path,
        default=Path(
            "/mnt/d/egoemg/data/dataset_egoemg_preview/data/pre-crop_webcam_videoframes"
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/fusion_resnet18_emgfusion_center.ckpt",
    )
    p.add_argument(
        "--vision-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/vision_resnet18.ckpt",
    )
    p.add_argument(
        "--norm-stats", type=Path, default=ROOT / "assets/per_dataset_norm_stats.json"
    )
    p.add_argument(
        "--history",
        type=Path,
        default=ROOT
        / "myself/outputs/single_subject_calibration_suite/2026-10-05_wmh_right_suite_full_02",
    )
    args = p.parse_args()
    args.smoke, args.epochs = args.mode == "smoke", 30
    args.episode, args.common_history = EPISODES[0], None
    resume = args.mode == "resume"
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / "myself/outputs"):
        raise ValueError("Run outputs must be under myself/outputs")
    run.mkdir(parents=True, exist_ok=resume)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(run / "run.log"), logging.StreamHandler()],
    )
    configure_precision()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    begin = time.perf_counter()
    try:
        identity, protocol = execution_identity(args, run, resume)
        if resume:
            state = json.loads((run / "state.json").read_text())
        else:
            state = dict(
                identity=identity,
                protocol=protocol,
                status="running",
                cells=planned_cells("fit"),
                started_at=now(),
            )
            save_json(run / "state.json", state)
        fusion, vision, hp = load_assets(args, run)
        frozen_before = dict(fusion=frozen_digest(fusion), vision=frozen_digest(vision))
        sets, parts, selection_audits = prepare_all(args, run, resume)
        padding = audit(args, fusion, hp, sets[EPISODES[0]]["T_all"].base, run)
        sample = sets[EPISODES[0]]["T_all"].emg(0)
        constant = torch.full(
            (1, *sample.shape), padding, dtype=torch.float32, device="cuda"
        )
        # Validate physical zero through the actual loader's sparse layout + norm.
        base = sets[EPISODES[0]]["T_all"].base
        raw_zero = np.zeros((WINDOW, 8), np.float32)
        layout, live_mask = base._convert_to_emg2pose_layout(
            raw_zero, interpolate_missing=False
        )
        assert int(live_mask.sum()) == 8
        normalized = (
            (layout - float(base._emg_mean)) / (float(base._emg_std) + 1e-6)
        ).T
        np.testing.assert_array_equal(constant[0].cpu().numpy(), normalized)
        with torch.no_grad():
            reference = fusion.decoder(fusion.featurizer(constant)).transpose(1, 2)
        npz_write(
            run / "reference.npz",
            normalized_input=normalized,
            encoded_reference=reference.cpu().numpy(),
        )
        json_write(
            run / "reference_audit.json",
            dict(
                raw_value=0,
                mean=float(base._emg_mean),
                std=float(base._emg_std),
                padding=padding,
                normalized_shape=list(normalized.shape),
                encoded_shape=list(reference.shape),
                raw_layout_reconstruction_exact=True,
                reference_sha256=sha256(run / "reference.npz"),
                no_encoded_reference_after_adapter_cache=True,
            ),
        )
        # Verify joint indices using both metadata and canonical repository constants.
        from egoemg.constants import JOINTS

        mf = json.loads((args.memmap_dir / "manifest.json").read_text())
        assert [j.name.lower() for j in JOINTS[:20]] == mf[
            "generated_joint_angles_semantics"
        ]
        json_write(
            run / "joint_mapping.json",
            dict(
                fingers=[
                    dict(
                        finger=f,
                        indices=list(range(i * 4, i * 4 + 4)),
                        joints=mf["generated_joint_angles_semantics"][
                            i * 4 : i * 4 + 4
                        ],
                    )
                    for i, f in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                ]
            ),
        )
        if args.smoke:
            ds = subset(
                parts["T_fit"],
                np.linspace(
                    0, len(parts["T_fit"]) - 1, min(64, len(parts["T_fit"])), dtype=int
                ),
            )
            bank = Bank(ds, fusion, vision, reference, batch=8)
            smoke_checks(fusion, vision, bank, run)
            assert frozen_before == dict(
                fusion=frozen_digest(fusion), vision=frozen_digest(vision)
            )
            save_json(
                run / "complete.json",
                dict(
                    status="smoke_complete",
                    frozen_before=frozen_before,
                    frozen_after=dict(
                        fusion=frozen_digest(fusion), vision=frozen_digest(vision)
                    ),
                    wall_seconds=time.perf_counter() - begin,
                    checks_sha256=sha256(run / "smoke_checks.json"),
                ),
            )
            return
        fit, val = Bank(parts["T_fit"], fusion, vision, reference), Bank(
            parts["T_val"], fusion, vision, reference
        )
        state.setdefault("bank_resources", []).extend(
            [dict(pool="T_fit", **fit.info), dict(pool="T_val", **val.info)]
        )
        if not (run / "profile.json").exists():
            profile = []
            for fam in ("R0", "R1", "R2"):
                h = init_head(fam, 42, reference.shape[-1])
                vals = fit.get(np.arange(min(TRAIN_BATCH, len(fit.ds))))
                torch.cuda.reset_peak_memory_stats()
                opt = torch.optim.Adam(
                    [q for q in h.parameters() if q.requires_grad],
                    lr=1e-4,
                    weight_decay=1e-4,
                )
                t = time.perf_counter()
                for _ in range(10):
                    opt.zero_grad(set_to_none=True)
                    pred = h(*vals, reference)[0]
                    pred.abs().mean().backward()
                    opt.step()
                torch.cuda.synchronize()
                step = (time.perf_counter() - t) / 10
                profile.append(
                    dict(
                        family=fam,
                        cached_training_step_seconds=step,
                        estimated_epoch_seconds=step
                        * ((len(fit.ds) + TRAIN_BATCH - 1) // TRAIN_BATCH),
                        cached_head_batch_seconds=bench(h, val),
                        **h.parameter_counts(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                )
            json_write(
                run / "profile.json",
                dict(
                    measurements=profile,
                    head_batch=TRAIN_BATCH,
                    encoding_batch=ENCODING_BATCH,
                    GPU=torch.cuda.get_device_name(),
                    synchronized=True,
                    feature_extraction=[fit.info, val.info],
                    estimated_54_cells_30_epochs_seconds=float(
                        np.mean([r["estimated_epoch_seconds"] for r in profile])
                    )
                    * 54
                    * 30,
                    note="head-only cache timing; pipeline bank timings separately include reads/backbones; projections 16 spatial vision tokens, EMG full vs one pooled token",
                ),
            )
            logging.info("PROFILE %s", profile)
        run_probes(args, fit, val, run, identity)
        for family in FAMILIES:
            for condition in ("B0", "B1", "B2"):
                for seed in SEEDS:
                    train_cell(
                        args,
                        family,
                        condition,
                        seed,
                        fit,
                        val,
                        "fit",
                        run,
                        identity,
                        state,
                    )
                    assert not any(m.training for m in fusion.modules()) and not any(
                        m.training for m in vision.modules()
                    )
            assert frozen_before == dict(
                fusion=frozen_digest(fusion), vision=frozen_digest(vision)
            )
        chosen = lock_selection(run, state, identity)
        del fit, val
        torch.cuda.empty_cache()
        score_evaluation(
            args, sets, fusion, vision, reference, state, "fit", FAMILIES, run, identity
        )
        families = sorted({"R0", chosen})
        if not any(k.startswith("all_") for k in state["cells"]):
            state["cells"].update(planned_cells("all", families))
            save_json(run / "state.json", state)
        allbank = Bank(sets[EPISODES[0]]["T_all"], fusion, vision, reference)
        state.setdefault("bank_resources", []).append(
            dict(pool="T_all", **allbank.info)
        )
        for family in families:
            for condition in ("B0", "B1", "B2"):
                for seed in SEEDS:
                    train_cell(
                        args,
                        family,
                        condition,
                        seed,
                        allbank,
                        None,
                        "all",
                        run,
                        identity,
                        state,
                    )
        del allbank
        score_evaluation(
            args, sets, fusion, vision, reference, state, "all", families, run, identity
        )
        frozen_after = dict(fusion=frozen_digest(fusion), vision=frozen_digest(vision))
        assert frozen_before == frozen_after
        state.update(
            status="training_and_scoring_complete",
            finished_at=now(),
            frozen_before=frozen_before,
            frozen_after=frozen_after,
            wall_seconds=time.perf_counter() - begin,
        )
        save_json(run / "state.json", state)
        save_json(
            run / "training_complete.json",
            dict(
                identity=identity,
                status=state["status"],
                selection_lock_sha256=sha256(run / "selection_lock.json"),
                cells=len(state["cells"]),
                all_complete=all(
                    c["status"] == "complete" for c in state["cells"].values()
                ),
                completed_at=now(),
            ),
        )
    except BaseException as exc:
        failures = run / "failures"
        failures.mkdir(exist_ok=True)
        save_json(
            failures / f"{time.time_ns()}.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                recorded_at=now(),
                wall_seconds=time.perf_counter() - begin,
            ),
        )
        logging.exception("Preserved failure; suite is NOT complete")
        raise


if __name__ == "__main__":
    main()
