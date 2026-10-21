#!/usr/bin/env python3
"""One controlled pooling change with frozen released fusion encoders."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from myself.scripts.analysis.common_modality_evaluation import (
    ROOT,
    SOURCE,
    configuration,
    make_dataset,
)
from myself.scripts.analysis.preview_calibration_smoke import open_field, sha256
from myself.scripts.analysis.single_subject_calibration_suite import (
    SEEDS,
    angular_mae,
    csv_write,
    json_write,
    npz_write,
    rgb_reason,
    strict_model,
)


class PoolingBranch(nn.Module):
    """Identical residual pathway, optionally condition temporal scores on RGB."""

    def __init__(self, released, guided=False):
        super().__init__()
        self.pool = copy.deepcopy(released.temporal_attn)
        self.vision_proj = copy.deepcopy(released.vision_proj)
        self.fusion_proj = copy.deepcopy(released.fusion_proj)
        self.head = copy.deepcopy(released.head)
        self.condition = None
        if guided:
            self.condition = nn.Linear(512, self.pool[0].out_features, bias=False)
            nn.init.zeros_(self.condition.weight)
        self.requires_grad_(True)

    def forward(self, emg, vision, visual_pose):
        tokens = emg.transpose(1, 2)
        if self.condition is None:
            scores = self.pool[2](self.pool[1](self.pool[0](tokens)))
        else:
            scores = self.pool[2](
                self.pool[1](self.pool[0](tokens) + self.condition(vision)[:, None])
            )
        weights = scores.softmax(dim=1)
        pooled = (emg * weights.transpose(1, 2)).sum(-1)
        fused = torch.cat([pooled, self.vision_proj(vision)], dim=-1).unsqueeze(-1)
        return visual_pose + self.head(self.fusion_proj(fused))[..., 0]


class DirectCenters(Dataset):
    def __init__(self, base, centers, references):
        self.base, self.centers, self.references = base, centers, references
        self.ep = base._episode_id.index("episode_000020")
        mf = json.loads((ROOT / "manifest.json").read_text())
        self.frame = open_field(ROOT, mf, "image_head_frame_index")[centers]
        stale = open_field(ROOT, mf, "image_head_stale")[centers]
        delta = open_field(ROOT, mf, "image_head_delta_ms")[centers]
        self.eligible = np.array(
            [
                rgb_reason(int(f), bool(s), float(d), 50.0) == "eligible"
                for f, s, d in zip(self.frame, stale, delta)
            ]
        )
        # Check decode availability before fitting; no image arrays are retained.
        for i in np.flatnonzero(self.eligible):
            rgb = base._read_episode_crops(
                "episode_000020", np.array([self.frame[i]]), "R"
            )
            self.eligible[i] = (
                rgb is not None
                and rgb.shape == (1, 256, 256, 3)
                and rgb.dtype == np.uint8
            )
        self.indices = np.flatnonzero(self.eligible)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        center = int(self.centers[i])
        sample = self.base._getitem_center_supervised_emg(
            self.ep, center - 3895, center + 3895
        )
        rgb = self.base._read_episode_crops(
            "episode_000020", np.array([self.frame[i]]), "R"
        )[0]
        img = (
            rgb.transpose(2, 0, 1).astype(np.float32)
            - self.base.vision_mean[:, None, None]
        ) / self.base.vision_std[:, None, None]
        assert np.isfinite(img).all() and np.isfinite(sample["emg"]).all()
        return sample["emg"], img, self.references[i, :20]


def frozen_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def extract(model, x, img):
    with torch.no_grad():
        emg = model.decoder(model.featurizer(x))
        vision = model._extract_vision_features(img)
        pose = model.head_vision(vision)
    return emg, vision, pose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(REPO / "myself/outputs"):
        raise ValueError(run)
    run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, run / "executed_workflow.py")
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
    m = np.load(SOURCE / "manifest.npz")
    train = m["pools"] == "T_all"
    test = m["pools"] == "E"
    base = make_dataset(True)
    train_ds = DirectCenters(base, m["centers"][train], m["reference"][train])
    test_ds = DirectCenters(base, m["centers"][test], m["reference"][test])
    test_ids = m["ids"][test][test_ds.indices]
    cfg = configuration("fusion/fusion_rn18_s_center_16ch_wl7790")
    cfg.module.vision_pretrained_checkpoint = str(
        REPO / "checkpoints/vision_resnet18.ckpt"
    )
    model, _ = strict_model(
        cfg.module, REPO / "checkpoints/fusion_resnet18_emgfusion_center.ckpt"
    )
    before = frozen_digest(model)
    settings = dict(
        epochs=args.epochs,
        seeds=list(SEEDS),
        train_centers=len(train_ds),
        test_centers=len(test_ds),
        batch_size=32,
        lr=0.0001,
        weight_decay=0.0001,
        optimizer="Adam",
        loss="20-finger MAE in radians",
        precision="FP32; TF32 disabled",
        selection="fixed final epoch, no tuning/early stopping",
        frozen="entire released model; featurizer, decoder, vision backbone and vision head reused in eval/no_grad",
        trainable="independent copies of temporal pool, vision projection, fusion projection and residual head",
        guided_change="zero-initialized 512-to-64 RGB condition added INSIDE tanh before temporal score projection",
        split="original complete-support train vs E intervals; gesture-disjoint, within wmh; not chronological",
        input_sha256={
            str(x): sha256(x)
            for x in [
                Path(__file__),
                REPO / "myself/scripts/analysis/common_modality_evaluation.py",
                SOURCE / "manifest.npz",
                REPO / "checkpoints/fusion_resnet18_emgfusion_center.ckpt",
            ]
        },
    )
    json_write(run / "settings.json", settings)
    predictions = {}
    history = []
    resource = []
    checks = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        branches = {
            "original": PoolingBranch(model).cuda(),
            "guided": PoolingBranch(model, True).cuda(),
        }
        opts = {
            k: torch.optim.Adam(branch.parameters(), lr=0.0001, weight_decay=0.0001)
            for k, branch in branches.items()
        }
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, generator=generator, num_workers=0
        )
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        for epoch in range(args.epochs):
            sums = {k: 0.0 for k in branches}
            for batch_index, (x, img, y) in enumerate(loader):
                x, img, y = x.cuda(), img.cuda(), y.cuda()
                features = extract(model, x, img)
                if epoch == 0 and batch_index == 0:
                    for branch in branches.values():
                        branch.eval()
                    with torch.no_grad():
                        a = branches["original"](*features)
                        b = branches["guided"](*features)
                        raw = model({"emg": x, "vision_img": img})[..., 0]
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(a, raw, atol=1e-6, rtol=1e-5)
                    checks.append(
                        dict(
                            seed=seed,
                            initial_original_released_max=float((a - raw).abs().max()),
                            initial_guided_original_max=float((a - b).abs().max()),
                        )
                    )
                for k, branch in branches.items():
                    branch.train()
                    # Match dropout draws between the two paired conditions.
                    torch.cuda.manual_seed(seed * 100000 + epoch * 1000 + batch_index)
                    opts[k].zero_grad(set_to_none=True)
                    out = branch(*features)
                    loss = (out[:, :20] - y).abs().mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    opts[k].step()
                    sums[k] += float(loss.detach()) * len(y)
            record = dict(
                seed=seed,
                epoch=epoch + 1,
                **{k: float(np.rad2deg(v / len(train_ds))) for k, v in sums.items()},
            )
            history.append(record)
            logging.info(
                "seed=%d epoch=%d train MAE original=%.4f guided=%.4f elapsed=%.1fs",
                seed,
                epoch + 1,
                record["original"],
                record["guided"],
                time.perf_counter() - begin,
            )
        torch.cuda.synchronize()
        resource.append(
            dict(
                seed=seed,
                paired_training_seconds=time.perf_counter() - begin,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                original_parameters=sum(
                    p.numel() for p in branches["original"].parameters()
                ),
                guided_parameters=sum(
                    p.numel() for p in branches["guided"].parameters()
                ),
            )
        )
        for k, branch in branches.items():
            branch.eval()
            torch.save(branch.state_dict(), run / f"{k}_seed{seed}.pt")
        accum = {k: [] for k in branches}
        with torch.inference_mode():
            for x, img, y in DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=0
            ):
                features = extract(model, x.cuda(), img.cuda())
                for k, branch in branches.items():
                    accum[k].append(branch(*features).cpu().numpy())
        for k in branches:
            predictions[f"{k}_seed{seed}"] = np.concatenate(accum[k])
    assert frozen_digest(model) == before, "Frozen weights or BatchNorm buffers changed"
    reference = m["reference"][test][test_ds.indices]
    gesture = m["gesture"][test][test_ds.indices]
    rows = []
    for key, pred in predictions.items():
        for g in [None] + sorted(np.unique(gesture).tolist()):
            ix = np.ones(len(gesture), bool) if g is None else gesture == g
            score = angular_mae(
                pred[ix, :20], reference[ix, :20], np.isfinite(reference[ix, :20])
            )
            rows.append(
                dict(
                    condition=key,
                    gesture="all" if g is None else g,
                    centers=int(ix.sum()),
                    mae_deg=score["mae_deg"],
                )
            )
    csv_write(run / "metrics.csv", rows)
    csv_write(run / "training.csv", history)
    json_write(run / "resources.json", resource)
    json_write(
        run / "verification.json",
        dict(
            frozen_model_sha256_before=before,
            frozen_model_sha256_after=frozen_digest(model),
            initial_equivalence=checks,
            all_predictions_finite=all(
                np.isfinite(x).all() for x in predictions.values()
            ),
        ),
    )
    npz_write(
        run / "predictions.npz",
        ids=test_ids,
        reference=reference,
        gesture=gesture,
        **predictions,
    )
    print(json.dumps([r for r in rows if r["gesture"] == "all"], indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
