"""Controlled frozen-feature finger-query residuals; all trainable dropout is zero."""

from __future__ import annotations

import torch
from torch import nn


FAMILIES = tuple(f"R{i}" for i in range(7))
DIFFERENCE = {"R2", "R3", "R4"}
POSE_AUX = {"R3", "R4", "R5", "R6"}
DYNAMIC_AUX = {"R4", "R5"}


def positions(length: int, width: int, device, dtype) -> torch.Tensor:
    """Ordered centered token indices, not validated physical receptive fields."""
    t = torch.linspace(-1, 1, length, device=device, dtype=dtype)[:, None]
    freq = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype) * (-9.210340372 / width)
    )
    p = t * freq[None] * 100
    return torch.stack((p.sin(), p.cos()), -1).flatten(-2)


class InnovationHead(nn.Module):
    def __init__(self, family: str, emg_dim: int = 256, width: int = 128):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.family = family
        # Create ALL shared modules in identical order, including unused pool/aux,
        # so reseeding makes corresponding tensors exactly identical across R0-R6.
        self.adapter = nn.Sequential(
            nn.Linear(emg_dim, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.vision_proj = nn.Linear(512, width)
        self.context_proj = nn.Linear(512 + 20, width)
        self.finger = nn.Parameter(torch.empty(5, width))
        nn.init.normal_(self.finger, std=0.02)
        self.vision_attention = nn.MultiheadAttention(
            width, 4, dropout=0, batch_first=True
        )
        self.qnorm = nn.LayerNorm(width)
        self.emg_attention = nn.MultiheadAttention(
            width, 4, dropout=0, batch_first=True
        )
        self.hnorm = nn.LayerNorm(width)
        self.interact = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.output = nn.Linear(width, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.pool = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1))
        self.aux_query = nn.Parameter(torch.empty(1, width))
        nn.init.normal_(self.aux_query, std=0.02)
        self.aux_attention = nn.MultiheadAttention(
            width, 4, dropout=0, batch_first=True
        )
        self.aux_readout = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 60)
        )
        # Disabled modules do not enter optimizer or declared inference parameters.
        self.pool.requires_grad_(family == "R0")
        for mod in (self.aux_attention, self.aux_readout):
            mod.requires_grad_(family in POSE_AUX)
        self.aux_query.requires_grad_(family in POSE_AUX)

    def condition(self, spatial, global_feature, anchor):
        b = len(spatial)
        grid = torch.linspace(-1, 1, 4, device=spatial.device, dtype=spatial.dtype)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        pos = torch.stack([xx, yy], -1).reshape(16, 2).repeat(1, 64)
        tokens = self.vision_proj(spatial) + pos
        q = (
            self.finger[None].expand(b, -1, -1)
            + self.context_proj(torch.cat((global_feature, anchor), -1))[:, None]
        )
        v = self.vision_attention(q, tokens, tokens, need_weights=False)[0]
        return self.qnorm(q + v)

    def adapted(self, emg):
        z = self.adapter(emg)
        return z + positions(z.shape[1], z.shape[2], z.device, z.dtype)[None]

    def read(self, q, z):
        if self.family == "R0":
            z = (self.pool(z).softmax(1) * z).sum(1, keepdim=True)
        a = self.emg_attention(q, z, z, need_weights=False)[0]
        h = self.hnorm(q + a)
        return h + self.interact(h)

    def forward(self, emg, spatial, global_feature, anchor, reference, auxiliary=False):
        q = self.condition(spatial, global_feature, anchor)
        z = self.adapted(emg)
        h = self.read(q, z)
        delta = self.output(h).reshape(-1, 20)
        if self.family in DIFFERENCE:
            # Recompute current adapter/H, share q, retain gradients on both paths.
            ref = self.adapted(reference.expand(len(emg), -1, -1))
            delta = delta - self.output(self.read(q, ref)).reshape(-1, 20)
        pred = anchor + delta
        aux = None
        if auxiliary and self.family in POSE_AUX:
            aq = self.aux_query[None].expand(len(emg), -1, -1)
            az = self.aux_attention(aq, z, z, need_weights=False)[0][:, 0]
            aux = self.aux_readout(az).reshape(-1, 3, 20)
        return pred, aux

    def parameter_counts(self):
        inference = [
            p
            for n, p in self.named_parameters()
            if not n.startswith("aux_")
            and (self.family == "R0" or not n.startswith("pool."))
        ]
        return dict(
            stored=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            inference=sum(p.numel() for p in inference),
        )


class TemporalProbe(nn.Module):
    """Same small ordered temporal reader for every target and condition."""

    def __init__(self, emg_dim=256, width=128, visual=False):
        super().__init__()
        self.adapter = nn.Linear(emg_dim, width)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.visual = visual
        self.head = nn.Sequential(
            nn.Linear(width + (512 + 20 if visual else 0), width),
            nn.GELU(),
            nn.Linear(width, 20),
        )

    def forward(self, e, global_feature, anchor):
        z = self.adapter(e)
        z = z + positions(z.shape[1], z.shape[2], z.device, z.dtype)[None]
        h = self.attn(self.query.expand(len(e), -1, -1), z, z, need_weights=False)[0][
            :, 0
        ]
        if self.visual:
            h = torch.cat((h, global_feature, anchor), -1)
        return self.head(h)
