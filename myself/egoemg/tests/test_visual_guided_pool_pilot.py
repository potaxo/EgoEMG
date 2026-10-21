"""Conditioning must preserve the baseline at initialization and affect time weights."""

from types import SimpleNamespace

import torch
from torch import nn

from egoemg.models.modules._pooling import TemporalAttentionPool
from myself.scripts.analysis.visual_guided_pool_pilot import PoolingBranch


def fixture():
    torch.manual_seed(42)
    released = SimpleNamespace(
        temporal_attn=TemporalAttentionPool(8),
        vision_proj=nn.Linear(512, 4),
        fusion_proj=nn.Conv1d(12, 8, 1),
        head=nn.Conv1d(8, 22, 1),
    )
    inputs = (torch.randn(3, 8, 7), torch.randn(3, 512), torch.randn(3, 22))
    return released, inputs


def test_zero_condition_preserves_released_pool_and_residual_path():
    released, (emg, vision, pose) = fixture()
    original = PoolingBranch(released)
    guided = PoolingBranch(released, True)
    expected = (
        pose
        + released.head(
            released.fusion_proj(
                torch.cat(
                    [released.temporal_attn(emg), released.vision_proj(vision)], -1
                )[..., None]
            )
        )[..., 0]
    )
    torch.testing.assert_close(original(emg, vision, pose), expected)
    torch.testing.assert_close(guided(emg, vision, pose), expected)
    assert (
        sum(p.numel() for p in guided.parameters())
        - sum(p.numel() for p in original.parameters())
        == 512 * 16
    )


def test_guidance_has_nonzero_gradient_and_changes_temporal_choice():
    released, (emg, vision, pose) = fixture()
    guided = PoolingBranch(released, True)
    loss = guided(emg, vision, pose).square().mean()
    loss.backward()
    assert guided.condition.weight.grad.abs().max() > 1e-7
    with torch.no_grad():
        guided.condition.weight.normal_(std=0.1)
    tokens = guided.pool[0](emg.transpose(1, 2))
    a = guided.pool[2](torch.tanh(tokens)).softmax(1)
    b = guided.pool[2](torch.tanh(tokens + guided.condition(vision)[:, None])).softmax(
        1
    )
    assert (a - b).abs().max() > 1e-5
