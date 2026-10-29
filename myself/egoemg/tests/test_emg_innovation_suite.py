"""Scientific invariants, using synthetic tensors rather than numerical outcomes."""

import numpy as np
import torch

from myself.egoemg.models.emg_innovation import FAMILIES, InnovationHead
from myself.scripts.analysis.emg_innovation_suite import support_partition


def inputs():
    torch.manual_seed(8)
    return (
        torch.randn(3, 13, 256),
        torch.randn(3, 16, 512),
        torch.randn(3, 512),
        torch.randn(3, 20),
        torch.randn(1, 13, 256),
    )


def test_shared_initialization_and_anchor():
    values = inputs()
    states = []
    for family in FAMILIES:
        torch.manual_seed(42)
        h = InnovationHead(family)
        states.append(h.state_dict())
        torch.testing.assert_close(h(*values)[0], values[3], atol=0, rtol=0)
    for state in states[1:]:
        assert all(torch.equal(state[k], v) for k, v in states[0].items())


def test_reference_cancellation_with_nonzero_output_and_query_interaction():
    e, sp, gl, anchor, ref = inputs()
    h = InnovationHead("R2")
    torch.nn.init.normal_(h.output.weight, std=0.02)
    torch.testing.assert_close(
        h(ref.expand_as(e), sp, gl, anchor, ref)[0], anchor, atol=0, rtol=0
    )
    q = h.condition(sp, gl, anchor)
    z, zr = h.adapted(e), h.adapted(ref.expand_as(e))
    dq = h.read(q, z) - h.read(q, zr)
    dq2 = h.read(q.flip(1), z) - h.read(q.flip(1), zr)
    assert (dq - dq2).abs().max() > 1e-6
    loss = h(e, sp, gl, anchor, ref)[0].square().mean()
    loss.backward()
    assert h.adapter[0].weight.grad.abs().sum() > 0
    # Reference adapter is not detached: explicit expression has identical grads.
    actual = h.adapter[0].weight.grad.clone()
    h.zero_grad()
    q = h.condition(sp, gl, anchor)
    explicit = anchor + (
        h.output(h.read(q, h.adapted(e)))
        - h.output(h.read(q, h.adapted(ref.expand_as(e))))
    ).reshape(-1, 20)
    explicit.square().mean().backward()
    torch.testing.assert_close(actual, h.adapter[0].weight.grad, atol=1e-7, rtol=1e-5)


def test_auxiliary_has_no_visual_input_and_reaches_adapter():
    values = inputs()
    for fam in ("R3", "R4", "R5", "R6"):
        h = InnovationHead(fam)
        _, a = h(*values, auxiliary=True)
        changed = list(values)
        changed[1:4] = [v + 3 for v in changed[1:4]]
        _, b = h(*changed, auxiliary=True)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward()
        assert h.adapter[0].weight.grad.abs().sum() > 0


def test_purge_covers_complete_support_not_just_center():
    ex = np.repeat(np.arange(10), 20000)
    centers = np.arange(4000, len(ex) - 4000, 200)
    labels, val = support_partition(dict(centers=centers), 0, len(ex), ex)
    for c, label in zip(centers, labels):
        chosen = np.isin(ex[c - 3895 : c + 3895], val)
        if label == "T_fit":
            assert not chosen.any()
        elif label == "T_val":
            assert chosen.all()
        else:
            assert chosen.any() and not chosen.all()
    fit, test = centers[labels == "T_fit"], centers[labels == "T_val"]
    assert not (np.abs(fit[:, None] - test[None]) < 7790).any()


def test_independent_metric_weights_angles_not_windows():
    from myself.scripts.analysis.summarize_emg_innovation import weighted

    target = np.zeros((2, 3))
    prediction = np.array([[1.0, 1.0, 1.0], [3.0, 99.0, 99.0]])
    mask = np.array([[True, True, True], [True, False, False]])
    # Four valid angles: mean 1.5, whereas mean of per-window means would be 2.
    assert abs(weighted(prediction, target, mask) - 1.5 * 180 / np.pi) < 1e-12


def test_adam_and_rng_resume_matches_uninterrupted(tmp_path):
    import copy
    import random

    from myself.scripts.analysis.emg_innovation_suite import restore_rng, rng_state

    torch.manual_seed(21)
    np.random.seed(21)
    random.seed(21)
    h = InnovationHead("R4")
    opt = torch.optim.Adam(
        [p for p in h.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )

    base_inputs = inputs()

    def step(model, optimizer):
        ee, sp, gl, anchor, ref = base_inputs
        # The next update depends on Python, NumPy and Torch RNG state.
        order = np.random.permutation(3)
        y = torch.randn(3, 20) * (0.5 + random.random())
        optimizer.zero_grad()
        p, a = model(
            ee[order], sp[order], gl[order], anchor[order], ref, auxiliary=True
        )
        loss = (p - y).abs().mean() + (a[:, 0] - y).abs().mean()
        loss.backward()
        optimizer.step()

    step(h, opt)
    path = tmp_path / "resume.pt"
    torch.save(
        dict(model=h.state_dict(), optimizer=opt.state_dict(), rng=rng_state()), path
    )
    step(h, opt)
    expected = copy.deepcopy(h.state_dict())
    resumed = InnovationHead("R4")
    resumed_opt = torch.optim.Adam(
        [p for p in resumed.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    saved = torch.load(path, weights_only=False)
    resumed.load_state_dict(saved["model"], strict=True)
    resumed_opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    step(resumed, resumed_opt)
    assert all(torch.equal(expected[k], v) for k, v in resumed.state_dict().items())
