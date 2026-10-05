"""Protocol, numerical fitting, supervision and report invariants."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from egoemg.models.modules.base import BaseModule
from myself.scripts.analysis.single_subject_calibration_suite import (
    SEEDS,
    aligned_center,
    angular_mae,
    apply_fingers,
    assert_support_separation,
    completed,
    coverage,
    enumerate_windows,
    evaluated_predictions,
    fit_affine,
    fit_offsets,
    json_write,
    mark,
    npz_write,
    rgb_reason,
    summarize_seeds,
    train_head,
    union_intervals,
)


def fixture_windows():
    split = np.repeat([0, 1, 0, 2, 3, 0], [25, 25, 25, 25, 25, 3])
    valid = np.ones(len(split), bool)
    # Context invalidity does not invalidate a center-only window.
    valid[0] = False
    return split, valid, np.zeros((len(split), 8)), np.zeros((len(split), 20))


def test_enumeration_interleaved_intervals_and_deterministic_order():
    split, valid, emg, ref = fixture_windows()
    args = (
        split,
        valid,
        emg,
        ref,
        {0: "train", 1: "user", 2: "gesture", 3: "both"},
        10,
        5,
    )
    starts, pools, intervals = enumerate_windows(*args)
    assert starts.tolist() == [
        0,
        5,
        10,
        15,
        25,
        30,
        35,
        40,
        50,
        55,
        60,
        65,
        75,
        80,
        85,
        90,
        100,
        105,
        110,
        115,
    ]
    assert intervals[-1]["insufficient_length"]
    assert intervals[-1]["candidate_windows"] == 0
    assert_support_separation(starts, pools, 10, split, 0)
    # Train occurring after evaluation must be accepted.
    assert starts[pools == "T_all"].max() > starts[pools == "E"].min()
    other = enumerate_windows(*args)
    np.testing.assert_array_equal(starts, other[0])
    assert intervals == other[2]


def test_nonfinite_input_and_only_center_reference_checks():
    split, valid, emg, ref = fixture_windows()
    emg[1, 0] = np.nan
    valid[30] = False  # center of start 25
    ref[60, 0] = np.nan  # center of start 55
    starts, _, intervals = enumerate_windows(
        split, valid, emg, ref, {0: "train", 1: "user", 2: "gesture", 3: "both"}, 10, 5
    )
    assert 0 not in starts and 25 not in starts and 55 not in starts
    assert 50 in starts  # NaN reference in context but valid center
    assert intervals[0]["exclusions"] == {"nonfinite_input": 1}
    assert intervals[1]["exclusions"] == {"invalid_center_supervision": 1}
    assert intervals[2]["exclusions"] == {"nonfinite_center_supervision": 1}


def test_cross_split_support_rejected():
    split = np.repeat([0, 1], 10)
    with pytest.raises(AssertionError, match="crosses split"):
        assert_support_separation(np.array([5]), np.array(["T_all"]), 10, split, 0)


def test_union_coverage_does_not_sum_overlaps():
    assert union_intervals([(0, 10), (5, 15), (30, 40)]) == [[0, 15], [30, 40]]
    result = coverage(np.array([0, 5, 30]), 10, 5)
    assert result["support_union_rows"] == 25
    assert result["nominal_center_seconds"] == 15 / 2000
    assert result["overlapping_supports"]


@pytest.mark.parametrize(
    "args,reason",
    [
        ((-1, False, 0), "mapping"),
        ((1, True, 0), "stale"),
        ((1, False, np.nan), "nonfinite_delta"),
        ((1, False, 51), "delta_tolerance"),
        ((1, False, -50), "eligible"),
    ],
)
def test_distinct_rgb_mapping_exclusions(args, reason):
    assert rgb_reason(*args, 50) == reason


@pytest.mark.parametrize("reason", ["missing_crop", "decode", "preprocessing"])
def test_rgb_crop_exclusions_and_no_label_argument(reason):
    assert rgb_reason(1, False, 0, 50, reason) == reason
    assert rgb_reason(1, False, 0, 50, "ok", False) == "nonfinite_prediction"
    # Both arbitrary reference labels give the same eligibility; no reference is read.
    assert rgb_reason(1, False, 0, 50) == "eligible"


def test_regularized_affine_known_solution_and_stationarity():
    p = np.linspace(-1, 1, 50)[:, None]
    target = 2 * p + 0.5
    result = fit_affine(p, target, np.ones_like(p, bool))
    expected_scale = 1 + p.var() / (p.var() + 0.01)
    assert result["scales"][0] == pytest.approx(expected_scale)
    assert result["offsets"][0] == pytest.approx(0.25)
    a, b = result["scales"][0], result["offsets"][0]
    residual = a * p[:, 0] + b - target[:, 0]
    assert np.mean(residual * p[:, 0]) + 0.01 * (a - 1) == pytest.approx(0, abs=1e-14)
    assert residual.mean() + b == pytest.approx(0, abs=1e-14)


def test_offset_empty_identity_low_variation_affine():
    p = np.ones((3, 2))
    target = np.tile([3, np.nan], (3, 1))
    mask = np.tile([True, False], (3, 1))
    offsets, counts = fit_offsets(p, target, mask, 1)
    np.testing.assert_array_equal(offsets, [1, 0])
    np.testing.assert_array_equal(counts, [3, 0])
    result = fit_affine(p, target, mask)
    np.testing.assert_array_equal(result["scales"], [1, 1])
    np.testing.assert_array_equal(result["offsets"], [1, 0])
    assert result["fallbacks"] == ["low_variation_offset", "empty_identity"]


def test_cache_original_linear_equivalence_after_full_alignment():
    class Timeline:
        left_context = 4
        right_context = 2
        align_predictions = BaseModule.align_predictions

    torch.manual_seed(42)
    model = Timeline()
    # Features are actual last-linear inputs after a nonlinear preceding MLP.
    features = torch.relu(torch.randn(2, 8, 11))
    weight, bias = torch.randn(22, 8), torch.randn(22)
    raw = F.linear(features.transpose(1, 2), weight, bias).transpose(1, 2)
    original = aligned_center(model, raw, 40)
    compact = F.linear(aligned_center(model, features, 40), weight, bias)
    torch.testing.assert_close(original, compact, atol=1e-5, rtol=1e-4)


def test_only_finger_rows_train_wrist_fixed_and_designated_targets():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(10, 4)).astype(np.float32)
    w = rng.normal(size=(22, 4)).astype(np.float32)
    b = rng.normal(size=22).astype(np.float32)
    w_before, b_before = w.copy(), b.copy()
    baseline = x @ w.T + b
    reference = baseline[:, :20] + 0.8
    teacher = baseline[:, :20] - 0.4
    ref_params, stats = train_head(x, reference, w, b, 42, epochs=2, device="cpu")
    rgb_params, _ = train_head(x, teacher, w, b, 42, epochs=2, device="cpu")
    assert stats["updated_parameters"] == 20 * (4 + 1)
    assert ref_params["weight"].shape == (20, 4)
    assert not np.array_equal(ref_params["bias"], rgb_params["bias"])
    np.testing.assert_array_equal(w, w_before)
    np.testing.assert_array_equal(b, b_before)
    for params in (ref_params, rgb_params):
        pred = evaluated_predictions(baseline, x, "head", params)
        np.testing.assert_array_equal(pred[:, 20:], baseline[:, 20:])
        # Evaluation ground truth remains reference, never the fitting teacher.
        expected = np.abs(pred[:, :20] - reference).mean()
        assert angular_mae(pred[:, :20], reference, np.ones_like(reference, bool))[
            "mae_rad"
        ] == pytest.approx(expected)


def test_empty_head_keeps_released_parameters():
    w, b = np.zeros((22, 4), np.float32), np.zeros(22, np.float32)
    params, stats = train_head(
        np.empty((0, 4), np.float32),
        np.empty((0, 20)),
        w,
        b,
        42,
        epochs=1,
        device="cpu",
    )
    np.testing.assert_array_equal(params["weight"], w[:20])
    assert stats["supervision_centers"] == 0


def test_evaluation_ids_wrist_and_masked_degrees():
    ids = np.array(["e:1", "e:2"])
    baseline = np.zeros((2, 22), np.float32)
    for offset in (0.1, 0.2):
        out = apply_fingers(baseline, np.full((2, 20), offset))
        np.testing.assert_array_equal(ids, ["e:1", "e:2"])
        np.testing.assert_array_equal(out[:, 20:], baseline[:, 20:])
    p = np.array([[np.pi, np.nan]])
    result = angular_mae(p, np.zeros_like(p), np.array([[True, False]]))
    assert result["mae_deg"] == pytest.approx(180)
    assert result["valid_entries"] == 1


def test_summary_uses_every_seed_sample_standard_deviation():
    rows = [
        dict(
            id="M3",
            key=f"M3_seed{s}",
            method="head",
            seed=s,
            mae_rad=float(i),
            mae_deg=float(i),
            delta_deg=float(i),
            delta_rad=float(i),
            fitting_seconds=1.0,
            peak_gpu_bytes=100,
        )
        for i, s in enumerate(SEEDS, 1)
    ]
    result = summarize_seeds(rows)[0]
    assert result["mae_deg"] == 2
    assert result["std_deg"] == 1
    with pytest.raises(AssertionError):
        summarize_seeds(rows[:2])


def test_atomic_completion_identity_and_corruption(tmp_path):
    path = tmp_path / "cache.npz"
    npz_write(path, ids=np.array([1, 2]))
    ident = {"version": 1}
    assert not completed(tmp_path, "emg", ident)
    mark(tmp_path, "emg", ident, ["cache.npz"])
    assert completed(tmp_path, "emg", ident)
    with pytest.raises(ValueError, match="Incompatible"):
        completed(tmp_path, "emg", {"version": 2})
    path.write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="corrupt"):
        completed(tmp_path, "emg", ident)


def test_phase_integration_keeps_fit_pools_targets_and_fixed_evaluation(
    tmp_path, monkeypatch
):
    from argparse import Namespace

    import myself.scripts.analysis.single_subject_calibration_suite as suite

    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 4)).astype(np.float32)
    w = rng.normal(size=(22, 4)).astype(np.float32)
    b = rng.normal(size=22).astype(np.float32)
    baseline = x @ w.T + b
    ref = baseline + 0.2
    rgb = baseline[:3] - 0.3
    ids = np.array(["T1", "E1", "T2", "E2", "T3"])
    pools = np.array(["T_all", "E", "T_all", "E", "T_all"])
    train, selected = np.array([0, 2, 4]), np.array([0, 4])
    rgb = baseline[train] - 0.3
    ident = {"version": 1, "finger_joints": [str(j) for j in range(20)]}
    npz_write(
        tmp_path / "manifest.npz",
        ids=ids,
        pools=pools,
        reference=ref,
        mask=np.ones_like(ref, bool),
        centers=np.arange(5),
        gesture=np.arange(5),
        active=np.ones(5, bool),
        gesture_name=np.array([str(j) for j in range(5)]),
        split=np.array(["train", "user", "train", "both", "train"]),
    )
    npz_write(
        tmp_path / "emg_cache.npz",
        ids=ids,
        prediction=baseline,
        features=x,
        weight=w,
        bias=b,
    )
    npz_write(
        tmp_path / "teacher_cache.npz",
        train_ids=ids[train],
        train_indices=train,
        eligible_indices=selected,
        frame=np.array([10, 20, 30]),
        reason=np.array(["eligible", "missing_crop", "eligible"]),
        target=rgb,
    )
    json_write(tmp_path / "emg_evidence.json", {"parameters": 1000})
    mark(tmp_path, "teacher", ident, ["teacher_cache.npz"])
    calls = []

    def fake_train(features, targets, w0, b0, seed, **kwargs):
        calls.append((features.copy(), targets.copy(), seed))
        return {"weight": w0[:20].copy(), "bias": b0[:20].copy()}, {
            "updated_parameters": 100,
            "peak_gpu_bytes": 0,
            "fitting_seconds": 0,
        }

    monkeypatch.setattr(suite, "train_head", fake_train)
    suite.fitting_stage(Namespace(feature_batch_size=256), tmp_path, ident)
    assert len(calls) == 9  # reference T_all, RGB T_rgb, reference T_rgb, every seed
    for i in range(3):
        np.testing.assert_array_equal(calls[i][0], x[train])
        np.testing.assert_array_equal(calls[i][1], ref[train, :20])
        np.testing.assert_array_equal(calls[i + 3][0], x[selected])
        np.testing.assert_array_equal(calls[i + 3][1], rgb[[0, 2], :20])
        np.testing.assert_array_equal(calls[i + 6][1], ref[selected, :20])
        assert calls[i][2] == SEEDS[i]
    for key, expected in [("M1", train), ("M4", selected), ("R1", selected)]:
        np.testing.assert_array_equal(
            np.load(tmp_path / f"parameters/{key}.npz")["fit_ids"], ids[expected]
        )
    suite.evaluation_stage(tmp_path, ident)
    pred = np.load(tmp_path / "evaluation_predictions.npz")
    np.testing.assert_array_equal(pred["ids"], ids[[1, 3]])
    np.testing.assert_array_equal(pred["reference"], ref[[1, 3]])
    scores = json.loads((tmp_path / "metrics.json").read_text())
    assert len(scores["all_seeds"]) == 16  # baseline + 6 analytic + 9 head seeds
    for row in scores["all_seeds"]:
        np.testing.assert_array_equal(pred[row["key"]][:, 20:], baseline[[1, 3], 20:])
        actual = angular_mae(
            pred[row["key"]][:, :20], ref[[1, 3], :20], np.ones((2, 20), bool)
        )
        assert row["mae_rad"] == actual["mae_rad"]
    assert scores["blocked"] == {}


def test_traces_stop_at_gaps_and_gesture_changes():
    from myself.scripts.analysis.single_subject_calibration_suite import (
        contiguous_trace_indices,
    )

    centers = np.array([100, 300, 500, 10000, 10200])
    gestures = np.array([1, 1, 2, 1, 1])
    np.testing.assert_array_equal(
        contiguous_trace_indices(np.array([0, 1, 3, 4]), centers, gestures, 200), [0, 1]
    )
    np.testing.assert_array_equal(
        contiguous_trace_indices(np.arange(5), centers, gestures, 200), [0, 1]
    )
