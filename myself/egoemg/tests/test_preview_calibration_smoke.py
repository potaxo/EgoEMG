import numpy as np
import pytest
import torch

from egoemg.models.modules.base import BaseModule
from myself.scripts.analysis.preview_calibration_smoke import (
    OUTPUT_PARENT,
    PERSONAL_ROOT,
    REPO,
    aligned_center,
    angular_mae,
    apply_offsets,
    assert_isolation,
    fit_offsets,
    select_windows,
)


def test_personal_workflow_resolves_original_assets_and_personal_outputs():
    assert PERSONAL_ROOT == REPO / "myself"
    assert (REPO / "config/base.yaml").is_file()
    assert OUTPUT_PARENT == PERSONAL_ROOT / "outputs/preview_calibration_smoke"


def test_center_train_does_not_allow_support_crossing_split():
    split = np.r_[np.ones(6, dtype=int), np.zeros(40, dtype=int)]
    starts = np.array([2, 20])  # first center (7) train, first support crosses boundary
    assert split[starts[0] + 5] == 0
    with pytest.raises(AssertionError, match="crosses split"):
        assert_isolation(
            starts, np.array(["calibration", "evaluation"]), 10, (0, 46), split, 0
        )


def test_phase_overlap_rejected_even_with_train_supports():
    with pytest.raises(AssertionError, match="overlap"):
        assert_isolation(
            np.array([0, 5]),
            np.array(["calibration", "evaluation"]),
            10,
            (0, 30),
            np.zeros(30),
            0,
        )


def test_earliest_valid_span_and_complete_supports():
    split = np.r_[np.ones(50), np.zeros(200)]
    valid = np.ones(250, dtype=bool)
    valid[55] = False
    starts, phases, selection = select_windows(
        split, valid, np.zeros((250, 8)), np.zeros((250, 20)), 0, 20, 10, 0.02, 0.03
    )
    assert starts[0] == 56
    assert selection["train_block_local"] == [50, 250]
    assert_isolation(starts, phases, 20, (50, 250), split, 0)
    assert (starts[phases == "calibration"] + 20).max() == starts[
        phases == "evaluation"
    ].min()


def test_nonfinite_input_excluded_before_interval_selection():
    emg = np.zeros((200, 8))
    emg[0, 2] = np.nan
    starts, _, _ = select_windows(
        np.zeros(200),
        np.ones(200, dtype=bool),
        emg,
        np.zeros((200, 20)),
        0,
        20,
        10,
        0.01,
        0.02,
    )
    assert starts[0] == 1


def test_requested_durations_reduce_only_if_full_pair_cannot_fit():
    starts, phases, selection = select_windows(
        np.zeros(41000),
        np.ones(41000, dtype=bool),
        np.zeros((41000, 8)),
        np.zeros((41000, 20)),
        0,
        500,
        200,
        15,
        30,
    )
    assert selection["reduced_durations"]
    assert selection["calibration_seconds"] == 5
    assert selection["evaluation_seconds"] == 15
    assert_isolation(starts, phases, 500, (0, 41000), np.zeros(41000), 0)


def test_constant_residual_regularization_and_missing_coverage():
    pred = np.zeros((4, 3))
    target = np.tile([2.0, -4.0, np.nan], (4, 1))
    mask = np.tile([True, True, False], (4, 1))
    offsets, counts = fit_offsets(pred, target, mask, 1.0)
    np.testing.assert_allclose(offsets, [1, -2, 0])
    np.testing.assert_array_equal(counts, [4, 4, 0])


def test_masked_aggregate_mae_matches_repository_and_degrees():
    from egoemg.metrics import AngleMAE

    pred = np.array([[np.pi, 500], [0, np.pi / 2]])
    ref = np.zeros((2, 2))
    mask = np.array([[True, False], [False, True]])
    result = angular_mae(pred, ref, mask)
    assert result["valid_entries"] == 2
    assert result["mae_deg"] == pytest.approx(135)
    existing = AngleMAE()(
        torch.tensor(pred)[..., None],
        torch.tensor(ref)[..., None],
        torch.tensor(mask)[..., None],
        "test",
    )["test_mae"]
    assert result["mae_rad"] == pytest.approx(existing.item())
    assert angular_mae(pred, ref, np.zeros_like(mask))["mae_rad"] is None


def test_later_predictions_use_fitted_parameters_unchanged():
    offsets, _ = fit_offsets(
        np.zeros((3, 2)), np.ones((3, 2)) * 2, np.ones((3, 2), dtype=bool), 1.0
    )
    original = offsets.copy()
    later = np.array([[100, -10], [-20, 30]])
    np.testing.assert_allclose(apply_offsets(later, offsets), later + 1)
    np.testing.assert_array_equal(offsets, original)
    # Wrist outputs are preserved when only finger offsets are supplied.
    padded = np.r_[offsets, np.zeros(2)]
    np.testing.assert_array_equal(
        apply_offsets(np.zeros((1, 4)), padded)[0, 2:], [0, 0]
    )


def test_zero_correction_metric_is_identical_across_prediction_dtypes():
    pred = np.array([[0.1, 0.4]], dtype=np.float32)
    ref = np.array([[0.7, -0.1]], dtype=np.float32)
    mask = np.ones_like(pred, dtype=bool)
    assert angular_mae(pred, ref, mask) == angular_mae(
        apply_offsets(pred, np.zeros(2)), ref, mask
    )


def test_alignment_uses_input_center_after_context_and_resampling():
    class Timeline:
        left_context = 4
        right_context = 0
        align_predictions = BaseModule.align_predictions

    raw = torch.arange(4.0).reshape(1, 1, 4)
    result = aligned_center(Timeline(), raw, 20)
    expected = torch.nn.functional.interpolate(
        raw, size=16, mode="linear", align_corners=False
    )[..., 6]
    torch.testing.assert_close(result, expected)
    assert result.item() != raw[..., raw.shape[-1] // 2].item()
