"""Guard against turning split changes or overlapping context into repetitions."""

import numpy as np
import pytest

from myself.scripts.analysis.repeated_gesture_diagnostic import (
    repetition_pairs,
    select_pairs,
)


def test_inactive_gap_defines_new_execution_but_contiguous_class_does_not():
    gesture = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
    active = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    assert repetition_pairs(gesture, active, origin=100) == {
        2: [(100, 103), (104, 109)]
    }
    assert repetition_pairs(gesture, np.ones(9, bool)) == {}


def test_complete_supports_do_not_cross_gesture_boundaries_and_labels_are_center_only():
    gesture = np.repeat([2, 3, 2, 3], 12)
    active = np.ones(48, bool)
    valid = np.ones(48, bool)
    ref = np.zeros((48, 20))
    emg = np.zeros((48, 8))
    valid[0] = False  # Context label invalidity does not discard a valid center.
    ref[28, 0] = np.nan  # Discard this center on the second execution.
    starts, phases, gestures, audit = select_pairs(
        gesture, active, valid, ref, emg, window=8, stride=2
    )
    assert starts.tolist() == [0, 2, 4, 12, 14, 16, 26, 28, 36, 38, 40]
    assert starts[phases == "fit"].max() + 8 <= starts[phases == "later"].min()
    for a, g in zip(starts, gestures):
        assert np.all(gesture[a : a + 8] == g)
    assert audit[0]["later_eligible"] == 2


def test_interleaved_first_and_second_executions_reject_pooled_time_leakage():
    gesture = np.repeat([2, 2, 3, 3], 12)
    active = np.ones(48, bool)
    active[[11, 35]] = False
    with pytest.raises(ValueError, match="global chronological"):
        select_pairs(
            gesture,
            active,
            np.ones(48, bool),
            np.zeros((48, 20)),
            np.zeros((48, 8)),
            window=8,
            stride=2,
        )


def test_pair_without_eligible_second_execution_is_excluded():
    gesture = np.repeat([2, 3, 2, 3], 12)
    valid = np.ones(48, bool)
    valid[24:36] = False
    starts, phases, gestures, audit = select_pairs(
        gesture,
        np.ones(48, bool),
        valid,
        np.zeros((48, 20)),
        np.zeros((48, 8)),
        window=8,
        stride=2,
    )
    assert set(gestures) == {3}
    assert not audit[0]["included"]
