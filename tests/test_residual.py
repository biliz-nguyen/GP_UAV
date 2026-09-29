"""The learning target is actual minus target, in the same frame and units."""

import numpy as np
import pytest

from src.log_pipeline.frames import ned_to_enu, residual


def test_residual_is_actual_minus_target_for_every_row():
    actual = np.array([[3.0, -2.0, 5.0], [1.0, 2.0, -4.0]])
    target = np.array([[1.0, 1.0, -2.0], [0.5, -2.0, -1.0]])
    result = residual(actual, target)
    np.testing.assert_array_equal(result, [[2, -3, 7], [0.5, 4, -3]])
    np.testing.assert_array_equal(target + result, actual)


def test_down_to_up_conversion_applies_to_actual_and_target_before_subtraction():
    result = residual(ned_to_enu([1, 2, 7]), ned_to_enu([4, 1, 3]))
    np.testing.assert_array_equal(result, [1, -3, -4])


@pytest.mark.parametrize("actual_frame,target_frame", [("NED", "ENU"), ("ENU", "NED")])
def test_residual_rejects_mismatched_frames(actual_frame, target_frame):
    with pytest.raises(ValueError, match="frame"):
        residual([1, 2, 3], [0, 0, 0], actual_frame=actual_frame, target_frame=target_frame)


def test_residual_rejects_unknown_frame_even_when_labels_match():
    with pytest.raises(ValueError, match="frame"):
        residual([1, 2, 3], [0, 0, 0], actual_frame="unknown", target_frame="unknown")


def test_residual_supports_axis_arrays_and_retains_missing_values():
    result = residual([2, np.nan, 4], [1, 3, np.nan], actual_frame="NED", target_frame="NED")
    np.testing.assert_allclose(result, [1, np.nan, np.nan], equal_nan=True)
