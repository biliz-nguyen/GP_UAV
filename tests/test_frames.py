"""Independent axis, norm, and heading checks for the ENU research frame."""

import numpy as np
import pytest

from src.log_pipeline.frames import enu_to_ned, ned_to_enu, yaw_ned_deg_to_enu_rad


def test_ned_to_enu_swaps_horizontal_axes_and_negates_down():
    source = np.array([[1, 2, 3], [-4, 5, -6]], dtype=np.int64)
    np.testing.assert_array_equal(ned_to_enu(source), [[2, 1, -3], [5, -4, 6]])
    np.testing.assert_array_equal(source, [[1, 2, 3], [-4, 5, -6]])


def test_enu_to_ned_accepts_single_vector_and_preserves_positive_up_sign():
    np.testing.assert_array_equal(enu_to_ned([7, -2, 9]), [-2, 7, -9])


def test_roundtrip_and_norm_preservation_for_arbitrary_batch_shape():
    vectors = np.random.default_rng(427).normal(size=(4, 11, 3))
    converted = ned_to_enu(vectors)
    assert converted.shape == vectors.shape
    np.testing.assert_array_equal(enu_to_ned(converted), vectors)
    np.testing.assert_allclose(np.linalg.norm(converted, axis=-1), np.linalg.norm(vectors, axis=-1))


@pytest.mark.parametrize("bad_shape", [1.0, [], [1, 2], [[1, 2], [3, 4]]])
@pytest.mark.parametrize("transform", [ned_to_enu, enu_to_ned])
def test_vector_transforms_reject_missing_three_axis_dimension(transform, bad_shape):
    with pytest.raises(ValueError, match="three|3"):
        transform(bad_shape)


def test_invalid_component_does_not_destroy_other_axes():
    np.testing.assert_allclose(ned_to_enu([np.nan, 2, -3]), [2, np.nan, 3], equal_nan=True)


def test_yaw_cardinal_directions_are_radians_counterclockwise_from_east():
    result = yaw_ned_deg_to_enu_rad([0, 90, 180, 270, 360])
    np.testing.assert_allclose(result, [np.pi / 2, 0, -np.pi / 2, -np.pi, np.pi / 2], atol=1e-15)


def test_yaw_wrap_is_half_open_and_preserves_small_motion_across_north():
    result = yaw_ned_deg_to_enu_rad([-90, 450, 359, 1, 720, -360])
    np.testing.assert_allclose(
        result,
        [-np.pi, 0, 91 * np.pi / 180, 89 * np.pi / 180, np.pi / 2, np.pi / 2],
        atol=1e-14,
    )
    assert np.all(result >= -np.pi)
    assert np.all(result < np.pi)


def test_yaw_nonfinite_input_remains_invalid():
    assert np.isnan(yaw_ned_deg_to_enu_rad(np.nan))
