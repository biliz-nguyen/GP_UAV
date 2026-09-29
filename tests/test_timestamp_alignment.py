"""Bounded causal alignment must preserve missing intervals and provenance."""

import numpy as np
import pytest

from src.log_pipeline.alignment import align_previous, align_source


def test_previous_selects_only_past_data_and_preserves_candidate_provenance():
    result = align_previous(
        [1, 1.1], [[10, 20], [11, 21]], [0.99, 1, 1.05, 1.1, 1.3],
        [1, 1], [1, 1, 1, 1, 1], max_age_ms=100, gap_threshold_ms=150,
    )
    np.testing.assert_array_equal(result["valid"], [False, True, True, True, False])
    np.testing.assert_allclose(result["values"], [[np.nan, np.nan], [10, 20], [10, 20], [11, 21], [np.nan, np.nan]], equal_nan=True)
    np.testing.assert_allclose(result["age_ms"], [np.nan, 0, 50, 0, 200], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(result["source_time_s"], [np.nan, 1, 1, 1.1, 1.1], equal_nan=True)


def test_matching_boundary_is_required_and_no_old_sample_is_reused():
    result = align_previous(
        [1, 1.1, 1.2], [[10], [11], [12]], [1.05, 1.15, 1.21],
        ["flight1", "flight2", "flight2"], ["flight2", "flight1", "flight2"],
        max_age_ms=1000, gap_threshold_ms=1000,
    )
    np.testing.assert_array_equal(result["valid"], [False, False, True])
    np.testing.assert_allclose(result["values"][:, 0], [np.nan, np.nan, 12], equal_nan=True)
    np.testing.assert_allclose(result["source_time_s"], [np.nan, np.nan, 1.2], equal_nan=True)


def test_large_gap_expires_hold_at_gap_limit_and_reacquires_at_next_sample():
    result = align_previous(
        [1, 3], [[7], [9]], [1.1, 1.2, 1.2001, 2, 3, 3.01],
        [1, 1], [1] * 6, max_age_ms=1000, gap_threshold_ms=200,
    )
    np.testing.assert_array_equal(result["valid"], [True, True, False, False, True, True])
    np.testing.assert_allclose(result["values"][:, 0], [7, 7, np.nan, np.nan, 9, 9], equal_nan=True)
    np.testing.assert_allclose(result["age_ms"], [100, 200, 200.1, 1000, 0, 10], atol=1e-9)


def test_age_limit_is_inclusive_and_zero_age_allows_exact_matches_only():
    result = align_previous([1], [[7]], [1, 1.1, 1.1001], [0], [0, 0, 0], 100, 500)
    np.testing.assert_array_equal(result["valid"], [True, True, False])
    exact = align_previous([1], [[7]], [1, 1.000001], [0], [0, 0], 0, 500)
    np.testing.assert_array_equal(exact["valid"], [True, False])


def test_duplicate_source_times_use_last_record_including_invalid_last_record():
    result = align_previous(
        [1, 1, 2, 2], [[2], [3], [4], [np.nan]], [1, 1, 2],
        [0, 0, 0, 0], [0, 0, 0], 100, 100,
    )
    np.testing.assert_array_equal(result["valid"], [True, True, False])
    np.testing.assert_allclose(result["values"][:, 0], [3, 3, np.nan], equal_nan=True)
    np.testing.assert_array_equal(result["source_time_s"], [1, 1, 2])


def test_invalid_field_invalidates_whole_selected_row_without_backward_fallback():
    result = align_previous(
        [1, 1.1, 1.2], [[1, 2], [3, np.nan], [np.inf, 4]], [1.05, 1.15, 1.25],
        [0, 0, 0], [0, 0, 0], 100, 100,
    )
    np.testing.assert_array_equal(result["valid"], [True, False, False])
    np.testing.assert_allclose(result["values"], [[1, 2], [np.nan, np.nan], [np.nan, np.nan]], equal_nan=True)
    np.testing.assert_allclose(result["age_ms"], [50, 50, 50], atol=1e-9)


def test_empty_source_produces_invalid_rows_with_no_provenance():
    result = align_previous([], np.empty((0, 2)), [1, 2], [], [0, 0], 100, 100)
    assert result["values"].shape == (2, 2)
    assert np.isnan(result["values"]).all()
    assert np.isnan(result["age_ms"]).all()
    assert np.isnan(result["source_time_s"]).all()
    assert not result["valid"].any()


def test_empty_target_preserves_column_count():
    result = align_previous([1], [[2, 3]], [], [0], [], 100, 100)
    assert result["values"].shape == (0, 2)
    assert result["age_ms"].shape == (0,)
    assert result["source_time_s"].shape == (0,)
    assert result["valid"].shape == (0,)


@pytest.mark.parametrize("source_times,target_times", [([2, 1], [1]), ([1, 2], [2, 1]), ([1, np.nan], [1]), ([1, 2], [np.inf])])
def test_backward_or_nonfinite_timestamps_are_rejected(source_times, target_times):
    with pytest.raises(ValueError, match="time"):
        align_previous(source_times, [[1], [2]], target_times, [0, 0], [0] * len(target_times), 100, 100)


@pytest.mark.parametrize("max_age,gap", [(-1, 100), (100, -1), (np.inf, 100), (100, np.inf), (np.nan, 100)])
def test_limits_must_be_finite_nonnegative_to_prevent_unlimited_hold(max_age, gap):
    with pytest.raises(ValueError, match="finite|nonnegative"):
        align_previous([1], [[2]], [1], [0], [0], max_age, gap)


@pytest.mark.parametrize("values,source_ids,target_ids", [([1, 2], [0, 0], [0]), ([[1]], [0, 0], [0]), ([[1], [2]], [0], [0]), ([[1], [2]], [0, 0], []), ([[], []], [0, 0], [0])])
def test_mismatched_shapes_cannot_silently_broadcast(values, source_ids, target_ids):
    with pytest.raises(ValueError):
        align_previous([1, 2], values, [1], source_ids, target_ids, 100, 100)


@pytest.mark.parametrize("source_times,target_times", [([[1]], [1]), ([1], [[1]])])
def test_timestamps_must_be_one_dimensional(source_times, target_times):
    with pytest.raises(ValueError, match="time"):
        align_previous(source_times, [[2]], target_times, [0], [0], 100, 100)


def test_dispatcher_previous_policy_retains_bounded_alignment():
    result = align_source([1], [[2]], [1, 2], [0], [0, 0], 100, 100, method="previous")
    np.testing.assert_array_equal(result["valid"], [True, False])


@pytest.mark.parametrize("method", ["linear", "nearest", "bogus"])
def test_dispatcher_rejects_unsupported_policy_instead_of_silently_holding(method):
    with pytest.raises(ValueError, match="method|policy"):
        align_source([1], [[2]], [1], [0], [0], 100, 100, method=method)
