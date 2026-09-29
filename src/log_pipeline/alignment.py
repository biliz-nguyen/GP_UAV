"""Causal, bounded previous-sample alignment with explicit interval identity.

All timestamps are seconds in one verified clock. Boundary IDs must identify
unique continuous intervals, shared by source and target. The caller constructs
them at arm/disarm, mode, reset, GPS/EKF validity and other semantic boundaries.
No boundary can be inferred from numeric values alone.
"""

import numpy as np


def _timestamps(values, name):
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{name} timestamps must be a finite one-dimensional array")
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{name} timestamps must be nondecreasing; input is not reordered")
    return times


def _boundaries(values, size, name):
    boundaries = np.asarray(values)
    if boundaries.ndim != 1 or boundaries.size != size:
        raise ValueError(f"{name} boundary IDs must match the timestamp array length")
    return boundaries


def _limit_ms(value, name):
    limit = float(value)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return limit


def align_previous(
    source_times,
    values,
    target_times,
    source_boundary_ids,
    target_boundary_ids,
    max_age_ms,
    gap_threshold_ms,
):
    """Align an (N, K) source matrix to target timestamps by bounded hold.

    Select the last source record at or before each target; duplicate source
    timestamps deterministically select the last record in input order. Both
    time arrays must be nondecreasing and finite. No future record is selected,
    and a candidate from another boundary is never reused.

    A candidate is valid only while its age is at most both max_age_ms and
    gap_threshold_ms, inclusive, and all K fields are finite. The gap threshold
    therefore bounds hold even without a later source sample. Acquisition of a
    fresh sample immediately restores validity, including after a large gap.

    Return a dict with values (M, K), age_ms (M,), source_time_s (M,) and valid
    (M,). Invalid values are NaN. An expired or nonfinite candidate retains its
    age and source timestamp. Before the first source or across a boundary,
    there is no candidate and provenance is NaN. Limits cannot be infinite.
    """
    source = _timestamps(source_times, "source")
    target = _timestamps(target_times, "target")
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] != source.size or data.shape[1] < 1:
        raise ValueError("values must have shape (N, K) with N source rows and K >= 1")
    source_boundaries = _boundaries(source_boundary_ids, source.size, "source")
    target_boundaries = _boundaries(target_boundary_ids, target.size, "target")
    hold_ms = min(_limit_ms(max_age_ms, "max_age_ms"), _limit_ms(gap_threshold_ms, "gap_threshold_ms"))

    result = {
        "values": np.full((target.size, data.shape[1]), np.nan),
        "age_ms": np.full(target.size, np.nan),
        "source_time_s": np.full(target.size, np.nan),
        "valid": np.zeros(target.size, dtype=bool),
    }
    if source.size == 0 or target.size == 0:
        return result

    indices = np.searchsorted(source, target, side="right") - 1
    target_rows = np.flatnonzero(indices >= 0)
    source_rows = indices[target_rows]
    same_boundary = source_boundaries[source_rows] == target_boundaries[target_rows]
    target_rows = target_rows[same_boundary]
    source_rows = source_rows[same_boundary]

    result["source_time_s"][target_rows] = source[source_rows]
    result["age_ms"][target_rows] = (target[target_rows] - source[source_rows]) * 1000
    # Compare absolute deadlines so representational error in (t - s) does not
    # reject an exactly-on-limit target such as 1.1 s with source 1 s and 100 ms.
    fresh = target[target_rows] <= source[source_rows] + hold_ms / 1000
    finite = np.isfinite(data[source_rows]).all(axis=1)
    valid_candidates = fresh & finite
    valid_rows = target_rows[valid_candidates]
    result["valid"][valid_rows] = True
    result["values"][valid_rows] = data[source_rows[valid_candidates]]
    return result


def align_source(
    source_times,
    values,
    target_times,
    source_boundary_ids,
    target_boundary_ids,
    max_age_ms,
    gap_threshold_ms,
    method="previous",
):
    """Apply the declared alignment policy; unsupported policies fail closed."""
    if method != "previous":
        raise ValueError(f"Unsupported alignment method {method!r}; only 'previous' is implemented")
    return align_previous(
        source_times, values, target_times, source_boundary_ids, target_boundary_ids,
        max_age_ms, gap_threshold_ms,
    )
