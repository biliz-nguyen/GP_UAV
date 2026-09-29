"""Central conversions for verified NED sources and the ENU research frame.

ENU means x East, y North, z Up. These functions do not infer a source frame
or unit: the caller must verify both from log metadata and source definitions.
The initial residual convention is d = a_actual - a_target in identical units.
"""

import numpy as np


def ned_to_enu(v):
    """Convert NED vector(s) of shape (..., 3) into ENU without mutating input."""
    vectors = np.asarray(v, dtype=float)
    if vectors.ndim == 0 or vectors.shape[-1] != 3:
        raise ValueError("Vectors must have a final dimension of three (3) axes")
    converted = vectors[..., [1, 0, 2]].copy()
    converted[..., 2] *= -1
    return converted


def enu_to_ned(v):
    """Convert ENU vector(s) of shape (..., 3) into NED; the map is its inverse."""
    return ned_to_enu(v)


def yaw_ned_deg_to_enu_rad(yaw):
    """Convert North-zero clockwise degrees to East-zero CCW radians.

    Output is wrapped to [-pi, pi). This applies to the verified ATT heading
    convention, not a full three-dimensional attitude transformation.
    """
    heading = np.asarray(yaw, dtype=float)
    with np.errstate(invalid="ignore"):
        return (np.pi / 2 - np.deg2rad(heading) + np.pi) % (2 * np.pi) - np.pi


def residual(actual, target, actual_frame="ENU", target_frame="ENU"):
    """Return d = a_actual - a_target; callers must verify identical units.

    Frame labels are checked explicitly to prevent mixing NED and ENU. Missing
    values propagate; NumPy broadcasting supports single-axis and batch inputs.
    The term actual denotes the selected source field, not a measurement claim.
    """
    if actual_frame not in ("ENU", "NED") or target_frame not in ("ENU", "NED"):
        raise ValueError("Residual frame must be explicitly ENU or NED")
    if actual_frame != target_frame:
        raise ValueError("Residual inputs must use the same coordinate frame")
    return np.asarray(actual, dtype=float) - np.asarray(target, dtype=float)
