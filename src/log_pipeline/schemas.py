"""Audited output schema and central field-to-frame mapping."""
from .reader import REVISION, SOURCE_ROOT

REQUIRED_COLUMNS = [
    "time_s", "px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az",
    "yaw", "ax_target", "ay_target", "az_target", "dx", "dy", "dz",
    "battery_v", "battery_i", "throttle", "motor1", "motor2", "motor3",
    "motor4", "flight_id", "mode", "valid_dx", "valid_dy", "valid_dz",
]
AXES = {"x": ("PSCE", "E", 0), "y": ("PSCN", "N", 1), "z": ("PSCD", "D", 2)}
SOURCE_UNITS = {"position": "m", "velocity": "m/s", "acceleration": "m/s/s"}
SEMANTICS = {
    "revision": REVISION,
    "acceleration": SOURCE_ROOT + "libraries/AC_AttitudeControl/AC_PosControl.cpp#L1504-L1534",
    "horizontal_proxy": SOURCE_ROOT + "libraries/AC_AttitudeControl/AC_PosControl.cpp#L1626-L1634",
    "vertical_acceleration": SOURCE_ROOT + "libraries/AC_AttitudeControl/AC_PosControl.h#L605-L611",
    "throttle": SOURCE_ROOT + "libraries/AP_Motors/AP_Motors_Class.h#L154-L159",
    "motor_mapping": SOURCE_ROOT + "libraries/SRV_Channel/SRV_Channel.h",
    "multiplier_sentinel": SOURCE_ROOT + "libraries/AP_Logger/README.md#L83-L99",
}


def column_mapping(columns, motor_mapping):
    """Give every column a lineage rule, including derived quality/provenance."""
    mapping = {}
    for axis, (message, suffix, _) in AXES.items():
        for prefix, units, quantity in [("p", "m", "P"), ("v", "m/s", "V"), ("a", "m/s/s", "A")]:
            col = prefix + axis
            mapping[col] = {"message": message, "field": quantity + suffix,
                "input_frame": "NED", "output_frame": "ENU", "input_unit": units, "output_unit": units,
                "conversion": "frames.ned_to_enu", "source_column": "source_" + col,
                "fallback": {"message": "XKF1", "field": quantity + suffix, "requires": "fresh healthy primary XKF4"} if prefix in "pv" else None,
                "meaning": "attitude-target-derived controller proxy" if prefix == "a" and axis in "xy" else "controller/estimator output"}
        mapping[f"a{axis}_target"] = {"message": message, "field": "TA" + suffix, "input_frame": "NED", "output_frame": "ENU", "input_unit": "m/s/s", "output_unit": "m/s/s", "conversion": "frames.ned_to_enu", "same_packet_as": "a" + axis}
        mapping["d" + axis] = {"message": message, "field": ["A" + suffix, "TA" + suffix], "input_frame": "ENU", "output_frame": "ENU", "input_unit": "m/s/s", "output_unit": "m/s/s", "conversion": f"a{axis} - a{axis}_target", "physical_residual_eligible": axis == "z"}
    mapping.update({
        "time_s": {"message": "ARM and verified TimeUS", "field": "TimeUS", "output_unit": "s", "conversion": "configured grid within each armed interval; vehicle startup time"},
        "yaw": {"message": "ATT", "field": "Yaw", "input_frame": "NED heading", "output_frame": "ENU", "input_unit": "degheading", "output_unit": "rad", "conversion": "frames.yaw_ned_deg_to_enu_rad; East-zero CCW [-pi,pi)"},
        "battery_v": {"message": "BAT", "field": "Volt", "output_unit": "V", "conversion": "verified decoder-to-unit factor"},
        "battery_i": {"message": "BAT", "field": "Curr", "output_unit": "A", "conversion": "verified decoder-to-unit factor; calibration unverified"},
        "throttle": {"message": "CTUN", "field": "ThO", "output_unit": "1", "conversion": "none; filtered normalized motor-controller throttle [0,1]"},
        "flight_id": {"message": "ARM/EV", "field": "ArmState", "output_unit": "id", "conversion": "inspector armed interval ID; does not establish airborne state"},
        "mode": {"message": "MODE", "field": "ModeNum", "output_unit": "category", "conversion": "inspector mode interval label"},
        "continuity_id": {"message": "ARM,MODE,GPS,XKF4,EV", "output_unit": "id", "conversion": "unique event/gap-bounded interval ID"},
        "continuity_start_s": {"message": "ARM,MODE,GPS,XKF4,EV", "output_unit": "s", "conversion": "inclusive continuity-interval start in vehicle startup seconds"},
        "continuity_end_s": {"message": "ARM,MODE,GPS,XKF4,EV", "output_unit": "s", "conversion": "exclusive continuity-interval end in vehicle startup seconds"},
        "split": {"message": "derived", "output_unit": "category", "conversion": "chronological unique-timestamp split or explicit held-out flight"},
    })
    for number in range(1, 5):
        channel = motor_mapping.get(str(number))
        mapping[f"motor{number}"] = {"message": "RCOU", "field": f"C{channel}" if channel else None, "output_unit": "us", "conversion": "PWM-equivalent command, unique stable SERVOx_FUNCTION mapping required", "mapping_verified": channel is not None}
    for col in columns:
        if col in mapping:
            continue
        if col.startswith("age_"):
            mapping[col] = {"message": col[4:-3].upper(), "field": "TimeUS", "output_unit": "ms", "conversion": "1000 * (time_s - selected source time); NaN if no candidate in continuity interval"}
        elif col.startswith("source_time_"):
            mapping[col] = {"message": col[12:-2].upper(), "field": "TimeUS", "output_unit": "s", "conversion": "verified TimeUS in seconds for selected previous source record"}
        elif col.startswith("source_"):
            mapping[col] = {"message": "derived lineage", "output_unit": "category", "conversion": "per-row selected field/core; empty when value invalid"}
        elif col.startswith("rcou_c"):
            mapping[col] = {"message": "RCOU", "field": col[5:].upper(), "output_unit": "us", "conversion": "verified PWM-equivalent raw output command"}
        elif col in ("gps_status", "ekf_primary_core", "ekf_solution_status", "ekf_fault_status"):
            msg, field = {"gps_status": ("GPS", "Status"), "ekf_primary_core": ("XKF4", "PI"), "ekf_solution_status": ("XKF4", "SS"), "ekf_fault_status": ("XKF4", "FS")}[col]
            mapping[col] = {"message": msg, "field": field, "output_unit": "enum/bitmask", "conversion": "bounded previous record"}
        else:
            mapping[col] = {"message": "derived quality", "output_unit": "bool", "conversion": "explicit numerical/eligibility/outlier policy in manifest; no implicit row deletion"}
    return mapping
