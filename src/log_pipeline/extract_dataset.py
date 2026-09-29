"""Read-only, source-audited DataFlash extraction; d = a_actual - a_target.

Horizontal A is an attitude-target-derived controller proxy, not a measured
physical acceleration. Numerical validity never overrides this limitation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
import yaml

from .alignment import align_source
from .frames import ned_to_enu, residual, yaw_ned_deg_to_enu_rad
from .reader import REVISION, ScanResult, scan_log
from .schemas import AXES, REQUIRED_COLUMNS, SEMANTICS, column_mapping
from .utils import git_revision, sha256_file, write_csv, write_json

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "configs/log_pipeline.yaml"
SOURCES = ("PSCN", "PSCE", "PSCD", "XKF1", "XKF4", "ATT", "BAT", "RCOU", "CTUN", "GPS")


def load_config(path=None):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        overrides = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    _validate_config(config)
    return config


def _validate_config(config):
    if config["frame_output"] != "ENU":
        raise ValueError("Only the audited ENU output frame is supported")
    rate = float(config["master_rate_hz"])
    if not math.isfinite(rate) or not 0 < rate <= 1000:
        raise ValueError("master_rate_hz must be finite, positive and <=1000")
    for source in SOURCES:
        if config["alignment_method_by_source"].get(source) != "previous":
            raise ValueError(f"Unaudited alignment method for {source}")
        for value in (config["maximum_source_age_ms"][source], _gap(config, source)):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Age/gap thresholds must be finite and nonnegative")
    fraction = config["split"]["train_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    if config["split"]["method"] != "chronological":
        raise ValueError("Only chronological splitting (or explicit holdout flight) is supported")


def _gap(config, source):
    gap = config["gap_threshold_ms"]
    return gap.get(source, gap["default"]) if isinstance(gap, dict) else gap


def validate_scan(scan, inspection, *, allow_truncated_tail=False):
    """Fail closed; the sole opt-in exception is one known incomplete EOF packet."""
    summary = scan.summary
    digest = summary.get("raw_sha256")
    if not digest or digest != inspection.get("raw_sha256") or digest != summary.get("raw_sha256_after") or not summary.get("raw_unchanged"):
        raise ValueError("Raw/inspection SHA256 mismatch or input changed during parsing")
    firmware = summary.get("firmware") or ""
    match = re.search(r"\(([0-9a-f]{8,40})\)", firmware)
    if not firmware.startswith("ArduCopter ") or match is None or not REVISION.startswith(match.group(1)):
        raise ValueError("Unaudited firmware: this extractor requires source profile 2b5cebb9")
    errors = summary.get("critical_errors", [])
    diagnostics = summary.get("parser_diagnostics", {})
    clean = not errors and summary.get("parser_error_count") == 0 and summary.get("scan_complete")
    if not clean:
        trailing = diagnostics.get("trailing_byte_count", 0)
        expected = diagnostics.get("trailing_expected_message_bytes", 0)
        permitted = (
            allow_truncated_tail and summary.get("parser_error_count") == 1
            and len(errors) == 1 and errors[0].startswith("Parser anomalies detected (1 diagnostic events)")
            and summary.get("whole_file_audited") and summary.get("reached_parser_eof")
            and diagnostics.get("trailing_kind") == "truncated_message"
            and bool(diagnostics.get("trailing_message")) and 3 <= trailing < expected
            and not diagnostics.get("skipped_regions") and not diagnostics.get("skipped_byte_count")
            and not diagnostics.get("stdout") and not diagnostics.get("stderr") and not diagnostics.get("exception")
            and diagnostics.get("last_decoded_byte_exclusive", -1) + trailing == diagnostics.get("file_size_bytes") == diagnostics.get("bytes_accounted_for")
        )
        if not permitted:
            raise ValueError("Parser corruption/truncation or unresolved metadata: " + "; ".join(errors))
    # Per-instance monotonicity, not an inappropriate merge of different cores.
    instance_messages = {r["message"] for r in summary.get("instance_message_stats", [])}
    stats = summary.get("instance_message_stats", []) + [r for r in summary.get("message_stats", []) if r["message"] not in instance_messages]
    consumed = set(SOURCES) | {"ARM", "EV", "MODE"}
    if any(r.get("backward_timestamp_count", 0) for r in stats if r["message"] in consumed):
        raise ValueError("Backward source timestamps require a separately audited clock segmentation")
    if not scan.flight_segments:
        raise ValueError("No explicit armed flight intervals; extraction cannot infer them")


def _finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def _rows(scan, name, start, end):
    return [r for r in scan.records.get(name, []) if _finite(r.get("time_s")) and start <= r["time_s"] < end]


def _factor(scan, name, field, allowed_units):
    metadata = next((r for r in scan.inventory if r["message"] == name and r["field"] == field), None)
    if metadata is None:
        return None
    factor = metadata.get("decoded_to_unit_factor")
    if not metadata.get("unit_verified") or metadata.get("unit") not in allowed_units or not _finite(factor) or factor <= 0:
        raise ValueError(f"Unverified or incompatible unit/scaling for {name}.{field}: {metadata.get('unit')!r}")
    return factor


def _primary_status_rows(rows):
    """Observe PI on every core; clear old status when a new primary is announced.

    A nonprimary row can announce the switch before the new primary logs its
    status. Its PI and timestamp are usable, but its C/SS/FS are not the new
    primary's status. Mask those fields instead of carrying the old core.
    """
    selected = []
    previous_primary = None
    primary_status_seen = False
    for row in sorted(rows, key=lambda r: (r["time_s"], r.get("_record_index", 0))):
        primary = row.get("PI")
        if primary != previous_primary:
            previous_primary = primary
            primary_status_seen = False
        if "C" in row and row.get("C") == primary:
            selected.append(row)
            primary_status_seen = True
        elif not primary_status_seen:
            selected.append({**row, **{field: np.nan for field in ("C", "SS", "FS", "TS", "OFN", "OFE")}})
    return selected


def _continuity(scan, segment, config):
    start, end = segment["arm_time_s"], segment["end_time_s"]
    boundaries = {start}
    for interval in scan.mode_intervals:
        if interval.get("flight_id") == segment["flight_id"] and start < interval["start_time_s"] < end:
            boundaries.add(interval["start_time_s"])
    for row in _rows(scan, "EV", start, end):
        if row.get("Id") in (60, 62, 85, 86, 87):
            boundaries.add(row["time_s"])
    for name in ("GPS", "XKF4"):
        rows = _rows(scan, name, start, end)
        if name == "GPS":
            rows = [r for r in rows if r.get("I", r.get("Instance")) == config["gps_instance"]]
            signature = lambda r: (r.get("Status"),)
        else:
            rows = _primary_status_rows(rows)
            signature = lambda r: tuple(r.get(k) if _finite(r.get(k)) else None for k in ("PI", "SS", "FS", "TS", "OFN", "OFE"))
        previous = None
        for row in rows:
            if previous is not None:
                if signature(row) != signature(previous):
                    boundaries.add(row["time_s"])
                if row["time_s"] - previous["time_s"] > config["maximum_source_age_ms"][name] / 1000:
                    boundaries.add(previous["time_s"] + config["maximum_source_age_ms"][name] / 1000)
                    boundaries.add(row["time_s"])
            previous = row
    return np.array(sorted(t for t in boundaries if start <= t < end), dtype=float)


def _aligned(scan, name, fields, target, bounds, config, rows):
    times = np.array([r["time_s"] for r in rows], dtype=float)
    source_ids = np.searchsorted(bounds, times, side="right") - 1
    target_ids = np.searchsorted(bounds, target, side="right") - 1
    kwargs = dict(source_times=times, target_times=target, source_boundary_ids=source_ids,
                  target_boundary_ids=target_ids, max_age_ms=config["maximum_source_age_ms"][name],
                  gap_threshold_ms=_gap(config, name), method=config["alignment_method_by_source"][name])
    result = align_source(values=np.ones((len(rows), 1)), **kwargs)
    result["fields"] = {}
    for field, allowed in fields.items():
        factor = _factor(scan, name, field, allowed) if rows else None
        values = [float(r[field]) * factor if factor is not None and _finite(r.get(field)) else np.nan for r in rows]
        result["fields"][field] = align_source(values=np.array(values).reshape(-1, 1), **kwargs)["values"][:, 0]
    return result


def motor_mapping(scan):
    result = {}
    history = scan.records.get("PARM", [])
    servo_names = {name for name in scan.params if re.fullmatch(r"SERVO\d+_FUNCTION", name)}
    servo_names.update(r["Name"] for r in history if re.fullmatch(r"SERVO\d+_FUNCTION", r.get("Name", "")))
    stable = all(len({r.get("Value") for r in history if r.get("Name") == name} | ({scan.params[name]} if name in scan.params else set())) <= 1 for name in servo_names)
    for motor in range(1, 5):
        function = motor + 32
        matches = [int(match.group(1)) for key, value in scan.params.items() if (match := re.fullmatch(r"SERVO(\d+)_FUNCTION", key)) and value == function]
        # Changed mappings are not interpreted using only the final parameter.
        result[str(motor)] = matches[0] if len(matches) == 1 and stable and matches[0] <= 14 else None
    return result


def _convert_axis(values, suffix, output_index):
    source_index = {"N": 0, "E": 1, "D": 2}[suffix]
    vector = np.full((len(values), 3), np.nan)
    vector[:, source_index] = values
    return ned_to_enu(vector)[:, output_index]


def build_dataset(scan: ScanResult, config):
    """Build numerical rows from an already validated scan; never mutate it."""
    _validate_config(config)
    tables = []
    mapping = motor_mapping(scan)
    epoch_offset = 0
    for segment in scan.flight_segments:
        start, end = segment["arm_time_s"], segment["end_time_s"]
        target = np.round(start + np.arange(max(0, math.ceil((end-start)*config["master_rate_hz"]))) / config["master_rate_hz"], 9)
        target = target[target < end]
        if not len(target):
            continue
        bounds = _continuity(scan, segment, config)
        interval_index = np.searchsorted(bounds, target, side="right") - 1
        data = pd.DataFrame({"time_s": target, "flight_id": segment["flight_id"],
                             "continuity_id": epoch_offset + interval_index + 1,
                             "continuity_start_s": bounds[interval_index],
                             "continuity_end_s": np.append(bounds[1:], end)[interval_index]})
        epoch_offset += len(bounds)
        data["mode"] = "UNKNOWN"
        for interval in scan.mode_intervals:
            if interval.get("flight_id") == segment["flight_id"]:
                data.loc[(target >= interval["start_time_s"]) & (target < interval["end_time_s"]), "mode"] = interval["mode"]
        source = {}
        def collect(name, fields, rows=None):
            if rows is None:
                rows = _rows(scan, name, start, end)
            result = _aligned(scan, name, fields, target, bounds, config, rows)
            source[name] = result
            data[f"age_{name.lower()}_ms"] = result["age_ms"]
            data[f"source_time_{name.lower()}_s"] = result["source_time_s"]
            return result["fields"]
        status_rows = _primary_status_rows(_rows(scan, "XKF4", start, end))
        ekf = collect("XKF4", {"C": {"instance", ""}, "PI": {""}, "SS": {""}, "FS": {""}}, status_rows)
        known = np.isfinite(ekf["C"]) & np.isfinite(ekf["PI"]) & np.isfinite(ekf["SS"]) & np.isfinite(ekf["FS"]) & (ekf["C"] == ekf["PI"])
        ss = np.nan_to_num(ekf["SS"], nan=0).astype(np.int64)
        healthy = known & (ekf["FS"] == 0) & ((ss & 1) != 0) & ((ss & 128) == 0)
        data["ekf_status_known"] = known
        data["ekf_primary_core"] = ekf["PI"]
        data["ekf_solution_status"] = ekf["SS"]
        data["ekf_fault_status"] = ekf["FS"]
        data["ekf_healthy"] = healthy
        gps_rows = [r for r in _rows(scan, "GPS", start, end) if r.get("I", r.get("Instance")) == config["gps_instance"]]
        gps = collect("GPS", {"Status": {""}}, gps_rows)
        data["gps_status"] = gps["Status"]
        data["gps_status_known"] = np.isfinite(gps["Status"]) & np.isin(gps["Status"], np.arange(9))
        data["gps_valid"] = np.isin(gps["Status"], [3, 4, 5, 6, 8])
        # Select cores using fresh status at each source acquisition time.
        status_times = np.array([r["time_s"] for r in status_rows])
        xkf_rows = []
        for row in _rows(scan, "XKF1", start, end):
            index = np.searchsorted(status_times, row["time_s"], side="right") - 1
            if index >= 0:
                status = status_rows[index]
                same_epoch = np.searchsorted(bounds, row["time_s"], side="right") == np.searchsorted(bounds, status["time_s"], side="right")
                if status.get("C") == status.get("PI") and row.get("C") == status.get("PI") and same_epoch and row["time_s"] - status["time_s"] <= config["maximum_source_age_ms"]["XKF4"] / 1000:
                    xkf_rows.append(row)
        xkf = collect("XKF1", {**{f"P{s}": {"m"} for s in "NED"}, **{f"V{s}": {"m/s"} for s in "NED"}, "C": {"instance", ""}}, xkf_rows)
        for axis, (message, suffix, output_index) in AXES.items():
            psc = collect(message, {"P"+suffix: {"m"}, "V"+suffix: {"m/s"}, "A"+suffix: {"m/s/s"}, "TA"+suffix: {"m/s/s"}})
            for prefix, field, bit in [("p", "P", 32 if axis == "z" else (8|16)), ("v", "V", 4 if axis == "z" else 2)]:
                psc_values = _convert_axis(psc[field+suffix], suffix, output_index)
                fallback = _convert_axis(xkf[field+suffix], suffix, output_index)
                valid_fallback = healthy & ((ss & bit) != 0) & (xkf["C"] == ekf["PI"]) & np.isfinite(fallback)
                values = np.where(np.isfinite(psc_values), psc_values, np.where(valid_fallback, fallback, np.nan))
                col = prefix+axis
                data[col] = values
                data["source_"+col] = [message+"."+field+suffix if np.isfinite(p) else f"XKF1.{field}{suffix}[C={int(c)}]" if good else "" for p,c,good in zip(psc_values, xkf["C"], valid_fallback)]
            actual = _convert_axis(psc["A"+suffix], suffix, output_index)
            desired = _convert_axis(psc["TA"+suffix], suffix, output_index)
            paired = np.isfinite(actual) & np.isfinite(desired)
            data["a"+axis] = np.where(paired, actual, np.nan)
            data["a"+axis+"_target"] = np.where(paired, desired, np.nan)
            data["d"+axis] = residual(data["a"+axis], data["a"+axis+"_target"])
            data["source_a"+axis] = np.where(paired, message+".A"+suffix, "")
            data["valid_d"+axis] = paired
        attitude = collect("ATT", {"Yaw": {"degheading", "deg"}})
        data["yaw"] = yaw_ned_deg_to_enu_rad(attitude["Yaw"])
        battery_rows = [r for r in _rows(scan, "BAT", start, end) if r.get("Inst", r.get("Instance")) == config["battery_instance"]]
        battery = collect("BAT", {"Volt": {"V"}, "Curr": {"A"}}, battery_rows)
        data["battery_v"], data["battery_i"] = battery["Volt"], battery["Curr"]
        data["throttle"] = collect("CTUN", {"ThO": {""}})["ThO"]
        fields = {r["field"]: {"us"} for r in scan.inventory if r["message"] == "RCOU" and re.fullmatch(r"C\d+", r["field"])}
        rcou = collect("RCOU", fields)
        for field in sorted(fields, key=lambda s: int(s[1:])):
            data["rcou_" + field.lower()] = rcou[field]
        for motor in range(1, 5):
            data[f"motor{motor}"] = rcou.get(f"C{mapping[str(motor)]}", np.nan)
        data["xy_acceleration_is_proxy"] = True
        _outliers(data, config)
        for axis in "xyz":
            features = config["learning_features_by_axis"][axis]
            finite_features = np.isfinite(data[features]).all(axis=1)
            data["eligible_d"+axis] = data["valid_d"+axis] & finite_features & ~data["outlier"] & (axis == "z")
        tables.append(data)
    if not tables:
        raise ValueError("No rows in explicit selected armed intervals")
    result = pd.concat(tables, ignore_index=True)
    return result[REQUIRED_COLUMNS + [c for c in result.columns if c not in REQUIRED_COLUMNS]]


def _outliers(data, config):
    bounds = config["battery_plausibility_bounds"]
    checks = {"battery_v": bounds.get("voltage", bounds.get("voltage_v")), "battery_i": bounds.get("current", bounds.get("current_a")), "throttle": [0, 1]}
    checks.update({col: config["acceleration_plausibility_bounds"] for col in ("ax", "ay", "az", "ax_target", "ay_target", "az_target")})
    checks.update({f"motor{i}": config["motor_output_plausibility_bounds"] for i in range(1, 5)})
    for col, (low, high) in checks.items():
        data["outlier_"+col] = np.isfinite(data[col]) & ((data[col] < low) | (data[col] > high))
    speed = np.sqrt((data[["vx", "vy", "vz"]] ** 2).sum(axis=1, min_count=3))
    low, high = config["velocity_plausibility_bounds"]
    data["outlier_speed"] = np.isfinite(speed) & ((speed < low) | (speed > high))
    data["outlier"] = data[[c for c in data if c.startswith("outlier_")]].any(axis=1)


def assign_splits(data, config):
    result = data.copy()
    heldout = config["split"].get("holdout_flight_id")
    if heldout is not None:
        flights = set(result.flight_id)
        if heldout not in flights or len(flights) < 2:
            raise ValueError("Explicit holdout requires that flight plus another suitable flight")
        if result.loc[result.flight_id != heldout, "time_s"].max() >= result.loc[result.flight_id == heldout, "time_s"].min():
            raise ValueError("The holdout flight must chronologically follow every training flight")
        result["split"] = np.where(result.flight_id == heldout, "test", "train")
    else:
        times = np.sort(result.time_s.unique())
        count = int(len(times) * config["split"]["train_fraction"])
        threshold = times[count] if count < len(times) else math.inf
        result["split"] = np.where(result.time_s < threshold, "train", "test")
    return result


def _protect_output_paths(destination, names, protected):
    allowed_roots = ((PROJECT / "data/processed").resolve(), (PROJECT / "results").resolve())
    for name in names:
        path = destination / name
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise ValueError("Derived output path escapes data/processed or results")
        for original in protected:
            if resolved == original.resolve() or (path.exists() and original.exists() and path.samefile(original)):
                raise ValueError(f"Refusing to overwrite protected input: {original}")


def extract_dataset(raw_path, inspection_path, output_dir, config_path=None, *, allow_truncated_tail=False, flight_ids=None, holdout_flight_id=None):
    raw = Path(raw_path).resolve()
    inspection_file = Path(inspection_path).resolve()
    inspection = json.loads(inspection_file.read_text(encoding="utf-8"))
    config = load_config(config_path)
    if holdout_flight_id is not None:
        config["split"]["holdout_flight_id"] = holdout_flight_id
    scan = scan_log(raw)
    validate_scan(scan, inspection, allow_truncated_tail=allow_truncated_tail)
    if flight_ids is not None:
        if set(flight_ids) - {r["flight_id"] for r in scan.flight_segments}:
            raise ValueError("Requested flight ID is absent from inspection")
        scan.flight_segments = [r for r in scan.flight_segments if r["flight_id"] in flight_ids]
    data = build_dataset(scan, config)
    candidates = assign_splits(data.loc[data[["valid_dx", "valid_dy", "valid_dz"]].any(axis=1)], config)
    destination = Path(output_dir).resolve()
    # Destination policy is checked before writing; the raw BIN is never opened for write.
    if not any(destination == allowed.resolve() or allowed.resolve() in destination.parents for allowed in (PROJECT / "data/processed", PROJECT / "results")):
        raise ValueError("Derived outputs must be inside data/processed or results")
    output_names = ["all_flights.csv", "gp_residual_dataset.csv", "battery_raw.csv", "esc_raw.csv", "dataset_manifest.json"]
    output_names.extend(f"flight_{int(flight):03d}.csv" for flight in data.flight_id.unique())
    _protect_output_paths(destination, output_names, [raw, inspection_file])
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    def save(name, table):
        path = destination / name
        table.to_csv(path, index=False, float_format="%.17g", na_rep="NaN", lineterminator="\n")
        outputs[name] = sha256_file(path)
    save("all_flights.csv", data)
    for flight, table in data.groupby("flight_id", sort=True):
        save(f"flight_{int(flight):03d}.csv", table)
    save("gp_residual_dataset.csv", candidates)
    for message, filename in (("BAT", "battery_raw.csv"), ("ESC", "esc_raw.csv")):
        rows = scan.records.get(message, [])
        fields = list(dict.fromkeys(["time_s"] + [k for r in rows for k in r]))
        write_csv(destination / filename, rows, fields)
        outputs[filename] = sha256_file(destination / filename)
    mapping = column_mapping([*data.columns, "split"], motor_mapping(scan))
    calibration = {key: value for key, value in scan.params.items() if key.startswith(("BATT", "MOT_BAT", "SERVO_BLH", "SERVO_DSHOT")) or key in ("MOT_THST_EXPO", "MOT_THST_HOVER", "FRAME_CLASS", "FRAME_TYPE") or re.fullmatch(r"SERVO\d+_FUNCTION", key)}
    warnings = list(scan.summary.get("warnings", [])) + [
        "Horizontal residuals are controller proxies: dx, dy and physical XYZ learning are NOT_READY.",
        "Armed intervals are not verified airborne intervals.",
        "Battery current calibration and ESC pole count/sensor trust are unverified; raw values preserved.",
        "Reset logs may be incomplete; identical repeated reset deltas cannot be detected from OFN/OFE alone.",
    ]
    if scan.summary["parser_error_count"]:
        warnings.append("Explicit --allow-truncated-tail exception: only complete prefix packets used; original raw file unchanged.")
    if any(channel is None for channel in motor_mapping(scan).values()):
        warnings.append("One or more logical motor mappings unresolved; raw RCOU channels retained.")
    manifest = {
        "schema_version": 1, "raw_log_filename": raw.name, "raw_path": str(raw), "raw_sha256": scan.summary["raw_sha256"],
        "raw_sha256_after": sha256_file(raw), "inspection_sha256": sha256_file(inspection_file),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_revision(PROJECT),
        "python_version": platform.python_version(), "pymavlink_version": scan.summary.get("pymavlink_version"),
        "code_sha256": {str(p.relative_to(PROJECT)).replace("\\", "/"): sha256_file(p) for p in sorted((PROJECT / "src/log_pipeline").glob("*.py"))},
        "extractor_config": config, "coordinate_frame": "ENU", "residual_sign_convention": "d = a_actual - a_target",
        "config_file_sha256": {str(path): sha256_file(path) for path in {DEFAULT_CONFIG.resolve(), Path(config_path).resolve() if config_path else DEFAULT_CONFIG.resolve()}},
        "firmware": scan.summary["firmware"], "audited_source_profile": SEMANTICS,
        "source_mapping": mapping, "column_lineage": mapping,
        "unit_mapping": {column: item["output_unit"] for column, item in mapping.items()},
        "field_inventory": scan.inventory, "motor_mapping": motor_mapping(scan),
        "alignment_policy": {"method_by_source": config["alignment_method_by_source"], "maximum_source_age_ms": config["maximum_source_age_ms"], "gap_threshold_ms": config["gap_threshold_ms"], "continuity_column": "continuity_id", "boundaries": "arm/disarm, mode, GPS status/gap, primary-core/status/reset-offset change/gap, EV reset/source events", "actual_target_pairing": "same original PSC packet; no differentiation or target fabrication"},
        "validity_thresholds": {k: config[k] for k in ("minimum_samples_for_learning", "minimum_train_samples", "minimum_test_samples", "maximum_source_age_ms", "gap_threshold_ms", "learning_features_by_axis")},
        "validity_policy": {"valid_d_axis": "finite same-packet actual and target, verified units/frame, bounded causal age in same continuity interval", "eligible_dx": "always false: controller proxy", "eligible_dy": "always false: controller proxy", "eligible_dz": "valid_dz and finite configured z features and not outlier", "outliers": "flagged and retained; excluded only from eligibility", "gps_valid": "Status in {3,4,5,6,8}; no GPS gate on vertical controller residual", "ekf_fallback": "XKF4 C==PI, FS==0, SS attitude, non-constant position and appropriate position/velocity validity bit"},
        "selected_flight_ids": sorted(int(v) for v in data.flight_id.unique()), "flight_segments": scan.flight_segments,
        "row_count": len(data), "gp_row_count": len(candidates), "valid_counts": {"d"+a: int(data["valid_d"+a].sum()) for a in "xyz"},
        "eligible_counts": {"d"+a: int(data["eligible_d"+a].sum()) for a in "xyz"},
        "xy_acceleration_is_proxy": True, "physical_xy_readiness": "NOT_READY", "physical_xyz_readiness": "NOT_READY",
        "battery_calibration_verified": False, "esc_rpm_physical_scale_verified": False,
        "calibration_parameters": calibration, "parameter_history": [r for r in scan.records.get("PARM", []) if r.get("Name") in calibration],
        "raw_sidecars": {message: {"file": filename, "semantics": "original pymavlink-decoded values; no additional scaling", "columns": {**{r["field"]: r for r in scan.inventory if r["message"] == message}, "time_s": {"unit": "s", "source": "verified TimeUS conversion"}, "_record_index": {"unit": "index", "source": "zero-based complete-packet parse order"}, "_offset": {"unit": "byte", "source": "original input packet start offset"}}} for message, filename in (("BAT", "battery_raw.csv"), ("ESC", "esc_raw.csv"))},
        "parser_diagnostics": scan.summary["parser_diagnostics"], "parser_critical_errors": scan.summary["critical_errors"],
        "allow_truncated_tail": bool(allow_truncated_tail), "complete_prefix_exception_used": bool(scan.summary["parser_error_count"]),
        "warnings": warnings, "file_hashes": outputs,
    }
    if manifest["raw_sha256_after"] != manifest["raw_sha256"]:
        raise ValueError("Raw SHA256 changed during extraction; outputs are not trusted")
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_log", type=Path)
    parser.add_argument("--inspection", type=Path, default=PROJECT / "results/log_inspection/log_summary.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-truncated-tail", action="store_true", help="explicitly permit only one known incomplete final packet; all interior/errors still rejected")
    parser.add_argument("--flight-id", type=int, action="append")
    parser.add_argument("--holdout-flight-id", type=int)
    args = parser.parse_args(argv)
    manifest = extract_dataset(args.raw_log, args.inspection, args.output, args.config, allow_truncated_tail=args.allow_truncated_tail, flight_ids=args.flight_id, holdout_flight_id=args.holdout_flight_id)
    print(json.dumps({"row_count": manifest["row_count"], "valid_counts": manifest["valid_counts"], "output": str(args.output), "physical_xyz_readiness": "NOT_READY"}, sort_keys=True))


if __name__ == "__main__":
    main()
