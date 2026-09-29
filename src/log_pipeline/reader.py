"""Whole-file pymavlink DataFlash reader with auditable scaling and boundaries.

Only the decoder applies DataFlash format-character scaling. FMTU multipliers
describe storage-to-unit scaling, so their ratio to decoder scaling is exposed
as ``decoded_to_unit_factor``. Records retain decoded values without rescaling.
"""
from __future__ import annotations

import contextlib
import io
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from .utils import sha256_file
from .inspection_metrics import inventory_statistics, load_inspection_config, distribution

REVISION = "2b5cebb933d91e92f7bab768268e41c2960b6ae4"
SOURCE_ROOT = f"https://github.com/ArduPilot/ardupilot/blob/{REVISION}/"
SEMANTIC_SOURCES = {
    "MULT.sentinels": SOURCE_ROOT + "libraries/AP_Logger/LogStructure.h#L93-L95",
    "ARM.ArmState": SOURCE_ROOT + "libraries/AP_Logger/LogStructure.h#L683-L691",
    "XKF4.SS_and_FS": SOURCE_ROOT + "libraries/AP_NavEKF/AP_Nav_Common.h",
    "XKF4.C_and_PI": SOURCE_ROOT + "libraries/AP_NavEKF3/LogStructure.h#L177-L215",
    "GPS.Status": SOURCE_ROOT + "libraries/AP_GPS/AP_GPS_FixType.h",
    "EV.armed_disarmed": "https://ardupilot.org/copter/docs/logmessages.html#ev",
}
# This is a reporting policy, not a claim about sampling latency or accuracy.
STATUS_MAX_HOLD_S = 1.0


@dataclass
class ScanResult:
    records: dict[str, list[dict]]
    formats: dict[str, dict]
    inventory: list[dict]
    message_stats: list[dict]
    summary: dict
    params: dict[str, float]
    flight_segments: list[dict]
    mode_intervals: list[dict]

    def field(self, message: str, field: str) -> dict:
        return next(row for row in self.inventory if row["message"] == message and row["field"] == field)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).split("\0", 1)[0]


def _metadata(records, decoder_formats):
    """Join final metadata after the ENTIRE stream, including late FMTU records."""
    from pymavlink.DFReader import FORMAT_TO_STRUCT

    units = {chr(int(r["Id"])): _text(r["Label"]) for r in records.get("UNIT", [])}
    raw_multipliers = {chr(int(r["Id"])): float(r["Mult"]) for r in records.get("MULT", [])}
    # ArduPilot logs float32 constants in the double MULT field. Match
    # pymavlink's documented seven-significant-digit normalization while
    # retaining the exact raw MULT value in each inventory entry.
    multipliers = {key: float(f"{value:.7g}") for key, value in raw_multipliers.items()}
    fmtu = {int(r["FmtType"]): r for r in records.get("FMTU", [])}
    definitions = {}
    conflicts = []
    for message_name, key_field, value_fields in (
        ("UNIT", "Id", ("Label",)), ("MULT", "Id", ("Mult",)),
        ("FMTU", "FmtType", ("UnitIds", "MultIds")),
    ):
        seen = {}
        for row in records.get(message_name, []):
            key = row[key_field]
            definition = tuple(row[field] for field in value_fields)
            if key in seen and seen[key] != definition:
                conflicts.append(f"{message_name} {key_field} {key} changed definition within file")
            seen[key] = definition
    for r in records.get("FMT", []):
        key = int(r["Type"])
        spec = (r["Name"], r["Length"], r["Format"], r["Columns"])
        if key in definitions and definitions[key] != spec:
            conflicts.append(f"FMT type {key} changed definition within file")
        definitions[key] = spec
    formats = {}
    inventory = []
    for type_id, fmt in sorted(decoder_formats.items(), key=lambda item: item[1].name):
        entry = fmtu.get(type_id, {})
        unit_ids, mult_ids = _text(entry.get("UnitIds", "")), _text(entry.get("MultIds", ""))
        instance_index = unit_ids.find("#")
        instance_field = fmt.columns[instance_index] if 0 <= instance_index < len(fmt.columns) else None
        formats[fmt.name] = {
            "type_id": type_id, "length": fmt.len, "format": fmt.format,
            "fields": list(fmt.columns), "unit_ids": unit_ids, "multiplier_ids": mult_ids,
            "instance_field": instance_field,
        }
        for index, field in enumerate(fmt.columns):
            char = fmt.format[index]
            unit_id = unit_ids[index] if index < len(unit_ids) else None
            mult_id = mult_ids[index] if index < len(mult_ids) else None
            unit = units.get(unit_id)
            multiplier = multipliers.get(mult_id)
            decoder_scale = FORMAT_TO_STRUCT[char][1] or 1.0
            # '-' is the documented no-multiplier sentinel, NOT numeric zero.
            # Preserve the decoder's already applied format-character scale.
            # '?' means unknown even though its numeric sentinel is one.
            no_multiplier = mult_id == "-" and multiplier == 0
            verified = unit is not None and (no_multiplier or mult_id not in (None, "?") and multiplier is not None and multiplier > 0)
            factor = 1.0 if verified and no_multiplier else multiplier / decoder_scale if verified else None
            if factor is not None and math.isclose(factor, round(factor), rel_tol=1e-6):
                factor = float(round(factor))
            values = [r.get(field) for r in records.get(fmt.name, [])]
            finite = [v for v in values if _finite(v)]
            invalid = sum(v is None or isinstance(v, (int, float)) and not _finite(v) for v in values)
            dtypes = sorted({type(v).__name__ for v in values if v is not None})
            inventory.append({
                "message": fmt.name, "field": field, "format_char": char,
                "decoded_dtype": "|".join(dtypes) or FORMAT_TO_STRUCT[char][2].__name__,
                "unit": unit, "unit_id": unit_id, "multiplier": multiplier,
                "multiplier_raw": raw_multipliers.get(mult_id),
                "multiplier_id": mult_id, "decoded_scale_applied": decoder_scale,
                "decoded_to_unit_factor": factor,
                "unit_verified": verified,
                "scale_semantics": "no additional multiplier; preserve decoded format scale" if no_multiplier else "FMTU storage-to-unit multiplier divided by decoder format scale" if verified else "unresolved multiplier or unit",
                "metadata_source": "log FMT/FMTU/UNIT/MULT + pymavlink FORMAT_TO_STRUCT" if unit_ids else "log FMT + pymavlink FORMAT_TO_STRUCT; unit metadata absent",
                "observed_min": min(finite) if finite else None,
                "observed_max": max(finite) if finite else None,
                "sample_count": len(values), "finite_sample_count": len(finite),
                "nan_invalid_count": invalid,
                "nonnumeric_sample_count": len(values) - len(finite) - invalid,
            })
    expanded = []
    for info in inventory:
        rows = records.get(info['message'], [])
        expanded.append(inventory_statistics(info, [r.get(info['field']) for r in rows]))
        instance_field = formats[info['message']]['instance_field']
        if instance_field:
            groups = defaultdict(list)
            for row in rows:
                groups[row.get(instance_field)].append(row.get(info['field']))
            for instance, values in sorted(groups.items(), key=lambda item: str(item[0])):
                expanded.append(inventory_statistics(info, values, instance, 'instance'))
    return formats, expanded, units, multipliers, list(dict.fromkeys(conflicts))


def _timing_stats(name, rows, instance=None, instance_field=None, rate_config=None):
    policy = rate_config or load_inspection_config()['rate']
    times = [r["time_s"] for r in rows if _finite(r.get("time_s"))]
    dt = [b - a for a, b in zip(times, times[1:])]
    positive = [d for d in dt if d > 0]
    median = statistics.median(positive) if positive else None
    threshold = max(policy['absolute_gap_threshold_s'], policy['gap_multiplier'] * median) if median is not None else policy['absolute_gap_threshold_s']
    active = [d for d in dt if 0 < d <= threshold]
    span = times[-1] - times[0] if times else 0
    out = {
        "message": name, "instance_field": instance_field, "instance": instance,
        "count": len(rows), "timestamped_count": len(times),
        "first_time_s": times[0] if times else None,
        "last_time_s": times[-1] if times else None,
        "mean_rate_hz": (len(times) - 1) / span if span > 0 else None,
        "aggregate_span_rate_hz": (len(times) - 1) / span if span > 0 else None,
        "nominal_rate_hz": 1 / median if median else None,
        "active_window_rate_hz": len(active) / sum(active) if active else None,
        "active_duration_s": sum(active),
        "active_window_count": 1 + sum(d < 0 or d > threshold for d in dt) if times else 0,
        "gap_threshold_s": threshold, "rate_scope": 'instance' if instance_field else 'message',
        "first_timestamp": times[0] if times else None,
        "last_timestamp": times[-1] if times else None,
        "median_dt_s": median,
        "minimum_dt_s": min(dt) if dt else None,
        "maximum_dt_s": max(dt) if dt else None,
        "duplicate_timestamp_count": sum(d == 0 for d in dt),
        "backward_timestamp_count": sum(d < 0 for d in dt),
    }
    if name == 'ESC':
        out['statistics_space'] = 'decoded; units and factors in field_inventory.csv'
        for field in ('RPM', 'RawRPM', 'Volt', 'Curr', 'Temp'):
            stats = distribution([r.get(field) for r in rows])
            out.update({f'{field}_{key}': stats[key] for key in ('min', 'max', 'median')})
    return out


def _build_segments(records, first_time, last_time, mode_names):
    warnings = []
    arm = [r for r in records.get("ARM", []) if _finite(r.get("time_s")) and r.get("ArmState") in (0, 1)]
    source = "ARM.ArmState"
    if not arm:
        arm = [dict(r, ArmState=int(r["Id"] == 10)) for r in records.get("EV", []) if r.get("Id") in (10, 11) and _finite(r.get("time_s"))]
        source = "EV.Id (10=armed, 11=disarmed)"
    if not arm:
        warnings.append("Armed segments unknown: no verified ARM or armed/disarmed EV events")
        source = "unknown"
    arm.sort(key=lambda r: (r["time_s"], r["_record_index"]))
    segments = []
    active = None
    for event in arm:
        if event["ArmState"]:
            if active is not None:
                warnings.append(f"Repeated armed event at {event['time_s']} s; continuous armed segment retained")
            else:
                active = {"flight_id": len(segments) + 1, "segment_id": len(segments) + 1,
                          "arm_time_s": event["time_s"], "arming_source": source}
        elif active is not None:
            active.update(disarm_time_s=event["time_s"], end_time_s=event["time_s"], open_ended=False)
            segments.append(active)
            active = None
    if active is not None:
        active.update(disarm_time_s=None, end_time_s=last_time, open_ended=True)
        segments.append(active)
        warnings.append("Final armed segment has no disarm event; duration is censored at final logged timestamp")
    modes = []
    for r in records.get("MODE", []):
        if not _finite(r.get("time_s")):
            continue
        number = r.get("ModeNum", r.get("Mode"))
        modes.append((r["time_s"], r["_record_index"], number, mode_names.get(number, f"UNKNOWN({number})")))
    modes.sort()
    # Unioning boundaries explicitly splits even the same mode across disarm.
    boundaries = sorted({first_time, last_time, *(m[0] for m in modes),
                         *(r["time_s"] for r in arm),
                         *(s["arm_time_s"] for s in segments), *(s["end_time_s"] for s in segments)})
    intervals = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        preceding = [m for m in modes if m[0] <= start]
        current = preceding[-1] if preceding else (None, None, None, "UNKNOWN")
        flight_id = next((s["flight_id"] for s in segments if s["arm_time_s"] <= start < s["end_time_s"]), None)
        arm_preceding = [r for r in arm if r["time_s"] <= start]
        row = {"start_time_s": start, "end_time_s": end, "duration_s": end - start,
               "flight_id": flight_id, "armed": bool(arm_preceding[-1]["ArmState"]) if arm_preceding else None,
               "mode_number": current[2], "mode": current[3]}
        # Only repeated MODE reports inside the same arm state may coalesce.
        if intervals and all(intervals[-1][key] == row[key] for key in ("flight_id", "armed", "mode_number")):
            intervals[-1]["end_time_s"] = end
            intervals[-1]["duration_s"] = end - intervals[-1]["start_time_s"]
        else:
            intervals.append(row)
    for segment in segments:
        start, end = segment["arm_time_s"], segment["end_time_s"]
        segment["duration_s"] = end - start
        segment["modes"] = list(dict.fromkeys(r["mode"] for r in intervals if r["flight_id"] == segment["flight_id"]))
        for name in ("PSCN", "PSCE", "PSCD"):
            segment[name.lower() + "_sample_count"] = sum(start <= r["time_s"] < end for r in records.get(name, []) if _finite(r.get("time_s")))
        segment.update(_segment_status(records, start, end))
    return segments, intervals, warnings, source


def _segment_status(records, start, end):
    gps = [r for r in records.get("GPS", []) if _finite(r.get("time_s")) and start <= r["time_s"] < end and "Status" in r]
    status_by_instance = defaultdict(set)
    for r in gps:
        status_by_instance[str(r.get("I", r.get("Instance", "unidentified")))].add(r["Status"])
    gps_status = {key: sorted(values) for key, values in status_by_instance.items()} if gps else "unknown: no GPS.Status samples"
    # Do not carry status samples across arm boundaries. A missing primary
    # sample after a core switch is unknown, even if the old core was healthy.
    ekf = [r for r in records.get("XKF4", []) if _finite(r.get("time_s")) and start <= r["time_s"] < end]
    ekf.sort(key=lambda r: (r["time_s"], r["_record_index"]))
    primary = [r for r in ekf if all(k in r for k in ("C", "PI", "SS", "FS")) and r["C"] == r["PI"]]
    if not primary:
        return {"gps_status": gps_status, "ekf_status": "unknown: no primary XKF4.C==PI status samples",
                "valid_xy_position_duration_s": None, "xy_status_known_duration_s": 0.0,
                "xy_status_unknown_duration_s": end - start}
    nodes = []
    current_primary = None
    for r in ekf:
        if not all(key in r for key in ("C", "PI", "SS", "FS")):
            continue
        if r["PI"] != current_primary:
            nodes.append((r["time_s"], None))
            current_primary = r["PI"]
        if r["C"] == current_primary:
            nodes.append((r["time_s"], r))
    valid_duration = known_duration = 0.0
    for index, (timestamp, r) in enumerate(nodes):
        if r is None:
            continue
        stop = min(end, r["time_s"] + STATUS_MAX_HOLD_S,
                   nodes[index + 1][0] if index + 1 < len(nodes) else end)
        duration = max(0.0, stop - max(start, r["time_s"]))
        known_duration += duration
        # A status screen, not an independent position accuracy measurement.
        ss = int(r["SS"])
        valid = int(r["FS"]) == 0 and bool(ss & 1) and bool(ss & (8 | 16)) and not bool(ss & 128)
        if valid:
            valid_duration += duration
    return {
        "gps_status": gps_status,
        "ekf_status": {"primary_cores": sorted({r["PI"] for r in primary}),
                       "solution_status_values": sorted({r["SS"] for r in primary}),
                       "fault_status_values": sorted({r["FS"] for r in primary})},
        "valid_xy_position_duration_s": valid_duration,
        "xy_status_known_duration_s": known_duration,
        "xy_status_unknown_duration_s": max(0.0, end - start - known_duration),
    }


def scan_log(path: str | Path, *, inspection_config=None) -> ScanResult:
    """Scan every decodable message. Classify known EOF truncation as warning.

    Consumers must check completeness and parser diagnostics as well as
    critical_errors: a warning does not authorize use of incomplete input.
    """
    from pymavlink import DFReader, mavutil
    config = inspection_config or load_inspection_config()

    raw = Path(path).resolve()
    if raw.stat().st_size == 0:
        raise ValueError(f"DataFlash input is empty: {raw}")
    before = sha256_file(raw)
    size = raw.stat().st_size
    records = defaultdict(list)
    stdout, stderr = io.StringIO(), io.StringIO()
    reader = None
    decoder_formats = {}
    skipped = []
    last_end = 0
    exception = None
    reached_eof = False
    count = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            reader = DFReader.DFReader_binary(str(raw), zero_time_base=True)
            while True:
                previous_offset = reader.offset
                message = reader.recv_msg()  # no filter: read the complete stream
                if message is None:
                    reached_eof = True
                    break
                end = reader.offset
                start = end - message.fmt.len
                if end <= previous_offset:
                    raise RuntimeError(f"Parser made no forward progress at byte {previous_offset}")
                if start > last_end:
                    skipped.append({"offset": last_end, "byte_count": start - last_end})
                row = message.to_dict()
                row.pop("mavpackettype", None)
                row.update(_record_index=count, _offset=start)
                records[message.get_type()].append(row)
                count += 1
                last_end = end
            decoder_formats = dict(reader.formats)
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
        if reader is not None:
            decoder_formats = dict(reader.formats)
    finally:
        if reader is not None:
            reader.close()
        after = sha256_file(raw)
    trailing_count = max(0, size - last_end)
    trailing_kind = None
    if trailing_count:
        with raw.open("rb") as stream:
            stream.seek(last_end)
            header = stream.read(3)
        if (len(header) >= 3 and header[:2] == b"\xa3\x95" and header[2] in decoder_formats
                and trailing_count < decoder_formats[header[2]].len):
            trailing_kind = "truncated_message"
        else:
            trailing_kind = "unparsed_trailing_bytes"
    formats, inventory, units, multipliers, metadata_errors = _metadata(records, decoder_formats)
    critical = list(metadata_errors)
    diagnostics = {
        "stdout": stdout.getvalue().splitlines(), "stderr": stderr.getvalue().splitlines(),
        "exception": exception, "skipped_regions": skipped,
        "skipped_byte_count": sum(r["byte_count"] for r in skipped),
        "trailing_byte_count": trailing_count, "trailing_kind": trailing_kind,
        "trailing_message": decoder_formats[header[2]].name if trailing_kind == "truncated_message" else None,
        "trailing_expected_message_bytes": decoder_formats[header[2]].len if trailing_kind == "truncated_message" else None,
        "last_decoded_byte_exclusive": last_end, "file_size_bytes": size,
        "bytes_accounted_for": last_end + trailing_count,
        "error_count_policy": "anomaly regions + trailing region + exception + parser diagnostic lines; lines can describe the same region",
    }
    diagnostic_lines = len(diagnostics["stdout"]) + len(diagnostics["stderr"])
    errors = len(skipped) + bool(trailing_count) + bool(exception) + diagnostic_lines
    benign_tail = (trailing_kind == 'truncated_message' and reached_eof
                   and not skipped and not exception and not diagnostic_lines)
    events = []
    if benign_tail:
        events.append({'severity': 'WARNING', 'code': 'TRAILING_TRUNCATED_MESSAGE_AT_EOF',
                       'offset': last_end, 'available_bytes': trailing_count,
                       'expected_bytes': diagnostics['trailing_expected_message_bytes']})
    elif errors:
        events.append({'severity': 'CRITICAL', 'code': 'PARSER_CORRUPTION_OR_UNRESOLVED_DIAGNOSTIC'})
        critical.append(f"Parser anomalies detected ({errors} diagnostic events); inspect parser_diagnostics")
    diagnostics['events'] = events
    parser_status = 'FAIL' if errors and not benign_tail else 'PASS_WITH_WARNINGS' if benign_tail else 'PASS'
    if before != after:
        critical.append("Raw SHA256 changed during parsing")
    if not count:
        critical.append("No messages decoded")
    timestamp_metadata = {}
    field_index = {(r["message"], r["field"]): r for r in inventory if r['inventory_scope'] == 'message'}
    for name, rows in records.items():
        if "TimeUS" in formats[name]["fields"]:
            info = field_index[name, "TimeUS"]
            verified = info["unit"] == "s" and info["decoded_to_unit_factor"] is not None and math.isclose(info["decoded_to_unit_factor"], 1e-6, rel_tol=1e-6)
            timestamp_metadata[name] = {"verified": verified, "field": "TimeUS", "unit": "s", "factor": info["decoded_to_unit_factor"]}
            if not verified:
                critical.append(f"Unresolved timestamp metadata for {name}.TimeUS")
            for row in rows:
                row["time_s"] = float(row["TimeUS"]) * info["decoded_to_unit_factor"] if verified and _finite(row.get("TimeUS")) else None
        else:
            for row in rows:
                row["time_s"] = None
    timed = [r["time_s"] for rows in records.values() for r in rows if _finite(r.get("time_s"))]
    first, last = (min(timed), max(timed)) if timed else (None, None)
    message_stats, instance_stats = [], []
    for name in sorted(formats):
        rows = records.get(name, [])
        aggregate = _timing_stats(name, rows, rate_config=config['rate'])
        message_stats.append(aggregate)
        instance_field = formats[name]["instance_field"]
        if instance_field:
            groups = defaultdict(list)
            for row in rows:
                groups[row.get(instance_field)].append(row)
            for instance, group in sorted(groups.items(), key=lambda item: str(item[0])):
                instance_stats.append(_timing_stats(name, group, instance, instance_field, config['rate']))
            if len(groups) > 1:
                aggregate.update(nominal_rate_hz=None, active_window_rate_hz=None,
                                 active_duration_s=None, median_dt_s=None,
                                 minimum_dt_s=None, maximum_dt_s=None,
                                 gap_threshold_s=None, active_window_count=None,
                                 rate_scope='mixed_instances_count_only')
    params = {r["Name"]: r["Value"] for r in records.get("PARM", [])}
    messages = [_text(r.get("Message", "")) for r in records.get("MSG", [])]
    firmware = next((m for m in messages if m.startswith(("ArduCopter", "ArduPlane", "ArduRover", "ArduSub"))), None)
    board = next((m for m in messages if any(token in m.lower() for token in ("speedybee", "pixhawk", "cube", "matek"))), None)
    frame = next((m for m in messages if "frame" in m.lower()), None)
    if first is not None:
        segments, intervals, warnings, arming_source = _build_segments(records, first, last, mavutil.mode_mapping_acm)
    else:
        segments, intervals, warnings, arming_source = [], [], ["No verified timestamps; event intervals unavailable"], "unknown"
    for row in [*message_stats, *instance_stats]:
        if row["backward_timestamp_count"]:
            warnings.append(f"{row['message']} instance {row['instance']}: {row['backward_timestamp_count']} backward timestamp transitions")
    summary = {
        "schema_version": 2, "input_path": str(raw), "file_size_bytes": size,
        "raw_sha256": before, "raw_sha256_after": after, "raw_unchanged": before == after,
        "parsing_library": "pymavlink", "pymavlink_version": version("pymavlink"),
        "firmware": firmware, "board": board, "frame": frame, "firmware_messages": messages,
        "first_timestamp_s": first, "last_timestamp_s": last,
        "duration_s": last - first if first is not None else None,
        "timestamp_reference": "seconds since vehicle system startup; verified log TimeUS metadata",
        "timestamp_metadata": timestamp_metadata,
        "parsed_message_count": count, "parser_error_count": errors,
        "scan_complete": reached_eof and last_end == size and exception is None and not skipped,
        "reached_parser_eof": reached_eof,
        "whole_file_audited": reached_eof and last_end + trailing_count == size,
        "parser_diagnostics": diagnostics, "parser_status": parser_status,
        "critical_errors": list(dict.fromkeys(critical)),
        "message_counts": {name: len(rows) for name, rows in sorted(records.items())},
        "message_stats": message_stats, "instance_message_stats": instance_stats,
        "formats": formats, "units": units, "multipliers": multipliers,
        "parameters": params, "flight_count": len(segments), "flight_segments": segments,
        "mode_intervals": intervals, "arming_source": arming_source,
        "segment_interpretation": "armed intervals; arming does not establish takeoff or airborne duration",
        "xy_position_validity_policy": {"source": "XKF4 where C==PI", "require": "FS==0; SS attitude bit 1 and horizontal position bit 8 or 16; SS constant-position bit 128 clear",
                                        "maximum_hold_s": STATUS_MAX_HOLD_S, "unknown_gaps_preserved": True,
                                        "cross_arm_hold": False, "invalidate_on_primary_core_change": True,
                                        "interpretation": "bounded status-flag duration, not independent position-accuracy validation"},
        "semantic_sources": SEMANTIC_SOURCES, "warnings": list(dict.fromkeys(warnings)),
    }
    return ScanResult(dict(records), formats, inventory, message_stats, summary, params, segments, intervals)
