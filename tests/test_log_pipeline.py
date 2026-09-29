"""Binary fixtures exercise the actual pymavlink decoder, not a fake reader."""
import hashlib
import json
import os
import struct
from pathlib import Path

import pytest


def _fmt(type_id, name, fmt, columns, size):
    return b"\xa3\x95\x80" + struct.pack(
        "<BB4s16s64s", type_id, size, name.encode(), fmt.encode(), columns.encode()
    )


def _packet(type_id, fmt, *values):
    return b"\xa3\x95" + bytes([type_id]) + struct.pack("<" + fmt, *values)


def synthetic_log(path, *, tail=b"", time_metadata=True, interstitial=b""):
    # ATT.Yaw uses the DataFlash C encoding: 12345 -> 123.45 degrees.
    data = b"".join([
        _fmt(128, "FMT", "BBnNZ", "Type,Length,Name,Format,Columns", 89),
        _fmt(129, "UNIT", "QBN", "TimeUS,Id,Label", 28),
        _fmt(130, "MULT", "QBd", "TimeUS,Id,Mult", 20),
        _fmt(131, "FMTU", "QBNN", "TimeUS,FmtType,UnitIds,MultIds", 44),
        _fmt(132, "ATT", "QBC", "TimeUS,I,Yaw", 14),
        _fmt(133, "ARM", "QB", "TimeUS,ArmState", 12),
        _fmt(134, "MODE", "QBB", "TimeUS,Mode,ModeNum", 13),
        _packet(129, "QB16s", 0, ord("s"), b"s"),
        _packet(129, "QB16s", 0, ord("d"), b"deg"),
        _packet(129, "QB16s", 0, ord("#"), b"instance"),
        _packet(130, "QBd", 0, ord("F"), 1e-6),
        _packet(130, "QBd", 0, ord("B"), .01),
        _packet(130, "QBd", 0, ord("-"), 0),
    ])
    if time_metadata:
        for type_id, units, mults in [
            (129, b"s--", b"F--"), (130, b"s--", b"F--"),
            (131, b"s---", b"F---"), (132, b"s#d", b"F-B"),
            (133, b"s-", b"F-"), (134, b"s--", b"F--"),
        ]:
            data += _packet(131, "QB16s16s", 0, type_id, units, mults)
    data += b"".join([
        _packet(134, "QBB", 1_000_000, 0, 0),
        _packet(133, "QB", 1_000_000, 1),
        _packet(132, "QBH", 1_000_000, 0, 12345),
        _packet(132, "QBH", 1_000_000, 1, 22345),
        interstitial,
        _packet(132, "QBH", 1_100_000, 0, 12346),
        _packet(132, "QBH", 1_100_000, 1, 22346),
        _packet(134, "QBB", 1_200_000, 5, 5),
        _packet(133, "QB", 1_500_000, 0),
        _packet(133, "QB", 2_000_000, 1),
        _packet(132, "QBH", 2_100_000, 0, 10000),
        _packet(133, "QB", 2_500_000, 0),
        tail,
    ])
    path.write_bytes(data)
    return path


def test_scan_entire_file_and_scale_once(tmp_path):
    from src.log_pipeline.reader import scan_log

    path = synthetic_log(tmp_path / "fixture.bin")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    scan = scan_log(path)
    assert len(scan.records["ATT"]) == 5
    assert scan.records["ATT"][0]["time_s"] == 1.0
    assert scan.records["ATT"][0]["Yaw"] == 123.45
    yaw = next(r for r in scan.inventory if r["message"] == "ATT" and r["field"] == "Yaw")
    assert yaw["unit"] == "deg"
    assert yaw["decoded_scale_applied"] == .01
    assert yaw["decoded_to_unit_factor"] == 1.0
    assert yaw["observed_min"] == 100.0
    assert scan.summary["raw_sha256"] == before
    assert scan.summary["raw_sha256_after"] == before
    assert scan.summary["parser_error_count"] == 0
    assert scan.summary["critical_errors"] == []
    assert scan.summary["scan_complete"] is True
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_float32_metadata_rounding_preserves_exact_microsecond_time(tmp_path):
    from src.log_pipeline.reader import scan_log

    path = synthetic_log(tmp_path / "fixture.bin")
    float32_scale = struct.unpack("<f", struct.pack("<f", 1e-6))[0]
    path.write_bytes(path.read_bytes().replace(struct.pack("<d", 1e-6), struct.pack("<d", float32_scale), 1))
    scan = scan_log(path)
    assert scan.records["ATT"][0]["time_s"] == 1.0
    assert scan.field("ATT", "TimeUS")["multiplier_raw"] == float32_scale


def test_no_multiplier_sentinel_does_not_zero_or_rescale_decoder_values(tmp_path):
    from src.log_pipeline.reader import scan_log

    path = synthetic_log(tmp_path / "fixture.bin")
    path.write_bytes(path.read_bytes().replace(b"F-B" + b"\0" * 13, b"F--" + b"\0" * 13))
    scan = scan_log(path)
    assert scan.field("ATT", "I")["decoded_to_unit_factor"] == 1.0
    assert scan.field("ATT", "Yaw")["decoded_to_unit_factor"] == 1.0
    assert scan.records["ATT"][0]["Yaw"] == 123.45


def test_arm_segments_and_mode_boundaries_never_merge_disarmed_gap(tmp_path):
    from src.log_pipeline.reader import scan_log

    scan = scan_log(synthetic_log(tmp_path / "fixture.bin"))
    assert [(r["arm_time_s"], r["disarm_time_s"]) for r in scan.flight_segments] == [(1, 1.5), (2, 2.5)]
    assert scan.flight_segments[0]["modes"] == ["STABILIZE", "LOITER"]
    assert scan.flight_segments[0]["valid_xy_position_duration_s"] is None
    assert scan.flight_segments[0]["gps_status"] == "unknown: no GPS.Status samples"
    armed = [r for r in scan.mode_intervals if r["flight_id"] is not None]
    assert [(r["start_time_s"], r["end_time_s"]) for r in armed] == [(1, 1.2), (1.2, 1.5), (2, 2.5)]
    stats = next(r for r in scan.summary["instance_message_stats"] if r["message"] == "ATT" and r["instance"] == 1)
    assert stats["median_dt_s"] == pytest.approx(.1)
    assert stats["duplicate_timestamp_count"] == 0


def test_modes_are_reported_even_without_arming_events(tmp_path):
    from src.log_pipeline.reader import scan_log

    path = synthetic_log(tmp_path / "fixture.bin")
    data = path.read_bytes()
    for timestamp, state in [(1_000_000, 1), (1_500_000, 0), (2_000_000, 1), (2_500_000, 0)]:
        data = data.replace(_packet(133, "QB", timestamp, state), b"")
    path.write_bytes(data)
    scan = scan_log(path)
    assert scan.flight_segments == []
    assert [r["mode"] for r in scan.mode_intervals if r["mode"] != "UNKNOWN"] == ["STABILIZE", "LOITER"]
    assert all(r["armed"] is None for r in scan.mode_intervals)


@pytest.mark.parametrize("tail", [b"\xa3\x95\x84\x00", b"garbage", b"\x00"])
def test_trailing_bytes_are_never_silently_ignored(tmp_path, tail):
    from src.log_pipeline.reader import scan_log

    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", tail=tail))
    assert scan.summary["parser_error_count"] > 0
    assert scan.summary["parser_diagnostics"]["trailing_byte_count"] == len(tail)
    if tail == b"\xa3\x95\x84\x00":
        assert not scan.summary["critical_errors"]
        assert scan.summary['parser_status'] == 'PASS_WITH_WARNINGS'
    else:
        assert scan.summary["critical_errors"]


def test_malformed_bytes_between_messages_are_reported_and_later_records_scanned(tmp_path):
    from src.log_pipeline.reader import scan_log

    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", interstitial=b"bad!"))
    assert len(scan.records["ATT"]) == 5
    assert scan.summary["parser_diagnostics"]["skipped_byte_count"] == 4
    assert scan.summary["parser_error_count"] > 0


def test_long_garbage_tail_is_not_misclassified_as_a_truncated_packet(tmp_path):
    from src.log_pipeline.reader import scan_log

    # One complete decodable ATT header/body, then bytes without valid framing.
    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", tail=b"\xa3\x95\x84" + b"\0" * 100))
    diagnostics = scan.summary["parser_diagnostics"]
    assert diagnostics["trailing_kind"] == "unparsed_trailing_bytes"
    assert diagnostics["trailing_message"] is None
    assert diagnostics["trailing_byte_count"] == 89


def test_unknown_timestamp_metadata_fails_closed(tmp_path):
    from src.log_pipeline.reader import scan_log

    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", time_metadata=False))
    assert scan.records["ATT"][0]["time_s"] is None
    assert any("timestamp" in e.lower() for e in scan.summary["critical_errors"])
    assert scan.flight_segments == []


@pytest.mark.parametrize("redefinition", [
    _packet(129, "QB16s", 0, ord("d"), b"rad"),
    _packet(130, "QBd", 0, ord("B"), .1),
    _packet(131, "QB16s16s", 0, 132, b"s#s", b"F-B"),
])
def test_conflicting_unit_metadata_is_not_silently_applied_to_earlier_samples(tmp_path, redefinition):
    from src.log_pipeline.reader import scan_log

    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", tail=redefinition))
    assert any("changed definition" in error for error in scan.summary["critical_errors"])


def test_identical_metadata_repetition_remains_valid(tmp_path):
    from src.log_pipeline.reader import scan_log

    repeated = b"".join([
        _packet(129, "QB16s", 0, ord("d"), b"deg"),
        _packet(130, "QBd", 0, ord("B"), .01),
        _packet(131, "QB16s16s", 0, 132, b"s#d", b"F-B"),
    ])
    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", tail=repeated))
    assert scan.summary["critical_errors"] == []


def test_primary_ekf_change_invalidates_old_core_and_prearm_status(tmp_path):
    from src.log_pipeline.reader import scan_log

    statuses = b"".join([
        _fmt(135, "XKF4", "QBbIH", "TimeUS,C,PI,SS,FS", 19),
        _packet(131, "QB16s16s", 0, 135, b"s#---", b"F----"),
        _packet(135, "QBbIH", 1_000_000, 0, 0, 9, 0),
        _packet(135, "QBbIH", 1_100_000, 0, 1, 9, 0),
        _packet(135, "QBbIH", 1_300_000, 1, 1, 9, 0),
        _packet(135, "QBbIH", 1_800_000, 1, 1, 9, 0),
        _packet(135, "QBbIH", 2_200_000, 1, 1, 9, 0),
    ])
    scan = scan_log(synthetic_log(tmp_path / "fixture.bin", tail=statuses))
    assert scan.flight_segments[0]["valid_xy_position_duration_s"] == pytest.approx(.3)
    assert scan.flight_segments[0]["xy_status_unknown_duration_s"] == pytest.approx(.2)
    assert scan.flight_segments[1]["valid_xy_position_duration_s"] == pytest.approx(.3)
    assert scan.flight_segments[1]["xy_status_unknown_duration_s"] == pytest.approx(.2)


def test_empty_file_gives_actionable_error(tmp_path):
    from src.log_pipeline.reader import scan_log

    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        scan_log(path)


def test_inspector_writes_parseable_outputs(tmp_path):
    from src.log_pipeline.inspect_log import inspect_log

    summary = inspect_log(synthetic_log(tmp_path / "fixture.bin"), tmp_path / "reports")
    for name in ["log_summary.json", "message_counts.csv", "flight_segments.csv", "field_inventory.csv", "log_summary.txt", "mode_intervals.csv"]:
        assert (tmp_path / "reports" / name).is_file()
    saved = json.loads((tmp_path / "reports/log_summary.json").read_text())
    assert saved["parsed_message_count"] == summary["parsed_message_count"]
    assert saved["flight_count"] == 2


def test_inspector_cannot_overwrite_an_input_named_like_an_output(tmp_path):
    from src.log_pipeline.inspect_log import inspect_log

    raw = synthetic_log(tmp_path / "log_summary.json")
    original = raw.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        inspect_log(raw, tmp_path)
    assert raw.read_bytes() == original


@pytest.mark.parametrize("output_name", [
    "log_summary.json", "log_summary.txt", "message_counts.csv",
    "message_instance_counts.csv", "field_inventory.csv",
    "flight_segments.csv", "mode_intervals.csv",
])
def test_inspector_rejects_hardlinked_output_before_any_write(tmp_path, output_name):
    from src.log_pipeline.inspect_log import inspect_log

    raw = synthetic_log(tmp_path / "fixture.bin")
    original = raw.read_bytes()
    output = tmp_path / "reports"
    output.mkdir()
    os.link(raw, output / output_name)
    with pytest.raises(ValueError, match="overwrite"):
        inspect_log(raw, output)
    assert raw.read_bytes() == original
    assert sorted(path.name for path in output.iterdir()) == [output_name]


@pytest.mark.integration
def test_real_log_inspection_and_raw_immutability(tmp_path):
    from src.log_pipeline.inspect_log import inspect_log

    if os.environ.get("RUN_REAL_LOG_TESTS") != "1":
        pytest.skip("Set RUN_REAL_LOG_TESTS=1 for the real DataFlash integration test")
    raw = Path(os.environ.get("REAL_LOG_PATH", "data/raw/flight.bin")).resolve()
    if not raw.is_file():
        pytest.skip("Real flight BIN is not available")
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    summary = inspect_log(raw, tmp_path / "inspection")
    assert summary["parsed_message_count"] > 0
    assert "FMT" in summary["message_counts"]
    assert any(name in summary["message_counts"] for name in ("ATT", "IMU", "GPS", "XKF1"))
    assert summary["whole_file_audited"]
    baseline = os.environ.get('INSPECTION_BASELINE_PATH')
    if baseline:
        previous = json.loads(Path(baseline).read_text(encoding='utf-8-sig'))
        assert summary['message_counts'] == previous['message_counts']
        assert summary['raw_sha256'] == previous['raw_sha256']
    diagnostics = summary["parser_diagnostics"]
    assert diagnostics["skipped_byte_count"] == 0
    if diagnostics["trailing_byte_count"]:
        assert diagnostics["trailing_kind"] == "truncated_message"
        assert 3 <= diagnostics["trailing_byte_count"] < diagnostics["trailing_expected_message_bytes"]
        assert not summary["critical_errors"]
        assert summary['parser_status'] == 'PASS_WITH_WARNINGS'
    else:
        assert summary["scan_complete"]
        assert not summary["critical_errors"]
    assert diagnostics["exception"] is None
    assert not diagnostics["stdout"] and not diagnostics["stderr"]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
