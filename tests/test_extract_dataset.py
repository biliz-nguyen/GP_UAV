"""Hand-checked fixtures for extraction safety and numerical provenance."""
from copy import deepcopy
import math
import json

import numpy as np
import pandas as pd
import pytest

from src.log_pipeline.reader import ScanResult


def fixture_scan():
    records = {
        "PSCE": [], "PSCN": [], "PSCD": [], "ATT": [], "XKF1": [],
        "XKF4": [], "GPS": [], "BAT": [], "CTUN": [], "RCOU": [],
        "MODE": [{"time_s": 1., "ModeNum": 5}], "EV": [], "PARM": [],
    }
    units = {"TimeUS": "s", "C": "instance", "PI": "", "SS": "", "FS": "", "TS": "", "OFN": "m", "OFE": "m", "Status": "", "I": "instance", "Inst": "instance", "Yaw": "degheading", "Volt": "V", "Curr": "A", "ThO": ""}
    for i in range(10):
        t = 1 + i / 10
        common = {"time_s": t, "TimeUS": int(t * 1e6), "_record_index": i}
        # Horizontal controller coverage exists for only one original packet.
        for name, suffix, p, v, a, target in [("PSCE", "E", 2, 5, 3, 1), ("PSCN", "N", 1, 4, 7, 2), ("PSCD", "D", 3, 6, 4, 1)]:
            if suffix == "D" or i == 0:
                records[name].append(dict(common, **{f"P{suffix}": p, f"V{suffix}": v, f"A{suffix}": a, f"TA{suffix}": target}))
            units.update({f"P{suffix}": "m", f"V{suffix}": "m/s", f"A{suffix}": "m/s/s", f"TA{suffix}": "m/s/s"})
        records["ATT"].append(dict(common, Yaw=0.))
        records["XKF1"].append(dict(common, C=0, PN=11, PE=12, PD=13, VN=14, VE=15, VD=16))
        records["XKF4"].append(dict(common, C=0, PI=0, FS=0, SS=1+2+4+8+32, TS=0, OFN=0., OFE=0.))
        records["GPS"].append(dict(common, I=0, Status=3))
        records["BAT"].append(dict(common, Inst=0, Volt=11.7, Curr=2.3))
        records["CTUN"].append(dict(common, ThO=.4))
        records["RCOU"].append(dict(common, C1=1100, C2=1200, C3=1300, C4=1400))
    units.update({f"C{i}": "us" for i in range(1, 5)})
    inventory = []
    for name, rows in records.items():
        fields = set().union(*(r.keys() for r in rows)) if rows else set()
        for field in sorted(fields - {"time_s", "_record_index"}):
            inventory.append({"message": name, "field": field, "unit": units.get(field, ""), "unit_verified": True, "decoded_to_unit_factor": 1e-6 if field == "TimeUS" else 1., "multiplier_id": "0"})
    summary = {"raw_sha256": "abc", "raw_sha256_after": "abc", "raw_unchanged": True, "firmware": "ArduCopter V4.8.0-dev (2b5cebb9)", "critical_errors": [], "parser_error_count": 0, "parser_diagnostics": {}, "scan_complete": True, "whole_file_audited": True, "reached_parser_eof": True, "warnings": [], "message_stats": [], "instance_message_stats": [], "pymavlink_version": "test"}
    segments = [{"flight_id": 1, "arm_time_s": 1., "end_time_s": 2., "disarm_time_s": 2., "open_ended": False}]
    modes = [{"flight_id": 1, "start_time_s": 1., "end_time_s": 2., "mode": "LOITER", "mode_number": 5}]
    return ScanResult(records, {}, inventory, [], summary, {f"SERVO{i}_FUNCTION": 32+i for i in range(1,5)}, segments, modes)


def config():
    from src.log_pipeline.extract_dataset import load_config
    return load_config()


def test_residual_sign_axes_proxy_and_no_invented_horizontal_targets():
    from src.log_pipeline.extract_dataset import build_dataset
    data = build_dataset(fixture_scan(), config())
    first = data.iloc[0]
    assert [first.px, first.py, first.pz] == [2, 1, -3]
    assert [first.dx, first.dy, first.dz] == [2, 5, -3]
    assert first.yaw == pytest.approx(math.pi/2)
    assert first.battery_v == 11.7 and first.battery_i == 2.3
    assert first.throttle == .4 and first.motor1 == 1100
    assert data.valid_dx.sum() == 2
    assert data.valid_dy.sum() == 2
    assert data.valid_dz.sum() == 10
    assert data.loc[2:, "dx"].isna().all()
    assert data.xy_acceleration_is_proxy.all()
    assert not data.eligible_dx.any() and not data.eligible_dy.any()
    assert data.iloc[2].px == 12 and data.iloc[2].source_px == "XKF1.PE[C=0]"


def test_mode_boundary_blocks_previous_values_then_reacquires_status():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    scan.mode_intervals[0]["end_time_s"] = 1.25
    scan.mode_intervals.append({"flight_id": 1, "start_time_s": 1.25, "end_time_s": 2., "mode": "ALT_HOLD", "mode_number": 2})
    # At 1.3 old samples must not leak; fresh 1.4 samples restore validity.
    for name in ["PSCD", "XKF1", "XKF4", "ATT"]:
        scan.records[name] = [r for r in scan.records[name] if r["time_s"] != 1.3]
    data = build_dataset(scan, config())
    assert not data.iloc[3].valid_dz
    assert np.isnan(data.iloc[3].pz)
    assert data.iloc[4].valid_dz and data.iloc[4].ekf_status_known
    assert data.iloc[4]["mode"] == "ALT_HOLD"
    assert data.iloc[4].continuity_start_s == 1.25
    assert data.iloc[4].continuity_end_s == 2.


def test_gps_invalid_does_not_erase_vertical_controller_residual():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    for row in scan.records["GPS"]:
        row["Status"] = 1
    data = build_dataset(scan, config())
    assert not data.gps_valid.any()
    assert data.valid_dz.all()


def test_unhealthy_core_cannot_supply_missing_psc_state():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    for row in scan.records["XKF4"]:
        row["FS"] = 2
    data = build_dataset(scan, config())
    assert data.iloc[0].px == 2
    assert data.loc[2:, "px"].isna().all()


def test_unverified_unit_rejects_conversion_and_unknown_build_rejects_profile():
    from src.log_pipeline.extract_dataset import build_dataset, validate_scan
    scan = fixture_scan()
    next(r for r in scan.inventory if r["message"] == "PSCE" and r["field"] == "AE")["unit"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unit"):
        build_dataset(scan, config())
    scan = fixture_scan()
    scan.summary["firmware"] = "ArduCopter V4.8.0-dev (deadbeef)"
    with pytest.raises(ValueError, match="firmware"):
        validate_scan(scan, {"raw_sha256": "abc"})


def test_hash_mismatch_and_corruption_fail_closed():
    from src.log_pipeline.extract_dataset import validate_scan
    scan = fixture_scan()
    with pytest.raises(ValueError, match="SHA256"):
        validate_scan(scan, {"raw_sha256": "different"})
    scan.summary["critical_errors"] = ["Parser anomalies detected (1 diagnostic events); inspect parser_diagnostics"]
    scan.summary["parser_error_count"] = 1
    scan.summary["scan_complete"] = False
    scan.summary["parser_diagnostics"] = {"trailing_kind": "truncated_message", "trailing_byte_count": 31, "trailing_expected_message_bytes": 54, "trailing_message": "IMU", "last_decoded_byte_exclusive": 100, "file_size_bytes": 131, "bytes_accounted_for": 131, "skipped_regions": [], "skipped_byte_count": 0, "stdout": [], "stderr": [], "exception": None}
    with pytest.raises(ValueError, match="Parser|corrupt|truncat"):
        validate_scan(scan, {"raw_sha256": "abc"})
    validate_scan(scan, {"raw_sha256": "abc"}, allow_truncated_tail=True)
    scan.summary["parser_diagnostics"]["skipped_byte_count"] = 4
    with pytest.raises(ValueError):
        validate_scan(scan, {"raw_sha256": "abc"}, allow_truncated_tail=True)


def test_motor_mapping_ambiguity_keeps_raw_channel_values():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    scan.params["SERVO2_FUNCTION"] = 33
    data = build_dataset(scan, config())
    assert data.motor1.isna().all()
    assert data.rcou_c1.eq(1100).all()


def test_historical_motor_duplicate_on_another_channel_prevents_logical_label():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    scan.records["PARM"] = [{"Name": "SERVO2_FUNCTION", "Value": 33}, {"Name": "SERVO2_FUNCTION", "Value": 34}]
    data = build_dataset(scan, config())
    assert data.motor1.isna().all() and data.motor2.isna().all()


def test_metadata_backward_time_is_reported_but_signal_backward_time_rejected():
    from src.log_pipeline.extract_dataset import validate_scan
    scan = fixture_scan()
    scan.summary["message_stats"] = [{"message": "FMTU", "backward_timestamp_count": 1}]
    validate_scan(scan, {"raw_sha256": "abc"})
    scan.summary["message_stats"].append({"message": "ATT", "backward_timestamp_count": 1})
    with pytest.raises(ValueError, match="Backward"):
        validate_scan(scan, {"raw_sha256": "abc"})


def test_holdout_must_follow_training_in_time():
    from src.log_pipeline.extract_dataset import assign_splits
    data = pd.DataFrame({"time_s": [1,2,3,4], "flight_id": [1,1,2,2]})
    cfg = config()
    cfg["split"]["holdout_flight_id"] = 1
    with pytest.raises(ValueError, match="holdout|chronolog"):
        assign_splits(data, cfg)
    cfg["split"]["holdout_flight_id"] = 2
    assert assign_splits(data, cfg).split.tolist() == ["train", "train", "test", "test"]


def test_chronological_split_cannot_divide_duplicate_timestamp_or_fake_holdout():
    from src.log_pipeline.extract_dataset import assign_splits
    data = pd.DataFrame({"time_s": [1,2,2,3,4,5], "flight_id": [1]*6})
    result = assign_splits(data, config())
    assert result.groupby("time_s").split.nunique().max() == 1
    assert result.loc[result.split.eq("train"), "time_s"].max() < result.loc[result.split.eq("test"), "time_s"].min()
    cfg = config()
    cfg["split"]["holdout_flight_id"] = 1
    with pytest.raises(ValueError, match="holdout"):
        assign_splits(data, cfg)


def test_extraction_is_deterministic_and_does_not_mutate_scan():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    original = deepcopy(scan.records)
    first = build_dataset(scan, config())
    pd.testing.assert_frame_equal(first, build_dataset(scan, config()))
    assert scan.records == original


def test_complete_synthetic_binary_exports_traceable_deterministic_files(tmp_path, monkeypatch):
    from test_log_pipeline import synthetic_log, _fmt, _packet
    from src.log_pipeline import extract_dataset as extractor
    from src.log_pipeline.reader import scan_log
    from src.log_pipeline.utils import sha256_file, write_json
    raw = synthetic_log(tmp_path / "flight.bin")
    # Actual decoded firmware message supplies the pinned semantic profile.
    raw.write_bytes(raw.read_bytes() + _fmt(140, "MSG", "QZ", "TimeUS,Message", 75)
                    + _packet(131, "QB16s16s", 0, 140, b"s-", b"F-")
                    + _packet(140, "Q64s", 0, b"ArduCopter V4.8.0-dev (2b5cebb9)"))
    before = sha256_file(raw)
    scan = scan_log(raw)
    inspection = tmp_path / "inspection.json"
    write_json(inspection, scan.summary)
    monkeypatch.setattr(extractor, "PROJECT", tmp_path)
    first = extractor.extract_dataset(raw, inspection, tmp_path / "results/one")
    second = extractor.extract_dataset(raw, inspection, tmp_path / "results/two")
    assert first["file_hashes"] == second["file_hashes"]
    assert sha256_file(raw) == before
    assert first["valid_counts"] == {"dx": 0, "dy": 0, "dz": 0}
    assert first["gp_row_count"] == 0
    table = pd.read_csv(tmp_path / "results/one/all_flights.csv")
    assert set(table.columns) <= set(first["source_mapping"])
    assert len(table.flight_id.unique()) == 2
    for name, digest in first["file_hashes"].items():
        assert sha256_file(tmp_path / "results/one" / name) == digest
    manifest = json.loads((tmp_path / "results/one/dataset_manifest.json").read_text())
    assert manifest["complete_prefix_exception_used"] is False
    assert all(first["source_mapping"][col]["output_frame"] == "ENU" for col in ["px", "vx", "ax", "ax_target", "dx", "yaw"])


def test_output_name_collision_cannot_overwrite_binary_input(tmp_path, monkeypatch):
    from test_log_pipeline import synthetic_log, _fmt, _packet
    from src.log_pipeline import extract_dataset as extractor
    from src.log_pipeline.reader import scan_log
    from src.log_pipeline.utils import sha256_file, write_json
    output = tmp_path / "results/out"
    output.mkdir(parents=True)
    raw = synthetic_log(output / "all_flights.csv")
    raw.write_bytes(raw.read_bytes() + _fmt(140, "MSG", "QZ", "TimeUS,Message", 75)
                    + _packet(131, "QB16s16s", 0, 140, b"s-", b"F-")
                    + _packet(140, "Q64s", 0, b"ArduCopter V4.8.0-dev (2b5cebb9)"))
    before = sha256_file(raw)
    inspection = tmp_path / "inspection.json"
    write_json(inspection, scan_log(raw).summary)
    monkeypatch.setattr(extractor, "PROJECT", tmp_path)
    with pytest.raises(ValueError, match="overwrite|protected|input"):
        extractor.extract_dataset(raw, inspection, output)
    assert sha256_file(raw) == before


def test_csv_preserves_submicrosecond_source_age_at_large_startup_time(tmp_path, monkeypatch):
    from src.log_pipeline import extract_dataset as extractor
    from src.log_pipeline.utils import sha256_file, write_json
    # Isolate serialization using a hand-computed table; parsing is exercised
    # separately by the complete synthetic-binary test above.
    raw = tmp_path / "fixture.bin"
    raw.write_bytes(b"serialization fixture")
    scan = fixture_scan()
    scan.summary["raw_sha256"] = scan.summary["raw_sha256_after"] = sha256_file(raw)
    inspection = tmp_path / "inspection.json"
    write_json(inspection, scan.summary)
    table = extractor.build_dataset(scan, config()).iloc[:1].copy()
    time_s, source_s = 12345.123456789, 12345.112345
    table.loc[:, "time_s"] = time_s
    table.loc[:, "source_time_pscd_s"] = source_s
    table.loc[:, "age_pscd_ms"] = 1000 * (time_s - source_s)
    monkeypatch.setattr(extractor, "PROJECT", tmp_path)
    monkeypatch.setattr(extractor, "scan_log", lambda path: scan)
    monkeypatch.setattr(extractor, "build_dataset", lambda scan, cfg: table)
    extractor.extract_dataset(raw, inspection, tmp_path / "results/precision")
    saved = pd.read_csv(tmp_path / "results/precision/all_flights.csv", float_precision="round_trip").iloc[0]
    assert abs(1000 * (saved.time_s - saved.source_time_pscd_s) - saved.age_pscd_ms) < 1e-5
    assert saved.time_s == time_s
    assert saved.source_time_pscd_s == source_s


def test_core_switch_never_reuses_old_core_and_later_source_recovers():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    for row in scan.records["XKF4"]:
        if row["time_s"] >= 1.3:
            row["C"] = row["PI"] = 1
    for row in scan.records["XKF1"]:
        if row["time_s"] >= 1.4:
            row["C"] = 1
            row["PE"] = 99
    result = build_dataset(scan, config())
    assert np.isnan(result.iloc[3].px)
    assert result.iloc[4].px == 99
    assert result.iloc[4].source_px == "XKF1.PE[C=1]"


@pytest.mark.parametrize("new_status_at", [1.6, None])
def test_nonprimary_core_announcing_switch_invalidates_status_until_new_core_arrives(new_status_at):
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    scan.records["PSCE"] = []
    scan.records["PSCN"] = []
    for row in scan.records["XKF4"]:
        if row["time_s"] >= 1.3:
            row["PI"] = 1
        if new_status_at is not None and row["time_s"] >= new_status_at:
            row["C"] = 1
    for row in scan.records["XKF1"]:
        if new_status_at is not None and row["time_s"] >= new_status_at:
            row["C"] = 1
            row["PE"] = 99
    result = build_dataset(scan, config())
    assert result.iloc[2].px == 12
    assert result.iloc[3].continuity_start_s == 1.3
    assert result.loc[3:5, "ekf_primary_core"].eq(1).all()
    assert not result.loc[3:5, "ekf_status_known"].any()
    assert not result.loc[3:5, "ekf_healthy"].any()
    assert result.loc[3:5, "px"].isna().all()
    if new_status_at is None:
        assert result.loc[3:, "ekf_primary_core"].eq(1).all()
        assert not result.loc[3:, "ekf_status_known"].any()
        assert result.loc[3:, "px"].isna().all()
    else:
        assert result.iloc[6].ekf_status_known
        assert result.iloc[6].px == 99
        assert result.iloc[6].source_px == "XKF1.PE[C=1]"


def test_regular_secondary_status_does_not_erase_current_primary_health():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    scan.records["PSCE"] = []
    scan.records["PSCN"] = []
    scan.records["XKF4"] = [packet for row in scan.records["XKF4"] for packet in
                            [row, {**row, "C": 1, "SS": 0, "FS": 128}]]
    result = build_dataset(scan, config())
    assert result.ekf_status_known.all()
    assert result.ekf_healthy.all()
    assert result.ekf_primary_core.eq(0).all()
    assert result.px.eq(12).all()


def test_outliers_are_retained_but_not_eligible_and_age_is_bounded():
    from src.log_pipeline.extract_dataset import build_dataset
    scan = fixture_scan()
    scan.records["PSCD"][0]["AD"] = 90
    result = build_dataset(scan, config())
    assert len(result) == 10
    assert result.iloc[0].valid_dz and result.iloc[0].outlier
    assert not result.iloc[0].eligible_dz
    assert result.loc[result.valid_dx, "age_psce_ms"].max() <= 180
