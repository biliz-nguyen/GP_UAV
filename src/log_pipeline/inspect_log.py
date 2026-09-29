"""Inspect a DataFlash BIN without modifying the input file."""
from __future__ import annotations

import argparse
from pathlib import Path

from .reader import scan_log
from .utils import write_csv, write_json
from .inspection_metrics import load_inspection_config
from .inspector_analysis import analyze_inspection


def _statuses(result, analysis, config):
    summary = result.summary
    unknown_units = [r for r in result.inventory if r['finite_count'] and r['canonical_unit'] is None]
    availability = config['inspector_availability']
    z_duration = sum(f.get('z_residual_candidate_duration_s') or 0 for f in result.flight_segments)
    z_count = sum(f.get('pscd_paired_sample_count', 0) for f in result.flight_segments)
    known = analysis['residual_semantics']['verified_profile']
    physical = analysis['physical_quality_warnings']
    def sensor_status(message):
        if not result.records.get(message):
            return 'UNKNOWN'
        fields = [('Volt', 'V'), ('Curr', 'A')]
        if any(not any(r['message'] == message and r['field'] == f and r.get('canonical_unit') == unit for r in result.inventory) for f, unit in fields):
            return 'UNKNOWN'
        if any(w['message'] == message for w in physical):
            return 'WARNING'
        checks = [r for r in analysis['physical_quality_checks'] if r['message'] == message]
        return 'NO_WARNING' if checks and all(r['sufficient_samples'] for r in checks) else 'UNKNOWN'
    return {'parser': summary['parser_status'],
            'structural_log': 'FAIL' if summary['critical_errors'] else 'PASS',
            'flight_segment_detection': 'UNKNOWN' if summary['arming_source'] == 'unknown' else 'PASS_WITH_WARNINGS' if any(f['open_ended'] for f in result.flight_segments) else 'PASS',
            'unit_metadata': 'FAIL' if any('metadata' in e.lower() or 'definition' in e.lower() for e in summary['critical_errors']) else 'PASS_WITH_WARNINGS' if unknown_units else 'PASS',
            'xy_residual_data_availability': 'INSUFFICIENT' if known else 'UNRESOLVED',
            'z_residual_data_availability': 'UNRESOLVED' if not known or any(f.get('z_residual_candidate_duration_s') is None for f in result.flight_segments) else 'AVAILABLE' if z_duration >= availability['minimum_candidate_duration_s'] and z_count >= availability['minimum_paired_samples'] else 'INSUFFICIENT',
            'battery_physical_quality': sensor_status('BAT'), 'esc_telemetry_physical_quality': sensor_status('ESC')}


def inspect_log(path: str | Path, output_dir: str | Path = "results/log_inspection", *, config_path=None) -> dict:
    output = Path(output_dir)
    raw = Path(path).resolve()
    output_names = ("log_summary.json", "log_summary.txt", "message_counts.csv", "message_instance_counts.csv",
                    "field_inventory.csv", "flight_segments.csv", "mode_intervals.csv",
                    "preliminary_audit.json", "vibration_summary.csv", "physical_quality_warnings.csv",
                    "psc_activity_by_mode.csv", "acceleration_dynamic_range.csv")
    # Resolve catches path aliases/symlinks; samefile also catches hardlinks,
    # whose different path names still reference the input's file contents.
    # Check every destination before opening any report for writing.
    for destination in (output, *(output / name for name in output_names)):
        if destination.resolve() == raw or (destination.exists() and destination.samefile(raw)):
            raise ValueError("Inspector output would overwrite the raw input file")
    config = load_inspection_config(config_path)
    result = scan_log(path, inspection_config=config)
    output.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    analysis = analyze_inspection(result, config)
    summary.update(analysis)
    summary['inspection_config'] = {k: config[k] for k in ('rate', 'physical_checks', 'vibration', 'inspector_availability')}
    summary['rate_policy'] = {
        'aggregate_span_rate_hz': '(timestamped_count - 1) / (last_timestamp - first_timestamp); not a nominal frequency',
        'nominal_rate_hz': '1 / median of strictly positive consecutive dt; duplicate/backward counts retained separately',
        'active_window_rate_hz': 'number of positive consecutive dt <= gap_threshold / sum of those dt',
        'gap_threshold': 'max(absolute_gap_threshold_s, gap_multiplier * median_positive_dt)',
        'mixed_instances': 'counts and aggregate span rate only; use per-instance rows for nominal/active rates',
        'legacy_mean_rate_hz': 'deprecated alias of aggregate_span_rate_hz',
        'event_messages': 'descriptive timing only; not a periodic sensor sampling claim'}
    summary['statuses'] = _statuses(result, analysis, config)
    summary['unit_metadata_diagnostics'] = [dict(message=r['message'], field=r['field'],
                                                 unit_id=r['unit_id'], multiplier_id=r['multiplier_id'],
                                                 finite_count=r['finite_count'], metadata_source=r['metadata_source'])
                                            for r in result.inventory if r['inventory_scope'] == 'message'
                                            and r['finite_count'] and r['canonical_unit'] is None]
    summary['warnings'].extend(f"{w['code']}: {w['message']} instance {w['instance']} ({w['flagged_count']}/{w['evaluated_count']})" for w in analysis['physical_quality_warnings'])
    summary['warnings'].extend(f"{e['code']}: {e['severity']}" for e in summary['parser_diagnostics']['events'] if e['severity'] == 'WARNING')
    summary['freeze_part1'] = 'YES' if not summary['critical_errors'] and analysis['residual_semantics']['verified_profile'] and all(
        any(r['message'] == name and r['field'] == field and r['canonical_unit'] == unit for r in result.inventory)
        for name, field, unit in [('PSCN', 'AN', 'm/s/s'), ('PSCN', 'TAN', 'm/s/s'), ('PSCE', 'AE', 'm/s/s'), ('PSCE', 'TAE', 'm/s/s'), ('PSCD', 'AD', 'm/s/s'), ('PSCD', 'TAD', 'm/s/s')]) else 'NO'
    summary['freeze_interpretation'] = 'Inspector scope only; sensor calibration and physical XY residual data remain unresolved/unavailable. Not ML readiness.'
    write_json(output / "log_summary.json", summary)
    write_csv(output / "message_counts.csv", result.message_stats)
    write_csv(output / "message_instance_counts.csv", summary["instance_message_stats"])
    write_csv(output / "field_inventory.csv", result.inventory)
    write_csv(output / "flight_segments.csv", result.flight_segments, fieldnames=list(result.flight_segments[0]) if result.flight_segments else ["flight_id", "arm_time_s", "disarm_time_s", "duration_s"])
    write_csv(output / "mode_intervals.csv", result.mode_intervals, fieldnames=list(result.mode_intervals[0]) if result.mode_intervals else ["start_time_s", "end_time_s", "flight_id", "mode"])
    write_json(output / 'preliminary_audit.json', dict(raw_sha256=summary['raw_sha256'], counts=summary['message_counts'],
                                                     statuses=summary['statuses'], inspection_config=summary['inspection_config'], **analysis))
    for key in ('vibration_summary', 'physical_quality_warnings', 'psc_activity_by_mode', 'acceleration_dynamic_range'):
        write_csv(output / (key + '.csv'), analysis[key], fieldnames=None if analysis[key] else ['message', 'instance', 'status'])
    lines = [
        "DataFlash log inspection", f"Input: {summary['input_path']}",
        f"Bytes: {summary['file_size_bytes']}", f"SHA256: {summary['raw_sha256']}",
        f"Raw unchanged: {summary['raw_unchanged']}",
        f"Parser: pymavlink {summary['pymavlink_version']}",
        f"Firmware: {summary['firmware'] or 'unknown'}", f"Board: {summary['board'] or 'unknown'}",
        f"Frame: {summary['frame'] or 'unknown'}",
        f"Time: {summary['first_timestamp_s']} to {summary['last_timestamp_s']} s since startup",
        f"Duration: {summary['duration_s']} s", f"Parsed messages: {summary['parsed_message_count']}",
        f"Parser diagnostic events: {summary['parser_error_count']}",
        f"Complete byte coverage: {summary['scan_complete']}",
        f"Armed segments: {summary['flight_count']} (not proof of takeoff)",
        "", "Message counts (nominal/active rates for multiple instances appear only in message_instance_counts.csv):",
    ]
    lines.extend(f"  {row['message']}: {row['count']}; aggregate span {row['aggregate_span_rate_hz']} Hz; nominal {row['nominal_rate_hz']} Hz; active-window {row['active_window_rate_hz']} Hz" for row in result.message_stats)
    lines.extend(["", "Armed segments:"])
    lines.extend(f"  {r['flight_id']}: modes={r['modes']}; EKF XY valid={r['ekf_xy_valid_duration_s']} s; PSC N/E/D active={r['pscn_active_duration_s']}/{r['psce_active_duration_s']}/{r['pscd_active_duration_s']} s; XY physical candidate={r['xy_residual_candidate_duration_s']} s; XY proxy overlap={r['xy_proxy_candidate_duration_s']} s; Z candidate={r['z_residual_candidate_duration_s']} s" for r in result.flight_segments)
    lines.extend(['', 'Observed PSC activity by mode interval (diagnostic, not mode-specific requirements):'])
    lines.extend(f"  {r['message']}: {r['count']} samples, {r['active_duration_s']} s; mode={r['mode']}, armed={r['armed']}, flight={r['flight_id']}" for r in analysis['psc_activity_by_mode'] if r['count'])
    lines.extend(["", "Critical errors:"] + ([f"  {e}" for e in summary["critical_errors"]] or ["  none"]))
    lines.extend(["", "Warnings:"] + [f"  {e}" for e in summary["warnings"]])
    lines.extend(["", "Parser stdout:"] + summary["parser_diagnostics"]["stdout"])
    lines.extend(["", "Parser stderr:"] + summary["parser_diagnostics"]["stderr"])
    lines.extend(['', 'Definitions:', *[f'  {key}: {value}' for key, value in summary['rate_policy'].items()],
                  '  Activity: ' + analysis['residual_semantics']['duration_policy'],
                  '  XY: ' + analysis['residual_semantics']['xy'], '  Z: ' + analysis['residual_semantics']['z'],
                  '  Inventory: observed_*_decoded retain decoder values; canonical values multiply by decoded_to_canonical_factor only when metadata resolves.',
                  '  Legacy unit/observed_min/observed_max mean canonical unit/decoded statistics respectively; use explicit new columns.',
                  '  Armed is not airborne. VIBE Clip totals are observed within-interval counter increments, not summed cumulative readings.',
                  '  Sensor screening uses configured thresholds; NO_WARNING is not calibrated/validated.',
                  '', 'STATUS SECTIONS (Inspector only; no ML readiness):'])
    labels = {'parser': 'PARSER STATUS', 'structural_log': 'STRUCTURAL LOG STATUS', 'flight_segment_detection': 'FLIGHT SEGMENT DETECTION',
              'unit_metadata': 'UNIT METADATA', 'xy_residual_data_availability': 'XY RESIDUAL DATA AVAILABILITY',
              'z_residual_data_availability': 'Z RESIDUAL DATA AVAILABILITY', 'battery_physical_quality': 'BATTERY PHYSICAL QUALITY',
              'esc_telemetry_physical_quality': 'ESC TELEMETRY PHYSICAL QUALITY'}
    lines.extend(f'{labels[k]}: {v}' for k, v in summary['statuses'].items())
    lines.extend([f"FREEZE_PART1 = {summary['freeze_part1']}", summary['freeze_interpretation']])
    (output / "log_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/log_inspection"))
    parser.add_argument('--config', type=Path, default=None)
    args = parser.parse_args()
    summary = inspect_log(args.log, args.output_dir, config_path=args.config)
    print(f"Scanned {summary['parsed_message_count']} messages; {summary['flight_count']} armed segments; {summary['parser_error_count']} parser diagnostic events")
    print(f"Report: {(args.output_dir / 'log_summary.json').resolve()}")
    for error in summary["critical_errors"]:
        print(f"CRITICAL: {error}")
    return 2 if summary["critical_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
