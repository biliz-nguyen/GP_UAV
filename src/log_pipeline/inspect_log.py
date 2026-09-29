"""Inspect a DataFlash BIN without modifying the input file."""
from __future__ import annotations

import argparse
from pathlib import Path

from .reader import scan_log
from .utils import write_csv, write_json


def inspect_log(path: str | Path, output_dir: str | Path = "results/log_inspection") -> dict:
    output = Path(output_dir)
    raw = Path(path).resolve()
    output_names = ("log_summary.json", "log_summary.txt", "message_counts.csv", "message_instance_counts.csv",
                    "field_inventory.csv", "flight_segments.csv", "mode_intervals.csv")
    # Resolve catches path aliases/symlinks; samefile also catches hardlinks,
    # whose different path names still reference the input's file contents.
    # Check every destination before opening any report for writing.
    for destination in (output, *(output / name for name in output_names)):
        if destination.resolve() == raw or (destination.exists() and destination.samefile(raw)):
            raise ValueError("Inspector output would overwrite the raw input file")
    result = scan_log(path)
    output.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    write_json(output / "log_summary.json", summary)
    write_csv(output / "message_counts.csv", result.message_stats)
    write_csv(output / "message_instance_counts.csv", summary["instance_message_stats"], fieldnames=list(result.message_stats[0]) if result.message_stats else [])
    write_csv(output / "field_inventory.csv", result.inventory)
    write_csv(output / "flight_segments.csv", result.flight_segments, fieldnames=list(result.flight_segments[0]) if result.flight_segments else ["flight_id", "arm_time_s", "disarm_time_s", "duration_s"])
    write_csv(output / "mode_intervals.csv", result.mode_intervals, fieldnames=list(result.mode_intervals[0]) if result.mode_intervals else ["start_time_s", "end_time_s", "flight_id", "mode"])
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
        "", "Message counts (aggregate rates can mix instances; see message_instance_counts.csv):",
    ]
    lines.extend(f"  {row['message']}: {row['count']}; mean {row['mean_rate_hz']} Hz; median dt {row['median_dt_s']} s" for row in result.message_stats)
    lines.extend(["", "Armed segments:"])
    lines.extend(f"  {r['flight_id']}: {r['arm_time_s']} to {r['disarm_time_s']} s, {r['duration_s']} s, modes={r['modes']}, valid XY status duration={r['valid_xy_position_duration_s']} s" for r in result.flight_segments)
    lines.extend(["", "Critical errors:"] + ([f"  {e}" for e in summary["critical_errors"]] or ["  none"]))
    lines.extend(["", "Warnings:"] + [f"  {e}" for e in summary["warnings"]])
    lines.extend(["", "Parser stdout:"] + summary["parser_diagnostics"]["stdout"])
    lines.extend(["", "Parser stderr:"] + summary["parser_diagnostics"]["stderr"])
    lines.extend(["", "Scaling: decoded DataFlash values already include format-character scaling; apply only decoded_to_unit_factor for verified UNIT values.",
                  "Unknown metadata and status coverage remain explicit; no motor threshold or unrestricted status fill is used."])
    (output / "log_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/log_inspection"))
    args = parser.parse_args()
    summary = inspect_log(args.log, args.output_dir)
    print(f"Scanned {summary['parsed_message_count']} messages; {summary['flight_count']} armed segments; {summary['parser_error_count']} parser diagnostic events")
    print(f"Report: {(args.output_dir / 'log_summary.json').resolve()}")
    for error in summary["critical_errors"]:
        print(f"CRITICAL: {error}")
    return 2 if summary["critical_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
