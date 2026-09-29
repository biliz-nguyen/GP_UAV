# Inspector reporting contract

This stage inventories DataFlash observations. It does not create a dataset, transform coordinates, subtract accelerations or assess ML readiness. Records retain pymavlink-decoded values; no warning filters, repairs or rescales the input.

## Diagnostics and statuses

`parser_diagnostics` retains parser stdout/stderr, exceptions, byte offsets, skipped regions and the exact trailing packet length. A known packet header with fewer than its declared bytes at EOF is WARNING only when no interior skips, exception or other parser output exists. Other unresolved parser anomalies remain CRITICAL. `parser_status` is PASS, PASS_WITH_WARNINGS or FAIL. `scan_complete=false` still distinguishes incomplete byte coverage from a successfully audited file. Metadata failures are tracked separately in `critical_errors` and structural/unit statuses.

`statuses` reports parser, structural, arm-event detection, metadata, physical residual-signal availability and battery/ESC screening separately. AVAILABLE means finite source-verified paired acceleration observations exceed `inspector_availability` counts/duration. It does not certify accuracy, excitation or learnability. UNKNOWN/UNRESOLVED are retained when evidence is absent. NO_WARNING means configured screening did not fire, not calibrated or healthy.

`FREEZE_PART1=YES` freezes only the Inspector implementation and its explicit reporting of limitations. Critical parser, timestamp or required acceleration-unit/semantic failures prevent this recommendation. Missing independent XY acceleration, unknown airborne state and sensor calibration remain visible limitations, not resolved by freezing Part 1.

## Rates and activity

All timestamps are seconds since vehicle startup. For each individual instance:

- `aggregate_span_rate_hz = (timestamped_count - 1) / (last_timestamp - first_timestamp)` when span is positive.
- `median_dt_s` is the median of **strictly positive** consecutive deltas in acquisition order. Duplicate/backward counts are reported separately; no observations are deleted.
- `nominal_rate_hz = 1 / median_dt_s`.
- `gap_threshold_s = max(rate.absolute_gap_threshold_s, rate.gap_multiplier * median_dt_s)`.
- Continuous active windows break at backward time or deltas above the threshold; duplicates contribute no duration or rate transitions.
- `active_window_rate_hz = count(0 < dt <= gap_threshold_s) / sum(dt in that set)`; `active_duration_s` is that denominator. Singleton windows contribute zero duration, with no extrapolation past the last sample.

Event/metadata message rates describe timing, not periodic sampling. Messages containing multiple instances retain total counts and an aggregate span rate; nominal and active metrics are blank in the aggregate row. Use `message_instance_counts.csv`. Its ESC values are decoded RPM/RawRPM/Volt/Curr/Temp min/max/median, not inferred mechanical speed or physical motor numbering.

PSC activity is independently calculated within each mode/armed interval. Candidate intervals additionally require finite acceleration pairs at both endpoints, resolved m/s/s metadata and the supported source profile. Missing/nonfinite samples break paired evidence. `ekf_xy_valid_duration_s` is a bounded primary-EKF status-flag duration; it does not imply any PSC observations. The legacy `valid_xy_position_duration_s` is only an alias.

Under the [audited firmware semantics](source_audit.md), AN/AE are attitude-target-derived controller proxies. `xy_proxy_candidate_duration_s` is the union duration of overlapping paired N/E evidence. Physical `x`, `y` and `xy_residual_candidate_duration_s` remain zero. `z_residual_candidate_duration_s` uses finite TAD/AD evidence. No A-minus-TA values are calculated by Inspector.

## Units and provenance

Inventory rows have `inventory_scope=message` (all observations) or `instance` (only that instance). Each row preserves FMT format character, FMTU unit/multiplier IDs, raw MULT value, decoder scale and metadata source. `decoded_unit` describes observed decoder output; `canonical_unit` describes the verified UNIT representation. `decoded_to_canonical_factor = log_multiplier / decoder_format_scale`, except the documented no-multiplier sentinel, which preserves decoder scale. Unknown metadata yields null canonical statistics, not guessed units.

Examples: TimeUS decoded microseconds multiply by 1e-6 to seconds; BAT.CurrTot decoded mAh multiply by 0.001 to Ah. Already-decoded centidegrees with matching log multiplier remain degrees (factor 1). Other known factors are displayed explicitly as scaled units if no conventional prefix is mapped. There is no implicit degrees/radians conversion.

`observed_*_decoded` and `observed_*_canonical` include min/max/mean/median/std/range/percentiles; finite, invalid and nonnumeric counts are distinct. `unit`, `decoded_to_unit_factor`, `observed_min`, `observed_max`, `finite_sample_count` and `nan_invalid_count` remain compatibility aliases; prefer the new explicit columns. Statistical spaces are labeled for vibration and acceleration-range reports too.

## Physical checks and vibration

All screening thresholds live in `configs/log_pipeline.yaml` under `physical_checks`; invalid policies fail before parsing. A warning needs at least `minimum_samples` and a flagged fraction at least `warning_fraction`. Every evaluated check, including those below threshold, appears in `physical_quality_checks` in JSON with counts and distributions.

Battery voltage uses configured absolute limits. Low-current screening uses known **armed** samples; `min_flying_current_a` is only a screening threshold because airborne state is unverified. ESC voltage uses absolute limits and the latest causal battery sample within `comparison_max_age_s` and the same mode/arm interval. ESC zero-current screening uses armed samples. RPM ratios require all observed ESC instances, positive RPM above `rpm_active_floor`, bounded age and the same armed/mode interval. Stale or uncertain identities are never assigned physical motors. Warning codes identify suspicious evidence without correcting any measurement.

VIBE distributions are split into armed, disarmed and unknown arm state for each IMU. Percentiles use NumPy's linear quantile interpolation, configured under `vibration.summary_percentiles`. `Clip` is a cumulative clipping counter: `clip_total_observed_increments` sums nonnegative consecutive differences within the same mode/arm interval and counts reset jumps separately. It excludes unassignable boundary increments, so is an observed lower bound. The same counter accompanies each axis; do not sum it across axes. No severity is assigned from a single maximum. [ArduPilot vibration and clipping explanation](https://en.ardupilot.org/dev/docs/common-measuring-vibration.html).

Airborne state is UNKNOWN because this Inspector has no independently verified detector. IMU acceleration/gyro distributions and unit provenance remain in the inventory for later analysis; no PSD is computed.

## Reproduction and compatibility

```powershell
python -m src.log_pipeline.inspect_log data/raw/flight.bin --config configs/log_pipeline.yaml --output-dir results/log_inspection
python -m pytest -q
$env:RUN_REAL_LOG_TESTS = '1'
$env:REAL_LOG_PATH = 'data/raw/flight.bin'
# Optional prior summary: compare every message count and the raw SHA256.
$env:INSPECTION_BASELINE_PATH = 'results/previous/log_summary.json'
python -m pytest -q
```

Reports are private, generated and excluded from Git. All report destinations are checked against raw path aliases and hardlinks before any write. The shared reader's warning-level EOF diagnostics intentionally change its report schema to version 2. The pre-existing extractor still fails closed on that new shape, even with its old tail opt-in; adapting that consumer is outside this Inspector-only task.
