"""Inspector evidence layers: activity, sensor quality and vibration, not ML readiness."""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
import statistics

from .inspection_metrics import distribution, finite


def _context(scan, timestamp):
    if not finite(timestamp):
        return None
    return next((i for i, row in enumerate(scan.mode_intervals)
                 if row['start_time_s'] <= timestamp < row['end_time_s']), None)


def _state(scan, timestamp):
    index = _context(scan, timestamp)
    armed = scan.mode_intervals[index]['armed'] if index is not None else None
    return 'unknown' if armed is None else 'armed' if armed else 'disarmed'


def _groups(scan, message):
    field = scan.formats.get(message, {}).get('instance_field')
    groups = defaultdict(list)
    for row in scan.records.get(message, []):
        groups[row.get(field) if field else None].append(row)
    return groups


def _canonical(scan, message, field, expected=None):
    info = next((r for r in scan.inventory if r['message'] == message and r['field'] == field
                 and r.get('inventory_scope', 'message') == 'message'), {})
    unit = info.get('canonical_unit')
    return info.get('decoded_to_canonical_factor') if unit is not None and (expected is None or unit == expected) else None


def _edges(rows, rate, predicate=lambda r: True):
    """Adjacent observations delimit evidence. No extrapolation past last sample."""
    times = [r.get('time_s') for r in rows if finite(r.get('time_s'))]
    dt = [b-a for a, b in zip(times, times[1:]) if b > a]
    threshold = max(rate['absolute_gap_threshold_s'], rate['gap_multiplier'] * statistics.median(dt)) if dt else rate['absolute_gap_threshold_s']
    return [(a['time_s'], b['time_s']) for a, b in zip(rows, rows[1:])
            if finite(a.get('time_s')) and finite(b.get('time_s'))
            and 0 < b['time_s'] - a['time_s'] <= threshold and predicate(a) and predicate(b)]


def _duration(edges):
    # Union intervals to avoid double counting after duplicate/backward timestamps.
    total, right = 0., None
    for a, b in sorted(edges):
        total += max(0., b - max(a, right if right is not None else a))
        right = max(b, right if right is not None else b)
    return total


def _overlap(left, right):
    return [(max(a, c), min(b, d)) for a, b in left for c, d in right if max(a, c) < min(b, d)]


def _psc_activity(scan, config):
    verified = '(2b5cebb9)' in (scan.summary.get('firmware') or '')
    rows_out, dynamics = [], []
    per_flight = defaultdict(lambda: defaultdict(list))
    paired_counts = defaultdict(lambda: defaultdict(int))
    fields = {'PSCN': ('TAN', 'AN'), 'PSCE': ('TAE', 'AE'), 'PSCD': ('TAD', 'AD')}
    for interval in scan.mode_intervals:
        for message, pair in fields.items():
            selected = [r for r in scan.records.get(message, []) if finite(r.get('time_s'))
                        and interval['start_time_s'] <= r['time_s'] < interval['end_time_s']]
            units_ok = all(_canonical(scan, message, f, 'm/s/s') is not None for f in pair)
            usable = lambda r: all(finite(r.get(f)) for f in pair)
            active = _edges(selected, config['rate'])
            paired = _edges(selected, config['rate'], usable) if verified and units_ok else []
            fid = interval['flight_id']
            per_flight[fid][message].extend(active)
            per_flight[fid][message + '_paired'].extend(paired)
            paired_counts[fid][message] += sum(usable(r) for r in selected) if verified and units_ok else 0
            rows_out.append(dict(interval, message=message, count=len(selected), active_duration_s=_duration(active),
                                 paired_acceleration_duration_s=_duration(paired),
                                 acceleration_semantics='controller_proxy' if verified and message != 'PSCD' else 'gravity_compensated_estimate' if verified else 'unverified'))
            for field in pair:
                dynamics.append(dict(flight_id=fid, mode=interval['mode'], start_time_s=interval['start_time_s'],
                                     message=message, field=field, statistics_space='decoded',
                                     **distribution([r.get(field) for r in selected], (1, 99))))
    for flight in scan.flight_segments:
        fid = flight['flight_id']
        evidence = per_flight[fid]
        flight['ekf_xy_valid_duration_s'] = flight['valid_xy_position_duration_s']
        for name in fields:
            flight[name.lower() + '_active_duration_s'] = _duration(evidence[name])
            flight[name.lower() + '_paired_sample_count'] = paired_counts[fid][name]
        proxy = _duration(_overlap(evidence['PSCN_paired'], evidence['PSCE_paired']))
        z_verified = verified and all(_canonical(scan, 'PSCD', f, 'm/s/s') is not None for f in fields['PSCD'])
        flight.update(xy_proxy_candidate_duration_s=proxy if verified else None,
                      xy_residual_candidate_duration_s=0. if verified else None,
                      x_residual_candidate_duration_s=0. if verified else None,
                      y_residual_candidate_duration_s=0. if verified else None,
                      z_residual_candidate_duration_s=_duration(evidence['PSCD_paired']) if z_verified else None)
    return rows_out, dynamics, {'verified_profile': verified,
        'profile': 'ArduCopter public revision 2b5cebb933d91e92f7bab768268e41c2960b6ae4' if verified else None,
        'xy': 'controller proxy; no independent measured XY acceleration' if verified else 'UNKNOWN',
        'z': 'gravity-compensated down acceleration estimate' if verified else 'UNKNOWN',
        'source': 'docs/source_audit.md',
        'duration_policy': 'sum adjacent positive timestamp differences <= max(absolute_gap_threshold_s, gap_multiplier * median_positive_dt), within each mode/arm interval; both endpoints finite for paired fields; no final-sample extension; XY proxy uses N/E interval intersection',
        'interpretation': 'candidate signal presence only, not ML readiness; physical XY candidates excluded for proxy semantics'}


def _physical_quality(scan, config):
    policy = config['physical_checks']
    warnings = []
    checks = []
    def emit(code, message, instance, bad, total, evidence):
        checks.append(dict(code=code, message=message, instance=instance, flagged_count=bad, evaluated_count=total,
                           fraction=bad / total if total else None, evidence=evidence,
                           sufficient_samples=total >= policy['minimum_samples']))
        if total >= policy['minimum_samples'] and bad / total >= policy['warning_fraction']:
            warnings.append(dict(severity='WARNING', code=code, message=message, instance=instance,
                                 flagged_count=bad, evaluated_count=total, fraction=bad / total, evidence=evidence,
                                 interpretation='screening only; calibration/telemetry/mapping unresolved; raw values unchanged'))
    bat, esc = policy['battery'], policy['esc']
    for instance, rows in _groups(scan, 'BAT').items():
        for field, unit in [('Volt', 'V'), ('Curr', 'A')]:
            factor = _canonical(scan, 'BAT', field, unit)
            if factor is None:
                continue
            values = [r[field]*factor for r in rows if finite(r.get(field)) and (field != 'Curr' or _state(scan, r.get('time_s')) == 'armed')]
            bad = sum(v < bat['min_plausible_voltage'] or v > bat['max_plausible_voltage'] for v in values) if field == 'Volt' else sum(v < bat['min_flying_current_a'] for v in values)
            emit('BAT_VOLTAGE_IMPLAUSIBLE' if field == 'Volt' else 'BAT_CURRENT_SUSPICIOUS', 'BAT', instance, bad, len(values),
                 dict(field=field, state='whole_log' if field == 'Volt' else 'armed (airborne unverified)', **distribution(values)))
    battery_rows = _groups(scan, 'BAT').get(bat['instance'], [])
    battery_factor = _canonical(scan, 'BAT', 'Volt', 'V')
    battery_rows = sorted([r for r in battery_rows if finite(r.get('time_s')) and finite(r.get('Volt'))], key=lambda r: r['time_s'])
    bt = [r['time_s'] for r in battery_rows]
    for instance, rows in _groups(scan, 'ESC').items():
        vf, cf = _canonical(scan, 'ESC', 'Volt', 'V'), _canonical(scan, 'ESC', 'Curr', 'A')
        if vf is not None:
            values = [r['Volt']*vf for r in rows if finite(r.get('Volt'))]
            bad = sum(v < esc['min_plausible_voltage'] or v > esc['max_plausible_voltage'] for v in values)
            ratios = []
            if battery_factor is not None:
                for r in rows:
                    if not finite(r.get('time_s')) or not finite(r.get('Volt')):
                        continue
                    i = bisect_right(bt, r['time_s']) - 1
                    if i >= 0 and 0 <= r['time_s'] - bt[i] <= esc['comparison_max_age_s'] and _context(scan, bt[i]) == _context(scan, r['time_s']) and battery_rows[i]['Volt']*battery_factor > 0:
                        ratios.append(r['Volt']*vf / (battery_rows[i]['Volt']*battery_factor))
            ratio_bad = sum(v < esc['min_voltage_ratio_to_battery'] or v > esc['max_voltage_ratio_to_battery'] for v in ratios)
            emit('ESC_VOLTAGE_IMPLAUSIBLE', 'ESC', instance, bad, len(values), dict(check='absolute_voltage', **distribution(values)))
            emit('ESC_VOLTAGE_IMPLAUSIBLE', 'ESC', instance, ratio_bad, len(ratios), dict(check='contemporaneous_battery_ratio', **distribution(ratios)))
        if cf is not None:
            values = [r['Curr']*cf for r in rows if finite(r.get('Curr')) and _state(scan, r.get('time_s')) == 'armed']
            emit('ESC_CURRENT_UNAVAILABLE', 'ESC', instance, sum(abs(v) <= esc['zero_current_epsilon_a'] for v in values), len(values),
                 dict(check='near_zero_armed_current; sensor availability unproven', **distribution(values)))
    # Compare only simultaneous fresh reports in the same known armed/mode interval.
    groups = _groups(scan, 'ESC')
    if len(groups) > 1 and _canonical(scan, 'ESC', 'RPM', 'rpm') is not None:
        latest, ratios = {}, []
        rf = _canonical(scan, 'ESC', 'RPM', 'rpm')
        field = scan.formats['ESC']['instance_field']
        records = sorted([r for r in scan.records['ESC'] if finite(r.get('time_s'))], key=lambda r: r['time_s'])
        for row in records:
            latest[row.get(field)] = row
            if len(latest) != len(groups) or _state(scan, row['time_s']) != 'armed':
                continue
            samples = list(latest.values())
            if all(finite(r.get('RPM')) and r['RPM']*rf > esc['rpm_active_floor']
                   and row['time_s'] - r['time_s'] <= esc['comparison_max_age_s']
                   and _context(scan, row['time_s']) == _context(scan, r['time_s']) for r in samples):
                speeds = [r['RPM']*rf for r in samples]
                ratios.append(max(speeds) / min(speeds))
        emit('ESC_RPM_CROSS_INSTANCE_INCONSISTENT', 'ESC', None,
             sum(v > esc['rpm_cross_instance_ratio_warning'] for v in ratios), len(ratios), distribution(ratios))
    return warnings, checks


def _vibration(scan, config):
    result = []
    for instance, rows in _groups(scan, 'VIBE').items():
        for state in ('disarmed', 'armed', 'unknown'):
            selected = [r for r in rows if _state(scan, r.get('time_s')) == state]
            clip_total, resets, increments = 0, 0, 0
            for a, b in zip(rows, rows[1:]):
                if (_state(scan, a.get('time_s')) != state or _state(scan, b.get('time_s')) != state
                        or _context(scan, a.get('time_s')) != _context(scan, b.get('time_s'))
                        or not finite(a.get('Clip')) or not finite(b.get('Clip'))
                        or not finite(a.get('time_s')) or not finite(b.get('time_s'))
                        or b['time_s'] <= a['time_s']):
                    continue
                delta = b['Clip'] - a['Clip']
                resets += delta < 0
                if delta >= 0:
                    clip_total += delta
                    increments += 1
            for field in ('VibeX', 'VibeY', 'VibeZ'):
                info = next((r for r in scan.inventory if r['message'] == 'VIBE' and r['field'] == field), {})
                result.append(dict(instance=instance, state=state, field=field, count=len(selected),
                    statistics_space='decoded', decoded_unit=info.get('decoded_unit'),
                    canonical_unit=info.get('canonical_unit'), decoded_to_canonical_factor=info.get('decoded_to_canonical_factor'),
                    **distribution([r.get(field) for r in selected], config['vibration']['summary_percentiles']),
                    clip_total_observed_increments=clip_total if increments else None,
                    clip_counter_resets=resets, clip_interval_count=increments,
                    clip_policy='nonnegative consecutive Clip differences within same mode/arm interval; excludes boundary increments and reset jumps; observed lower bound, not sum of counters'))
    return result


def analyze_inspection(scan, config):
    activity, dynamics, semantics = _psc_activity(scan, config)
    warnings, checks = _physical_quality(scan, config)
    return {'psc_activity_by_mode': activity, 'acceleration_dynamic_range': dynamics,
            'residual_semantics': semantics, 'physical_quality_warnings': warnings, 'physical_quality_checks': checks,
            'vibration_summary': _vibration(scan, config), 'airborne_status': 'UNKNOWN',
            'airborne_reason': 'No independently verified airborne-state detector configured; armed does not establish takeoff.'}
