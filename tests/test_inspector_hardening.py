"""Inspector reporting regressions; hand-computed examples, real BIN decoder."""
import copy
import math

import pytest

from test_log_pipeline import synthetic_log, _fmt, _packet
from src.log_pipeline.reader import scan_log, _timing_stats


def analysis_scan(records, *, firmware='ArduCopter V4.8.0-dev (2b5cebb9)'):
    from types import SimpleNamespace
    fields = {'PSCN': ('TAN', 'AN'), 'PSCE': ('TAE', 'AE'), 'PSCD': ('TAD', 'AD'),
              'BAT': ('Volt', 'Curr'), 'ESC': ('Volt', 'Curr', 'RPM'),
              'VIBE': ('VibeX', 'VibeY', 'VibeZ', 'Clip')}
    inventory = []
    for name, names in fields.items():
        for field in names:
            unit = 'V' if field == 'Volt' else 'A' if field == 'Curr' else 'rpm' if field == 'RPM' else 'm/s/s'
            inventory.append(dict(message=name, field=field, inventory_scope='message', canonical_unit=unit,
                                  decoded_to_canonical_factor=1.0, unit_verified=True))
    return SimpleNamespace(records=records, inventory=inventory,
        formats={n: {'instance_field': 'Instance' if n == 'ESC' else 'IMU' if n == 'VIBE' else 'I' if n == 'BAT' else None} for n in fields},
        flight_segments=[{'flight_id': 1, 'arm_time_s': 1., 'end_time_s': 3., 'valid_xy_position_duration_s': 2.}],
        mode_intervals=[{'start_time_s': 0., 'end_time_s': 1., 'armed': False, 'flight_id': None, 'mode': 'STABILIZE'},
                        {'start_time_s': 1., 'end_time_s': 3., 'armed': True, 'flight_id': 1, 'mode': 'LOITER'}],
        summary={'firmware': firmware})


def test_psc_duration_is_separate_from_ekf_and_xy_proxy_is_not_physical():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    scan = analysis_scan({'PSCN': [{'time_s': t, 'TAN': 0., 'AN': .01} for t in [1., 1.1, 1.2]],
                          'PSCE': [{'time_s': t, 'TAE': 0., 'AE': .01} for t in [1.05, 1.15, 1.25]],
                          'PSCD': [{'time_s': t, 'TAD': 0., 'AD': .01} for t in [1., 1.1, 1.2, 2.9]]})
    analyze_inspection(scan, load_inspection_config())
    f = scan.flight_segments[0]
    assert f['ekf_xy_valid_duration_s'] == 2
    assert f['pscn_active_duration_s'] == pytest.approx(.2)
    assert f['xy_proxy_candidate_duration_s'] == pytest.approx(.15)
    assert f['xy_residual_candidate_duration_s'] == 0
    assert f['z_residual_candidate_duration_s'] == pytest.approx(.2)


def test_unknown_firmware_does_not_verify_acceleration_semantics():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    scan = analysis_scan({'PSCD': [{'time_s': t, 'TAD': 0., 'AD': 1.} for t in [1, 1.1]]}, firmware='unknown')
    result = analyze_inspection(scan, load_inspection_config())
    assert scan.flight_segments[0]['z_residual_candidate_duration_s'] is None
    assert result['residual_semantics']['verified_profile'] is False


def test_physical_warnings_use_thresholds_without_changing_samples():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    records = {'BAT': [{'time_s': 1 + i / 100, 'I': 0, 'Volt': 12., 'Curr': .2} for i in range(20)],
               'ESC': [{'time_s': 1 + i / 100, 'Instance': j, 'Volt': 1.2, 'Curr': 0., 'RPM': 3000. * (1 + 3 * j)} for i in range(20) for j in (0, 1)]}
    saved = copy.deepcopy(records)
    result = analyze_inspection(analysis_scan(records), load_inspection_config())
    codes = {w['code'] for w in result['physical_quality_warnings']}
    assert {'BAT_CURRENT_SUSPICIOUS', 'ESC_VOLTAGE_IMPLAUSIBLE', 'ESC_CURRENT_UNAVAILABLE', 'ESC_RPM_CROSS_INSTANCE_INCONSISTENT'} <= codes
    assert records == saved
    config = load_inspection_config()
    config['physical_checks']['minimum_samples'] = 100
    assert not analyze_inspection(analysis_scan(records), config)['physical_quality_warnings']


def test_vibration_separates_arm_state_and_differences_clip_counters():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    scan = analysis_scan({'VIBE': [{'time_s': t, 'IMU': 0, 'VibeX': x, 'VibeY': x, 'VibeZ': x, 'Clip': clip}
                                   for t, x, clip in [(.1, 100, 5), (.2, 100, 7), (1, 1, 10), (1.1, 3, 12), (1.2, 5, 15)]]})
    out = analyze_inspection(scan, load_inspection_config())
    armed = next(r for r in out['vibration_summary'] if r['state'] == 'armed' and r['field'] == 'VibeX')
    assert armed['mean'] == 3
    assert armed['p95'] == pytest.approx(4.8)
    assert armed['clip_total_observed_increments'] == 5
    assert armed['max'] == 5
    assert out['airborne_status'] == 'UNKNOWN'


def test_only_known_eof_truncation_is_warning(tmp_path):
    scan = scan_log(synthetic_log(tmp_path / 'tail.bin', tail=b'\xa3\x95\x84\x00'))
    assert scan.summary['parser_status'] == 'PASS_WITH_WARNINGS'
    assert not scan.summary['critical_errors']
    assert scan.summary['parser_diagnostics']['events'][0]['severity'] == 'WARNING'
    assert not scan.summary['scan_complete']
    bad = scan_log(synthetic_log(tmp_path / 'bad.bin', interstitial=b'bad!'))
    assert bad.summary['parser_status'] == 'FAIL'
    assert bad.summary['critical_errors']


def test_rates_distinguish_span_nominal_and_active_windows():
    r = _timing_stats('ATT', [{'time_s': t} for t in [0, .1, .2, 10, 10.1]])
    assert r['aggregate_span_rate_hz'] == pytest.approx(4 / 10.1)
    assert r['nominal_rate_hz'] == pytest.approx(10)
    assert r['active_window_rate_hz'] == pytest.approx(10)
    assert r['active_duration_s'] == pytest.approx(.3)
    assert r['active_window_count'] == 2


def test_rates_do_not_mix_instances(tmp_path):
    scan = scan_log(synthetic_log(tmp_path / 'fixture.bin'))
    instances = {r['instance']: r for r in scan.summary['instance_message_stats'] if r['message'] == 'ATT'}
    assert instances[1]['nominal_rate_hz'] == pytest.approx(10)
    assert instances[0]['aggregate_span_rate_hz'] == pytest.approx(2 / 1.1)
    aggregate = next(r for r in scan.message_stats if r['message'] == 'ATT')
    assert aggregate['nominal_rate_hz'] is None
    assert aggregate['rate_scope'] == 'mixed_instances_count_only'


def test_inventory_distinguishes_decoded_and_canonical_units(tmp_path):
    extra = b''.join([
        _fmt(135, 'BAT', 'Qf', 'TimeUS,CurrTot', 15),
        _packet(129, 'QB16s', 0, ord('a'), b'Ah'),
        _packet(130, 'QBd', 0, ord('C'), .001),
        _packet(131, 'QB16s16s', 0, 135, b'sa', b'FC'),
        _packet(135, 'Qf', 15_310_661, 20),
    ])
    scan = scan_log(synthetic_log(tmp_path / 'units.bin', tail=extra))
    time = scan.field('BAT', 'TimeUS')
    assert time['decoded_unit'] == 'us'
    assert time['canonical_unit'] == 's'
    assert time['observed_min_decoded'] == 15310661
    assert time['observed_min_canonical'] == pytest.approx(15.310661)
    charge = scan.field('BAT', 'CurrTot')
    assert charge['decoded_unit'] == 'mAh'
    assert charge['canonical_unit'] == 'Ah'
    assert charge['observed_mean_canonical'] == pytest.approx(.02)
    assert scan.records['BAT'][0]['CurrTot'] == 20
    per_instance = [r for r in scan.inventory if r['message'] == 'ATT' and r['field'] == 'Yaw' and r.get('instance') == 1]
    assert per_instance[0]['observed_min_decoded'] == 223.45


def test_unknown_units_never_get_canonical_statistics(tmp_path):
    scan = scan_log(synthetic_log(tmp_path / 'unknown.bin', time_metadata=False))
    field = scan.field('ATT', 'TimeUS')
    assert field['canonical_unit'] is None
    assert field['observed_min_canonical'] is None
    assert field['observed_min_decoded'] == 1000000


def test_inspector_emits_status_sections_and_new_audits(tmp_path):
    from src.log_pipeline.inspect_log import inspect_log
    summary = inspect_log(synthetic_log(tmp_path / 'fixture.bin', tail=b'\xa3\x95\x84\x00'), tmp_path / 'reports')
    assert summary['statuses']['parser'] == 'PASS_WITH_WARNINGS'
    assert summary['statuses']['structural_log'] == 'PASS'
    assert summary['statuses']['xy_residual_data_availability'] == 'UNRESOLVED'
    for name in ('vibration_summary.csv', 'physical_quality_warnings.csv', 'psc_activity_by_mode.csv', 'preliminary_audit.json'):
        assert (tmp_path / 'reports' / name).exists()
    assert 'PARSER STATUS' in (tmp_path / 'reports/log_summary.txt').read_text()


@pytest.mark.parametrize('name', ['preliminary_audit.json', 'vibration_summary.csv', 'physical_quality_warnings.csv', 'psc_activity_by_mode.csv', 'acceleration_dynamic_range.csv'])
def test_new_report_destinations_cannot_overwrite_raw_hardlinks(tmp_path, name):
    import os
    from src.log_pipeline.inspect_log import inspect_log
    raw = synthetic_log(tmp_path / 'fixture.bin')
    before = raw.read_bytes()
    out = tmp_path / 'reports'
    out.mkdir()
    os.link(raw, out / name)
    with pytest.raises(ValueError, match='overwrite'):
        inspect_log(raw, out)
    assert raw.read_bytes() == before


def test_inspector_cli_config_controls_active_gap(tmp_path):
    from src.log_pipeline.inspect_log import inspect_log
    cfg = tmp_path / 'policy.yaml'
    cfg.write_text('rate:\n  gap_multiplier: 1\n  absolute_gap_threshold_s: 0.01\n')
    summary = inspect_log(synthetic_log(tmp_path / 'fixture.bin'), tmp_path / 'reports', config_path=cfg)
    instance = next(r for r in summary['instance_message_stats'] if r['message'] == 'ATT' and r['instance'] == 0)
    assert instance['active_duration_s'] == pytest.approx(.1)


@pytest.mark.parametrize('text', [
    'physical_checks:\n  minimum_samples: 0',
    'physical_checks:\n  warning_fraction: 2',
    'physical_checks:\n  esc:\n    comparison_max_age_s: -.1',
    'physical_checks:\n  battery:\n    min_plausible_voltage: 99',
    'vibration:\n  summary_percentiles: [101]',
    'inspector_availability:\n  minimum_paired_samples: 1.5',
])
def test_invalid_inspector_thresholds_fail_early(tmp_path, text):
    from src.log_pipeline.inspection_metrics import load_inspection_config
    path = tmp_path / 'bad.yaml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_inspection_config(path)


def test_sensor_without_finite_data_is_unknown_not_no_warning(tmp_path):
    from src.log_pipeline.inspect_log import inspect_log
    extra = b''.join([
        _fmt(135, 'BAT', 'Qff', 'TimeUS,Volt,Curr', 19),
        _packet(129, 'QB16s', 0, ord('v'), b'V'),
        _packet(129, 'QB16s', 0, ord('A'), b'A'),
        _packet(131, 'QB16s16s', 0, 135, b'svA', b'F--'),
        _packet(135, 'Qff', 1_000_000, float('nan'), float('nan')),
    ])
    summary = inspect_log(synthetic_log(tmp_path / 'nan.bin', tail=extra), tmp_path / 'reports')
    assert summary['statuses']['battery_physical_quality'] == 'UNKNOWN'


def test_mode_boundaries_and_nonfinite_acceleration_break_candidates():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    scan = analysis_scan({'PSCD': [{'time_s': t, 'TAD': 0., 'AD': a} for t, a in [(1, 1), (1.1, float('nan')), (1.2, 1), (1.3, 1)]]})
    scan.mode_intervals[1]['end_time_s'] = 1.25
    scan.mode_intervals.append({'start_time_s': 1.25, 'end_time_s': 3., 'armed': True, 'flight_id': 1, 'mode': 'ALT_HOLD'})
    analyze_inspection(scan, load_inspection_config())
    assert scan.flight_segments[0]['z_residual_candidate_duration_s'] == 0


def test_missing_psc_units_never_produces_proxy_or_z_candidates():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    scan = analysis_scan({'PSCD': [{'time_s': t, 'TAD': 0., 'AD': 1.} for t in [1, 1.1]]})
    scan.inventory = []
    analyze_inspection(scan, load_inspection_config())
    assert scan.flight_segments[0]['z_residual_candidate_duration_s'] is None


def test_inventory_legacy_invalid_alias_is_scoped_to_instance():
    from src.log_pipeline.inspection_metrics import inventory_statistics
    info = {'decoded_to_unit_factor': 1, 'unit_verified': True, 'unit': 'V', 'unit_id': 'v', 'nan_invalid_count': 4}
    row = inventory_statistics(info, [1., 2.], instance=1, scope='instance')
    assert row['invalid_count'] == row['nan_invalid_count'] == 0


def test_clip_resets_do_not_create_negative_totals_and_unknown_arm_is_preserved():
    from src.log_pipeline.inspector_analysis import analyze_inspection
    from src.log_pipeline.inspection_metrics import load_inspection_config
    scan = analysis_scan({'VIBE': [{'time_s': t, 'IMU': 0, 'VibeX': 1, 'VibeY': 1, 'VibeZ': 1, 'Clip': c}
                                   for t, c in [(.1, 100), (.2, 110), (1, 15), (1.1, 0), (1.2, 3)]]})
    scan.mode_intervals[0]['armed'] = None
    out = analyze_inspection(scan, load_inspection_config())
    rows = {r['state']: r for r in out['vibration_summary'] if r['field'] == 'VibeX'}
    assert rows['armed']['clip_total_observed_increments'] == 3
    assert rows['armed']['clip_counter_resets'] == 1
    assert rows['disarmed']['count'] == 0
    assert rows['unknown']['count'] == 2
