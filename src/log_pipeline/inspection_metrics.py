"""Report-only statistics. Never change decoded records or fill missing data."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml


def load_inspection_config(path=None):
    default = Path(__file__).resolve().parents[2] / 'configs/log_pipeline.yaml'
    config = yaml.safe_load(default.read_text(encoding='utf-8'))
    if path is not None:
        override = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        def merge(base, other):
            for key, value in other.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merge(base[key], value)
                else:
                    base[key] = value
        merge(config, override)
    for key in ('gap_multiplier', 'absolute_gap_threshold_s'):
        value = config['rate'][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'rate.{key} must be finite and positive')
    def number(value, name, minimum=0., strict=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or strict and value == minimum:
            raise ValueError(f'{name} must be finite and {">" if strict else ">="} {minimum}')
    p = config['physical_checks']
    for name, value in [('minimum_samples', p['minimum_samples']),
                        ('minimum_paired_samples', config['inspector_availability']['minimum_paired_samples'])]:
        number(value, name, 1)
        if int(value) != value:
            raise ValueError(f'{name} must be an integer')
    number(p['warning_fraction'], 'warning_fraction', 0, True)
    if p['warning_fraction'] > 1:
        raise ValueError('warning_fraction must be <= 1')
    for group in ('battery', 'esc'):
        for name, value in p[group].items():
            number(value, f'{group}.{name}')
        if p[group]['min_plausible_voltage'] >= p[group]['max_plausible_voltage']:
            raise ValueError(f'{group} voltage bounds are reversed')
    if p['esc']['min_voltage_ratio_to_battery'] >= p['esc']['max_voltage_ratio_to_battery']:
        raise ValueError('ESC voltage ratio bounds are reversed')
    number(config['inspector_availability']['minimum_candidate_duration_s'], 'minimum_candidate_duration_s', 0, True)
    percentiles = config['vibration']['summary_percentiles']
    if not isinstance(percentiles, list) or not percentiles:
        raise ValueError('summary_percentiles must be a nonempty list')
    for value in percentiles:
        number(value, 'summary_percentiles')
        if value > 100:
            raise ValueError('summary_percentiles must be <= 100')
    return config


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def distribution(values, percentiles=(95, 99)):
    values = [v for v in values if finite(v)]
    if not values:
        return {key: None for key in ('min', 'max', 'median', 'mean', 'std', 'range', *(f'p{p:g}' for p in percentiles))} | {'finite_count': 0}
    a = np.asarray(values, dtype=float)
    return {'finite_count': len(values), 'min': float(a.min()), 'max': float(a.max()),
            'median': float(np.median(a)), 'mean': float(a.mean()),
            'std': float(a.std()), 'range': float(np.ptp(a)),
            **{f'p{p:g}': float(np.percentile(a, p)) for p in percentiles}}


def inventory_statistics(info, values, instance=None, scope='message'):
    """Units describe the decoder output, not the underlying packed integer."""
    good = [v for v in values if finite(v)]
    factor = info['decoded_to_unit_factor'] if info['unit_verified'] else None
    unit = info['unit'] if factor is not None and info['unit_id'] != '?' else None
    if unit is None:
        factor = None
    decoded = unit
    if factor is not None and factor != 1:
        known_prefixes = {('s', 1e-6): 'us', ('s', .001): 'ms', ('Ah', .001): 'mAh'}
        decoded = next((name for (base, scale), name in known_prefixes.items()
                        if unit == base and math.isclose(factor, scale, rel_tol=1e-9)),
                       f'{factor:g} {unit}')
    canonical = [v * factor for v in good] if factor is not None else []
    stats = distribution(good, (1, 95, 99))
    cstats = distribution(canonical, (1, 95, 99))
    return dict(info, instance=instance, inventory_scope=scope,
                decoded_unit=decoded, canonical_unit=unit,
                decoded_to_canonical_factor=factor,
                sample_count=len(values), finite_count=len(good), finite_sample_count=len(good),
                invalid_count=sum(v is None or isinstance(v, (float, int)) and not finite(v) for v in values),
                nan_invalid_count=sum(v is None or isinstance(v, (float, int)) and not finite(v) for v in values),
                nonnumeric_sample_count=sum(v is not None and not isinstance(v, (float, int)) for v in values),
                observed_min=stats['min'], observed_max=stats['max'],
                **{f'observed_{key}_decoded': value for key, value in stats.items() if key != 'finite_count'},
                **{f'observed_{key}_canonical': value for key, value in cstats.items() if key != 'finite_count'})
