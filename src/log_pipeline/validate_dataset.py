"""Validate scientific usability independently of successful CSV generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .frames import enu_to_ned, ned_to_enu, residual
from .utils import sha256_file, write_json

STATE_COLUMNS = ['px','py','pz','vx','vy','vz','ax','ay','az','yaw']
NUMERIC_COLUMNS = ['time_s', *STATE_COLUMNS, 'ax_target','ay_target','az_target',
                   'dx','dy','dz','battery_v','battery_i','throttle','motor1','motor2','motor3','motor4']
REQUIRED = [*NUMERIC_COLUMNS, 'flight_id','mode','valid_dx','valid_dy','valid_dz']
AXIS_SOURCE = {'x':'psce', 'y':'pscn', 'z':'pscd'}


def _flags(series, errors, name):
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    converted = series.astype(str).str.lower().map(mapping)
    if converted.isna().any():
        errors.append(f'{name}: non-boolean or missing validity flags')
    return converted.fillna(False).astype(bool)


def _stats(series):
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return {'count_valid': len(values), 'mean': float(values.mean()) if len(values) else None,
            'std': float(values.std()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'p01': float(np.quantile(values,.01)) if len(values) else None,
            'p99': float(np.quantile(values,.99)) if len(values) else None,
            'min': float(values.min()) if len(values) else None,
            'max': float(values.max()) if len(values) else None,
            'range': float(np.ptp(values)) if len(values) else None}


def _limit(config, key, source, default):
    value = config.get(key, default)
    return float(value.get(source.upper(), value.get(source.lower(), value.get('default',default)))) if isinstance(value,dict) else float(value)


def _physical_flags(df, config):
    flags = df[['time_s','flight_id']].copy()
    batt = config.get('battery_plausibility_bounds', {})
    vlo, vhi = batt.get('voltage', batt.get('voltage_v', [9,12.8]))
    ilo, ihi = batt.get('current', batt.get('current_a', [.5,150]))
    flags['outlier_battery_voltage'] = df.battery_v.notna() & ~df.battery_v.between(vlo,vhi)
    flags['outlier_battery_current'] = df.battery_i.notna() & ~df.battery_i.between(ilo,ihi)
    alo, ahi = config.get('acceleration_plausibility_bounds', [-40,40])
    speed = max(abs(x) for x in config.get('velocity_plausibility_bounds', [0,30]))
    flags['outlier_acceleration'] = (df[['ax','ay','az']] < alo).any(axis=1) | (df[['ax','ay','az']] > ahi).any(axis=1)
    flags['outlier_velocity'] = (df[['vx','vy','vz']].abs() > speed).any(axis=1)
    flags['outlier_throttle'] = df.throttle.notna() & ~df.throttle.between(0,1)
    # PWM-equivalent command limits, not a statement about DShot wire encoding.
    motor_bounds = config.get('motor_output_plausibility_bounds', config.get('motor_output_bounds', [0,2200]))
    flags['outlier_motor_output'] = ((df[['motor1','motor2','motor3','motor4']] < motor_bounds[0]) | (df[['motor1','motor2','motor3','motor4']] > motor_bounds[1])).any(axis=1)
    flags['outlier_position_jump'] = False
    flags['outlier_velocity_jump'] = False
    flags['outlier_yaw_jump'] = False
    group_columns = ['flight_id'] + ([c for c in ['continuity_id','boundary_id'] if c in df][:1])
    for _, part in df.groupby(group_columns, sort=False, dropna=False):
        dt = part.time_s.diff()
        valid_dt = dt.where(dt > 0)
        pos_speed = part[['px','py','pz']].diff().abs().div(valid_dt,axis=0)
        vel_acc = part[['vx','vy','vz']].diff().abs().div(valid_dt,axis=0)
        flags.loc[part.index,'outlier_position_jump'] = (pos_speed > speed).any(axis=1)
        flags.loc[part.index,'outlier_velocity_jump'] = (vel_acc > max(abs(alo),abs(ahi))).any(axis=1)
        angular_difference = (part.yaw.diff() + np.pi) % (2*np.pi) - np.pi
        flags.loc[part.index,'outlier_yaw_jump'] = (angular_difference.abs()/valid_dt > config.get('maximum_yaw_rate_rad_s', 12)).fillna(False)
    flags['outlier'] = flags.filter(like='outlier_').any(axis=1)
    return flags


def _frame_checks(manifest, errors):
    vector = np.array([[1.,2.,3.],[-4.,0.,9.]])
    checks = {'round_trip': np.allclose(enu_to_ned(ned_to_enu(vector)), vector),
              'positive_down_is_negative_up': ned_to_enu([0,0,2])[2] == -2,
              'norm_invariant': np.allclose(np.linalg.norm(vector,axis=1),np.linalg.norm(ned_to_enu(vector),axis=1)),
              'residual_sign': np.array_equal(residual([2.,-1.,3.],[1.,2.,4.]),[1.,-3.,-1.])}
    if not all(checks.values()): errors.append('Frame/sign numerical self-check failed')
    if manifest.get('coordinate_frame') != 'ENU': errors.append('Dataset coordinate frame must be ENU')
    if manifest.get('residual_sign_convention') != 'd = a_actual - a_target': errors.append('Residual sign convention is unresolved or inconsistent')
    for col in ['ax','ay','az','ax_target','ay_target','az_target','dx','dy','dz']:
        spec = manifest.get('source_mapping', {}).get(col, {})
        if spec.get('output_frame') != 'ENU': errors.append(f'{col}: missing or mixed output frame')
        if spec.get('output_unit') not in ('m/s^2','m/s/s','m/s²'): errors.append(f'{col}: acceleration output unit unresolved')
    return {k: bool(v) for k,v in checks.items()}


def _split_checks(data_dir, df, errors, config):
    path = data_dir/'gp_residual_dataset.csv'
    if not path.is_file():
        errors.append('Missing chronological GP split file')
        return pd.DataFrame()
    gp = pd.read_csv(path)
    if not {'time_s','flight_id','split'} <= set(gp):
        errors.append('GP split columns missing'); return pd.DataFrame()
    for axis in 'xyz':
        if f'valid_d{axis}' not in gp:
            errors.append(f'GP split missing valid_d{axis}');gp[f'valid_d{axis}']=False
        else: gp[f'valid_d{axis}']=_flags(gp[f'valid_d{axis}'],errors,f'valid_d{axis}')
    _validate_eligibility(gp,config,errors)
    if not gp['split'].isin(['train','test']).all(): errors.append('Unknown GP split labels')
    if gp.duplicated(['flight_id','time_s']).any(): errors.append('GP split duplicates a timestamp')
    if (gp.groupby('time_s')['split'].nunique() > 1).any(): errors.append('One timestamp occurs in multiple splits')
    train, test = gp[gp.split=='train'], gp[gp.split=='test']
    if len(train) and len(test) and train.time_s.max() >= test.time_s.min():
        errors.append('Temporal split leakage: training is not strictly before testing')
    keys = pd.MultiIndex.from_frame(df[['flight_id','time_s']])
    if not pd.MultiIndex.from_frame(gp[['flight_id','time_s']]).isin(keys).all(): errors.append('GP split contains rows outside extracted flights')
    # Match residual values, not only timestamps, to detect independently edited GP data.
    common = [c for c in NUMERIC_COLUMNS if c in gp and c != 'time_s']
    matched = gp.merge(df[['flight_id','time_s',*common]],on=['flight_id','time_s'],how='left',suffixes=('_gp','_all'))
    for col in common:
        if not np.allclose(pd.to_numeric(matched[col+'_gp'],errors='coerce'), matched[col+'_all'],equal_nan=True):
            errors.append(f'GP split {col} differs from all_flights')
    flag_columns = [c for c in df if c.startswith(('valid_d','eligible_d'))]
    flag_matched = gp.merge(df[['flight_id','time_s',*flag_columns]],on=['flight_id','time_s'],how='left',suffixes=('_gp','_all'))
    for col in flag_columns:
        if col not in gp:
            errors.append(f'GP split missing canonical flag {col}')
        elif not np.array_equal(_flags(flag_matched[col+'_gp'],errors,col),_flags(flag_matched[col+'_all'],errors,col)):
            errors.append(f'GP split {col} differs from canonical dataset')
    return gp


def _validate_source_provenance(df, manifest, config, errors):
    """Every exported signal with source-age evidence must honor that evidence."""
    used_by_source = {
        'att': df.yaw.notna(), 'bat':df[['battery_v','battery_i']].notna().any(axis=1),
        'rcou':df[['motor1','motor2','motor3','motor4']].notna().any(axis=1),
        'ctun':df.throttle.notna(),
    }
    raw_rcou=[c for c in df if c.startswith('rcou_c')]
    if raw_rcou: used_by_source['rcou'] |= df[raw_rcou].notna().any(axis=1)
    if 'gps_status' in df: used_by_source['gps']=df.gps_status.notna()
    ekf_columns=[c for c in ['ekf_primary_core','ekf_solution_status','ekf_fault_status'] if c in df]
    if ekf_columns: used_by_source['xkf4']=df[ekf_columns].notna().any(axis=1)
    state_lineage = [c for c in df if c.startswith(('source_p','source_v')) and not c.startswith('source_time_')]
    used_by_source['xkf1'] = (pd.concat([df[c].astype(str).str.upper().str.startswith('XKF1') for c in state_lineage],axis=1).any(axis=1)
                              if state_lineage else df[['px','py','pz','vx','vy','vz']].notna().any(axis=1))
    for axis, source in AXIS_SOURCE.items():
        used_by_source[source] = df[[f'a{axis}',f'a{axis}_target',f'd{axis}']].notna().any(axis=1)
        if state_lineage:
            used_by_source[source] |= pd.concat([df[c].astype(str).str.upper().str.startswith(source.upper()) for c in state_lineage],axis=1).any(axis=1)
    for source,used in used_by_source.items():
        if used.any() and (f'age_{source}_ms' not in df or f'source_time_{source}_s' not in df):
            errors.append(f'{source}: used signal lacks source provenance')
    for agecol in [c for c in df if c.startswith('age_') and c.endswith('_ms')]:
        source = agecol[4:-3]; timecol = f'source_time_{source}_s'
        if timecol not in df:
            errors.append(f'{source}: missing source timestamp'); continue
        used = used_by_source.get(source,pd.Series(False,index=df.index))
        valid_source_col = f'valid_{source}'
        if valid_source_col in df: used |= _flags(df[valid_source_col],errors,valid_source_col)
        age = pd.to_numeric(df[agecol],errors='coerce'); st = pd.to_numeric(df[timecol],errors='coerce')
        limit=min(_limit(config,'maximum_source_age_ms',source,180),_limit(config,'gap_threshold_ms',source,500))
        if (used & (~np.isfinite(age) | (age < -1e-6) | (age > limit+1e-6))).any(): errors.append(f'{source}: used signal violates source-age limit')
        if (used & (~np.isfinite(st) | (st > df.time_s+1e-9) | (((df.time_s-st)*1000-age).abs()>1e-5))).any(): errors.append(f'{source}: used signal timestamp/age inconsistent')
        for seg in manifest.get('flight_segments',[]):
            within = used & df.flight_id.eq(seg['flight_id'])
            end = seg.get('end_time_s',seg.get('disarm_time_s'))
            if end is not None and (within & ((st < seg['arm_time_s']-1e-9) | (st >= end))).any(): errors.append(f'{source}: source outside current armed flight')
        for prefix in ['continuity','boundary']:
            startcol,endcol = f'{prefix}_start_s',f'{prefix}_end_s'
            if startcol in df and endcol in df and (used & ((st < df[startcol]-1e-9) | (st >= df[endcol]))).any(): errors.append(f'{source}: source crosses {prefix} boundary')


def _validate_eligibility(df, config, errors):
    for axis in 'xyz':
        name=f'eligible_d{axis}'
        eligible=_flags(df[name],errors,name) if name in df else None
        expected=df[f'valid_d{axis}'].copy()
        defaults=['pz','vz','yaw'] if axis=='z' else ['px','py','vx','vy','yaw']
        features=config.get('learning_features_by_axis',{}).get(axis,defaults)
        if features:
            if any(c not in df for c in features):
                errors.append(f'{axis}: missing configured learning features'); expected &= False
            else: expected &= np.isfinite(df[features]).all(axis=1)
        if 'outlier' in df: expected &= ~_flags(df.outlier,errors,'outlier')
        if eligible is not None and (eligible & ~expected).any(): errors.append(f'{name}: true with invalid residual/features or outlier')
        df[name]=expected if eligible is None else eligible & expected


def _protect_outputs(dataset_path, manifest_path, manifest, output, plots, make_plots):
    protected=[dataset_path,manifest_path,dataset_path.with_name('gp_residual_dataset.csv')]
    raw=manifest.get('raw_path')
    if raw: protected.append(Path(raw))
    protected.extend(dataset_path.parent/name for name in manifest.get('file_hashes',{}))
    targets=[output/name for name in ['validation_report.json','validation_report.txt','column_stats.csv','quality_by_flight.csv','quality_by_mode.csv','outlier_flags.csv']]
    if make_plots:
        names=['position','velocity','acceleration','battery','flight_modes','source_age']
        names += [name for a in 'xyz' for name in [f'acceleration_{a}_actual_target',f'residual_{a}_time',f'residual_{a}_histogram']]
        targets += [plots/(name+'.png') for name in names]
    for target in targets:
        for original in protected:
            if target.resolve()==original.resolve() or (target.exists() and original.exists() and target.samefile(original)):
                raise ValueError(f'Output would overwrite protected input: {target}')


def validate_dataset(dataset_path, manifest_path=None, output_dir='results/validation', plots_dir='results/plots', *, make_plots=True):
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path) if manifest_path else dataset_path.with_name('dataset_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = manifest.get('extractor_config', {})
    df = pd.read_csv(dataset_path)
    output = Path(output_dir)
    _protect_outputs(dataset_path,manifest_path,manifest,output,Path(plots_dir),make_plots)
    output.mkdir(parents=True, exist_ok=True)
    errors, warnings = [], list(manifest.get('warnings', []))
    missing = sorted(set(REQUIRED)-set(df))
    if missing: errors.append('Missing required columns: '+', '.join(missing))
    for column in missing:
        df[column] = False if column.startswith('valid_') else ('' if column=='mode' else np.nan)
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors='coerce')
        if (df[column].notna() & converted.isna()).any(): errors.append(f'{column}: nonnumeric values')
        if np.isinf(converted.to_numpy(dtype=float)).any(): errors.append(f'{column}: infinite values')
        df[column] = converted
    if not len(df): errors.append('Dataset is empty')
    if not np.isfinite(df.time_s).all(): errors.append('Nonfinite timestamps')
    if df.flight_id.isna().any(): errors.append('Missing flight IDs')
    if df.duplicated(['flight_id','time_s']).any(): errors.append('Duplicate timestamps within flight')
    sampling = []
    for fid, group in df.groupby('flight_id',sort=False):
        delta = group.time_s.diff().dropna()
        if (delta <= 0).any(): errors.append(f'Flight {fid}: timestamps not strictly monotonic')
        large = int((delta > 1.5/float(config.get('master_rate_hz',10))).sum())
        if large: warnings.append(f'Flight {fid}: {large} large master-grid gaps')
        sampling.append({'flight_id':fid,'median_dt_s':delta.median() if len(delta) else None,'min_dt_s':delta.min() if len(delta) else None,'max_dt_s':delta.max() if len(delta) else None,'large_gaps':large})
        segments = [s for s in manifest.get('flight_segments',[]) if s.get('flight_id')==fid]
        if len(segments)!=1: errors.append(f'Flight {fid}: no unique manifest boundary')
        else:
            segment = segments[0]; end = segment.get('end_time_s',segment.get('disarm_time_s'))
            if end is None or not ((group.time_s >= segment['arm_time_s']) & (group.time_s < end)).all(): errors.append(f'Flight {fid}: samples outside arm/disarm boundaries')
    frame_checks = _frame_checks(manifest, errors)
    axis_stats = {}
    for axis, source in AXIS_SOURCE.items():
        flag = f'valid_d{axis}'; df[flag] = _flags(df[flag],errors,flag)
        columns = [f'a{axis}', f'a{axis}_target', f'd{axis}']
        finite = np.isfinite(df[columns]).all(axis=1)
        if (df[flag] & ~finite).any(): errors.append(f'{flag}: true with nonfinite actual/target/residual')
        if (~df[flag] & df[f'd{axis}'].notna()).any(): errors.append(f'{flag}: invalid residual is not NaN')
        good = df[flag] & finite
        if not np.allclose(df.loc[good,f'd{axis}'],df.loc[good,f'a{axis}']-df.loc[good,f'a{axis}_target'],atol=1e-9,rtol=1e-9): errors.append(f'{axis}: residual identity actual-target failed')
        agecol, timecol = f'age_{source}_ms', f'source_time_{source}_s'
        if agecol not in df or timecol not in df: errors.append(f'{axis}: missing source age/timestamp')
        else:
            age = pd.to_numeric(df[agecol],errors='coerce'); st = pd.to_numeric(df[timecol],errors='coerce')
            limit = min(_limit(config,'maximum_source_age_ms',source,180),_limit(config,'gap_threshold_ms',source,500))
            if (good & (~np.isfinite(age) | (age < -1e-6) | (age > limit+1e-6))).any(): errors.append(f'{axis}: valid sample violates source-age limit')
            if (good & (~np.isfinite(st) | (st > df.time_s+1e-9) | ((df.time_s-st)*1000-age).abs().gt(1e-5))).any(): errors.append(f'{axis}: source timestamp/age inconsistent or in future')
        axis_stats[axis] = _stats(df.loc[good,f'd{axis}'])
        axis_stats[axis]['RMSE_actual_target'] = float(np.sqrt(np.mean(df.loc[good,f'd{axis}']**2))) if good.any() else None
        recorded = manifest.get('valid_counts',{}).get(f'd{axis}')
        if recorded is not None and recorded != int(df[flag].sum()): errors.append(f'{axis}: valid count differs from manifest')
    for name, expected in manifest.get('file_hashes',{}).items():
        path = dataset_path.parent/name
        if not path.is_file() or sha256_file(path) != expected: errors.append(f'Artifact checksum mismatch: {name}')
    _validate_source_provenance(df,manifest,config,errors)
    _validate_eligibility(df,config,errors)
    gp = _split_checks(dataset_path.parent,df,errors,config)
    stats = []
    for column in NUMERIC_COLUMNS:
        row = {'column':column, **_stats(df[column]), 'nan_percentage':float(df[column].isna().mean()*100)}
        stats.append(row)
        if column in STATE_COLUMNS and row['count_valid'] and row['range'] < config.get('constant_range_epsilon',1e-3): warnings.append(f'{column}: essentially constant; inadequate coverage for learning its influence')
    pd.DataFrame(stats).to_csv(output/'column_stats.csv',index=False)
    flags = _physical_flags(df,config)
    flags.to_csv(output/'outlier_flags.csv',index=False)
    for column in flags.filter(like='outlier_'):
        count = int(flags[column].sum())
        if count: warnings.append(f'{column}: {count} rows; retained without correction or deletion')
    for grouping, filename in [('flight_id','quality_by_flight.csv'),('mode','quality_by_mode.csv')]:
        rows = []
        for value, group in df.groupby(grouping,sort=False,dropna=False):
            rows.append({grouping:value,'row_count':len(group),**{f'valid_d{a}':int(group[f'valid_d{a}'].sum()) for a in 'xyz'},'outlier_count':int(flags.loc[group.index,'outlier'].sum()),'state_nan_percentage':float(group[STATE_COLUMNS].isna().mean().mean()*100)})
        pd.DataFrame(rows,columns=[grouping,'row_count','valid_dx','valid_dy','valid_dz','outlier_count','state_nan_percentage']).to_csv(output/filename,index=False)
    warnings.extend(['Armed segments do not establish airborne time; ground/transition samples remain visible.',
                     'Chronological splits reduce leakage but do not establish independence or excitation.',
                     'Battery current calibration has not been independently validated.',
                     'ESC physical rotor speed, pole count and cross-sensor calibration remain unverified.',
                     'Readiness is a configured data-screening result, not evidence of GP model performance.'])
    if manifest.get('xy_acceleration_is_proxy',True): warnings.append('Horizontal A-TA is a controller proxy, not measured acceleration residual; XY/XYZ physical learning is NOT_READY.')
    if axis_stats['z']['count_valid'] <= max(axis_stats['x']['count_valid'],axis_stats['y']['count_valid']): warnings.append('Z valid coverage is not greater than XY; investigate mode selection/source validity.')
    esc_summary = {}
    esc_path = dataset_path.parent/manifest.get('raw_sidecars',{}).get('ESC',{}).get('file','esc_raw.csv')
    if esc_path.is_file():
        esc = pd.read_csv(esc_path)
        if 'Instance' in esc and 'RPM' in esc:
            esc_summary = {str(k):_stats(g.RPM) for k,g in esc.groupby('Instance')}
            means = [s['mean'] for s in esc_summary.values() if s['mean'] is not None and s['mean'] > 0]
            if len(means)>1 and max(means)/min(means) > config.get('esc_disagreement_ratio',2): warnings.append('ESC mean RPM differs across instances beyond configured ratio; dynamics/mapping/scaling require investigation.')
        if 'Volt' in esc and esc.Volt.notna().any() and df.battery_v.notna().any():
            ratio=float(esc.Volt.median()/df.battery_v.median())
            if not .5 <= ratio <= 1.5: warnings.append(f'ESC/BAT median voltage ratio {ratio:.3g}: telemetry units/calibration disagree; no correction applied.')
        if 'Curr' in esc and len(esc) and pd.to_numeric(esc.Curr,errors='coerce').fillna(0).eq(0).all(): warnings.append('ESC current is zero throughout the logged telemetry; it cannot validate battery current.')
    errors = list(dict.fromkeys(errors)); warnings = list(dict.fromkeys(warnings))
    readiness = {}
    min_samples = int(config.get('minimum_samples_for_learning',200))
    min_train = int(config.get('minimum_train_samples',100)); min_test = int(config.get('minimum_test_samples',30))
    for axis in 'xyz':
        count = axis_stats[axis]['count_valid']; reasons = []
        eligible_col = f'eligible_d{axis}'
        eligible_count = int(_flags(df[eligible_col],[],eligible_col).sum()) if eligible_col in df else count
        split_counts = {}
        for split in ('train','test'):
            subset = gp[gp.split==split] if 'split' in gp else pd.DataFrame()
            valid_column = eligible_col if eligible_col in subset else f'valid_d{axis}'
            split_counts[split] = int(_flags(subset[valid_column],[],valid_column).sum()) if valid_column in subset else 0
        if errors: reasons.append('Structural/provenance checks failed')
        if axis in 'xy' and manifest.get('xy_acceleration_is_proxy',True): reasons.append('Horizontal acceleration source is a controller proxy')
        if eligible_count < min_samples: reasons.append(f'Eligible samples {eligible_count} < {min_samples}')
        if split_counts['train'] < min_train or split_counts['test'] < min_test: reasons.append(f'Eligible train/test {split_counts["train"]}/{split_counts["test"]} below {min_train}/{min_test}')
        readiness['d'+axis] = {'status':'NOT_READY' if reasons else ('READY_WITH_WARNINGS' if warnings else 'READY'), 'reasons':reasons, 'count_valid':count,'count_eligible':eligible_count, 'split_counts':split_counts}
    joint_columns=[f'eligible_d{a}' if f'eligible_d{a}' in df else f'valid_d{a}' for a in 'xyz']
    joint_count=int(df[joint_columns].all(axis=1).sum())
    joint_splits={}
    for split in ['train','test']:
        subset=gp[gp.split==split] if 'split' in gp else pd.DataFrame()
        joint_splits[split]=int(pd.concat([_flags(subset[c],[],c) for c in joint_columns],axis=1).all(axis=1).sum()) if len(subset) and set(joint_columns)<=set(subset) else 0
    xyz_bad=any(readiness['d'+a]['status']=='NOT_READY' for a in 'xyz') or joint_count<min_samples or joint_splits['train']<min_train or joint_splits['test']<min_test
    readiness['XYZ'] = {'status':'NOT_READY' if xyz_bad else 'READY_WITH_WARNINGS','reasons':['Requires all three physical residual models and joint eligible total/train/test thresholds'],'joint_valid_count':int(df[['valid_dx','valid_dy','valid_dz']].all(axis=1).sum()),'joint_eligible_count':joint_count,'split_counts':joint_splits}
    readiness['battery'] = {'status':'NOT_READY','reasons':['Current sensor calibration is not independently verified; voltage/current exports are observational only']}
    readiness['ESC'] = {'status':'NOT_READY','reasons':['Physical rotor RPM/pole count and telemetry agreement are not independently verified']}
    plot_files = []
    if make_plots:
        from .plots import make_validation_plots
        plot_files = make_validation_plots(df, Path(plots_dir))
    report = {'structural_pass':not errors,'row_count':len(df),'dataset_sha256':sha256_file(dataset_path),
              'errors':errors,'warnings':warnings,'frame_checks':frame_checks,'axis_statistics':axis_stats,
              'sampling_by_flight':sampling,'readiness':readiness,'esc_statistics':esc_summary,
              'readiness_criteria':{'minimum_samples':min_samples,'minimum_train':min_train,'minimum_test':min_test,'require_physical_acceleration':True,'require_verified_sensor_calibration':True},
              'outlier_row_count':int(flags.outlier.sum()),'plots':plot_files}
    write_json(output/'validation_report.json',report)
    lines = ['Dataset validation',f'Rows: {len(df)}',f'Structural checks: {"PASS" if not errors else "FAIL"}',
             'Research frame ENU: x East, y North, z Up; d = a_actual - a_target','', 'Readiness:']
    lines.extend(f'  {k}: {v["status"]}; '+ '; '.join(v['reasons']) for k,v in readiness.items())
    lines.extend(['','Errors:'] + [f'  {e}' for e in errors] + ['','Warnings:'] + [f'  {w}' for w in warnings])
    (output/'validation_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path,nargs='?',default=Path('data/processed/all_flights.csv'))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/validation'))
    parser.add_argument('--plots-dir',type=Path,default=Path('results/plots'))
    parser.add_argument('--no-plots',action='store_true')
    args = parser.parse_args()
    report = validate_dataset(args.dataset,args.manifest,args.output_dir,args.plots_dir,make_plots=not args.no_plots)
    print(json.dumps({'structural_pass':report['structural_pass'],'readiness':report['readiness'],'errors':report['errors']},indent=2))
    return 0 if report['structural_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
