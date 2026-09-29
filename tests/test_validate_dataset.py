"""Independent failure cases: a readable CSV must not imply readiness."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def fixture_dataset(tmp_path):
    n = 20
    d = pd.DataFrame({'time_s': np.arange(n) / 10 + 1, 'flight_id': 1, 'mode': 'ALT_HOLD'})
    for field in ('px','py','pz','vx','vy','vz','ax','ay','az','yaw','ax_target','ay_target','az_target','dx','dy','dz','battery_v','battery_i','throttle','motor1','motor2','motor3','motor4'):
        d[field] = 0.0
    d['az'] = np.linspace(-1, 1, n)
    d['az_target'] = 0.2
    d['dz'] = d.az - d.az_target
    d['battery_v'] = 11.4
    d['battery_i'] = 5.0
    for axis, source in [('x','psce'),('y','pscn'),('z','pscd')]:
        d[f'valid_d{axis}'] = True
        d[f'age_{source}_ms'] = 20.0
        d[f'source_time_{source}_s'] = d.time_s - .02
    d['outlier'] = False
    d['age_att_ms']=20.0
    d['source_time_att_s']=d.time_s-.02
    for src in ['bat','rcou','ctun']:
        d[f'age_{src}_ms']=20.0
        d[f'source_time_{src}_s']=d.time_s-.02
    for axis,src,suffix in [('x','PSCE','E'),('y','PSCN','N'),('z','PSCD','D')]:
        d['source_p'+axis]=src+'.P'+suffix
        d['source_v'+axis]=src+'.V'+suffix
    path = tmp_path / 'all_flights.csv'
    d.to_csv(path, index=False)
    gp = d.copy()
    gp['split'] = ['train'] * 14 + ['test'] * 6
    gp.to_csv(tmp_path/'gp_residual_dataset.csv', index=False)
    manifest = {
        'coordinate_frame': 'ENU',
        'residual_sign_convention': 'd = a_actual - a_target',
        'xy_acceleration_is_proxy': True,
        'extractor_config': {
            'master_rate_hz': 10, 'gap_threshold_ms': 500,
            'maximum_source_age_ms': {'PSCN':180,'PSCE':180,'PSCD':180},
            'minimum_samples_for_learning': 10,
            'minimum_train_samples': 5, 'minimum_test_samples': 3,
            'battery_plausibility_bounds': {'voltage_v':[9,12.8], 'current_a':[.5,150]},
            'velocity_plausibility_bounds': [-30,30],
            'acceleration_plausibility_bounds': [-40,40],
        },
        'source_mapping': {c: {'output_unit': 'm/s^2', 'output_frame':'ENU'} for c in ['ax','ay','az','ax_target','ay_target','az_target','dx','dy','dz']},
        'flight_segments': [{'flight_id':1,'arm_time_s':.9,'end_time_s':3,'disarm_time_s':3}],
        'warnings': [], 'valid_counts': {'dx':n,'dy':n,'dz':n},
    }
    mp = tmp_path/'dataset_manifest.json'
    mp.write_text(json.dumps(manifest), encoding='utf-8')
    return d, path, manifest, mp


def run_validation(path, mp, tmp_path):
    from src.log_pipeline.validate_dataset import validate_dataset
    return validate_dataset(path, mp, tmp_path/'validation', tmp_path/'plots', make_plots=False)


def test_proxy_xy_and_uncalibrated_sensors_are_not_ready(tmp_path):
    _, path, _, mp = fixture_dataset(tmp_path)
    report = run_validation(path, mp, tmp_path)
    assert report['structural_pass']
    assert report['readiness']['dx']['status'] == 'NOT_READY'
    assert report['readiness']['dy']['status'] == 'NOT_READY'
    assert report['readiness']['XYZ']['status'] == 'NOT_READY'
    assert report['readiness']['battery']['status'] == 'NOT_READY'
    assert report['readiness']['ESC']['status'] == 'NOT_READY'
    assert report['readiness']['dz']['status'] == 'READY_WITH_WARNINGS'
    assert report['axis_statistics']['z']['count_valid'] == 20
    for name in ['validation_report.json','validation_report.txt','column_stats.csv','quality_by_flight.csv','quality_by_mode.csv','outlier_flags.csv']:
        assert (tmp_path/'validation'/name).is_file()


@pytest.mark.parametrize('mutation', ['residual_sign','duplicate','backwards','stale','negative_age','outside_flight','invalid_flag','nonnumeric','infinite','source_future'])
def test_structural_corruption_fails_loudly(tmp_path, mutation):
    d, path, _, mp = fixture_dataset(tmp_path)
    if mutation == 'residual_sign': d.loc[0,'dz'] *= -1
    if mutation == 'duplicate': d.loc[1,'time_s'] = d.loc[0,'time_s']
    if mutation == 'backwards': d.loc[1,'time_s'] = 0.9
    if mutation == 'stale': d.loc[0,'age_pscd_ms'] = 181
    if mutation == 'negative_age': d.loc[0,'age_pscd_ms'] = -1
    if mutation == 'outside_flight': d.loc[0,'time_s'] = .5
    if mutation == 'invalid_flag': d.loc[0,'az'] = np.nan
    if mutation == 'nonnumeric': d['dz'] = d.dz.astype(object); d.loc[0,'dz'] = 'broken'
    if mutation == 'infinite': d.loc[0,'az'] = np.inf
    if mutation == 'source_future': d.loc[0,'source_time_pscd_s'] = 2
    d.to_csv(path,index=False)
    report = run_validation(path,mp,tmp_path)
    assert not report['structural_pass']
    assert report['errors']
    assert report['readiness']['dz']['status'] == 'NOT_READY'


def test_mixed_frames_fail_even_when_equation_matches(tmp_path):
    _, path, m, mp = fixture_dataset(tmp_path)
    m['source_mapping']['az_target']['output_frame'] = 'NED'
    mp.write_text(json.dumps(m))
    report = run_validation(path,mp,tmp_path)
    assert not report['structural_pass']
    assert any('frame' in e.lower() for e in report['errors'])


def test_temporal_split_leakage_fails(tmp_path):
    _, path, _, mp = fixture_dataset(tmp_path)
    gp = pd.read_csv(tmp_path/'gp_residual_dataset.csv')
    gp.loc[0,'split'] = 'test'
    gp.to_csv(tmp_path/'gp_residual_dataset.csv',index=False)
    report = run_validation(path,mp,tmp_path)
    assert not report['structural_pass']
    assert any('split' in e.lower() or 'leak' in e.lower() for e in report['errors'])


def test_outlier_is_flagged_and_never_deleted(tmp_path):
    d, path, _, mp = fixture_dataset(tmp_path)
    d.loc[0,'battery_v'] = 50
    d.to_csv(path,index=False)
    report = run_validation(path,mp,tmp_path)
    assert report['row_count'] == 20
    flags = pd.read_csv(tmp_path/'validation/outlier_flags.csv')
    assert bool(flags.loc[0,'outlier_battery_voltage'])
    assert len(pd.read_csv(path)) == 20


def test_missing_required_columns_returns_failure_report(tmp_path):
    d, path, _, mp = fixture_dataset(tmp_path)
    d.drop(columns=['az']).to_csv(path,index=False)
    assert not run_validation(path,mp,tmp_path)['structural_pass']


def test_eligibility_cannot_override_invalid_residual(tmp_path):
    d,path,m,mp = fixture_dataset(tmp_path)
    d['eligible_dz'] = True
    d['valid_dz'] = False
    d[['az','az_target','dz']] = np.nan
    d.to_csv(path,index=False)
    gp=d.copy(); gp['split']=['train']*14+['test']*6
    gp.to_csv(tmp_path/'gp_residual_dataset.csv',index=False)
    m['valid_counts']['dz']=0; mp.write_text(json.dumps(m))
    report=run_validation(path,mp,tmp_path)
    assert not report['structural_pass']
    assert report['readiness']['dz']['status']=='NOT_READY'


@pytest.mark.parametrize('scope',['flight','continuity'])
def test_source_must_be_inside_current_flight_and_continuity(tmp_path,scope):
    d,path,m,mp=fixture_dataset(tmp_path)
    if scope=='flight':
        m['flight_segments'][0]['arm_time_s']=1
        mp.write_text(json.dumps(m))
    else:
        d['continuity_start_s']=d.time_s-.01
        d['continuity_end_s']=3.0
    d.to_csv(path,index=False)
    assert not run_validation(path,mp,tmp_path)['structural_pass']


def test_xyz_requires_joint_samples(tmp_path):
    d,path,m,mp=fixture_dataset(tmp_path)
    m['xy_acceleration_is_proxy']=False
    m['extractor_config'].update(minimum_samples_for_learning=3,minimum_train_samples=1,minimum_test_samples=1)
    for i,axis in enumerate('xyz'):
        good=np.arange(len(d))%3==i
        d[f'valid_d{axis}']=good
        d.loc[~good,[f'a{axis}',f'a{axis}_target',f'd{axis}']]=np.nan
        m['valid_counts'][f'd{axis}']=int(good.sum())
    d.to_csv(path,index=False)
    gp=d.copy();gp['split']=['train']*14+['test']*6
    gp.to_csv(tmp_path/'gp_residual_dataset.csv',index=False)
    mp.write_text(json.dumps(m))
    report=run_validation(path,mp,tmp_path)
    assert report['structural_pass']
    assert report['readiness']['XYZ']['status']=='NOT_READY'


def test_gp_cannot_invent_eligibility_flags(tmp_path):
    d,path,_,mp=fixture_dataset(tmp_path)
    d['eligible_dz']=False;d.to_csv(path,index=False)
    gp=pd.read_csv(tmp_path/'gp_residual_dataset.csv');gp['eligible_dz']=True
    gp.to_csv(tmp_path/'gp_residual_dataset.csv',index=False)
    assert not run_validation(path,mp,tmp_path)['structural_pass']


@pytest.mark.parametrize('source',['att','xkf1'])
def test_state_feature_source_age_is_validated(tmp_path,source):
    d,path,m,mp=fixture_dataset(tmp_path)
    d[f'age_{source}_ms']=5000
    d[f'source_time_{source}_s']=d.time_s-5
    if source=='xkf1': d['source_px']='XKF1.PE[C=0]'
    m['extractor_config']['maximum_source_age_ms'][source.upper()]=180
    d.to_csv(path,index=False);mp.write_text(json.dumps(m))
    assert not run_validation(path,mp,tmp_path)['structural_pass']


def test_real_extractor_schema_validates_without_interface_assumptions(tmp_path):
    from test_extract_dataset import fixture_scan, config
    from src.log_pipeline.extract_dataset import build_dataset, assign_splits, motor_mapping
    from src.log_pipeline.schemas import column_mapping
    scan=fixture_scan();cfg=config()
    d=build_dataset(scan,cfg)
    path=tmp_path/'all_flights.csv';d.to_csv(path,index=False,float_format='%.12g')
    assign_splits(d,cfg).to_csv(tmp_path/'gp_residual_dataset.csv',index=False,float_format='%.12g')
    m={'coordinate_frame':'ENU','residual_sign_convention':'d = a_actual - a_target',
       'extractor_config':cfg,'xy_acceleration_is_proxy':True,
       'source_mapping':column_mapping(d.columns,motor_mapping(scan)),
       'flight_segments':scan.flight_segments,'warnings':[],
       'valid_counts':{f'd{a}':int(d[f'valid_d{a}'].sum()) for a in 'xyz'}}
    mp=tmp_path/'dataset_manifest.json';mp.write_text(json.dumps(m))
    report=run_validation(path,mp,tmp_path)
    assert report['structural_pass'],report['errors']


def test_missing_eligibility_cannot_make_missing_features_ready(tmp_path):
    d,path,m,mp=fixture_dataset(tmp_path)
    m['extractor_config']['learning_features_by_axis']={'z':['pz','vz','yaw']}
    d['yaw']=np.nan;d.to_csv(path,index=False)
    gp=d.copy();gp['split']=['train']*14+['test']*6
    gp.to_csv(tmp_path/'gp_residual_dataset.csv',index=False)
    mp.write_text(json.dumps(m))
    report=run_validation(path,mp,tmp_path)
    assert report['readiness']['dz']['status']=='NOT_READY'
    assert report['readiness']['dz']['count_eligible']==0


def test_missing_attitude_provenance_fails_when_yaw_is_used(tmp_path):
    d,path,_,mp=fixture_dataset(tmp_path)
    d.drop(columns=['age_att_ms','source_time_att_s']).to_csv(path,index=False)
    assert not run_validation(path,mp,tmp_path)['structural_pass']


def test_validator_output_hardlink_cannot_overwrite_raw_input(tmp_path):
    import os
    _,path,m,mp=fixture_dataset(tmp_path)
    raw=tmp_path/'raw.bin';raw.write_bytes(b'raw data must remain immutable')
    m['raw_path']=str(raw);mp.write_text(json.dumps(m))
    output=tmp_path/'validation';output.mkdir()
    os.link(raw,output/'column_stats.csv')
    with pytest.raises(ValueError,match='overwrite|input|protected'):
        run_validation(path,mp,tmp_path)
    assert raw.read_bytes()==b'raw data must remain immutable'
