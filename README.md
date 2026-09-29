# TinyLB_F405 data foundation

Python tools to inspect ArduPilot DataFlash logs, extract traceable datasets, and assess their suitability for residual-learning research on an F450 / SpeedyBee F405 V5. No GP, MPC, control, communication or training code is included.

## Install

Python 3.11 or newer; Python 3.12 is the verified environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

On Linux/macOS, activate with `source .venv/bin/activate`. `requirements-lock.txt` pins the tested library versions. A development installation can also use `python -m pip install -e ".[test]"`.

Place an unchanged copy of your log at `data/raw/flight.bin`. Raw logs, processed telemetry, private audit documents, inspection reports and plots are excluded from the public repository because they may contain GPS locations and device identifiers.

## Run

From the repository root:

```powershell
python -m src.log_pipeline.inspect_log data/raw/flight.bin
python -m src.log_pipeline.extract_dataset data/raw/flight.bin --inspection results/log_inspection/log_summary.json --config configs/log_pipeline.yaml
python -m src.log_pipeline.validate_dataset data/processed/all_flights.csv
python -m pytest -q
$env:RUN_REAL_LOG_TESTS = "1"
$env:REAL_LOG_PATH = "data/raw/flight.bin"
python -m pytest -q
```

The inspector writes JSON/text summaries, message/field inventories, per-instance timing, armed segments and mode intervals to `results/log_inspection/`. It returns exit code 2 if parsing or metadata has a critical error while still writing diagnostic reports. Do not treat that exit as permission to discard errors.

Extraction consumes the raw log and inspector summary, verifies checksums and metadata, then writes separate flight CSVs, `all_flights.csv`, `gp_residual_dataset.csv`, raw battery/ESC telemetry and a lineage manifest to `data/processed/`. Validation writes reports/statistics/outlier flags to `results/validation/` and separate matplotlib PNGs to `results/plots/`. Validator exit code 2 means structural failure; per-model `NOT_READY` is reported separately from structural validity.

An incomplete final packet is a critical diagnostic and extraction rejects it by default. The raw file is never repaired or truncated. An explicit `--allow-truncated-tail` exception may use only the complete-message prefix when authorized; it cannot permit interior corruption, arbitrary garbage, unresolved metadata or timestamp errors. All evidence remains in the local manifest and warnings.

Only after accepting that exception, append `--allow-truncated-tail` to the extraction command above. Use `--flight-id 3` to select one armed interval; repeat the flag to select several. `--holdout-flight-id 3` chooses a later suitable flight for testing instead of the default chronological row split.

## Scientific conventions

Research frame: **ENU**, x East, y North, z Up. Verified NED vectors convert as `[E, N, -D]`. ATT heading uses North=0 and clockwise positive degrees; research yaw is `wrap(pi/2 - radians(ATT.Yaw))`, East=0, counterclockwise positive, radians in `[-pi,pi)`.

Initial residual convention is exactly **`d = a_actual - a_target`**, in identical frames and m/s². Targets are PSC `TA*` fields, never substituted from desired `DA*` fields. Transformations and residual checks live centrally in `frames.py`.

**Horizontal acceleration caveat:** in the supported ArduPilot source profile, PSCN.AN/PSCE.AE come from target roll/pitch and current yaw. They are controller proxies, not independent acceleration measurements. `dx/dy` retain numeric A−TA values with quality flags, while physical XY/XYZ learning remains `NOT_READY`. PSCD.AD is an AHRS Earth-frame gravity-compensated acceleration estimate, with different provenance. See the [public source-code audit](docs/source_audit.md). Extraction rejects firmware revisions outside that audited profile.

Logical motor mapping requires logged SERVOx_FUNCTION parameters. RCOU outputs are PWM-equivalent commands even with DShot; they are not measured thrust. Battery current and ESC RPM require independent calibration. Logged RPM has already passed through backend conversion; no extra pole-count scaling is applied.

## Alignment and validity

The default 10 Hz master grid is created separately within each armed interval. A bounded causal previous-sample policy retains source time and age; unsupported interpolation policies fail explicitly. Nothing is interpolated across disarm, mode, GPS validity, estimator-core/source or detected reset boundaries. Unknown or expired fields remain NaN. Reset-event logging is not exhaustive; absence of logged evidence cannot guarantee no reset occurred.

Position/velocity prefer PSC per axis and may fall back to a healthy selected primary EKF3 core. Target/actual acceleration always share a PSC packet. Numeric validity, feature eligibility and scientific readiness are separate. Outliers are flagged and retained. Chronological 70/30 splitting avoids random mixing; a flight holdout is available when suitable flights actually exist. Configuration lives in `configs/log_pipeline.yaml`.

## Future data collection

Armed intervals do not establish airborne time. Controller messages can be sparse or absent depending on mode and logging configuration; the pipeline preserves that missing coverage. Do not extrapolate brief XY targets across an entire flight.

For future data collection, verify required PSC logging in the intended position-control modes, retain explicit arming/mode/EKF/GPS health and reset information, obtain independently measured and frame-verified horizontal acceleration for physical residual targets, and confirm sensor calibration and ESC pole configuration. Collect repeatable trajectories with meaningful state/target variation over multiple separate flights, plus a genuine holdout flight. These are data requirements, not flight-control instructions.

The same raw bytes, code revision and configuration produce the same numerical CSV data. Manifest timestamps record provenance and naturally differ between runs. Readiness thresholds are explicit screening criteria, not a guarantee of learned-model quality.
