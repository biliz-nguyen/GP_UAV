"""Render real PNGs and inspect plotted geometry for hidden data gaps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.log_pipeline.plots import make_validation_plots


@pytest.fixture
def saved_figures(monkeypatch):
    """Observe actual figures without replacing their rendering or disk output."""
    saved = {}
    original = Figure.savefig

    def save_and_observe(figure, filename, *args, **kwargs):
        saved[Path(filename).name] = [
            {
                "lines": [(np.array(line.get_xdata()), np.array(line.get_ydata())) for line in axis.lines],
                "texts": [text.get_text() for text in axis.texts],
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "patches": len(axis.patches),
            }
            for axis in figure.axes
        ]
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_and_observe)
    return saved


def test_plots_render_all_required_figures_without_connecting_flights_or_invalid_samples(tmp_path, saved_figures):
    df = pd.DataFrame({
        "time_s": [1.0, 1.1, 1.2, 2.0, 5.0, 5.1],
        "flight_id": [1, 1, 1, 1, 2, 2],
        "continuity_id": [1, 1, 1, 2, 3, 3],
        "mode": [5, 5, 5, 0, 0, 0],
        "valid_dx": [True, False, True, False, True, True],
        "valid_dy": [True, False, True, False, True, True],
        "valid_dz": [True, False, True, False, True, True],
        "xy_acceleration_is_proxy": [True] * 6,
        "age_pscn_ms": [0, 100, 0, 300, 0, 0],
        "battery_v": [12.2, 12.1, 12, 11.9, 12, 11.9],
        "battery_i": [2, 2.1, 2.2, 3, 2.2, 2.3],
    })
    for axis in "xyz":
        df[f"p{axis}"] = [1, np.nan, 2, 3, 4, 5]
        df[f"v{axis}"] = [0, np.nan, 1, 2, 3, 4]
        df[f"a{axis}"] = [1, np.nan, 2, np.nan, 3, 4]
        df[f"a{axis}_target"] = [0, np.nan, 1, np.nan, 2, 3]
        df[f"d{axis}"] = [1, np.nan, 1, np.nan, 1, 1]
    before = df.copy(deep=True)
    files = make_validation_plots(df, tmp_path)
    expected = {"position.png", "velocity.png", "acceleration.png", "battery.png", "flight_modes.png", "source_age.png"}
    expected |= {f"acceleration_{axis}_actual_target.png" for axis in "xyz"}
    expected |= {f"residual_{axis}_time.png" for axis in "xyz"}
    expected |= {f"residual_{axis}_histogram.png" for axis in "xyz"}
    assert {Path(path).name for path in files} == expected
    assert len(files) == 15
    for path in files:
        assert Path(path).is_file()
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert Path(path).stat().st_size > 1000
    pd.testing.assert_frame_equal(df, before)
    assert matplotlib.get_backend().lower() == "agg"
    position_lines = saved_figures["position.png"][0]["lines"]
    assert any(np.isnan(y).any() for _, y in position_lines)
    for x, _ in position_lines:
        assert not (np.any(x < 2) and np.any(x >= 2))
        assert not (np.any(x < 5) and np.any(x >= 5))
    assert saved_figures["acceleration_x_actual_target.png"][0]["patches"] > 0
    assert saved_figures["residual_x_time.png"][0]["patches"] > 0
    assert "seconds since boot" in saved_figures["source_age.png"][0]["xlabel"].lower()


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"time_s": [1.0, 2.0]})])
def test_empty_or_missing_columns_render_labeled_no_valid_samples(tmp_path, saved_figures, df):
    paths = make_validation_plots(df, tmp_path)
    assert len(paths) == 15
    assert all(Path(path).is_file() for path in paths)
    for axes in saved_figures.values():
        assert any("No valid samples" in text for axis in axes for text in axis["texts"])
