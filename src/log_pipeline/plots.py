"""Separate, headless validation figures with visible missing-data intervals."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


_AXES = (("x", "East"), ("y", "North"), ("z", "Up"))
_TIME_LABEL = "Time (seconds since boot)"


def _numeric(df, column):
    if column not in df:
        return np.full(len(df), np.nan)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _flag(df, column):
    if column not in df:
        return np.zeros(len(df), dtype=bool)
    values = df[column]
    return (values.eq(True) | values.astype("string").str.lower().isin(["true", "1"])).fillna(False).to_numpy(dtype=bool)


def _runs(df, times):
    """Keep input order and break lines at declared boundaries or clock gaps."""
    if not len(times):
        return []
    starts = np.zeros(len(times), dtype=bool)
    starts[0] = True
    for column in ("flight_id", "continuity_id", "boundary_id", "mode"):
        if column in df:
            starts |= df[column].ne(df[column].shift()).fillna(True).to_numpy(dtype=bool)
    differences = np.diff(times)
    starts[1:] |= ~np.isfinite(differences) | (differences <= 0)
    positive_steps = differences[np.isfinite(differences) & (differences > 0)]
    if len(positive_steps):
        # A plotting safeguard for omitted boundary metadata; explicit interval
        # IDs remain authoritative, and no data is interpolated or resampled.
        starts[1:] |= differences > 3 * np.median(positive_steps)
    edges = np.append(np.flatnonzero(starts), len(times))
    return [slice(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:])]


def _figure(title, rows=1):
    figure = Figure(figsize=(10, 3.2 * rows + 0.7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(rows, 1, sharex=rows > 1))
    figure.suptitle(title)
    return figure, axes


def _draw_series(axis, times, values, runs, label, color=None, drawstyle="default"):
    drew = False
    for run in runs:
        if not (np.isfinite(times[run]) & np.isfinite(values[run])).any():
            continue
        axis.plot(
            times[run], values[run], color=color, linewidth=1, marker=".", markersize=3,
            label=label if not drew else "_nolegend_", drawstyle=drawstyle,
        )
        drew = True
    return drew


def _time_axis(axis, times, ylabel, drew):
    axis.set_xlabel(_TIME_LABEL)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        first, last = float(finite_times.min()), float(finite_times.max())
        margin = max((last - first) * 0.01, 0.05)
        axis.set_xlim(first - margin, last + margin)
    if not drew:
        axis.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=axis.transAxes)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="best", fontsize=8)


def _shade_invalid(axis, times, valid, runs):
    """Shade contiguous invalid rows without filling gaps between intervals."""
    dt = np.diff(times)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    half_step = float(np.median(positive) / 2) if len(positive) else 0.05
    labeled = False
    for run in runs:
        invalid = ~valid[run] & np.isfinite(times[run])
        changes = np.diff(np.r_[False, invalid, False].astype(int))
        for left_index, right_index in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            start, stop = run.start + int(left_index), run.start + int(right_index) - 1
            left = (times[start - 1] + times[start]) / 2 if start > run.start else times[start] - half_step
            right = (times[stop] + times[stop + 1]) / 2 if stop + 1 < run.stop else times[stop] + half_step
            axis.axvspan(left, right, color="#bcbcbc", alpha=0.25, linewidth=0, label="Residual invalid" if not labeled else "_nolegend_")
            labeled = True


def make_validation_plots(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Write fifteen separate PNG diagnostics without modifying the dataframe.

    Preserve NaNs and split lines by flight, continuity/boundary ID and mode.
    Mark individual samples so isolated points remain visible. Gray shading on
    acceleration and residual plots means valid_d<axis> is false or unknown;
    numeric diagnostic values are retained even when shaded. Histograms use
    only explicitly valid, finite residuals and state the excluded row count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = _numeric(df, "time_s")
    runs = _runs(df, times)
    files = []
    proxy = _flag(df, "xy_acceleration_is_proxy").any()

    def save(figure, name):
        path = output_dir / name
        figure.savefig(path, dpi=140, facecolor="white")
        figure.clear()
        files.append(str(path))

    for name, prefix, unit in (("position", "p", "m"), ("velocity", "v", "m/s"), ("acceleration", "a", "m/s²")):
        figure, axes = _figure(f"{name.capitalize()} (ENU)", rows=3)
        for plot_axis, (letter, direction) in zip(axes, _AXES):
            label = f"{letter} {direction}"
            if name == "acceleration":
                _shade_invalid(plot_axis, times, _flag(df, f"valid_d{letter}"), runs)
                if proxy and letter in "xy":
                    label += " (controller proxy)"
            drew = _draw_series(plot_axis, times, _numeric(df, prefix + letter), runs, label, color="#2563a6")
            _time_axis(plot_axis, times, f"{letter} {direction} ({unit})", drew)
        save(figure, f"{name}.png")

    for letter, direction in _AXES:
        actual, target = _numeric(df, f"a{letter}"), _numeric(df, f"a{letter}_target")
        values = _numeric(df, f"d{letter}")
        valid = _flag(df, f"valid_d{letter}")
        proxy_text = " (controller proxy)" if proxy and letter in "xy" else ""
        figure, axes = _figure(f"Acceleration {letter}: logged actual and target{proxy_text}")
        plot_axis = axes[0]
        _shade_invalid(plot_axis, times, valid, runs)
        drew_actual = _draw_series(plot_axis, times, actual, runs, "Logged actual" + proxy_text, color="#2563a6")
        drew_target = _draw_series(plot_axis, times, target, runs, "Target", color="#cb5b22")
        _time_axis(plot_axis, times, f"{letter} {direction} (m/s²)", drew_actual or drew_target)
        save(figure, f"acceleration_{letter}_actual_target.png")

        figure, axes = _figure(f"Residual {letter}: actual − target{proxy_text}")
        plot_axis = axes[0]
        _shade_invalid(plot_axis, times, valid, runs)
        drew = _draw_series(plot_axis, times, values, runs, f"d{letter}", color="#6b48a8")
        _time_axis(plot_axis, times, f"d{letter} (m/s²)", drew)
        save(figure, f"residual_{letter}_time.png")

        figure, axes = _figure(f"Residual {letter} distribution{proxy_text}")
        plot_axis = axes[0]
        included = valid & np.isfinite(values) & np.isfinite(times)
        samples = values[included]
        if samples.size:
            plot_axis.hist(samples, bins=min(40, max(1, int(np.sqrt(samples.size)))), color="#6b48a8", alpha=0.8)
        else:
            plot_axis.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=plot_axis.transAxes)
        plot_axis.text(0.99, 0.98, f"Valid: {samples.size} / {len(df)} rows\nExcluded: {len(df) - samples.size}", ha="right", va="top", transform=plot_axis.transAxes, fontsize=9)
        plot_axis.set_xlabel(f"d{letter} = a{letter} − a{letter}_target (m/s²)")
        plot_axis.set_ylabel("Sample count")
        plot_axis.grid(True, alpha=0.25)
        save(figure, f"residual_{letter}_histogram.png")

    figure, axes = _figure("Battery telemetry", rows=2)
    for plot_axis, (column, label, color) in zip(axes, (("battery_v", "Voltage (V)", "#2563a6"), ("battery_i", "Current (A)", "#cb5b22"))):
        drew = _draw_series(plot_axis, times, _numeric(df, column), runs, label, color=color)
        _time_axis(plot_axis, times, label, drew)
    save(figure, "battery.png")

    figure, axes = _figure("Flight mode timeline")
    plot_axis = axes[0]
    mode_values = np.full(len(df), np.nan)
    labels = []
    if "mode" in df:
        labels = list(dict.fromkeys(str(value) for value in df["mode"].dropna()))
        mapping = {label: index for index, label in enumerate(labels)}
        mode_values = df["mode"].map(lambda value: mapping.get(str(value), np.nan)).to_numpy(dtype=float)
    drew = _draw_series(plot_axis, times, mode_values, runs, "Logged mode", color="#39794f", drawstyle="steps-post")
    plot_axis.set_yticks(range(len(labels)), labels)
    _time_axis(plot_axis, times, "Flight mode", drew)
    save(figure, "flight_modes.png")

    figure, axes = _figure("Source-data age and alignment coverage")
    plot_axis = axes[0]
    drew = False
    colors = ("#2563a6", "#cb5b22", "#39794f", "#6b48a8", "#a33f64", "#777020")
    age_columns = [column for column in df if str(column).startswith("age_") and str(column).endswith("_ms")]
    for index, column in enumerate(age_columns):
        drew = _draw_series(plot_axis, times, _numeric(df, column), runs, str(column), color=colors[index % len(colors)]) or drew
    _time_axis(plot_axis, times, "Source age (ms)", drew)
    save(figure, "source_age.png")
    return files
