"""
    python plot_tanh_models.py
"""

import os
import re
import glob
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tanh_step_fit import build_design_matrix, lstsq_fit
from read_pos_files import read_pos_file
from rotate_ts import read_vel_file, station_azimuths, rotate_dataframe
from gnss_timeseries_manipulation.processing import hampel_identifier

STATIONS = ["ALBH", "SC03","P403"]

# "east", "north", "perp", "para", "both" (east+north), or "perp_para"
COMPONENT = "perp"

VEL_FILE = "loading.vel"

TIME_RESOLUTIONS = ["24h", "12h", "08h", "06h"] # Models to be plotted

DATA_RESOLUTIONS_TO_PLOT = ["24h","06h"] # Raw data to overlay the models on

SEGMENT_INDEX = 2

GPS_DATA_ROOT = "gps_for_dash"
SUBFOLDER = "20230608"

TANH_RESULTS_ROOT_TEMPLATE = "tanh_fit_results_{res}" # Location of model outputs

OUTPUT_DIR = "model_comparison_plots"

HAMPEL_WINDOW = 11
OFFSET_SIGMA = 4.0

MODEL_CURVE_POINTS = 400

TICK_MARKER_SIZE = 14

MJD_EPOCH = pd.Timestamp("1858-11-17")

COMPONENT_MAP = {"east": "dE", "north": "dN", "perp": "dPerp", "para": "dPara"}

_STATION_AZIMUTHS = None


def get_station_azimuths():
    """Lazily load + cache the per-station azimuth table from VEL_FILE."""
    global _STATION_AZIMUTHS
    if _STATION_AZIMUTHS is None:
        _STATION_AZIMUTHS = station_azimuths(read_vel_file(VEL_FILE))
    return _STATION_AZIMUTHS

_FLOAT = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_LAT_RE = re.compile(rf"^Latitude:\s*({_FLOAT})")
_LON_RE = re.compile(rf"^Longitude:\s*({_FLOAT})")
_SEGMENT_RE = re.compile(r"^Segment\s+(\d+):\s*(.+?)\s*->\s*(.+?)\s*\((\d+) points\)")
_COMPONENT_RE = re.compile(r"^\s*Component:\s*(\S+)")
_STEP_RE = re.compile(
    rf"^\s*Step\s+(\d+):\s*start=(\S+\s+\S+)\s+end=(\S+\s+\S+)\s+"
    rf"duration=({_FLOAT})\s*days\s+amplitude=({_FLOAT})\s*\+/-\s*({_FLOAT})\s*m"
)


def datetime_to_mjd(ts):
    return (pd.Timestamp(ts) - MJD_EPOCH).total_seconds() / 86400.0


def parse_summary(path):
    """Parse a fitting_tanh_function.py summary.txt file.

    Returns
    -------
    dict with:
      'latitude', 'longitude'
      'segments' : {seg_idx: {"start": Timestamp, "end": Timestamp,
                               "components": {comp: [step, ...]}}}
        each step is a dict with start/end (Timestamp), duration (days),
        amplitude, amplitude_std.
    """
    lat = lon = None
    segments = {}
    current_seg = None
    current_comp = None

    with open(path) as fp:
        for line in fp:
            m = _LAT_RE.match(line)
            if m:
                lat = float(m.group(1))
                continue
            m = _LON_RE.match(line)
            if m:
                lon = float(m.group(1))
                continue
            m = _SEGMENT_RE.match(line)
            if m:
                current_seg = int(m.group(1))
                segments[current_seg] = {
                    "start": pd.to_datetime(m.group(2)),
                    "end": pd.to_datetime(m.group(3)),
                    "n_points": int(m.group(4)),
                    "components": {},
                }
                current_comp = None
                continue
            m = _COMPONENT_RE.match(line)
            if m and current_seg is not None:
                current_comp = m.group(1)
                segments[current_seg]["components"].setdefault(current_comp, [])
                continue
            m = _STEP_RE.match(line)
            if m and current_seg is not None and current_comp is not None:
                segments[current_seg]["components"][current_comp].append({
                    "start": pd.to_datetime(m.group(2)),
                    "end": pd.to_datetime(m.group(3)),
                    "duration": float(m.group(4)),
                    "amplitude": float(m.group(5)),
                    "amplitude_std": float(m.group(6)),
                })
                continue

    return {"latitude": lat, "longitude": lon, "segments": segments}


def find_summary(station, resolution):
    results_root = TANH_RESULTS_ROOT_TEMPLATE.format(res=resolution)
    path = os.path.join(results_root, station, "summary.txt")
    if os.path.isfile(path):
        return path
    return None


def find_pos_file(station, resolution):
    """Locate the station's .pos file for the given resolution/date."""
    base_dir = os.path.join(GPS_DATA_ROOT, resolution, SUBFOLDER)

    # Expected naming convention (see read_pos_files.py / plot_multi_resolution_map.py)
    expected = os.path.join(base_dir, f"{station}.mit.final_nam20.pos")
    if os.path.isfile(expected):
        return expected

    # Fall back: station has its own subfolder.
    candidates = glob.glob(os.path.join(base_dir, station, "*.pos"))
    if candidates:
        return candidates[0]

    # Fall back: any .pos file starting with the station code.
    candidates = glob.glob(os.path.join(base_dir, f"{station}*.pos"))
    if candidates:
        return candidates[0]

    return None


def slice_segment(df, seg_start, seg_end, pad_hours=1):
    """Slice df to the datetime window [seg_start, seg_end] (small pad to
    absorb rounding between summary.txt's timestamps and the data's own)."""
    pad = pd.Timedelta(hours=pad_hours)
    mask = (df["datetime"] >= seg_start - pad) & (df["datetime"] <= seg_end + pad)
    return df.loc[mask].sort_values("datetime").reset_index(drop=True)


def reconstruct_model(seg_df, component, steps):
    """Refit constant + linear + fixed tanh-step(s) against this segment's
    data (same outlier removal as fitting_tanh_function.py), and return a
    dense (t_dense, y_dense) curve in MJD/meters for plotting, plus the
    step start/end tick points (t_ticks, y_ticks) evaluated on that same
    fitted curve.

    Returns None if there isn't enough data to fit.
    """
    t = seg_df["MJD"].to_numpy(dtype=float)
    y = seg_df[component].to_numpy(dtype=float)

    y_clean = hampel_identifier(y, window=HAMPEL_WINDOW, k=OFFSET_SIGMA)
    mask = np.isfinite(y_clean)
    if mask.sum() < 3:
        return None
    t_fit, y_fit = t[mask], y_clean[mask]

    step_list = [(datetime_to_mjd(s["start"]), s["duration"]) for s in steps]

    fit = lstsq_fit(t_fit, y_fit, step_list=step_list)

    t_dense = np.linspace(t.min(), t.max(), MODEL_CURVE_POINTS)
    G_dense, _ = build_design_matrix(t_dense, step_list)
    y_dense = G_dense @ fit["params"]

    # Step start/end tick points, evaluated on the fitted curve itself so
    # they sit exactly on the line.
    if steps:
        t_ticks = np.array(
            [datetime_to_mjd(s["start"]) for s in steps]
            + [datetime_to_mjd(s["end"]) for s in steps],
            dtype=float,
        )
        G_ticks, _ = build_design_matrix(t_ticks, step_list)
        y_ticks = G_ticks @ fit["params"]
    else:
        t_ticks = np.array([])
        y_ticks = np.array([])

    return t_dense, y_dense, t_ticks, y_ticks


def components_to_plot(component_setting):
    setting = component_setting.lower()
    if setting == "both":
        return ["east", "north"]
    if setting == "perp_para":
        return ["perp", "para"]
    if setting in COMPONENT_MAP:
        return [setting]
    raise ValueError(
        "COMPONENT must be one of 'east', 'north', 'perp', 'para', "
        f"'both', or 'perp_para', got {component_setting!r}"
    )


def plot_station_component(station, component_label, resolutions, data_resolutions,
                            output_dir, segment_index):
    col = COMPONENT_MAP[component_label]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    plotted_any = False

    for resolution in resolutions:
        color = next(colors)

        summary_path = find_summary(station, resolution)
        if summary_path is None:
            print(f"  [{station}/{component_label}] {resolution}: no summary.txt found, skipping.")
            continue

        pos_path = find_pos_file(station, resolution)
        if pos_path is None:
            print(f"  [{station}/{component_label}] {resolution}: no .pos file found, skipping.")
            continue

        parsed = parse_summary(summary_path)
        df = read_pos_file(pos_path)

        if col in ("dPerp", "dPara"):
            azimuths = get_station_azimuths()
            if station.upper() not in azimuths:
                print(f"  [{station}/{component_label}] {resolution}: no azimuth "
                      f"in '{VEL_FILE}' -- can't rotate raw data, skipping.")
                continue
            df = rotate_dataframe(df, azimuths[station.upper()])

        seg = parsed["segments"].get(segment_index)
        if seg is None:
            available = sorted(parsed["segments"].keys())
            print(f"  [{station}/{component_label}] {resolution}: no segment {segment_index} "
                  f"in summary.txt (available: {available}), skipping.")
            continue

        seg_df = slice_segment(df, seg["start"], seg["end"])
        if seg_df.empty:
            print(f"  [{station}/{component_label}] {resolution}: segment {segment_index} "
                  f"has no matching data rows, skipping.")
            continue

        # Raw data for this resolution/segment (only for the resolution(s)
        # selected in DATA_RESOLUTIONS_TO_PLOT).
        if resolution in data_resolutions:
            ax.scatter(
                seg_df["datetime"], seg_df[col],
                s=8, color=color, alpha=0.35,
                label=f"{resolution} data",
            )

        if col not in seg["components"]:
            # This segment/component wasn't fit (e.g. too few points).
            plotted_any = True
            continue

        steps = seg["components"][col]
        result = reconstruct_model(seg_df, col, steps)
        if result is None:
            plotted_any = True
            continue
        t_dense, y_dense, t_ticks, y_ticks = result
        dt_dense = MJD_EPOCH + pd.to_timedelta(t_dense, unit="D")

        ax.plot(
            dt_dense, y_dense,
            color=color, lw=2.2,
            label=f"{resolution} model",
        )

        if t_ticks.size:
            dt_ticks = MJD_EPOCH + pd.to_timedelta(t_ticks, unit="D")
            ax.plot(
                dt_ticks, y_ticks,
                marker="|", linestyle="none",
                color=color, markersize=TICK_MARKER_SIZE, markeredgewidth=2,
                zorder=5,
            )
        plotted_any = True

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{col} displacement (m)")
    ax.set_title(f"{station} — {component_label.capitalize()} component ({col}) — "
                 f"segment {segment_index} — model comparison across resolutions")
    ax.grid(alpha=0.3)
    if plotted_any:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, f"{station}_{component_label}_segment{segment_index}_resolution_comparison.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    comps = components_to_plot(COMPONENT)

    unknown = set(DATA_RESOLUTIONS_TO_PLOT) - set(TIME_RESOLUTIONS)
    if unknown:
        raise ValueError(
            f"DATA_RESOLUTIONS_TO_PLOT {sorted(unknown)} not in TIME_RESOLUTIONS "
            f"{TIME_RESOLUTIONS} -- add them there too if you want their models plotted."
        )

    print(f"Comparing resolutions {TIME_RESOLUTIONS} (models) / "
          f"{DATA_RESOLUTIONS_TO_PLOT} (data shown), segment {SEGMENT_INDEX}, "
          f"components {comps}, for stations: {STATIONS}")
    for station in STATIONS:
        for comp_label in comps:
            plot_station_component(
                station, comp_label, TIME_RESOLUTIONS, DATA_RESOLUTIONS_TO_PLOT,
                OUTPUT_DIR, SEGMENT_INDEX,
            )


if __name__ == "__main__":
    main()