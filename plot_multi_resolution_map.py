"""
Plots the vectors of motion from SSEs on a map

python plot_multi_resolution_map.py tanh_fit_results_06h tanh_fit_results_08h tanh_fit_results_12h tanh_fit_results_24h
"""

import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from gnss_timeseries_manipulation.processing import hampel_identifier
from rotate_ts import read_vel_file, station_azimuths, perp_para_to_en
 
 
SEGMENT_INDEX = 0 # If there's multiple segments in the data/summary.txt files

SCALE = 1.5
 
ARROW_HEADWIDTH = 1
ARROW_HEADLENGTH = 1.0
ARROW_HEADAXISLENGTH = .8
 
DRAW_ERROR_ELLIPSES = False
 
POS_DATA_ROOT = "gps_for_dash" # Change to your folder
POS_SUBFOLDER = "20230608"
 
POS_INTERVAL = "08h"
POS_INTERVALS = None 
 
 
# 2.44 ~ 95%
ELLIPSE_N_SIGMA = 1.5
ELLIPSE_ALPHA = 0.25
 
ELLIPSE_EXAGGERATION = 1
 
#Excludes unrealistic sigma values caused by outliers
MAX_REASONABLE_SIGMA_M = 1
 
#Outlier removal
HAMPEL_WINDOW = 11
OFFSET_SIGMA = 4.0
 
DRAW_DURATION_BARS = True
 
DURATION_AGG = "max"

DURATION_BAR_WIDTH_DEG = 0.10
DURATION_BAR_HEIGHT_DEG = 0.10
 
DURATION_BAR_OFFSET = (1.2, -1.0)
 
DEFAULT_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

VEL_FILE = "loading.vel"
 
EN_COMPONENTS = ("dE", "dN")
ROTATED_COMPONENTS = ("dPerp", "dPara")
 
_STATION_AZIMUTHS = None
 
 
def get_station_azimuths():
    """Lazily load + cache the per-station azimuth table from VEL_FILE."""
    global _STATION_AZIMUTHS
    if _STATION_AZIMUTHS is None:
        try:
            _STATION_AZIMUTHS = station_azimuths(read_vel_file(VEL_FILE))
        except FileNotFoundError:
            print(f"Warning: '{VEL_FILE}' not found -- dPerp/dPara amplitudes "
                  "can't be converted back to east/north.")
            _STATION_AZIMUTHS = pd.Series(dtype=float)
    return _STATION_AZIMUTHS
 
 
def resolve_en_vector(parsed, station, component_pair, agg="sum",
                       segment_index=SEGMENT_INDEX, azimuths=None):
    def combine(comp):
        steps = parsed["amplitudes"].get(comp, [])
        if segment_index is not None:
            steps = [s for s in steps if s[0] == segment_index]
        if not steps:
            return 0.0
        amps = [amp for (_, amp, _, _) in steps]
        return float(np.sum(amps)) if agg == "sum" else max(amps, key=abs)

    en = {"dE": 0.0, "dN": 0.0}
    rot = {"dPerp": 0.0, "dPara": 0.0}

    for comp in component_pair:
        if comp in en:
            en[comp] = combine(comp)
        elif comp in rot:
            rot[comp] = combine(comp)
        else:
            raise ValueError(
                f"Unknown component {comp!r} in component_pair; expected "
                "some combination of 'dE', 'dN', 'dPerp', 'dPara'."
            )

    ve, vn = en["dE"], en["dN"]

    if rot["dPerp"] or rot["dPara"]:
        azi = None if azimuths is None else azimuths.get(station.upper())
        if azi is None:
            print(f"  {station}: has dPerp/dPara steps but no azimuth "
                  f"available -- skipping that contribution.")
        else:
            e_rot, n_rot = perp_para_to_en(rot["dPerp"], rot["dPara"], azi)
            ve += e_rot
            vn += n_rot

    return ve, vn
 
_FLOAT = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_LAT_RE = re.compile(rf"^Latitude:\s*({_FLOAT})")
_LON_RE = re.compile(rf"^Longitude:\s*({_FLOAT})")
_SEGMENT_RE = re.compile(r"^Segment\s+(\d+):\s*(.+?)\s*->\s*(.+?)\s*\(\d+ points\)")
_COMPONENT_RE = re.compile(r"^\s*Component:\s*(\S+)")
_STEP_RE = re.compile(
    rf"duration=({_FLOAT})\s*days\s*amplitude=({_FLOAT})\s*\+/-\s*({_FLOAT})"
)
 
MJD_EPOCH = pd.Timestamp("1858-11-17")
 
 
def parse_summary(summary_path):
    lat = lon = None
    amplitudes = {}
    segments = {}
    current_component = None
    current_segment = None
 
    with open(summary_path) as fp:
        for line in fp:
            m = _LAT_RE.match(line)
            if m:
                lat = float(m.group(1))
                continue
            m = _LON_RE.match(line)
            if m:
                lon = float(m.group(1))
                if lon > 180:
                    lon -= 360
                continue
            m = _SEGMENT_RE.match(line)
            if m:
                current_segment = int(m.group(1))
                try:
                    start_ts = pd.to_datetime(m.group(2))
                    end_ts = pd.to_datetime(m.group(3))
                    segments[current_segment] = (start_ts, end_ts)
                except (ValueError, TypeError):
                    pass
                continue
            m = _COMPONENT_RE.match(line)
            if m:
                current_component = m.group(1)
                amplitudes.setdefault(current_component, [])
                continue
            m = _STEP_RE.search(line)
            if m and current_component is not None:
                duration = float(m.group(1))
                amp = float(m.group(2))
                amp_std = float(m.group(3))
                amplitudes[current_component].append(
                    (current_segment, amp, amp_std, duration)
                )
 
    return {"lat": lat, "lon": lon, "amplitudes": amplitudes, "segments": segments}
 
 
def detect_component_pair(folders):
    """
    Scans summary.txt files (stopping as soon as one has a recognized
    horizontal component) and returns, in priority order:
      ("dPerp", "dPara") if both are present
      ("dPerp",)         if only dPerp is present
      ("dPara",)         if only dPara is present
      ("dE", "dN")       if dE/dN are present (and no dPerp/dPara)
    """
    if isinstance(folders, str):
        folders = [folders]

    found = set()
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "*", "summary.txt"))):
            parsed = parse_summary(path)
            found |= set(parsed["amplitudes"]) & {"dE", "dN", "dPerp", "dPara"}
            if found:
                break
        if found:
            break

    if {"dPerp", "dPara"} <= found:
        return ("dPerp", "dPara")
    if "dPerp" in found:
        return ("dPerp",)
    if "dPara" in found:
        return ("dPara",)
    if found & {"dE", "dN"}:
        return tuple(c for c in ("dE", "dN") if c in found)

    raise ValueError(
        f"Couldn't auto-detect a horizontal component_pair from any "
        f"summary.txt under {folders!r} -- pass component_pair explicitly "
        "(e.g. component_pair=('dE', 'dN') or ('dPerp',))."
    )
 
 
def segment_time_range(parsed, segment_index):
    segments = parsed.get("segments", {})
    if not segments:
        return None
    if segment_index is not None:
        return segments.get(segment_index)
    starts = [s for s, _ in segments.values()]
    ends = [e for _, e in segments.values()]
    return min(starts), max(ends)
 
 
def resolve_pos_path(station, pos_data_root, interval, date):
    """Builds the expected .pos path for one station/interval/date."""
    return os.path.join(pos_data_root, interval, date, f"{station}.mit.final_nam20.pos")
 
 
def load_pos_sigmas(pos_path):
    from read_pos_files import read_pos_file
 
    df = read_pos_file(pos_path)
    for col in ("MJD", "Sn", "Se", "Rne"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    return df
 
 
def average_sigma_for_window(pos_df, start_ts, end_ts, trim_pct=0.05):
    mjd0 = (start_ts - MJD_EPOCH).total_seconds() / 86400.0
    mjd1 = (end_ts - MJD_EPOCH).total_seconds() / 86400.0
    window = pos_df[(pos_df["MJD"] >= mjd0) & (pos_df["MJD"] <= mjd1)]
    window = window.dropna(subset=["Sn", "Se", "Rne"])
    if window.empty:
        return None
 
    def trimmed_mean(s, pct):
        if pct <= 0 or len(s) < 3:
            return float(s.mean())
        lo, hi = s.quantile(pct), s.quantile(1 - pct)
        clipped = s[(s >= lo) & (s <= hi)]
        return float(clipped.mean()) if not clipped.empty else float(s.mean())
 
    sigma_n = trimmed_mean(window["Sn"], trim_pct)
    sigma_e = trimmed_mean(window["Se"], trim_pct)
    rne = trimmed_mean(window["Rne"], trim_pct)
    return sigma_n, sigma_e, rne
 
 
def error_ellipse_geometry(sigma_n, sigma_e, rne, n_sigma=ELLIPSE_N_SIGMA):
    """
    Builds a 2D error ellipse from (sigma_north, sigma_east, correlation).
    """
    cov = np.array([[sigma_e ** 2, rne * sigma_e * sigma_n],
                     [rne * sigma_e * sigma_n, sigma_n ** 2]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = np.clip(eigvals, 0, None)
 
    width = 2.0 * n_sigma * np.sqrt(eigvals[0])
    height = 2.0 * n_sigma * np.sqrt(eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle
 
 
def collect_station_vectors(folder, component_pair=("dE", "dN"), agg="sum",
                             segment_index=SEGMENT_INDEX):
    needs_azimuths = any(c in ROTATED_COMPONENTS for c in component_pair)
    azimuths = get_station_azimuths() if needs_azimuths else None
    stations = {}
 
    summary_paths = sorted(glob.glob(os.path.join(folder, "*", "summary.txt")))
    for path in summary_paths:
        station = os.path.basename(os.path.dirname(path))
        parsed = parse_summary(path)
        if parsed["lat"] is None or parsed["lon"] is None:
            continue
 
        ve, vn = resolve_en_vector(parsed, station, component_pair, agg=agg,
                                    segment_index=segment_index, azimuths=azimuths)
        if ve == 0.0 and vn == 0.0:
            continue
        if abs(ve) > 1:
            ve = 0
        if abs(vn) > 1:
            vn = 0
 
        stations[station] = (parsed["lat"], parsed["lon"], ve, vn)
      
    return stations
 
 
def collect_station_ellipses(folder, interval, date, segment_index,
                              pos_data_root=POS_DATA_ROOT,
                              n_sigma=ELLIPSE_N_SIGMA,
                              max_reasonable_sigma_m=MAX_REASONABLE_SIGMA_M):
    ellipses = {}
    summary_paths = sorted(glob.glob(os.path.join(folder, "*", "summary.txt")))
    for path in summary_paths:
        station = os.path.basename(os.path.dirname(path))
        parsed = parse_summary(path)
        time_range = segment_time_range(parsed, segment_index)
        if time_range is None:
            continue
        start_ts, end_ts = time_range
 
        pos_path = resolve_pos_path(station, pos_data_root, interval, date)
        pos_df = load_pos_sigmas(pos_path)
        if pos_df is None:
            continue
 
        sigma_stats = average_sigma_for_window(pos_df, start_ts, end_ts)
        if sigma_stats is None:
            continue
        sigma_n, sigma_e, rne = sigma_stats
 
        if sigma_n > max_reasonable_sigma_m or sigma_e > max_reasonable_sigma_m:
            continue
 
        width, height, angle = error_ellipse_geometry(sigma_n, sigma_e, rne, n_sigma=n_sigma)
        ellipses[station] = (width, height, angle)
 
    return ellipses
 
 
def collect_station_durations(folder, component_pair=("dE", "dN"),
                               segment_index=SEGMENT_INDEX, agg=DURATION_AGG):
    durations = {}
 
    summary_paths = sorted(glob.glob(os.path.join(folder, "*", "summary.txt")))
    for path in summary_paths:
        station = os.path.basename(os.path.dirname(path))
        parsed = parse_summary(path)
 
        steps = []
        for comp in component_pair:
            comp_steps = parsed["amplitudes"].get(comp, [])
            if segment_index is not None:
                comp_steps = [s for s in comp_steps if s[0] == segment_index]
            steps.extend(comp_steps)
 
        if not steps:
            continue
 
        if agg == "max":
            dur = max(dur for (_, _, _, dur) in steps)
        elif agg == "mean":
            dur = float(np.mean([dur for (_, _, _, dur) in steps]))
        else:  # "largest_amplitude"
            dur = max(steps, key=lambda s: abs(s[1]))[3]
 
        durations[station] = dur
 
    return durations
 
 
_INTERVAL_IN_NAME_RE = re.compile(r"(\d{1,2}h)\b")
 
 
def _infer_interval_from_folder(folder):
    m = _INTERVAL_IN_NAME_RE.search(os.path.basename(os.path.normpath(folder)))
    return m.group(1) if m else None
 
 
def _pos_interval_for(i, pos_interval, pos_intervals, folder=None):
    if pos_intervals is not None and i < len(pos_intervals):
        return pos_intervals[i]
    if folder is not None:
        inferred = _infer_interval_from_folder(folder)
        if inferred is not None:
            return inferred
    return pos_interval
def add_scale_reference(ax, quiver_obj, ref_mm=(5, 10), x0=0.06, y0=0.05,
                         dy=0.035, color="k"):
    for i, mm in enumerate(ref_mm):
        U = mm / 1000.0
        ax.quiverkey(quiver_obj, X=x0, Y=y0 + i * dy, U=U,
                     label=f"{mm} mm", labelpos="E", coordinates="axes",
                     color=color, fontproperties={"size": 8})
 
 
def draw_duration_bars(ax, x0, y0, values, colors, width=DURATION_BAR_WIDTH_DEG,
                        height=DURATION_BAR_HEIGHT_DEG, max_value=None,
                        gap_frac=0.15, baseline=True, zorder=60, transform=None):
    n = len(values)
    if n == 0:
        return
 
    finite_vals = [v for v in values if v is not None]
    if not finite_vals:
        return
    if max_value is None or max_value <= 0:
        max_value = max(finite_vals)
 
    bar_w = width / n * (1 - gap_frac)
    gap = width / n * gap_frac / max(n - 1, 1) if n > 1 else 0
 
    if baseline:
        base_kwargs = dict(xy=(x0, y0), width=width, height=height * 0.0 + height * 0.02,
                            facecolor="0.3", edgecolor="none", alpha=0.6, zorder=zorder)
        if transform is not None:
            base_kwargs["transform"] = transform
        ax.add_patch(mpatches.Rectangle(**base_kwargs))
 
    for i, (val, color) in enumerate(zip(values, colors)):
        if val is None:
            continue
        h = height * min(val / max_value, 1.0)
        x = x0 + i * (bar_w + gap)
        rect_kwargs = dict(xy=(x, y0), width=bar_w, height=h,
                            facecolor=color, edgecolor="k", linewidth=0.3,
                            zorder=zorder + 1)
        if transform is not None:
            rect_kwargs["transform"] = transform
        ax.add_patch(mpatches.Rectangle(**rect_kwargs))
 
    # thin outline box so the chart reads as a discrete little widget
    box_kwargs = dict(xy=(x0, y0), width=width, height=height, fill=False,
                       edgecolor="0.3", linewidth=0.4, zorder=zorder + 2)
    if transform is not None:
        box_kwargs["transform"] = transform
    ax.add_patch(mpatches.Rectangle(**box_kwargs))
 
 
def add_duration_bar_legend(ax, colors, labels, width=DURATION_BAR_WIDTH_DEG,
                             loc="lower right"):
    ax.text(
        0.995, 0.005 if loc == "lower right" else 0.995,
        "bars: event duration by dataset, RELATIVE to that station's "
        "longest-duration fit (tallest bar = that station's max)",
        transform=ax.transAxes, ha="right",
        va="bottom" if loc == "lower right" else "top",
        fontsize=7, color="0.3",
    )
 
 
def plot_multi_resolution_map(folders, labels=None, colors=None,
                               component_pair=None, agg="sum",
                               segment_index=SEGMENT_INDEX,
                               scale=None, arrow_width=0.005, ax=None,
                               annotate_stations=True, legend=True,
                               use_cartopy=True, show=True,
                               draw_error_ellipses=DRAW_ERROR_ELLIPSES,
                               pos_data_root=POS_DATA_ROOT,
                               pos_interval=POS_INTERVAL, pos_intervals=POS_INTERVALS,
                               pos_subfolder=POS_SUBFOLDER,
                               ellipse_n_sigma=ELLIPSE_N_SIGMA,
                               ellipse_alpha=ELLIPSE_ALPHA,
                               ellipse_exaggeration=ELLIPSE_EXAGGERATION,
                               max_reasonable_sigma_m=MAX_REASONABLE_SIGMA_M,
                               draw_duration_bars_flag=DRAW_DURATION_BARS,
                               duration_agg=DURATION_AGG,
                               duration_bar_width=DURATION_BAR_WIDTH_DEG,
                               duration_bar_height=DURATION_BAR_HEIGHT_DEG,
                               duration_bar_offset=DURATION_BAR_OFFSET):
    """
    Parameters
    ----------
    folders : list of str
        output_root directories from fitting_tanh_function.py, each
        containing <STATION>/summary.txt subfolders.
    labels : list of str, optional
        Legend label per folder/dataset. Defaults to each folder's name.
    colors : list, optional
        One color per folder/dataset. Defaults to a qualitative palette.
    component_pair : tuple of summary.txt component names, optional
        Which components form the plotted (E, N) vector. If None
        (the default), auto-detected from the summary.txt files
        themselves via detect_component_pair() -- so this works
        whether the folders were produced with HORIZONTAL_MODE="en",
        "perp", or "perp_para" in fitting_tanh_function.py, without
        needing to know that in advance. Explicit values accepted:
        ("dE", "dN"), ("dPerp", "dPara"), ("dPerp",), or ("dPara",);
        if dPerp/dPara are used, amplitudes are rotated back to
        east/north using each station's azimuth from VEL_FILE before
        plotting (see resolve_en_vector). Pass e.g. ("dE", "dU") for a
        vertical-vs-east view instead.
    agg : "sum" or "largest"
        How multiple qualifying steps at a station are combined into one
        vector -- see collect_station_vectors.
    segment_index : None or int
        Which segment's steps to use -- defaults to the module-level
        SEGMENT_INDEX constant at the top of this file. Pass None to pool
        steps across every segment instead of restricting to one.
    scale : float, optional
        Quiver length scale (data-units per arrow-length-unit, since
        angles/scale_units="xy" is used). If None, auto-picked so the
        single largest vector in the whole plot spans ~15% of the
        latitude range -- pass your own value to keep scale consistent
        across separately-generated figures.
    arrow_width : float
        Passed straight to quiver's `width`.
    ax : matplotlib (Geo)Axes, optional
        Existing axes to draw on. If use_cartopy=True this should be a
        cartopy GeoAxes; a new one is created if not supplied.
    annotate_stations : bool
        Label each station with its name.
    legend : bool
        Draw a color legend mapping dataset -> folder/label.
    use_cartopy : bool
        Try to draw coastlines/borders with cartopy if it's installed.
        Falls back to a plain longitude/latitude scatter axes if cartopy
        isn't available.
    show : bool
        Call plt.show() before returning (default True) so the figure
        pops up interactively. Set False if you'd rather just get the
        fig/ax back (e.g. to save it yourself) without blocking.
    draw_error_ellipses : bool
        Defaults to the module-level DRAW_ERROR_ELLIPSES constant. If
        True, draws a semi-transparent error ellipse (from the raw .pos
        files' Sn/Se/Rne columns, averaged over the selected segment's
        time window) around each vector's tip, in that dataset's color.
    pos_data_root, pos_interval, pos_intervals, pos_subfolder, pos_subfolders :
        Where to find the original .pos files -- see the POS_* constants
        at the top of this file. pos_intervals/pos_subfolders (lists, one per
        folder) take priority over the singular pos_interval/pos_subfolder
        when given.
    ellipse_n_sigma, ellipse_alpha, ellipse_exaggeration :
        Ellipse size (in multiples of sigma), fill transparency, and a
        purely-visual size multiplier on top of that (since real GPS
        sigmas are often sub-pixel next to displacement vectors) --
        see ELLIPSE_N_SIGMA / ELLIPSE_ALPHA / ELLIPSE_EXAGGERATION at the
        top of this file.
    max_reasonable_sigma_m : float
        Stations whose averaged Sn/Se exceeds this (meters) are treated
        as bad .pos data and skipped rather than drawn -- see
        MAX_REASONABLE_SIGMA_M at the top of this file.
    draw_duration_bars_flag : bool
        Defaults to the module-level DRAW_DURATION_BARS constant. If
        True, draws a tiny bar-chart next to each station showing the
        fitted event DURATION (in days, from each dataset's summary.txt)
        as one bar per folder/dataset, colored to match that dataset --
        so you can see at a glance whether e.g. the 06h/08h/12h/24h tanh
        models agree on how long the event took. Each station's chart is
        scaled to ITS OWN longest bar (that dataset's duration fills the
        whole chart height); the other bars in that same chart show their
        duration relative to it. This is a RELATIVE, not absolute, scale
        -- bar heights are only comparable within one station's chart,
        not across different stations.
    duration_agg : "largest_amplitude", "max", or "mean"
        How a station's per-dataset duration is picked when more than one
        component (of component_pair) has a qualifying step -- see
        collect_station_durations. Defaults to DURATION_AGG.
    duration_bar_width, duration_bar_height : float
        Size of each station's bar-chart box, in degrees (lon/lat) --
        see DURATION_BAR_WIDTH_DEG / DURATION_BAR_HEIGHT_DEG.
    duration_bar_offset : (float, float)
        Where each bar-chart sits relative to its station, in units of
        (duration_bar_width, duration_bar_height) -- see
        DURATION_BAR_OFFSET at the top of this file.
 
    Returns
    -------
    fig, ax
    """
    if labels is None:
        labels = [os.path.basename(os.path.normpath(f)) for f in folders]
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(folders))]
    if not (len(folders) == len(labels) == len(colors)):
        raise ValueError("folders, labels, and colors must be the same length")

    if component_pair is None:
        component_pair = detect_component_pair(folders)
        print(f"Auto-detected component_pair={component_pair!r} from summary.txt files.")
 
    per_dataset = [
        collect_station_vectors(f, component_pair, agg=agg, segment_index=segment_index)
        for f in folders
    ]
    per_dataset_ellipses = [{} for _ in folders]
    ellipse_coverage_lines = []
    if draw_error_ellipses:
        for i, f in enumerate(folders):
            interval_i = _pos_interval_for(i, pos_interval, pos_intervals, folder=f)
            date_i = pos_subfolder
            per_dataset_ellipses[i] = collect_station_ellipses(
                f, interval_i, date_i, segment_index,
                pos_data_root=pos_data_root, n_sigma=ellipse_n_sigma,
                max_reasonable_sigma_m=max_reasonable_sigma_m,
            )
            label_i = labels[i] if labels else f
            n_vec = len(per_dataset[i])
            n_ell = len(set(per_dataset_ellipses[i]) & set(per_dataset[i]))
            line = (f"{label_i} ({interval_i}): {n_ell}/{n_vec} plotted vectors have ellipses")
            ellipse_coverage_lines.append(line)
            
    else:
        print("[draw ellipses is false")
 
    per_dataset_durations = [{} for _ in folders]
    if draw_duration_bars_flag:
        for i, f in enumerate(folders):
            per_dataset_durations[i] = collect_station_durations(
                f, component_pair, segment_index=segment_index, agg=duration_agg
            )
 
    all_stations = sorted(set().union(*[d.keys() for d in per_dataset])) if per_dataset else []
    if not all_stations:
        raise ValueError(
            "No stations with qualifying offsets found in the given folders "
            f"(segment_index={segment_index!r}). Try segment_index=None to pool "
            "across all segments."
        )
 
    all_entries = [(lat, lon, ve, vn) for d in per_dataset for (lat, lon, ve, vn) in d.values()]
    all_mags = [np.hypot(ve, vn) for (_, _, ve, vn) in all_entries]
    
 
    if scale is None:
        lats = [lat for (lat, lon, ve, vn) in all_entries]
        lat_span = (max(lats) - min(lats)) if len(lats) > 1 else 1.0
        ref_mag = np.percentile(all_mags, 95)
        scale = ref_mag / (0.15 * max(lat_span, 1e-6)) * SCALE
    
    print(scale)
    
    lon_pts = [lon for (lat, lon, ve, vn) in all_entries] + \
              [lon + ve / scale for (lat, lon, ve, vn) in all_entries]
    lat_pts = [lat for (lat, lon, ve, vn) in all_entries] + \
              [lat + vn / scale for (lat, lon, ve, vn) in all_entries]
    lon_min, lon_max = min(lon_pts), max(lon_pts)
    lat_min, lat_max = min(lat_pts), max(lat_pts)
    lon_pad = max((lon_max - lon_min) * 0.1, 0.05)
    lat_pad = max((lat_max - lat_min) * 0.1, 0.05)
    plot_extent = (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)
 
    proj = None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            proj = ccrs.PlateCarree()
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": proj})
            else:
                fig = ax.figure
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        except ImportError:
            print("cartopy not installed -- falling back to a plain lon/lat plot.", flush=True)
            use_cartopy = False
 
    if not use_cartopy:
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.figure
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
 
    quiver_kwargs = dict(angles="xy", scale_units="xy", scale=scale,
                          width=arrow_width, headwidth=ARROW_HEADWIDTH,
                          headlength=ARROW_HEADLENGTH,
                          headaxislength=ARROW_HEADAXISLENGTH)
    if proj is not None:
        quiver_kwargs["transform"] = proj
 
    # Layering smallest to largest vectors
    plotted_labels = set()
    for station in all_stations:
        entries = []  # (dataset_idx, lat, lon, ve, vn, magnitude)
        for i, d in enumerate(per_dataset):
            if station in d:
                lat, lon, ve, vn = d[station]
                entries.append((i, lat, lon, ve, vn, np.hypot(ve, vn)))
        if not entries:
            continue
 
        # Largest magnitude first -> plotted first -> ends up on the bottom
        entries.sort(key=lambda e: e[5], reverse=True)
 
        for rank, (i, lat, lon, ve, vn, mag) in enumerate(entries):
            zorder_base = 3 + rank * 2
            label = labels[i] if labels[i] not in plotted_labels else None
 
            if draw_error_ellipses and station in per_dataset_ellipses[i]:
                width_m, height_m, angle = per_dataset_ellipses[i][station]
                tip_lon, tip_lat = lon + ve / scale, lat + vn / scale
                ellipse_kwargs = dict(
                    xy=(tip_lon, tip_lat),
                    width=width_m / scale * ellipse_exaggeration,
                    height=height_m / scale * ellipse_exaggeration,
                    angle=angle, facecolor=colors[i], edgecolor=colors[i],
                    alpha=ellipse_alpha, zorder=zorder_base, linewidth=0.8,
                )
                if proj is not None:
                    ellipse_kwargs["transform"] = proj
                ax.add_patch(mpatches.Ellipse(**ellipse_kwargs))
 
            last_quiver = ax.quiver(lon, lat, ve, vn, color=colors[i], zorder=zorder_base + 1,
                         label=label, **quiver_kwargs)
            plotted_labels.add(labels[i])
 
        if annotate_stations:
            lat0, lon0 = entries[0][1], entries[0][2]
            txt_kwargs = dict(fontsize=7, ha="left", va="bottom", zorder=50)
            if proj is not None:
                txt_kwargs["transform"] = proj
            ax.text(lon0, lat0, f"  {station}", **txt_kwargs)
 
        if draw_duration_bars_flag:
            lat0, lon0 = entries[0][1], entries[0][2]
            station_durations = [per_dataset_durations[i].get(station) for i in range(len(folders))]
            if any(v is not None for v in station_durations):
                off_x, off_y = duration_bar_offset
                bar_x0 = lon0 + off_x * duration_bar_width
                bar_y0 = lat0 + off_y * duration_bar_height
                bar_kwargs = dict(zorder=60)
                if proj is not None:
                    bar_kwargs["transform"] = proj
                draw_duration_bars(
                    ax, bar_x0, bar_y0, station_durations, colors,
                    width=duration_bar_width, height=duration_bar_height,
                    **bar_kwargs,
                )
 
    if legend:
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(folders))]
        ax.legend(handles=handles, loc="best", fontsize=9)
 
    seg_label = "all segments" if segment_index is None else f"segment {segment_index}"
    ax.set_title(f"Offset vectors ({seg_label}): " + " vs ".join(labels))
 
    if proj is not None:
        ax.set_extent(plot_extent, crs=proj)
    else:
        ax.set_xlim(plot_extent[0], plot_extent[1])
        ax.set_ylim(plot_extent[2], plot_extent[3])
 
    if last_quiver is not None:
        add_scale_reference(ax, last_quiver, ref_mm=(5, 10))
 
    if draw_duration_bars_flag:
        add_duration_bar_legend(ax, colors, labels)
 
    fig.tight_layout()
 
    if show:
        plt.show()
 
    return fig, ax
 
 
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_multi_resolution_map.py <folder1> [folder2 ...]", flush=True)
        sys.exit(1)
    plot_multi_resolution_map(sys.argv[1:])