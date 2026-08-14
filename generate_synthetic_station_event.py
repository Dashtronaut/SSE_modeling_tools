"""
Uses the actual noise from a station and adds a synthetic tangent step to it


    <OUTPUT_ROOT>/<STATION>_synthetic/
        24h/20230608/<STATION>....pos     
        12h/20230608/<STATION>....pos       
        08h/20230608/<STATION>....pos      
        06h/20230608/<STATION>....pos
        ground_truth.txt                 
        <STATION>_perp_resolution_comparison.png


    python generate_synthetic_station_event.py
    python generate_synthetic_station_event.py P403 --duration-days 8 --amplitude-mm -6

"""

import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_pos_files import read_pos_file, COLUMNS
from rotate_ts import read_vel_file, station_azimuths, rotate_dataframe, perp_para_to_en
from tanh_step_fit import tanh_basis
from gnss_timeseries_manipulation.processing import hampel_identifier

from fitting_tanh_function import greedy_multi_step_search


STATION = "ALBH"

GPS_DATA_ROOT = "gps_for_dash"
DATA_DATE = "20230608"
RESOLUTIONS = ["24h", "12h", "08h", "06h"]
VEL_FILE = "loading.vel"

OUTPUT_ROOT = "synthetic_events"


EVENT_CENTER_DATE = None
EVENT_DURATION_DAYS = 10.0
EVENT_AMPLITUDE_M = -0.006

NOISE_WINDOW_START = "2013-08-01"
NOISE_WINDOW_END = "2013-11-01"

AUTO_NOISE_WINDOW = True
NOISE_RESULTS_RESOLUTION = "08h"
TANH_RESULTS_ROOT_TEMPLATE = "tanh_fit_results_{res}"

NOISE_MARGIN_DAYS = 5.0

NOISE_SEGMENT = 0
MIN_QUIET_WINDOW_DAYS = 30.0

# Outlier flagging, matching fitting_tanh_function.py's own settings.
HAMPEL_WINDOW = 11
OFFSET_SIGMA = 4.0

MJD_EPOCH = pd.Timestamp("1858-11-17")


_FLOAT = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_SEGMENT_RE = re.compile(r"^Segment\s+(\d+):\s*(.+?)\s*->\s*(.+?)\s*\((\d+) points\)")
_COMPONENT_RE = re.compile(r"^\s*Component:\s*(\S+)")
_STEP_RE = re.compile(
    rf"^\s*Step\s+(\d+):\s*start=(\S+\s+\S+)\s+end=(\S+\s+\S+)\s+"
    rf"duration=({_FLOAT})\s*days\s+amplitude=({_FLOAT})\s*\+/-\s*({_FLOAT})\s*m"
)


def parse_tanh_summary(path):
    """Parse a fitting_tanh_function.py summary.txt into
    {seg_idx: {"start", "end", "components": {comp: [step, ...]}}}.
    """
    segments = {}
    current_seg = None
    current_comp = None

    with open(path) as fp:
        for line in fp:
            m = _SEGMENT_RE.match(line)
            if m:
                current_seg = int(m.group(1))
                segments[current_seg] = {
                    "start": pd.to_datetime(m.group(2)),
                    "end": pd.to_datetime(m.group(3)),
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
                })
                continue

    return segments


def find_quiet_windows(station, resolution=NOISE_RESULTS_RESOLUTION,
                        margin_days=NOISE_MARGIN_DAYS,
                        min_length_days=MIN_QUIET_WINDOW_DAYS,
                        segment_idx=NOISE_SEGMENT):
    """Find event-free stretches for `station` from an already-run
    fitting_tanh_function.py results folder (tanh_fit_results_{resolution}).

    A step chosen in ANY component (dN, dE, or dU) marks that window
    (padded by `margin_days` on both sides) as "busy" -- the remaining gaps
    within each segment, if long enough, are candidate noise windows.

    Returns a list of dicts sorted longest-first:
        {"segment": int, "start": Timestamp, "end": Timestamp, "length_days": float}
    """
    results_root = TANH_RESULTS_ROOT_TEMPLATE.format(res=resolution)
    summary_path = os.path.join(results_root, station, "summary.txt")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"No summary.txt for station '{station}' at {summary_path} -- "
            f"run fitting_tanh_function.py on the {resolution} data for "
            f"this station first, or set AUTO_NOISE_WINDOW=False and "
            f"specify NOISE_WINDOW_START/END by hand."
        )

    segments = parse_tanh_summary(summary_path)

    if segment_idx is not None:
        if segment_idx not in segments:
            raise ValueError(
                f"segment {segment_idx} not found for '{station}' in "
                f"{summary_path} (available: {sorted(segments)})"
            )
        segments = {segment_idx: segments[segment_idx]}

    candidates = []
    for seg_idx, seg in segments.items():
        busy = []
        for comp, steps in seg["components"].items():
            for step in steps:
                busy.append((
                    step["start"] - pd.Timedelta(days=margin_days),
                    step["end"] + pd.Timedelta(days=margin_days),
                ))
        busy.sort()

        merged = []
        for b_start, b_end in busy:
            if merged and b_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b_end))
            else:
                merged.append((b_start, b_end))

        cursor = seg["start"]
        gaps = []
        for b_start, b_end in merged:
            if b_start > cursor:
                gaps.append((cursor, min(b_start, seg["end"])))
            cursor = max(cursor, b_end)
        if cursor < seg["end"]:
            gaps.append((cursor, seg["end"]))

        for g_start, g_end in gaps:
            length_days = (g_end - g_start).total_seconds() / 86400.0
            if length_days >= min_length_days:
                candidates.append({
                    "segment": seg_idx, "start": g_start, "end": g_end,
                    "length_days": length_days,
                })

    candidates.sort(key=lambda c: -c["length_days"])
    return candidates


def find_station_pos_file(resolution, station):
    """Locate the real .pos file for `station` at `resolution`, tolerant of
    exact filename suffix (".mit.final_nam20.pos" etc. varies by network
    solution)."""
    directory = os.path.join(GPS_DATA_ROOT, resolution, DATA_DATE)
    patterns = [f"{station}*.pos", f"{station.lower()}*.pos", f"{station.upper()}*.pos"]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No .pos file found for station '{station}' in {directory} "
        f"(tried patterns: {patterns})"
    )


def datetime_to_mjd(ts):
    return (pd.Timestamp(ts) - MJD_EPOCH).total_seconds() / 86400.0


def build_synthetic_series(pos_filepath, azimuth_deg, t0_mjd,
                            duration_days, amplitude_m,
                            noise_start, noise_end):
    """Read one real .pos file, rotate to perp/para, slice to the noise
    window, inject the synthetic tanh step into dPerp, rotate back to
    dE/dN.

    """
    with open(pos_filepath) as fp:
        raw_lines = fp.readlines()
    header_lines = []
    for line in raw_lines:
        header_lines.append(line)
        if line.strip().startswith("*YYYYMMDD"):
            break

    df = read_pos_file(pos_filepath)
    df = rotate_dataframe(df, azimuth_deg)  # adds dPerp, dPara

    if noise_start is not None or noise_end is not None:
        lo = pd.Timestamp(noise_start) if noise_start else df["datetime"].min()
        hi = pd.Timestamp(noise_end) if noise_end else df["datetime"].max()
        df = df[(df["datetime"] >= lo) & (df["datetime"] <= hi)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No data left in {pos_filepath} after slicing to "
            f"[{noise_start}, {noise_end}] -- check the noise window "
            f"against this station's actual record."
        )

    t_start = t0_mjd - duration_days / 2.0
    step = tanh_basis(df["MJD"].to_numpy(dtype=float), t_start, duration_days)
    dperp_synth = df["dPerp"].to_numpy(dtype=float) + amplitude_m * step
    dpara_real = df["dPara"].to_numpy(dtype=float)

    e_synth, n_synth = perp_para_to_en(dperp_synth, dpara_real, azimuth_deg)
    df["dE"] = e_synth
    df["dN"] = n_synth
    df["dPerp"] = dperp_synth

    return df, header_lines


def write_pos_file(out_path, df, header_lines):
    """Write df (as produced by build_synthetic_series) out as a valid
    .pos file: original header verbatim, then data rows in the same 25
    whitespace-separated columns read_pos_file() expects.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        fp.writelines(header_lines)
        for _, row in df.iterrows():
            fp.write(
                f"{int(row['YYYYMMDD']):08d} {int(row['HHMMSS']):06d} "
                f"{row['MJD']:.5f} "
                f"{row['X']:.5f} {row['Y']:.5f} {row['Z']:.5f} "
                f"{row['Sx']:.5f} {row['Sy']:.5f} {row['Sz']:.5f} "
                f"{row['Rxy']:.5f} {row['Rxz']:.5f} {row['Ryz']:.5f} "
                f"{row['NLat']:.9f} {row['Elong']:.9f} {row['Height']:.5f} "
                f"{row['dN']:.5f} {row['dE']:.5f} {row['dU']:.5f} "
                f"{row['Sn']:.5f} {row['Se']:.5f} {row['Su']:.5f} "
                f"{row['Rne']:.5f} {row['Rnu']:.5f} {row['Reu']:.5f} "
                f"{row['Soln']}\n"
            )

def fit_dperp(df):
    t = df["MJD"].to_numpy(dtype=float)
    y = hampel_identifier(df["dPerp"].to_numpy(dtype=float),
                           window=HAMPEL_WINDOW, k=OFFSET_SIGMA)
    mask = np.isfinite(y)
    t, y = t[mask], y[mask]
    result = greedy_multi_step_search(t, y, max_steps=1, verbose=False)
    return t, y, result


def plot_comparison(station, results_by_res, t0_mjd, duration_days, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"24h": "tab:blue", "12h": "tab:orange", "08h": "tab:green", "06h": "tab:red"}

    true_start = t0_mjd - duration_days / 2.0
    true_end = t0_mjd + duration_days / 2.0
    ax.axvspan(true_start, true_end, color="black", alpha=0.08,
               label=f"true event ({duration_days:.3g}d)")
    ax.axvline(t0_mjd, color="black", ls=":", lw=1)

    for res, (t, y, result) in results_by_res.items():
        color = colors.get(res, None)
        ax.scatter(t, y, s=8, alpha=0.15, color=color)
        best = result["best"]
        ax.plot(t, best["fitted"], color=color, lw=2, label=f"{res} model")
        if best["step_list"]:
            for (rt_start, rdur) in best["step_list"]:
                ax.axvline(rt_start + rdur / 2.0, color=color, ls="--", lw=1, alpha=0.6)
                print(f"  {res}: recovered t_start={rt_start:.3f} MJD, "
                      f"duration={rdur:.3g} d "
                      f"(true: t_start={true_start:.3f} MJD, duration={duration_days:.3g} d)")
        else:
            print(f"  {res}: NO STEP CHOSEN (best_k=0) -- model preferred "
                  f"no event over any tested step "
                  f"(true: t_start={true_start:.3f} MJD, duration={duration_days:.3g} d)")

    ax.set_xlabel("MJD")
    ax.set_ylabel("dPerp (m)")
    ax.set_title(f"{station} -- synthetic SSE recovery across resolutions "
                 f"(true duration={duration_days:.3g}d)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("station", nargs="?", default=STATION)
    ap.add_argument("--duration-days", type=float, default=EVENT_DURATION_DAYS)
    ap.add_argument("--amplitude-mm", type=float, default=EVENT_AMPLITUDE_M * 1000.0,
                     help="Perpendicular amplitude in mm (converted to m internally).")
    ap.add_argument("--center-date", default=EVENT_CENTER_DATE,
                     help="ISO date for the event center. If omitted, "
                          "automatically uses the midpoint of the chosen "
                          "quiet window.")
    ap.add_argument("--noise-start", default=NOISE_WINDOW_START)
    ap.add_argument("--noise-end", default=NOISE_WINDOW_END)
    ap.add_argument("--auto-noise-window", dest="auto_noise_window",
                     action="store_true", default=AUTO_NOISE_WINDOW)
    ap.add_argument("--no-auto-noise-window", dest="auto_noise_window",
                     action="store_false")
    ap.add_argument("--noise-results-resolution", default=NOISE_RESULTS_RESOLUTION)
    ap.add_argument("--noise-margin-days", type=float, default=NOISE_MARGIN_DAYS)
    ap.add_argument("--segment", type=int, default=NOISE_SEGMENT,
                     help="Restrict quiet-window search to this segment "
                          "number only (see summary.txt). Use this to "
                          "exclude a segment that looks quiet only because "
                          "of known data corruption, not a real absence of "
                          "events.")
    args = ap.parse_args()

    station = args.station.upper()
    duration_days = args.duration_days
    amplitude_m = args.amplitude_mm / 1000.0

    noise_start, noise_end = args.noise_start, args.noise_end
    if args.auto_noise_window:
        candidates = find_quiet_windows(
            station, resolution=args.noise_results_resolution,
            margin_days=args.noise_margin_days,
            segment_idx=args.segment,
        )
        if not candidates:
            sys.exit(
                f"AUTO_NOISE_WINDOW is on but no quiet window >= "
                f"{MIN_QUIET_WINDOW_DAYS}d was found for '{station}' in "
                f"{args.noise_results_resolution} results"
                f"{f' (segment {args.segment})' if args.segment is not None else ''}. "
                f"Either lower MIN_QUIET_WINDOW_DAYS/NOISE_MARGIN_DAYS, or "
                f"pass --no-auto-noise-window and set --noise-start/"
                f"--noise-end by hand."
            )
        print(f"Quiet-window candidates for {station} "
              f"(from {args.noise_results_resolution} results"
              f"{f', segment {args.segment} only' if args.segment is not None else ''}"
              f", longest first):")
        for c in candidates[:5]:
            print(f"  segment {c['segment']}: {c['start']:%Y-%m-%d} -> "
                  f"{c['end']:%Y-%m-%d}  ({c['length_days']:.0f} days)")
        chosen = candidates[0]
        noise_start = f"{chosen['start']:%Y-%m-%d}"
        noise_end = f"{chosen['end']:%Y-%m-%d}"
        print(f"Using: {noise_start} -> {noise_end} "
              f"(segment {chosen['segment']}, {chosen['length_days']:.0f} days)")

    if noise_start is None or noise_end is None:
        if args.center_date is None:
            sys.exit(
                "Can't auto-center: noise_start/noise_end are both None "
                "(whole-file mode), so there's no window to take a "
                "midpoint of. Either set --noise-start/--noise-end (or "
                "enable auto-detection), or pass --center-date explicitly."
            )
        t0_mjd = datetime_to_mjd(args.center_date)
        center_date_str = args.center_date
        window_lo_mjd, window_hi_mjd = -np.inf, np.inf
    else:
        window_lo_mjd = datetime_to_mjd(noise_start)
        window_hi_mjd = datetime_to_mjd(noise_end)
        if args.center_date is not None:
            t0_mjd = datetime_to_mjd(args.center_date)
            center_date_str = args.center_date
        else:
            t0_mjd = (window_lo_mjd + window_hi_mjd) / 2.0
            center_date_str = f"{MJD_EPOCH + pd.Timedelta(days=t0_mjd):%Y-%m-%d}"
            print(f"No --center-date given -- using quiet-window midpoint: "
                  f"{center_date_str}")

    true_start = t0_mjd - duration_days / 2.0
    true_end = t0_mjd + duration_days / 2.0
    if not (window_lo_mjd <= true_start and true_end <= window_hi_mjd):
        sys.exit(
            f"center-date={center_date_str} +/- duration/2 falls outside "
            f"the quiet window [{noise_start}, {noise_end}]. Pick a "
            f"--center-date inside that window (or drop --center-date to "
            f"auto-center), or choose a different "
            f"--noise-results-resolution/--segment."
        )

    azimuths = station_azimuths(read_vel_file(VEL_FILE))
    if station not in azimuths:
        sys.exit(f"No azimuth for station '{station}' in {VEL_FILE} -- can't rotate.")
    azimuth_deg = azimuths[station]

    out_station_dir = os.path.join(OUTPUT_ROOT, f"{station}_synthetic")
    os.makedirs(out_station_dir, exist_ok=True)

    print(f"Station {station}: azimuth={azimuth_deg:.2f} deg, "
          f"true event center={center_date_str} (MJD {t0_mjd:.3f}), "
          f"duration={duration_days:.3g} d, amplitude={amplitude_m * 1000:.2f} mm")

    results_by_res = {}
    for res in RESOLUTIONS:
        src_path = find_station_pos_file(res, station)
        df, header_lines = build_synthetic_series(
            src_path, azimuth_deg, t0_mjd, duration_days, amplitude_m,
            noise_start, noise_end,
        )

        out_path = os.path.join(out_station_dir, res, DATA_DATE,
                                 os.path.basename(src_path))
        write_pos_file(out_path, df, header_lines)
        print(f"  [{res}] {len(df)} pts, source={src_path} -> {out_path}")

        t, y, result = fit_dperp(df)
        results_by_res[res] = (t, y, result)

    truth_path = os.path.join(out_station_dir, "ground_truth.txt")
    with open(truth_path, "w") as fp:
        fp.write(f"station: {station}\n")
        fp.write(f"azimuth_deg: {azimuth_deg:.4f}\n")
        fp.write(f"event_center_date: {center_date_str}\n")
        fp.write(f"event_center_mjd: {t0_mjd:.4f}\n")
        fp.write(f"event_duration_days: {duration_days:.4f}\n")
        fp.write(f"event_t_start_mjd: {t0_mjd - duration_days / 2.0:.4f}\n")
        fp.write(f"event_t_end_mjd: {t0_mjd + duration_days / 2.0:.4f}\n")
        fp.write(f"event_amplitude_m: {amplitude_m:.5f}\n")
        fp.write(f"noise_window: {noise_start} -> {noise_end}\n")
    print(f"Ground truth -> {truth_path}")

    print("\nRecovered vs. true (per resolution):")
    plot_path = os.path.join(out_station_dir, f"{station}_perp_resolution_comparison.png")
    plot_comparison(station, results_by_res, t0_mjd, duration_days, plot_path)
    print(f"\nComparison plot -> {plot_path}")

    print(f"\nThese are also plain valid .pos files -- you can point the "
          f"standard fitting_tanh_function.py CLI at {out_station_dir} "
          f"(set EXPECTED_SEGMENTS=1 there first, since these are single, "
          f"gap-free synthetic segments) to get the usual dN/dE/dU "
          f"summary.txt/plots as a second comparison.")


if __name__ == "__main__":
    main()
