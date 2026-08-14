"""
Reads PBO-style GNSS station position (.pos) files into pandas DataFrames.

Usage:
    python read_pos_files.py /path/to/folder

Or import and use directly:

    from read_pos_files import read_pos_file, read_pos_folder

    df = read_pos_file("ALBH.mit.final_nam20.pos")
    all_stations = read_pos_folder("./gps_for_dash/24h/20230608")
"""

import glob
import os
import sys

import pandas as pd

# Column names, in order, as defined in the PBO .pos header
COLUMNS = [
    "YYYYMMDD", "HHMMSS", "MJD",
    "X", "Y", "Z", "Sx", "Sy", "Sz", "Rxy", "Rxz", "Ryz",
    "NLat", "Elong", "Height",
    "dN", "dE", "dU", "Sn", "Se", "Su", "Rne", "Rnu", "Reu",
    "Soln",
]


def read_pos_file(filepath):
    meta = {}
    data_start = None

    with open(filepath) as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Grab a few useful metadata fields from the header
        if stripped.startswith("4-character ID"):
            meta["station"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Station name"):
            meta["station_name"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("NEU Reference position"):
            neu_ref = stripped.split(":", 1)[1].strip()
            meta["neu_reference"] = neu_ref
            # neu_ref looks like: "48.3897810428  236.5125295037   31.75148 (NAM14/WGS84)"
            parts = neu_ref.replace("(", " ").split()
            try:
                meta["latitude"] = float(parts[0])
                meta["longitude"] = float(parts[1])
                meta["height"] = float(parts[2])
            except (IndexError, ValueError):
                pass
        elif stripped.startswith("XYZ Reference position"):
            meta["xyz_reference"] = stripped.split(":", 1)[1].strip()

        # The line starting with '*YYYYMMDD' marks the column header;
        # actual data starts on the next line.
        if stripped.startswith("*YYYYMMDD"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data header ('*YYYYMMDD...') in {filepath}")

    df = pd.read_csv(
        filepath,
        skiprows=data_start,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    df["datetime"] = pd.to_datetime(
        df["YYYYMMDD"].astype(str) + df["HHMMSS"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
    )

    df.attrs.update(meta)
    df.attrs["source_file"] = filepath
    if "station" in meta:
        df["station"] = meta["station"]
    if "latitude" in meta:
        df["latitude"] = meta["latitude"]
        df["longitude"] = meta["longitude"]

    return df


def describe_gaps(df, time_col="datetime", top_n=15):
    df = df.sort_values(time_col).reset_index(drop=True)
    gap_hours = df[time_col].diff().dt.total_seconds() / 3600.0

    gaps = pd.DataFrame({
        "gap_hours": gap_hours,
        "before": df[time_col].shift(1),
        "after": df[time_col],
    }).dropna().sort_values("gap_hours", ascending=False)

    print(gaps.head(top_n).to_string(index=False))
    return gaps.head(top_n)


def split_on_gaps(df, max_gap_hours=None, n_segments=None, time_col="datetime"):
    df = df.sort_values(time_col).reset_index(drop=True)
    gap_hours = df[time_col].diff().dt.total_seconds() / 3600.0

    if n_segments is not None:
        n_splits = max(int(n_segments) - 1, 0)
        if n_splits == 0 or len(df) <= 1:
            return [df]

        # Row indices where the n_splits largest gaps occur (a new segment
        # starts AT that row, since gap_hours[i] is the gap before row i).
        split_positions = set(gap_hours.nlargest(n_splits).index)

        segment_id = pd.Series(0, index=df.index)
        current = 0
        for i in range(1, len(df)):
            if i in split_positions:
                current += 1
            segment_id.iloc[i] = current

    else:
        if max_gap_hours is None:
            max_gap_hours = 24
        segment_id = (gap_hours > max_gap_hours).cumsum()

    segments = [
        seg.reset_index(drop=True)
        for _, seg in df.groupby(segment_id)
    ]

    return segments


def read_pos_folder(folder, pattern="*.pos", recursive=False):
    abs_folder = os.path.abspath(folder)

    if not os.path.isdir(abs_folder):
        print(f"'{abs_folder}' is not a directory (check the path / your current working directory).")
        return {}

    search_pattern = os.path.join(abs_folder, "**", pattern) if recursive else os.path.join(abs_folder, pattern)
    fnames = sorted(glob.glob(search_pattern, recursive=recursive))

    if not fnames:
        all_files = glob.glob(os.path.join(abs_folder, "**", "*"), recursive=True) if recursive \
            else glob.glob(os.path.join(abs_folder, "*"))
        ext = pattern.lstrip("*").lower()
        fnames = sorted(f for f in all_files if os.path.isfile(f) and f.lower().endswith(ext))

    if not fnames:
        contents = os.listdir(abs_folder)[:10]
        print(f"No files matching '{pattern}' found in {abs_folder}")
        print(f"First few items actually in that folder: {contents}")
        return {}

    stations = {}
    for fname in fnames:
        try:
            df = read_pos_file(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        station = df.attrs.get("station") or os.path.splitext(os.path.basename(fname))[0]

        # Append the GPS coordinates to the key, e.g. "ALBH_48.3898_236.5125",
        # so the station's location is visible/accessible straight from the key.
        if "latitude" in df.attrs and "longitude" in df.attrs:
            key = f"{station}_{df.attrs['latitude']:.4f}_{df.attrs['longitude']:.4f}"
        else:
            key = station

        stations[key] = df
        print(f"Loaded {key}: {len(df)} epochs from {os.path.basename(fname)}")

    return stations


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pos_files.py /path/to/folder [--recursive]")
        sys.exit(1)

    folder = sys.argv[1]
    recursive = "--recursive" in sys.argv[2:]

    stations = read_pos_folder(folder, recursive=recursive)

    print(f"\nLoaded {len(stations)} station(s).")
    for station, df in stations.items():
        print(f"\n{station}:")
        print(df[["datetime", "dN", "dE", "dU"]].head())