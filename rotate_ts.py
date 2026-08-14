"""
I originally miscoded a as the angle of strike, not the perpendicular vector, so everything might seem swapped, it was easier to switch the variable names than rewrite all the math

a is the azimuth, clockwise from north, of the trench-perpendicular axis

    / perpendicular \   / sin(a)  cos(a) 0 \ / e \
    |    parallel    | = | cos(a) -sin(a) 0 | | n |
    \       up       /   \   0       0    1 / \ u /

    perpendicular = e*sin(a) + n*cos(a)     (projection onto bearing a)
    parallel      = e*cos(a) - n*sin(a)     (projection onto bearing a+90)

Propagating the 2x2 east/north covariance matrix through the same
rotation gives:

    /        sig_perp^2        rho_pp*sig_perp*sig_para \ = / sin(a)  cos(a) \ /      sig_e^2       rho_en*sig_e*sig_n \ / sin(a) cos(a) \
    \ rho_pp*sig_para*sig_perp        sig_para^2        /   \ cos(a) -sin(a) / \ rho_ne*sig_n*sig_e      sig_n^2       / \ cos(a) -sin(a) /

    sig_perp = sqrt[sig_e^2*sin^2(a) + 2*rho_en*sig_e*sig_n*cos(a)*sin(a) + sig_n^2*cos^2(a)]
    sig_para = sqrt[sig_e^2*cos^2(a) - 2*rho_en*sig_e*sig_n*cos(a)*sin(a) + sig_n^2*sin^2(a)]
    rho_pp   = [(sig_e^2-sig_n^2)*cos(a)*sin(a) + rho_en*sig_e*sig_n*(cos^2(a)-sin^2(a))]/(sig_perp*sig_para)


    azi = atan(v_e / v_n) = atan2(v_e, v_n)   [degrees, clockwise from north]


1. read_vel_file()      -- parse a TSFIT/HECTOR-style .vel file (e.g.
                            loading.vel) into a DataFrame, one row per
                            station.
2. station_azimuths()   -- azi = atan2(Evel, Nvel) for every station
                            in that DataFrame.
3. rotate_en()           -- the core vectorized rotation (scalars or
                            numpy arrays/Series in, arrays out).
4. rotate_dataframe()    -- wrapper for a station DataFrame such as
                            the one returned by
                            read_pos_files.read_pos_file(): rotates
                            the dE/dN (+ Se/Sn/Rne) columns and
                            appends dPerp/dPara (+ Sperp/Spara/Rpp).
5. rotate_station_file() -- end-to-end: read one .pos file, look up
                            its station's azimuth, rotate.
6. process_folder()      -- batch version, used by the CLI below.


    python sh_rotate_ts.py <pos_file_or_folder> <vel_file> [output_folder]

Or import and use directly:

    from sh_rotate_ts import read_vel_file, station_azimuths, rotate_dataframe
    from read_pos_files import read_pos_file

    vel = read_vel_file("loading.vel")
    azis = station_azimuths(vel)

    df = read_pos_file("ALBH.mit.final_nam20.pos")
    df = rotate_dataframe(df, azis["ALBH"])
    # df now also has dPerp, dPara, Sperp, Spara, Rpp columns
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

VEL_COLUMNS = [
    "lon", "lat", "Evel", "Nvel", "dEv", "dNv",
    "Evel_sig", "Nvel_sig", "Rne_vel",
    "Hvel", "dHv", "Hvel_sig", "site",
]

def read_vel_file(filepath):
    """Read a TSFIT/HECTOR-style .vel file (e.g. loading.vel) into a
    DataFrame indexed by 4-character station code.

    Header/comment lines (starting with '*' or '#') are skipped. The
    trailing "_GPS" suffix on the site column is stripped so station
    codes line up with the 4-character IDs used in .pos files.

    :param filepath: path to a .vel file
    :return: pandas.DataFrame indexed by station code, columns per
        VEL_COLUMNS (minus 'site', which becomes the index source)
    """
    rows = []
    with open(filepath) as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", "#")):
                continue
            rows.append(stripped.split())

    df = pd.DataFrame(rows, columns=VEL_COLUMNS)

    numeric_cols = [c for c in VEL_COLUMNS if c != "site"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df["station"] = df["site"].str.replace(r"_GPS$", "", regex=True).str.upper()
    df = df.set_index("station")

    return df

def station_azimuths(vel_df, evel_col="Evel", nvel_col="Nvel"):
    vel_df = vel_df[~vel_df.index.duplicated(keep="first")]
    azi = np.degrees(np.arctan2(vel_df[evel_col], vel_df[nvel_col])) % 360.0
    azi.name = "azimuth_deg"
    return azi

def rotate_en(e, n, azimuth_deg, se=None, sn=None, rho_en=None):
    """Rotate east/north values (and optionally their uncertainties)
    into the perpendicular/parallel frame defined by `azimuth_deg`,
    where "perpendicular" is aligned WITH `azimuth_deg` (e.g.
    trench-perpendicular/convergence direction) and "parallel" is 90
    degrees away from it (e.g. along-strike). See the module docstring
    ("A note on the labeling") for why it's this way round.

    `e`, `n`, `se`, `sn`, `rho_en` may be scalars or numpy
    arrays/Series of matching shape; `azimuth_deg` may be a scalar
    (one azimuth for the whole series -- the normal case for a single
    station) or an array matching `e`/`n`.

    This transform is a reflection (determinant -1), not a pure
    rotation, and is its own inverse: calling rotate_en(perp, para,
    azimuth_deg) recovers (e, n) exactly (see perp_para_to_en(), a
    thin wrapper for that use).

    :param e: east component (any units, e.g. mm)
    :param n: north component (same units as e)
    :param azimuth_deg: azimuth "a" of the perpendicular axis
        (i.e. the trench-perpendicular/convergence direction),
        degrees clockwise from north
    :param se: sigma_e, 1-sigma uncertainty of e; optional
    :param sn: sigma_n, 1-sigma uncertainty of n; optional
    :param rho_en: correlation coefficient between e and n; optional
    :return: (perp, para) as numpy arrays if se/sn/rho_en are not all
        given, else (perp, para, sig_perp, sig_para, rho_pp)
    """
    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)
    a = np.radians(np.asarray(azimuth_deg, dtype=float))

    cos_a = np.cos(a)
    sin_a = np.sin(a)

    perp = e * sin_a + n * cos_a
    para = e * cos_a - n * sin_a

    if se is None or sn is None or rho_en is None:
        return perp, para

    se = np.asarray(se, dtype=float)
    sn = np.asarray(sn, dtype=float)
    rho_en = np.asarray(rho_en, dtype=float)

    cross = rho_en * se * sn

    var_perp = se**2 * sin_a**2 + 2.0 * cross * cos_a * sin_a + sn**2 * cos_a**2
    var_para = se**2 * cos_a**2 - 2.0 * cross * cos_a * sin_a + sn**2 * sin_a**2

    # Guard against tiny negative variances from floating-point round-off.
    sig_perp = np.sqrt(np.clip(var_perp, 0.0, None))
    sig_para = np.sqrt(np.clip(var_para, 0.0, None))

    cov_pp = (se**2 - sn**2) * cos_a * sin_a + cross * (cos_a**2 - sin_a**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_pp = cov_pp / (sig_perp * sig_para)

    return perp, para, sig_perp, sig_para, rho_pp


def perp_para_to_en(perp, para, azimuth_deg):
    """Inverse of the (e, n) -> (perp, para) rotation: recover east and
    north from perpendicular/parallel values.

    rotate_en()'s transform is a reflection, not a pure rotation, and
    is its own inverse -- so this is just rotate_en() again, called
    with the SAME (not negated) azimuth. Useful downstream (e.g. map
    plotting) when a station was fit in the perp/para frame (possibly
    with only one of the two components fit) and you need an actual
    geographic (e, n) vector back. Pass 0 for whichever of perp/para
    wasn't fit.

    :param perp: perpendicular component
    :param para: parallel component
    :param azimuth_deg: the SAME azimuth "a" used in rotate_en() to
        produce perp/para from e/n originally
    :return: (e, n)
    """
    e, n = rotate_en(perp, para, azimuth_deg)
    return e, n


def rotate_dataframe(df, azimuth_deg, e_col="dE", n_col="dN",
                      se_col="Se", sn_col="Sn", rho_col="Rne",
                      out_prefix=""):
    """Rotate the east/north displacement columns of a station
    DataFrame (e.g. from read_pos_files.read_pos_file()) into
    perpendicular/parallel columns, propagating uncertainties.

    Adds columns 'dPerp', 'dPara', 'Sperp', 'Spara', 'Rpp' (prefixed
    with `out_prefix` if given). The 'dU'/'Su' (up) columns are left
    untouched -- only e/n rotate.

    :param df: DataFrame with e_col/n_col (and, if uncertainties are
        to be propagated, se_col/sn_col/rho_col) columns
    :param azimuth_deg: azimuth "a" of the perpendicular axis, degrees
        clockwise from north (scalar -- one azimuth per station)
    :param e_col, n_col: names of the east/north displacement columns
    :param se_col, sn_col, rho_col: names of the east sigma, north
        sigma, and east-north correlation columns. Pass None for any
        of these to skip uncertainty propagation.
    :param out_prefix: optional prefix for the new column names
    :return: a copy of df with the new columns appended
    """
    df = df.copy()

    have_unc = (
        se_col is not None and sn_col is not None and rho_col is not None
        and {se_col, sn_col, rho_col}.issubset(df.columns)
    )

    if have_unc:
        perp, para, sig_perp, sig_para, rho_pp = rotate_en(
            df[e_col], df[n_col], azimuth_deg,
            df[se_col], df[sn_col], df[rho_col],
        )
        df[f"{out_prefix}Sperp"] = sig_perp
        df[f"{out_prefix}Spara"] = sig_para
        df[f"{out_prefix}Rpp"] = rho_pp
    else:
        perp, para = rotate_en(df[e_col], df[n_col], azimuth_deg)

    df[f"{out_prefix}dPerp"] = perp
    df[f"{out_prefix}dPara"] = para
    df.attrs["rotation_azimuth_deg"] = azimuth_deg

    return df

def rotate_station_file(pos_filepath, azimuths, station=None):
    from read_pos_files import read_pos_file

    df = read_pos_file(pos_filepath)
    station = station or df.attrs.get("station") or \
        os.path.splitext(os.path.basename(pos_filepath))[0]
    station = station.upper()

    if station not in azimuths:
        print(f"Warning: no azimuth found for station '{station}' in the "
              f"vel file -- skipping {os.path.basename(pos_filepath)}.")
        return None

    azi = azimuths[station]
    df = rotate_dataframe(df, azi)
    df.attrs["station"] = station

    return df


def _resolve_pos_files(source):
    """Turn `source` into a flat list of .pos file paths.
    """
    if isinstance(source, (list, tuple)):
        files = []
        for item in source:
            files.extend(_resolve_pos_files(item))
        return files

    if os.path.isdir(source):
        return sorted(glob.glob(os.path.join(source, "**", "*.pos"), recursive=True))

    if os.path.isfile(source):
        return [source]

    print(f"Warning: '{source}' is not a valid file or folder, skipping.")
    return []


def process_folder(pos_source, vel_file, output_folder=None):
    vel = read_vel_file(vel_file)
    azimuths = station_azimuths(vel)

    files = _resolve_pos_files(pos_source)
    if not files:
        print("No .pos files found for the given input.")
        return {}

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    results = {}
    for filepath in files:
        try:
            df = rotate_station_file(filepath, azimuths)
        except Exception as e:
            print(f"Failed on {filepath}: {e}")
            continue

        if df is None:
            continue

        station = df.attrs["station"]
        results[station] = df
        print(f"Rotated {station}: azimuth={azimuths[station]:.2f} deg "
              f"clockwise-from-north, {len(df)} epochs")

        if output_folder:
            out_path = os.path.join(output_folder, f"{station}_rotated.csv")
            df.to_csv(out_path, index=False)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sh_rotate_ts.py <pos_file_or_folder> <vel_file> [output_folder]")
        sys.exit(1)

    pos_source_arg = sys.argv[1]
    vel_file_arg = sys.argv[2]
    output_folder_arg = sys.argv[3] if len(sys.argv) > 3 else None

    results = process_folder(pos_source_arg, vel_file_arg, output_folder_arg)

    print(f"\nRotated {len(results)} station(s).")