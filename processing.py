"""
gnss_tools.processing
=====================
Low-level signal processing routines used by :class:`~gnss_tools.TimeSeries`.

Outlier removal — Hampel identifier
------------------------------------
The **Hampel identifier** (also called the Hampel filter or robust
median-based outlier detector) flags a sample as an outlier when it
deviates from the **rolling median** by more than k × 1.4826 × MAD,
where 1.4826 is the consistency factor that makes MAD a consistent
estimator of σ under a Gaussian distribution.

This is the standard preprocessing method in the GNSS geodesy community:

* Klos et al. (2015), "On the Handling of Outliers in the GNSS Time Series
  by Means of the Noise and Probability Analysis", *IAG Symposia* 143.
  Cited widely as the reference implementation for GPS daily position series.
* Langbein & Bock (2004) introduced the 3 × IQR threshold that the Hampel
  approach supersedes for time series because IQR is global whereas the
  Hampel window is local.
* Langbein & Svarc (2019), *JGR Solid Earth*, justify the doubled threshold
  for the Up component given its ~4× larger coloured-noise amplitude.

The Up component uses ``k_eff = k × up_scale`` (default 2 ×) to avoid
over-rejection in the noisier vertical channel.

Outliers are **set to NaN** (not interpolated), preserving the original
data distribution while marking suspect epochs transparently.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Hampel identifier (core function)
# ---------------------------------------------------------------------------

def hampel_identifier(
    y:       np.ndarray,
    window:  int   = 11,
    k:       float = 3.0,
) -> np.ndarray:
    """
    Apply the Hampel identifier to a 1-D array, returning a copy with
    outliers replaced by NaN.

    Algorithm
    ---------
    For each sample i, compute the median and MAD of the surrounding
    ``2 * window + 1`` samples.  Flag the sample as an outlier if::

        |y[i] - median| > k × 1.4826 × MAD

    Parameters
    ----------
    y      : 1-D float array (may contain NaN).
    window : half-width of the rolling window (samples).
             Full window = 2 * window + 1.
    k      : MAD multiplier (default 3.0 → ~3σ for Gaussian data).

    Returns
    -------
    np.ndarray with outliers set to NaN.

    References
    ----------
    Klos et al. (2015) IAG Symposia 143.
    Hampel (1974) *Technometrics* 16(1).
    """
    y    = np.asarray(y, dtype=float).copy()
    n    = len(y)
    full = 2 * window + 1

    s = pd.Series(y)
    rolling_median = s.rolling(full, center=True, min_periods=3).median()
    rolling_mad    = (
        s.subtract(rolling_median)
        .abs()
        .rolling(full, center=True, min_periods=3)
        .median()
    )

    threshold = k * 1.4826 * rolling_mad.values
    residual  = np.abs(y - rolling_median.values)
    outlier   = residual > threshold

    y[outlier] = np.nan
    return y


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------

def detrend_series(
    y:       np.ndarray,
    t:       np.ndarray,
    method:  str = "linear",
) -> np.ndarray:
    """
    Remove a linear trend from *y* versus *t*.

    Parameters
    ----------
    y      : displacement values (mm), may contain NaN.
    t      : time axis (decimal years, already centred if desired).
    method : ``"linear"`` (OLS via scipy.stats.linregress) or
             ``"robust"`` (Theil-Sen via scipy.stats.theilslopes).

    Returns
    -------
    np.ndarray — residuals with trend removed.
    """
    y = np.asarray(y, dtype=float).copy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y

    if method == "robust":
        slope, intercept, *_ = stats.theilslopes(y[mask], t[mask])
    elif method == "linear":
        slope, intercept, *_ = stats.linregress(t[mask], y[mask])
    else:
        raise ValueError(
            f"Unknown detrend method {method!r}. "
            "Use 'linear' or 'robust'."
        )

    y -= slope * t + intercept
    return y


# ---------------------------------------------------------------------------
# Offset estimation
# ---------------------------------------------------------------------------

def estimate_offset(
    series:       pd.Series,
    offset_time:  pd.Timestamp,
    window_days:  int = 30,
) -> float:
    """
    Estimate an instantaneous step offset at *offset_time*.

    Computes the median displacement in a ``window_days``-day window
    immediately before and immediately after the event, and returns
    the difference (after − before).

    Parameters
    ----------
    series      : pd.Series with DatetimeIndex (mm).
    offset_time : timestamp of the offset.
    window_days : half-window size in days.

    Returns
    -------
    float : estimated jump in mm (positive = series rises after event).
            Returns 0.0 with a warning if insufficient data exist on
            either side.
    """
    idx = series.index
    td  = pd.Timedelta(days=window_days)

    before = series.loc[(idx >= offset_time - td) & (idx < offset_time)]
    after  = series.loc[(idx >= offset_time) & (idx <= offset_time + td)]

    if len(before) < 2 or len(after) < 2:
        print(
            f"Warning: insufficient data around {offset_time:%Y-%m-%d}. "
            "Offset set to 0."
        )
        return 0.0

    jump = float(np.nanmedian(after.values) - np.nanmedian(before.values))
    if np.isnan(jump):
        print(
            f"Warning: NaN offset at {offset_time:%Y-%m-%d}. "
            "Offset set to 0."
        )
        return 0.0

    return jump


def estimate_interval_offset(
    series:      pd.Series,
    start_time:  pd.Timestamp,
    end_time:    pd.Timestamp,
    window_days: int = 30,
) -> float:
    """
    Estimate an offset that spans an interval [start_time, end_time]
    (e.g. a slow-slip event, a multi-day antenna campaign).

    Uses a ``window_days``-day window *before* start_time and *after*
    end_time to estimate pre- and post-event medians.

    Parameters
    ----------
    series      : pd.Series with DatetimeIndex (mm).
    start_time  : beginning of the offset interval.
    end_time    : end of the offset interval.
    window_days : window size in days.

    Returns
    -------
    float : estimated jump in mm.
    """
    if end_time < start_time:
        raise ValueError("end_time must be >= start_time")

    idx = series.index
    td  = pd.Timedelta(days=window_days)

    before = series.loc[(idx >= start_time - td) & (idx < start_time)]
    after  = series.loc[(idx > end_time) & (idx <= end_time + td)]

    if len(before) < 2 or len(after) < 2:
        print(
            f"Warning: insufficient data around "
            f"{start_time:%Y-%m-%d}–{end_time:%Y-%m-%d}. "
            "Offset set to 0."
        )
        return 0.0

    jump = float(np.nanmedian(after.values) - np.nanmedian(before.values))
    if np.isnan(jump):
        return 0.0

    return jump
