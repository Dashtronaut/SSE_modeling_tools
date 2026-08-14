"""
Fits a constant + linear trend + hyperbolic-tangent step(s) decomposition
model to a time series.

    y(t) = a + b*t + sum_i  q_i * tanh_step(t; t_start_i, duration_i)

`tanh_step` is a smooth 0->1 ramp (see `tanh_basis`), so each `q_i` is the
step's total (permanent) amplitude. Because the model is linear in
(a, b, q_1, q_2, ...) once the shape parameters (t_start_i, duration_i) of
every step are fixed, the amplitudes never need to be searched -- for any
candidate placement of the step(s), ordinary least squares finds the
best-fit amplitude(s) in closed form. Only the nonlinear shape parameters
(start time + duration of each step) are found by grid search.


    from tanh_step_fit import greedy_multi_step_search, plot_result
    result = greedy_multi_step_search(t, y, max_steps=4)
    best = result["best"]                      # the recommended fit
    print(best["params"], best["step_list"])

    python tanh_step_fit.py                        # synthetic self-test
    python tanh_step_fit.py --input series.csv      # columns: t,y (t optional)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize




def tanh_basis(t, t_start, duration):
    """
    Hyperbolic-tangent step, normalized to run from ~0 well before
    `t_start` to ~1 well after. `t_start` is the step's ONSET, `duration` is the span from the 5% point to the 95%
    point. Same shape as a tanh centered at t_start + duration/2.
    """
    t = np.asarray(t, dtype=float)
    duration = max(float(duration), 1e-9)
    t_center = t_start + duration / 2.0
    k = np.log(19.0) / duration  # tanh(k*duration/2) = 0.9 -> 0.5*(1+0.9) = 0.95
    return 0.5 * (1.0 + np.tanh(k * (t - t_center)))


def build_design_matrix(t, step_list=None):
    t = np.asarray(t, dtype=float)
    G = np.column_stack([np.ones_like(t), t])
    names = ["constant", "linear"]
    for i, (t_start, duration) in enumerate(step_list or []):
        G = np.column_stack([G, tanh_basis(t, t_start, duration)])
        names.append(f"step{i+1}_amp[t_start={t_start:.4g},dur={duration:.4g}]")
    return G, names


def lstsq_fit(t, y, step_list=None, sigma=None):
    """Linear least squares fit of the constant+linear+steps model.

    Returns a dict: params, param_names, param_std, fitted, residuals, rss,
    n, k, step_list. `param_std` is the formal 1-sigma standard error of
    each linear parameter (constant, linear, and every step's amplitude)
    from the least-squares covariance matrix
    """
    y = np.asarray(y, dtype=float)
    G, names = build_design_matrix(t, step_list)

    if sigma is not None:
        w = 1.0 / np.asarray(sigma, dtype=float)
        Gw, yw = G * w[:, None], y * w
        m, *_ = np.linalg.lstsq(Gw, yw, rcond=None)
        fitted = G @ m
        rss = float(np.sum((w * (y - fitted)) ** 2))
        Cm = np.linalg.pinv(Gw.T @ Gw)
    else:
        m, *_ = np.linalg.lstsq(G, y, rcond=None)
        fitted = G @ m
        rss = float(np.sum((y - fitted) ** 2))
        n_, k_ = len(y), G.shape[1]
        dof = max(n_ - k_, 1)
        Cm = np.linalg.pinv(G.T @ G) * (rss / dof)

    param_std = np.sqrt(np.clip(np.diag(Cm), 0, None))

    n = len(y)
    k_lin = G.shape[1]  # linear parameters actually solved for (a, b, amplitudes)

    return {
        "params": m,
        "param_names": names,
        "param_std": param_std,
        "fitted": fitted,
        "residuals": y - fitted,
        "rss": rss,
        "n": n,
        "k_linear": k_lin,
        "step_list": list(step_list or []),
    }

def params_per_step():
    """How many effective free parameters one tanh step costs.

    Each step contributes 1 LINEAR parameter (its amplitude, solved for by
    least squares) plus 2 NONLINEAR shape parameters (t_start and duration,
    chosen by grid search rather than least squares). Grid-searched
    parameters still "use up" degrees of freedom -- searching harder makes
    it easier to fit noise -- so both are counted here for a fair
    RSS-vs-parameter-count comparison across step counts.
    """
    return 3


def info_criteria(rss, n, k_linear, n_steps):
    """AIC / BIC / AICc for a Gaussian-residual least squares fit.

    k_total counts linear params (k_linear) already, then adds 2 more
    per step for the grid-searched (t_start, duration) pair -- see
    `params_per_step`.
    """
    k_total = k_linear + 2 * n_steps
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k_total
    bic = n * np.log(rss / n) + k_total * np.log(n)
    denom = n - k_total - 1
    aicc = aic + (2 * k_total * (k_total + 1)) / denom if denom > 0 else np.inf
    return {"aic": aic, "bic": bic, "aicc": aicc, "k_total": k_total}



def default_duration_grid(t, n_durations=10, min_duration=None):
    t = np.asarray(t, dtype=float)
    dt = np.median(np.diff(np.sort(t)))
    span = t.max() - t.min()
    lo = float(min_duration) if min_duration is not None else max(3 * dt, span / 200.0)
    hi = max(lo * 1.5, span / 2.0)
    return np.geomspace(lo, hi, n_durations)


def default_start_candidates(t, stride=1):
    t = np.asarray(t, dtype=float)
    return np.sort(t)[::stride]


def count_points_in_step(t, t_start, duration):
    """Number of data points (in `t`) that fall within a step's
    [t_start, t_start + duration] window -- a direct measure of how well
    supported that step's shape actually is by the data."""
    t = np.asarray(t, dtype=float)
    return int(np.sum((t >= t_start) & (t <= t_start + duration)))


def steps_satisfy_min_points(t, step_list, min_points):
    """True if every step in `step_list` has at least `min_points` data
    points inside its [t_start, t_start + duration] window."""
    if min_points is None:
        return True
    return all(
        count_points_in_step(t, t_start, duration) >= min_points
        for t_start, duration in step_list
    )


def refine_step(t, y, existing_steps, t_start0, duration0, sigma=None,
                 min_duration=None, max_duration=None):
    """
    Refines step components (t_start, duration) beyond the coarse grid's
    resolution, via continuous (Nelder-Mead) optimization of RSS starting
    from the grid's best point. Duration is optimized in log-space so it
    can never go negative.

    min_duration / max_duration : optional hard bounds (same units as t) on
        the refined duration. Passed to Nelder-Mead's `bounds` argument (in
        log-duration space) -- without this, refinement is otherwise
        unconstrained and, when there's little data inside the transition,
        will happily walk duration down toward zero even below the coarse
        grid's own floor, producing an abrupt near-step-function jump
        instead of a smooth one. Setting `min_duration` is what stops that.

    Returns a fit dict (same shape as lstsq_fit's output) at the refined
    placement -- or the original grid-point fit if refinement fails to
    improve on it (can happen at the edge of the series).
    """
    existing_steps = list(existing_steps or [])

    def neg_obj(x):
        t_start, log_dur = x
        duration = np.exp(log_dur)
        return lstsq_fit(t, y, existing_steps + [(t_start, duration)], sigma=sigma)["rss"]

    log_dur_lo = np.log(min_duration) if min_duration is not None else -np.inf
    log_dur_hi = np.log(max_duration) if max_duration is not None else np.inf
    bounds = [(-np.inf, np.inf), (log_dur_lo, log_dur_hi)]

    x0 = [float(t_start0), float(np.log(max(duration0, 1e-6)))]
    x0[1] = min(max(x0[1], log_dur_lo), log_dur_hi)  # start feasible

    res = minimize(neg_obj, x0, method="Nelder-Mead", bounds=bounds,
                    options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 500})

    t_start_ref, duration_ref = res.x[0], float(np.exp(res.x[1]))
    refined_fit = lstsq_fit(t, y, existing_steps + [(t_start_ref, duration_ref)], sigma=sigma)

    coarse_fit = lstsq_fit(t, y, existing_steps + [(float(t_start0), float(duration0))], sigma=sigma)
    return refined_fit if refined_fit["rss"] <= coarse_fit["rss"] else coarse_fit


def sweep_single_step(t, y, existing_steps=None, duration_grid=None,
                       start_candidates=None, sigma=None, refine=True,
                       min_points=None, growth_factor=1.5, max_duration_retries=8):
    """
    Grid-search one additional tanh step's (t_start, duration) on top of
    whatever steps are already in `existing_steps`, refitting the full
    linear model (constant + linear + existing steps + candidate step) by
    least squares at every grid point and keeping the placement with the
    lowest RSS. If `refine` (default), the coarse grid winner is then
    locally refined with continuous optimization (`refine_step`) so the
    reported duration/center aren't limited to the grid's spacing.

    min_points : if set, requires the resulting step to have at least this
        many data points inside its [t_start, t_start+duration] window (see
        `count_points_in_step`). If the winning placement doesn't clear
        that bar -- the low-resolution degeneracy where too few points
        inside the transition let RSS collapse duration toward an abrupt
        jump -- the duration floor is raised by `growth_factor` and BOTH
        the grid and the refinement's Nelder-Mead bound (`refine_step`'s
        min_duration) are rebuilt with it, then re-solved. This repeats up
        to `max_duration_retries` times or until the floor reaches half the
        series' span, whichever comes first; the widest-floor result found
        is returned even if it still can't clear `min_points` (a step that
        genuinely isn't supported by roughly that many points anywhere in
        its neighborhood -- callers can check `steps_satisfy_min_points`
        on the result and fall back to fewer steps if so).

    Returns (best_fit, rss_grid) where rss_grid is a DataFrame of every
    (t_start, duration) tried on the last COARSE grid used and its
    resulting RSS -- useful for diagnosing how sharply-peaked / ambiguous
    the best placement is (a flat RSS surface means the duration/center are
    poorly constrained by the data, however precise the refined number
    looks).
    """
    existing_steps = list(existing_steps or [])
    if start_candidates is None:
        start_candidates = default_start_candidates(t)

    t_arr = np.asarray(t, dtype=float)
    span = float(t_arr.max() - t_arr.min())
    max_duration_cap = span / 2.0

    user_grid = duration_grid
    if user_grid is not None:
        cur_min_duration = float(np.min(user_grid))
    else:
        dt = np.median(np.diff(np.sort(t_arr)))
        cur_min_duration = max(3 * dt, span / 200.0)

    best_fit, rss_grid = None, None
    for attempt in range(max_duration_retries + 1):
        grid = user_grid if (user_grid is not None and attempt == 0) \
            else default_duration_grid(t_arr, min_duration=cur_min_duration)

        best_rss, best_t0, best_d0 = np.inf, None, None
        rows = []
        for t_start in start_candidates:
            for duration in grid:
                trial_steps = existing_steps + [(float(t_start), float(duration))]
                fit = lstsq_fit(t, y, trial_steps, sigma=sigma)
                rows.append((float(t_start), float(duration), fit["rss"]))
                if fit["rss"] < best_rss:
                    best_rss, best_fit = fit["rss"], fit
                    best_t0, best_d0 = float(t_start), float(duration)

        rss_grid = pd.DataFrame(rows, columns=["t_start", "duration", "rss"])

        if refine:
            best_fit = refine_step(t, y, existing_steps, best_t0, best_d0, sigma=sigma,
                                    min_duration=cur_min_duration, max_duration=max_duration_cap)

        if min_points is None:
            return best_fit, rss_grid

        new_step = best_fit["step_list"][-1]
        if steps_satisfy_min_points(t, [new_step], min_points) or cur_min_duration >= max_duration_cap:
            return best_fit, rss_grid

        cur_min_duration = min(cur_min_duration * growth_factor, max_duration_cap)

    return best_fit, rss_grid


def greedy_multi_step_search(t, y, max_steps=5, duration_grid=None,
                              start_stride=1, sigma=None, criterion="bic",
                              verbose=True, min_points=None,
                              growth_factor=1.5, max_duration_retries=8):
    """
    Fit with 0, 1, 2, ..., max_steps tanh steps (each additional step being
    the single best greedy addition to the previous model), and report the
    RSS / AIC / BIC at every step count

    min_points, growth_factor, max_duration_retries : forwarded to
        `sweep_single_step` at every greedy addition -- see its docstring.
        When `min_points` is set, each new step is required to have that
        many data points inside its window, growing the duration floor
        (grid + Nelder-Mead bound together) rather than accepting an
        under-supported, abrupt-jump step.

    Returns
    -------
    dict with:
      'fits'    : list of fit dicts, index k = number of steps (0..max_steps)
      'table'   : DataFrame with columns n_steps, rss, aic, bic, aicc
      'best'    : the fit dict at the step count that minimizes `criterion`
      'best_k'  : that step count
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    start_candidates = default_start_candidates(t, stride=start_stride)
    if duration_grid is None:
        duration_grid = default_duration_grid(t)

    fits = [lstsq_fit(t, y, step_list=[], sigma=sigma)]
    if verbose:
        print(f"k=0 (no steps): RSS={fits[0]['rss']:.5g}")

    for k in range(1, max_steps + 1):
        existing = fits[-1]["step_list"]
        best_fit, _ = sweep_single_step(
            t, y, existing_steps=existing, duration_grid=duration_grid,
            start_candidates=start_candidates, sigma=sigma,
            min_points=min_points, growth_factor=growth_factor,
            max_duration_retries=max_duration_retries,
        )
        fits.append(best_fit)
        if verbose:
            new_step = best_fit["step_list"][-1]
            print(f"k={k}: added step t_start={new_step[0]:.4g}, "
                  f"duration={new_step[1]:.4g}  ->  RSS={best_fit['rss']:.5g}")

    rows = []
    for k, fit in enumerate(fits):
        ic = info_criteria(fit["rss"], fit["n"], fit["k_linear"], k)
        rows.append({"n_steps": k, "rss": fit["rss"], **ic})
    table = pd.DataFrame(rows)

    best_idx = int(table[criterion].idxmin())
    if verbose:
        print(f"\nBest model by {criterion.upper()}: {best_idx} step(s)")
        print(table.to_string(index=False))

    return {
        "fits": fits,
        "table": table,
        "best": fits[best_idx],
        "best_k": best_idx,
        "criterion": criterion,
    }


def summarize_steps(fit):
    rows = []
    for i, (t_start, duration) in enumerate(fit["step_list"]):
        amp = fit["params"][2 + i]
        amp_std = fit["param_std"][2 + i]
        rows.append({
            "step": i + 1,
            "t_start": t_start,
            "center": t_start + duration / 2.0,
            "duration": duration,
            "amplitude": amp,
            "amplitude_std": amp_std,
            "amplitude_z": amp / amp_std if amp_std > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def plot_result(t, y, result, title="Tanh-step decomposition"):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    table = result["table"]
    best_k = result["best_k"]
    best = result["best"]
    criterion = result["criterion"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].scatter(t, y, s=10, color="0.5", alpha=0.6, label="data")
    axes[0].plot(t, best["fitted"], color="crimson", lw=1.8,
                 label=f"fit (k={best_k} step(s))")
    for i, (t_start, duration) in enumerate(best["step_list"]):
        axes[0].axvspan(t_start, t_start + duration, color="orange", alpha=0.15)
        axes[0].axvline(t_start + duration / 2.0, color="orange", ls="--", lw=0.8)
    axes[0].set_ylabel("value")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(table["n_steps"], table[criterion], "o-", color="steelblue")
    axes[1].axvline(best_k, color="crimson", ls="--", lw=0.8,
                     label=f"chosen k={best_k} (min {criterion.upper()})")
    axes[1].set_xticks(table["n_steps"])
    axes[1].set_xlabel("number of tanh steps")
    axes[1].set_ylabel(criterion.upper())
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def load_series(path):
    try:
        df = pd.read_csv(path, sep=r"[,\s]+", engine="python", comment="#")
        # if the sniffed header row wasn't actually numeric, re-read headerless
        if not any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
            raise ValueError("header row not numeric")
    except Exception:
        df = pd.read_csv(path, sep=r"[,\s]+", engine="python", comment="#", header=None)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric columns found in {path}")
    if len(numeric_cols) == 1:
        y = df[numeric_cols[0]].to_numpy(dtype=float)
        t = np.arange(len(y), dtype=float)
    else:
        t = df[numeric_cols[0]].to_numpy(dtype=float)
        y = df[numeric_cols[1]].to_numpy(dtype=float)
    return t, y

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                     help="CSV/whitespace file with column(s) [t,]y. "
                          "If omitted, runs a synthetic self-test.")
    ap.add_argument("--max-steps", type=int, default=5,
                     help="Maximum number of tanh steps to try adding.")
    ap.add_argument("--start-stride", type=int, default=1,
                     help="Stride (in samples) between candidate start times "
                          "tried in the sweep -- raise this to speed up long series.")
    ap.add_argument("--n-durations", type=int, default=10,
                     help="Number of candidate durations in the (log-spaced) grid.")
    ap.add_argument("--criterion", choices=["aic", "bic", "aicc"], default="bic",
                     help="Information criterion used to pick the best step count.")
    ap.add_argument("--out-prefix", default="tanh_step_fit")
    args = ap.parse_args()

    if args.input:
        t, y = load_series(args.input)
        title = f"Tanh-step decomposition: {args.input}"

    duration_grid = default_duration_grid(t, n_durations=args.n_durations)

    result = greedy_multi_step_search(
        t, y, max_steps=args.max_steps, duration_grid=duration_grid,
        start_stride=args.start_stride, criterion=args.criterion, verbose=True,
    )

    best = result["best"]
    print(f"\nChosen model ({result['best_k']} step(s)):")
    for name, val, std in zip(best["param_names"], best["params"], best["param_std"]):
        print(f"  {name:45s}: {val:+.5g} +/- {std:.3g}")

    steps_df = summarize_steps(best)
    if len(steps_df):
        print("\nPer-event summary (center time, duration, amplitude):")
        print(steps_df.to_string(index=False))
        steps_csv = f"{args.out_prefix}_events.csv"
        steps_df.to_csv(steps_csv, index=False)
        print(f"Saved per-event summary -> {steps_csv}")

    table_csv = f"{args.out_prefix}_model_selection.csv"
    result["table"].to_csv(table_csv, index=False)
    print(f"\nSaved model-selection table -> {table_csv}")

    out_ts = pd.DataFrame({"t": t, "y": y, "fit": best["fitted"], "residual": best["residuals"]})
    ts_csv = f"{args.out_prefix}_best_fit_timeseries.csv"
    out_ts.to_csv(ts_csv, index=False)
    print(f"Saved best-fit time series -> {ts_csv}")

    fig = plot_result(t, y, result, title=title)
    plot_path = f"{args.out_prefix}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()