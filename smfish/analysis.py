"""Downstream analysis helpers.

Conventions:
- Cells come from ``aggregated/cell_counts_long`` (zero-filled; non-expressing cells kept).
- QC is explicit: missing columns raise, nothing is skipped silently.
- Uncertainty is computed over fields of view (FOVs, the unit closest to a replicate here), not over cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def require(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}")


def qc_cells(long: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Attach cell morphology and keep QC-passing cells (has nucleus, not on an XY/Z border)."""
    require(cells, ["fov_id", "cell_id", "qc_pass", "cell_volume_um3", "nucleus_volume_um3"])
    keep = ["fov_id", "cell_id", "qc_pass", "cell_volume_um3", "nucleus_volume_um3", "cytoplasm_volume_um3",
            "nucleus_dapi_sum"]
    m = long.merge(cells[[c for c in keep if c in cells.columns]], on=["fov_id", "cell_id"], how="left")
    return m[m["qc_pass"]].drop(columns="qc_pass")


def collapse_cytoplasm(long: pd.DataFrame) -> pd.DataFrame:
    """Nucleus / cytoplasm view: cytoplasm = cell minus nucleus (perinuclear rows summed into cytoplasm).

    The perinuclear split is kept in the per-FOV outputs (and ``dist_nucleus_um`` per spot), so it can
    be analysed separately later without re-running anything.
    """
    d = long.copy()
    d["compartment"] = d["compartment"].replace({"perinuclear": "cytoplasm"})
    keys = [c for c in d.columns if c not in ("n_spots", "n_rna")]
    return d.groupby(keys, dropna=False)[["n_spots", "n_rna"]].sum(min_count=1).reset_index()


def compartment_totals(long: pd.DataFrame) -> pd.DataFrame:
    """Add a ``cell`` compartment row (sum over nucleus + perinuclear + cytoplasm)."""
    keys = [c for c in long.columns if c not in ("compartment", "n_spots", "n_rna")]
    tot = long.groupby(keys, dropna=False)[["n_spots", "n_rna"]].sum(min_count=1).reset_index()
    tot["compartment"] = "cell"
    return pd.concat([long, tot], ignore_index=True)


def per_fov(df: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Mean of ``value`` per FOV within each group (one row per group x FOV)."""
    return df.groupby([*by, "fov_id"])[value].agg(["mean", "size"]).rename(columns={"size": "n_cells"}).reset_index()


def bootstrap_over_fovs(df: pd.DataFrame, value: str, by: list[str], n_boot: int = 2000, seed: int = 0,
                        stat=np.mean) -> pd.DataFrame:
    """Cell-level statistic per group with a 95% CI from resampling whole FOVs."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(by):
        fovs = g.groupby("fov_id")[value].apply(np.asarray)
        arrays = list(fovs.values)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(arrays), len(arrays))
            boots[b] = stat(np.concatenate([arrays[i] for i in pick]))
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "estimate": stat(g[value].to_numpy()),
                     "ci_low": np.nanpercentile(boots, 2.5), "ci_high": np.nanpercentile(boots, 97.5),
                     "n_cells": len(g), "n_fov": len(arrays)})
    return pd.DataFrame(rows, columns=[*by, "estimate", "ci_low", "ci_high", "n_cells", "n_fov"])



# --------------------------------------------------------------------------- size scaling and noise


VOLUME_COLUMNS = {"cell": "cell_volume_um3", "nucleus": "nucleus_volume_um3", "cytoplasm": "cytoplasm_volume_um3"}


def per_cell_counts(long: pd.DataFrame, cells: pd.DataFrame, probe: str, channel: str | None = None,
                    compartments=("cell",)) -> pd.DataFrame:
    """One row per QC cell and compartment for one probe: counts plus volumes in pL.

    Non-expressing cells are KEPT (counts of 0): dropping them, as a log-log fit needs, biases both
    the mean and the Fano factor. Compartments are nucleus / cytoplasm (perinuclear folded in) / cell.
    """
    d = long[long["probe"] == probe]
    if channel is not None:
        d = d[d["channel"] == channel]
    if d.empty:
        raise KeyError(f"No rows for probe {probe!r}" + (f" in {channel}" if channel else ""))
    d = compartment_totals(collapse_cytoplasm(qc_cells(d, cells)))
    d = d[d["compartment"].isin(compartments)].copy()
    for comp, col in VOLUME_COLUMNS.items():
        if col in d:
            d[col.replace("_um3", "_pL")] = d[col] / 1000.0
    return d.reset_index(drop=True)


def rolling_moments(df: pd.DataFrame, x: str, y: str, window: float = 0.1, step: float | None = None,
                    min_cells: int = 50, within: str | None = None) -> pd.DataFrame:
    """Mean, variance and Fano factor of ``y`` in a sliding window of cells ordered by ``x``.

    The window holds a fixed NUMBER of cells (``window`` as a fraction of the cells, at least
    ``min_cells``) rather than a fixed span of ``x``, so every point is equally noisy. ``step`` is the
    stride in cells (default a fifth of the window). Returns one row per window: ``x`` (median of the
    window), ``mean``, ``var`` (ddof 1), ``fano``, ``cv2`` (var/mean^2), ``excess_cv2`` ((var - mean)/mean^2,
    the CV^2 above Poisson counting noise) and ``n``. Empty if there are fewer than ``min_cells`` cells.

    The Fano factor compares noise between groups only at equal means: at a flat Fano factor the excess
    CV^2 falls as 1/mean, so compare compartments (whose means differ) on ``excess_cv2``.

    ``within`` (e.g. ``"well"``) pools only the variance WITHIN each group: the window's variance is
    the pooled within-group variance and its mean the mean of the group means it averages over. Without
    it, wells whose mean differs (staining, segmentation) add their spread to the variance - on 250806
    the Streaming wells give Fano ~4-5 each but ~6.3 pooled.
    """
    if within is not None:
        d = df[[x, y, within]].dropna().sort_values(x)
        groups = d[within].to_numpy()
    else:
        d = df[[x, y]].dropna().sort_values(x)
        groups = None
    n = len(d)
    if n < min_cells:
        return pd.DataFrame(columns=["x", "mean", "var", "fano", "cv2", "excess_cv2", "n"])
    w = max(int(round(window * n)), min_cells) if window < 1 else int(window)
    w = min(w, n)
    s = max(int(step if step is not None else w / 5), 1)
    xs, ys = d[x].to_numpy(float), d[y].to_numpy(float)
    rows = []
    for start in range(0, n - w + 1, s):
        v = ys[start:start + w]
        if groups is None:
            m, var = v.mean(), v.var(ddof=1)
        else:
            g = pd.Series(v).groupby(groups[start:start + w])
            k, means, sizes = g.var(ddof=1), g.mean(), g.size()
            ok = sizes > 1
            dof = (sizes[ok] - 1).sum()
            if dof < 1:
                continue
            var = float(((sizes[ok] - 1) * k[ok]).sum() / dof)
            m = float((means[ok] * sizes[ok]).sum() / sizes[ok].sum())
        rows.append({"x": float(np.median(xs[start:start + w])), "mean": m, "var": var,
                     "fano": var / m if m > 0 else np.nan, "cv2": var / m ** 2 if m > 0 else np.nan,
                     "excess_cv2": (var - m) / m ** 2 if m > 0 else np.nan, "n": w})
    return pd.DataFrame(rows)


def _resample_fovs(df: pd.DataFrame, n_boot: int, seed: int):
    """Yield ``n_boot`` copies of ``df`` built from whole FOVs drawn with replacement (none if < 2 FOVs)."""
    fovs = [f for _, f in df.groupby("fov_id")] if "fov_id" in df else []
    if len(fovs) < 2:
        return
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield pd.concat([fovs[i] for i in rng.integers(0, len(fovs), len(fovs))], ignore_index=True)


def bootstrap_moments(df: pd.DataFrame, x: str, y: str, window: float = 0.1, within: str | None = "well",
                      n_boot: int = 200, seed: int = 0, stats=("fano", "excess_cv2"), min_cells: int = 50) -> pd.DataFrame:
    """:func:`rolling_moments` plus a 95% band per window from resampling whole FOVs.

    Each replicate's windows fall at slightly different ``x``, so its values are interpolated onto the point
    estimate's ``x`` (never extrapolated). Adds ``<stat>_lo`` / ``<stat>_hi`` for each of ``stats``.
    """
    import warnings

    within = within if within is not None and within in df else None
    est = rolling_moments(df, x, y, window=window, min_cells=min_cells, within=within)
    for st in stats:
        est[f"{st}_lo"] = est[f"{st}_hi"] = np.nan
    if len(est) < 2:
        return est
    xs = est["x"].to_numpy(float)
    reps = {st: [] for st in stats}
    for d in _resample_fovs(df, n_boot, seed):
        r = rolling_moments(d, x, y, window=window, min_cells=min_cells, within=within)
        if len(r) < 2:
            continue
        for st in stats:
            reps[st].append(np.interp(xs, r["x"].to_numpy(float), r[st].to_numpy(float), left=np.nan, right=np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns at the ends
        for st in stats:
            if reps[st]:
                lo, hi = np.nanpercentile(np.array(reps[st]), [2.5, 97.5], axis=0)
                est[f"{st}_lo"], est[f"{st}_hi"] = lo, hi
    return est


def noise_by_bins(df: pd.DataFrame, x: str, y: str, edges, window: float = 0.1, within: str | None = "well",
                  n_boot: int = 200, seed: int = 0, min_cells: int = 50) -> pd.DataFrame:
    """Fano factor and excess CV^2 per bin of ``x`` (e.g. volume terciles), with FOV-bootstrap 95% CIs.

    Values are medians of the :func:`rolling_moments` windows whose median ``x`` falls in the bin, so the spread of
    ``x`` inside a wide bin does not inflate the variance. ``excess_cv2_const_fano`` is what the excess CV^2 would be
    if the Fano factor stayed at its population value (median over all windows): (F - 1) / mean, per window. A
    measured excess CV^2 above it means noise that does not fall with the mean.
    """
    import warnings

    within = within if within is not None and within in df else None
    edges = np.asarray(edges, float)

    def _summ(d):
        r = rolling_moments(d, x, y, window=window, min_cells=min_cells, within=within)
        f_pop = r["fano"].median() if len(r) else np.nan
        out = []
        for b in range(len(edges) - 1):
            w = r[(r["x"] >= edges[b]) & (r["x"] <= edges[b + 1])] if len(r) else r
            out.append({"n_windows": len(w), "fano": w["fano"].median(), "excess_cv2": w["excess_cv2"].median(),
                        "excess_cv2_const_fano": ((f_pop - 1) / w["mean"]).median() if len(w) else np.nan,
                        "fano_pop": f_pop})
        return out

    est = _summ(df)
    boots = [_summ(d) for d in _resample_fovs(df, n_boot, seed)]
    rows = []
    for b, e in enumerate(est):
        in_b = df[(df[x] >= edges[b]) & (df[x] <= edges[b + 1])]
        row = {"bin": b + 1, "lo": edges[b], "hi": edges[b + 1], "n_cells": len(in_b), "mean": in_b[y].mean(), **e}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for st in ("fano", "excess_cv2"):
                v = np.array([bt[b][st] for bt in boots], float)
                row[f"{st}_lo"], row[f"{st}_hi"] = (np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any()
                                                    else (np.nan, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def scaling_slope(df: pd.DataFrame, x: str, y: str, by: list[str], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Log-log slope of ``y`` against ``x`` per group, with a 95% CI from resampling whole FOVs.

    Slope ~1 means the count rises in proportion to size (constant concentration). Cells with ``y`` = 0
    cannot enter a log fit; they are left out of the fit only, and counted in ``n_zero``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(by):
        pos = g[(g[y] > 0) & (g[x] > 0)]
        fovs = [f for _, f in pos.groupby("fov_id")]

        def fit(d):
            return np.polyfit(np.log(d[x].to_numpy(float)), np.log(d[y].to_numpy(float)), 1)[0]

        est = fit(pos) if len(pos) > 2 else np.nan
        boots = []
        if len(fovs) > 1:
            for _ in range(n_boot):
                d = pd.concat([fovs[i] for i in rng.integers(0, len(fovs), len(fovs))])
                if len(d) > 2:
                    boots.append(fit(d))
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "slope": est,
                     "ci_low": np.percentile(boots, 2.5) if boots else np.nan,
                     "ci_high": np.percentile(boots, 97.5) if boots else np.nan,
                     "n_cells": len(g), "n_zero": int((g[y] <= 0).sum()), "n_fov": g["fov_id"].nunique()})
    return pd.DataFrame(rows)


def fano_window_check(df: pd.DataFrame, x: str, y: str, windows=(0.05, 0.1, 0.2), min_cells: int = 30) -> pd.DataFrame:
    """Median rolling Fano factor at several window sizes.

    Within a window the count still rises with size, so a wider window adds the size spread to the
    variance; and cells whose volume is mis-segmented (merged, clipped) scatter away from the trend.
    If the Fano factor climbs steeply with window size, or only settles at the narrowest window, it is
    reporting size spread and segmentation error rather than expression noise - drop the panel.
    """
    rows = []
    for w in windows:
        r = rolling_moments(df, x, y, window=w, min_cells=min_cells)
        rows.append({"window": w, "median_fano": float(r["fano"].median()) if len(r) else np.nan,
                     "n_windows": len(r)})
    return pd.DataFrame(rows)


def ratio_over_fovs(per_fov: pd.DataFrame, num: str, den: str, by: list[str], n_boot: int = 2000,
                    seed: int = 0) -> pd.DataFrame:
    """sum(num) / sum(den) per group, with a 95% CI from resampling whole FOVs."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in per_fov.groupby(by):
        n, d = g[num].to_numpy(float), g[den].to_numpy(float)
        est = n.sum() / d.sum() if d.sum() else np.nan
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(g), len(g))
            if d[i].sum():
                boots.append(n[i].sum() / d[i].sum())
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "estimate": est,
                     "ci_low": np.percentile(boots, 2.5) if boots else np.nan,
                     "ci_high": np.percentile(boots, 97.5) if boots else np.nan,
                     "numerator": n.sum(), "denominator": d.sum(), "n_fov": len(g)})
    return pd.DataFrame(rows)
