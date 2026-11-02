"""Spot-based flat-field: correct spot intensities for illumination fall-off across the FOV.

On 250806, isolated single mRNAs were ~30% dimmer in the FOV corners than in the centre, while
the blimp illumination reference (dominated by camera offset in the FISH channels) predicted
only ~8%. Single-transcript spots are the right calibration objects, so the gain map is fitted
on them directly:

    log(I) = c_well + b1*u + b2*v + b3*(u^2 + v^2)        u, v in [-1, 1] across the FOV

i.e. radial fall-off plus a linear tilt, with a free intercept per FOV (absorbs focus / staining /
probe brightness differences between FOVs and wells) and a shared spatial shape per channel. A free-form quadratic over-fitted
the sparsely sampled corners (Atto633: ~200 spots) - this 3-parameter shape cannot. Intensities are
divided by ``gain(u, v) = exp(f(u, v))``, normalised to mean 1 over the field and clipped to
[MIN_GAIN, 1/MIN_GAIN]. Robust: two passes, dropping |residual| > 2.5 MAD (clusters, noise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TERMS = ["u", "v", "r2"]
MIN_GAIN = 0.5


def _design(y_px, x_px, shape_yx) -> np.ndarray:
    u = 2 * np.asarray(y_px, float) / (shape_yx[0] - 1) - 1
    v = 2 * np.asarray(x_px, float) / (shape_yx[1] - 1) - 1
    return np.column_stack([u, v, u * u + v * v])


def _mean_gain_log(coef: np.ndarray) -> float:
    g = np.linspace(-1, 1, 101)
    u, v = np.meshgrid(g, g, indexing="ij")
    f = _design((u.ravel() + 1) / 2 * 100, (v.ravel() + 1) / 2 * 100, (101, 101)) @ coef
    return float(np.log(np.mean(np.exp(f))))


def fit_spot_flatfield(spots: pd.DataFrame, shape_yx, min_spots: int = 200) -> dict:
    """Fit one gain model per channel from reference-quality spots (see :func:`reference_spots`).

    Returns ``{channel: {"coef": [...], "n": int, "corner_over_centre": float}}``.
    """
    models = {}
    for ch, d in spots.groupby("channel"):
        d = d[d["intensity_bgsub"] > 0]
        if len(d) < min_spots:
            continue
        X = _design(d["y_px"], d["x_px"], shape_yx)
        # one intercept per FOV (absorbs per-FOV focus / staining differences), per well if no fov_id
        group = d["fov_id"] if "fov_id" in d else d["well"]
        wells = pd.get_dummies(group.astype(str)).to_numpy(float)
        y = np.log(d["intensity_bgsub"].to_numpy(float))
        keep = np.ones(len(y), bool)
        for _ in range(2):
            A = np.hstack([wells[keep], X[keep]])
            beta, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            resid = y - np.hstack([wells, X]) @ beta
            mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) * 1.4826
            keep = np.abs(resid) <= 2.5 * mad
        coef = beta[wells.shape[1]:]
        norm = _mean_gain_log(coef)
        prof = {}
        for name, r in (("centre", 0.0), ("edge_mid", 1.0), ("corner", np.sqrt(2))):  # r in units of FOV half-width
            prof[name] = float(np.exp(coef[2] * r * r))
        models[ch] = {"terms": TERMS, "coef": coef.tolist(), "log_norm": norm, "n": int(keep.sum()),
                      "edge_over_centre": prof["edge_mid"] / prof["centre"],
                      "corner_over_centre": prof["corner"] / prof["centre"]}
    return models


def gain(models: dict, channel: pd.Series, y_px, x_px, shape_yx) -> np.ndarray:
    """Multiplicative gain at each position (1 where the channel has no model)."""
    out = np.ones(len(channel))
    X = _design(y_px, x_px, shape_yx)
    for ch, m in models.items():
        sel = (channel == ch).to_numpy()
        if sel.any():
            out[sel] = np.clip(np.exp(X[sel] @ np.asarray(m["coef"]) - m["log_norm"]), MIN_GAIN, 1 / MIN_GAIN)
    return out


def apply_spot_flatfield(spots: pd.DataFrame, models: dict, shape_yx) -> pd.DataFrame:
    """Divide spot (and partner-channel-at-spot) intensities by the gain; keeps raw values as ``*_raw``."""
    s = spots.copy()
    s["intensity_bgsub_raw"] = s["intensity_bgsub"]
    g = gain(models, s["channel"], s["y_px"], s["x_px"], shape_yx)
    s["flatfield_gain"] = g
    s["intensity_bgsub"] = s["intensity_bgsub"] / g
    if "partner_channel" in s:
        has = s["partner_channel"].notna()
        pg = np.ones(len(s))
        pg[has.to_numpy()] = gain(models, s.loc[has, "partner_channel"], s.loc[has, "y_px"], s.loc[has, "x_px"], shape_yx)
        s["partner_intensity_bgsub_raw"] = s["partner_intensity_bgsub"]
        s["partner_intensity_bgsub"] = s["partner_intensity_bgsub"] / pg
    return s
