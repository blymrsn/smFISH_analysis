"""Per-spot intensity with local background, and single-transcript references.

Each spot's signal is the sum over a fixed physical ellipsoid around its centre
minus (local background x voxels). Local background is the median of an
ellipsoidal shell around the spot, so the result does not depend on a global
background constant or on spot-mask shape (unlike the legacy watershed masks /
overwritten balls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde


def ellipsoid_offsets(radius_px) -> np.ndarray:
    """Integer (dz, dy, dx) offsets inside an ellipsoid with the given semi-axes (px)."""
    r = np.asarray(radius_px, dtype=float)
    ext = np.ceil(r).astype(int)
    grid = np.mgrid[-ext[0] : ext[0] + 1, -ext[1] : ext[1] + 1, -ext[2] : ext[2] + 1].reshape(3, -1).T
    inside = ((grid / np.maximum(r, 1e-9)) ** 2).sum(1) <= 1.0
    return grid[inside].astype(np.int32)


def shell_offsets(inner_px, outer_px) -> np.ndarray:
    outer = ellipsoid_offsets(outer_px)
    r_in = np.asarray(inner_px, dtype=float)
    keep = ((outer / np.maximum(r_in, 1e-9)) ** 2).sum(1) > 1.0
    return outer[keep]


def nm_to_px(radius_nm, voxel_um) -> np.ndarray:
    return np.asarray(radius_nm, dtype=float) / (np.asarray(voxel_um, dtype=float) * 1000)


def _gather(img: np.ndarray, coords: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Values at coords + offsets, NaN outside the image. Shape (N, K)."""
    pts = coords[:, None, :].astype(np.int64) + offsets[None, :, :]
    shape = np.asarray(img.shape)
    valid = np.all((pts >= 0) & (pts < shape), axis=2)
    pts = np.clip(pts, 0, shape - 1)
    vals = img[pts[..., 0], pts[..., 1], pts[..., 2]].astype(np.float32)
    vals[~valid] = np.nan
    return vals


def measure_spots(
    img: np.ndarray,
    coords: np.ndarray,
    window_px,
    bg_inner_px,
    bg_outer_px,
    chunk: int = 2000,
) -> pd.DataFrame:
    """Integrated intensity with local background for each spot.

    Columns: ``intensity_sum`` (raw sum in window), ``n_voxels``, ``background``
    (median of shell), ``intensity_bgsub`` = sum - background*n_voxels, ``peak``.
    Spots whose window is clipped by the image edge get fewer voxels (``n_voxels``).
    """
    coords = np.asarray(coords).reshape(-1, 3)
    win = ellipsoid_offsets(window_px)
    shell = shell_offsets(bg_inner_px, bg_outer_px)
    out = {k: np.full(len(coords), np.nan, dtype=np.float64) for k in ["intensity_sum", "n_voxels", "background", "peak"]}
    for s in range(0, len(coords), chunk):
        c = np.round(coords[s : s + chunk]).astype(np.int64)
        w = _gather(img, c, win)
        b = _gather(img, c, shell)
        out["intensity_sum"][s : s + chunk] = np.nansum(w, axis=1)
        out["n_voxels"][s : s + chunk] = np.sum(~np.isnan(w), axis=1)
        out["peak"][s : s + chunk] = np.nanmax(w, axis=1)
        out["background"][s : s + chunk] = np.nanmedian(b, axis=1)
    df = pd.DataFrame(out)
    df["intensity_bgsub"] = df["intensity_sum"] - df["background"] * df["n_voxels"]
    return df


def nearest_neighbour_um(coords_px: np.ndarray, voxel_um) -> np.ndarray:
    """Distance (µm) from each spot to its nearest neighbour in the same set."""
    if len(coords_px) < 2:
        return np.full(len(coords_px), np.inf)
    pts = np.asarray(coords_px, dtype=float) * np.asarray(voxel_um)
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1]


def single_rna_reference(values, max_samples: int = 20000, seed: int = 0) -> dict:
    """Single-transcript intensity: mode of the (positive) intensity distribution.

    The mode is robust to the right tail of clusters / transcription sites that
    pulls the mean and median up. Returns mode, median and n.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 10:
        return {"mode": np.nan, "median": np.nan, "n": int(v.size)}
    lo, hi = np.percentile(v, [1, 99])
    v_in = v[(v >= lo) & (v <= hi)]
    rng = np.random.default_rng(seed)
    if v_in.size > max_samples:
        v_in = rng.choice(v_in, max_samples, replace=False)
    grid = np.linspace(lo, hi, 512)
    dens = gaussian_kde(v_in)(grid)
    return {"mode": float(grid[np.argmax(dens)]), "median": float(np.median(v)), "n": int(v.size)}
