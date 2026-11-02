"""Spot detection (big-fish LoG + local maxima) with deferred thresholding.

Spots are detected permissively (``permissive_factor`` x the per-FOV automated
threshold) and each spot keeps its LoG score. The final threshold is applied at
aggregation time, so thresholds can be calibrated on negative controls
(mrcB, NeoR in parental cells) without re-running detection.
"""

from __future__ import annotations

import numpy as np
from bigfish import detection, stack


def sigma_px(voxel_um: tuple, spot_radius_nm) -> tuple[float, float, float]:
    voxel_nm = tuple(float(v) * 1000 for v in voxel_um)
    return tuple(float(s) for s in detection.get_object_radius_pixel(voxel_nm, tuple(spot_radius_nm), ndim=3))


def elbow_curve(values: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Number of local maxima above each threshold (for threshold calibration plots)."""
    if values.size == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    # dense grid over the bulk + a grid up to the maximum: sparse bright spots (TSSs) live in the tail
    p = float(np.percentile(values, 99.99))
    thresholds = np.unique(np.concatenate([np.linspace(0, p, n_points), np.linspace(p, float(values.max()), n_points // 2)]))
    sorted_v = np.sort(values)
    counts = sorted_v.size - np.searchsorted(sorted_v, thresholds, side="right")
    return thresholds, counts


def detect_channel(img: np.ndarray, voxel_um: tuple, spot_radius_nm, permissive_factor: float, elbow_points: int = 100,
                   fixed_threshold: float | None = None, exclude_z_edges: int = 1):
    """Detect spots in one corrected channel (ZYX float32).

    Spots are stored down to ``permissive_factor`` x the automated threshold, or down to
    ``fixed_threshold`` (an absolute LoG value, e.g. below the mrcB-calibrated floor) if given.
    Local maxima in the first / last ``exclude_z_edges`` slices are ignored: their integration
    window is truncated, and on 250806 the first slice of Atto633 carries a camera artifact that
    produced ~48k false maxima and dragged the automated threshold into the noise.

    Returns ``(coords, log_score, info)`` where ``coords`` is (N, 3) int ZYX,
    ``log_score`` the LoG response at each spot and ``info`` holds the
    automated threshold, the permissive threshold used and the elbow curve.
    """
    sig = sigma_px(voxel_um, spot_radius_nm)
    filtered = stack.log_filter(img.astype(np.float32, copy=False), sig)
    mask_lm = detection.local_maximum_detection(filtered, min_distance=sig)
    if exclude_z_edges > 0:
        mask_lm[:exclude_z_edges] = False
        mask_lm[-exclude_z_edges:] = False
    auto = detection.automated_threshold_setting(filtered, mask_lm)
    lm_values = filtered[mask_lm]
    thresholds, counts = elbow_curve(lm_values, elbow_points)
    info = {
        "sigma_px": sig,
        "excluded_z_edges": exclude_z_edges,
        "auto_threshold": None if auto is None else float(auto),
        "elbow_thresholds": thresholds.tolist(),
        "elbow_counts": counts.tolist(),
    }
    if fixed_threshold is not None:
        thr = float(fixed_threshold)
    elif auto is None:
        info["detection_threshold"] = None
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.float32), info
    else:
        thr = float(auto) * permissive_factor
    coords, _ = detection.spots_thresholding(filtered, mask_lm, thr, remove_duplicate=True)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    info["detection_threshold"] = thr
    score = filtered[tuple(coords.T)].astype(np.float32)
    return coords, score, info


def subpixel(img: np.ndarray, coords: np.ndarray, voxel_um: tuple, spot_radius_nm) -> np.ndarray:
    """Optional Gaussian sub-pixel refinement (float ZYX in pixels)."""
    if len(coords) == 0:
        return coords.astype(float)
    voxel_nm = tuple(float(v) * 1000 for v in voxel_um)
    return detection.fit_subpixel(img.astype(np.float32, copy=False), coords, voxel_nm, tuple(spot_radius_nm))
