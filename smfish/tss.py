"""Colocalisation between channels that target the same gene.

For every spot we record its nearest spot in the partner channel (distance in
µm) and the partner channel's background-subtracted intensity measured at the
spot itself. For an intron spot paired with an exon channel, that partner
intensity divided by the exon single-mRNA reference is the nascent-transcript
estimate at the transcription site (computed at aggregation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def nearest_partner(coords_a_px: np.ndarray, coords_b_px: np.ndarray, voxel_um) -> tuple[np.ndarray, np.ndarray]:
    """For each spot in A, distance (µm) and index of the nearest spot in B (-1 if B is empty)."""
    n = len(coords_a_px)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    if len(coords_b_px) == 0:
        return np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
    v = np.asarray(voxel_um, dtype=float)
    d, i = cKDTree(np.asarray(coords_b_px, float) * v).query(np.asarray(coords_a_px, float) * v, k=1)
    return d, i.astype(np.int64)


def recompute_partners(spots: pd.DataFrame, max_dist_nm: float, offsets_um: dict | None = None) -> pd.DataFrame:
    """Redo nearest-partner matching using only spots that pass the final threshold,
    after shifting each channel by its chromatic offset (``offsets_um[channel]`` = (dz, dy, dx)
    added to that channel's coordinates).

    Per-FOV matching at detection time used every permissively detected partner spot
    and no offset; this is the version all colocalisation / TSS calls should use.
    ``partner_intensity_bgsub`` (partner image measured at the spot) is unchanged.
    """
    if "partner_channel" not in spots:
        return spots
    offsets_um = offsets_um or {}
    s = spots.reset_index(drop=True).copy()
    pos = s[["z_um", "y_um", "x_um"]].to_numpy(float)
    pos = pos + np.array([offsets_um.get(c, (0.0, 0.0, 0.0)) for c in s["channel"]], dtype=float)
    has = s["partner_channel"].notna().to_numpy()
    dist = np.full(len(s), np.nan)
    pidx = np.full(len(s), -1, dtype=np.int64)
    dist[has] = np.inf

    passing = s.index[s["passes"].to_numpy()]
    targets = s.loc[passing].groupby(["fov_id", "channel"]).indices
    for (fov, ch, pch), rows in s[has].groupby(["fov_id", "channel", "partner_channel"]).indices.items():
        rows = s[has].index[rows]
        key = (fov, pch)
        if key not in targets:
            continue
        dst = passing[targets[key]]
        d, j = cKDTree(pos[dst]).query(pos[rows], k=1)
        dist[rows] = d
        pidx[rows] = s.loc[dst, "spot_index"].to_numpy()[j]
    s["partner_dist_um"] = dist
    s["partner_index"] = pidx
    s["partner_coloc"] = np.where(has, dist <= max_dist_nm / 1000.0, False)
    return s


def partner_columns(
    coords_a_px: np.ndarray,
    coords_b_px: np.ndarray,
    partner_channel: str,
    partner_measure: pd.DataFrame,
    voxel_um,
    max_dist_nm: float,
) -> pd.DataFrame:
    """Partner info for spots in channel A given spots in channel B.

    ``partner_measure`` is :func:`smfish.spotquant.measure_spots` of the partner
    image at A's coordinates (same window / background settings).
    """
    d, i = nearest_partner(coords_a_px, coords_b_px, voxel_um)
    return pd.DataFrame(
        {
            "partner_channel": partner_channel,
            "partner_dist_um": d,
            "partner_index": i,
            "partner_coloc": d <= max_dist_nm / 1000.0,
            "partner_intensity_bgsub": partner_measure["intensity_bgsub"].to_numpy(),
            "partner_background": partner_measure["background"].to_numpy(),
        }
    )
