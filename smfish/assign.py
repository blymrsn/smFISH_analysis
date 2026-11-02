"""Assign spots to cells and compartments, with physical distances.

Assignment is a direct label lookup at each spot coordinate (no label-image
re-quantification). Distances use an anisotropic Euclidean distance transform
(``sampling = voxel size``) computed inside each cell's bounding box only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

COMPARTMENTS = ["nucleus", "perinuclear", "cytoplasm"]


def lookup(labels: np.ndarray, coords: np.ndarray) -> np.ndarray:
    c = np.round(np.asarray(coords)).astype(np.int64).reshape(-1, 3)
    c = np.clip(c, 0, np.asarray(labels.shape) - 1)
    return labels[c[:, 0], c[:, 1], c[:, 2]]


def spot_geometry(
    coords: np.ndarray,
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    voxel_um,
    perinuclear_um: float,
) -> pd.DataFrame:
    """Per-spot cell id, compartment and distances.

    Columns:
      cell_id            0 = outside any cell
      in_nucleus         bool
      compartment        nucleus | perinuclear | cytoplasm | extracellular
      dist_nucleus_um    signed distance to the nuclear envelope (negative inside), NaN if the cell has no nucleus
      dist_membrane_um   distance to the cell boundary (inside the cell)
    """
    coords = np.round(np.asarray(coords)).astype(np.int64).reshape(-1, 3)
    n = len(coords)
    cell_id = lookup(cell_labels, coords) if n else np.zeros(0, dtype=np.int32)
    in_nuc = (lookup(nuc_labels, coords) > 0) if n else np.zeros(0, dtype=bool)
    d_nuc = np.full(n, np.nan)
    d_mem = np.full(n, np.nan)
    sampling = tuple(float(v) for v in voxel_um)

    slices = ndimage.find_objects(cell_labels)
    for cid in np.unique(cell_id[cell_id > 0]):
        sl = slices[cid - 1]
        idx = np.flatnonzero(cell_id == cid)
        local = coords[idx] - np.array([s.start for s in sl])
        cell = np.pad(cell_labels[sl] == cid, 1)  # pad so the bbox edge counts as outside
        local = local + 1
        loc = tuple(local.T)
        d_mem[idx] = ndimage.distance_transform_edt(cell, sampling=sampling)[loc]
        nuc = np.pad(nuc_labels[sl] == cid, 1)
        if nuc.any():
            inside = ndimage.distance_transform_edt(nuc, sampling=sampling)[loc]
            outside = ndimage.distance_transform_edt(~nuc, sampling=sampling)[loc]
            d_nuc[idx] = np.where(nuc[loc], -inside, outside)

    comp = np.full(n, "extracellular", dtype=object)
    inside_cell = cell_id > 0
    comp[inside_cell] = "cytoplasm"
    comp[inside_cell & (d_nuc <= perinuclear_um) & ~in_nuc] = "perinuclear"
    comp[inside_cell & in_nuc] = "nucleus"
    return pd.DataFrame(
        {
            "cell_id": cell_id.astype(np.int32),
            "in_nucleus": in_nuc,
            "compartment": comp,
            "dist_nucleus_um": d_nuc,
            "dist_membrane_um": d_mem,
        }
    )


def cell_table(
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    voxel_um,
    dapi: np.ndarray | None = None,
    n_nuclei: dict | None = None,
) -> pd.DataFrame:
    """Per-cell morphology: volumes, centroid, border flags, nuclear DAPI content."""
    n = int(cell_labels.max())
    vox = float(np.prod(voxel_um))
    flat_c = cell_labels.ravel()
    cell_vox = np.bincount(flat_c, minlength=n + 1)
    nuc_vox = np.bincount(nuc_labels.ravel(), minlength=n + 1)[: n + 1]
    df = pd.DataFrame({"cell_id": np.arange(1, n + 1)})
    df["cell_volume_um3"] = cell_vox[1:] * vox
    df["nucleus_volume_um3"] = nuc_vox[1:] * vox
    df["cytoplasm_volume_um3"] = df["cell_volume_um3"] - df["nucleus_volume_um3"]
    df["has_nucleus"] = df["nucleus_volume_um3"] > 0
    if n_nuclei is not None:
        df["n_nuclei"] = df["cell_id"].map(n_nuclei).fillna(0).astype(int)

    cent = np.full((n, 3), np.nan)
    for i, sl in enumerate(ndimage.find_objects(cell_labels)):
        if sl is not None:
            pts = np.nonzero(cell_labels[sl] == i + 1)
            cent[i] = [p.mean() + s.start for p, s in zip(pts, sl)]
    for axis, name in enumerate(["z", "y", "x"]):
        df[f"centroid_{name}_um"] = cent[:, axis] * voxel_um[axis]

    def touching(planes):
        return set(np.unique(np.concatenate([p.ravel() for p in planes]))) - {0}

    border_xy = touching([cell_labels[:, 0, :], cell_labels[:, -1, :], cell_labels[:, :, 0], cell_labels[:, :, -1]])
    border_z = touching([cell_labels[0], cell_labels[-1]])
    nuc_border_z = touching([nuc_labels[0], nuc_labels[-1]])
    df["border_xy"] = df["cell_id"].isin(border_xy)
    df["border_z"] = df["cell_id"].isin(border_z)
    df["nucleus_border_z"] = df["cell_id"].isin(nuc_border_z)

    if dapi is not None:
        s = np.bincount(nuc_labels.ravel(), weights=dapi.ravel().astype(np.float64), minlength=n + 1)[: n + 1]
        df["nucleus_dapi_sum"] = s[1:]
        df["nucleus_dapi_mean"] = s[1:] / np.maximum(nuc_vox[1:], 1)
    return df[cell_vox[1:] > 0].reset_index(drop=True)
