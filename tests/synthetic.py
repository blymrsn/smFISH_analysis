"""Synthetic 3D smFISH field of view with known ground truth."""

from __future__ import annotations

import numpy as np

VOXEL = (0.3, 0.065, 0.065)  # z, y, x µm


def ellipsoid_mask(shape, centre, radii):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((zz - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2 + ((xx - centre[2]) / radii[2]) ** 2 <= 1


def add_spot(vol, pos, amplitude, sigma=(1.0, 1.6, 1.6)):
    ext = [int(np.ceil(4 * s)) for s in sigma]
    p = np.round(pos).astype(int)
    sl = tuple(slice(max(0, p[i] - ext[i]), min(vol.shape[i], p[i] + ext[i] + 1)) for i in range(3))
    grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in sl], indexing="ij")
    g = np.exp(-sum((grids[i] - pos[i]) ** 2 / (2 * sigma[i] ** 2) for i in range(3)))
    vol[sl] += amplitude * g


def make_fov(seed=0, n_cyto=70, n_tss=4, n_lone_intron=1, tss_copies=4.0, amp=300.0, intron_amp=200.0, shape=(24, 200, 200)):
    """One cell with a nucleus. Exon spots: ``n_cyto`` single mRNAs in the cytoplasm plus
    ``n_tss`` nuclear transcription sites of ``tss_copies`` x single brightness, each
    with a colocalised intron spot, plus ``n_lone_intron`` nuclear intron spots without exon signal."""
    rng = np.random.default_rng(seed)
    centre = (12, 100, 100)
    cell = ellipsoid_mask(shape, centre, (10, 90, 90))
    nuc = ellipsoid_mask(shape, centre, (5, 40, 40))
    cells = cell.astype(np.int32)
    nuclei = (nuc & cell).astype(np.int32)

    um = np.asarray(VOXEL)
    cyto_pts = np.argwhere(cell & ~nuc & ~ellipsoid_mask(shape, centre, (6, 48, 48)))
    cyto_pts = cyto_pts[(cyto_pts[:, 0] > 3) & (cyto_pts[:, 0] < shape[0] - 4)]
    chosen = []
    for idx in rng.permutation(len(cyto_pts)):
        p = cyto_pts[idx]
        if all(np.linalg.norm((p - q) * um) > 1.3 for q in chosen):
            chosen.append(p)
        if len(chosen) == n_cyto:
            break
    nuc_pts = np.argwhere(ellipsoid_mask(shape, centre, (3, 28, 28)))
    tss = []
    for idx in rng.permutation(len(nuc_pts)):
        p = nuc_pts[idx]
        if all(np.linalg.norm((p - q) * um) > 1.3 for q in tss):
            tss.append(p)
        if len(tss) == n_tss + n_lone_intron:
            break
    lone, tss = tss[n_tss:], tss[:n_tss]

    exon = np.zeros(shape, np.float32)
    intron = np.zeros(shape, np.float32)
    for p in chosen:
        add_spot(exon, p.astype(float), amp)
    for p in tss:
        add_spot(exon, p.astype(float), amp * tss_copies)
        add_spot(intron, p.astype(float), intron_amp)
    for p in lone:  # intron spot with no exon signal: must NOT be called a TSS
        add_spot(intron, p.astype(float), intron_amp)
    for v in (exon, intron):
        v += 20.0
        v[:] = rng.poisson(v).astype(np.float32)

    dapi = (nuc * 500 + 20).astype(np.float32)
    cona = (cell * 50 + 20).astype(np.float32)
    return {
        "exon": exon,
        "intron": intron,
        "dapi": dapi,
        "cona": cona,
        "cells": cells,
        "nuclei": nuclei,
        "cyto_spots": np.array(chosen),
        "tss_spots": np.array(tss),
        "lone_intron_spots": np.array(lone),
    }
