"""3D cell + nucleus segmentation.

Used setting (``segmentation.cell_method: cpsam``, :func:`segment_fov_cpsam`): Cellpose-SAM (Cellpose 4) on
a downscaled, near-isotropic stack, z-cropped to the cell layer without the fibronectin substrate plane and
corrected for depth attenuation; its masks seed a watershed that snaps boundaries to ConA ridges (cells) and
DAPI edges (nuclei). Labels are padded back to the full z stack; nucleus label == cell label.

Also available: ``cellpose`` (Cellpose 3 cyto3 / nuclei models, the same watershed refinement) and
``nucleus_expansion`` (cells grown from nuclei, for plates without a usable membrane stain).

Throughout: diameters and shrink distances are in µm; seeds shrink by distance transforms; per-object cleanup
works on ``ndimage.find_objects`` crops; nuclei keep their own instances and take their cell's label.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
from skimage import filters, segmentation, transform

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- resampling


def downscale_xy(vol: np.ndarray, factor: float) -> np.ndarray:
    """Downscale Y and X only (Z untouched), anti-aliased, float32."""
    if factor == 1:
        return vol.astype(np.float32)
    out = transform.rescale(
        vol.astype(np.float32), (1, factor, factor), order=1, anti_aliasing=True, preserve_range=True
    )
    return out.astype(np.float32)


def upscale_labels_xy(labels: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour upscaling of a label volume to ``shape`` (Z must already match)."""
    if labels.shape[0] != shape[0]:
        raise ValueError(f"Z mismatch: {labels.shape} vs {shape}")
    iy = np.minimum((np.arange(shape[1]) * labels.shape[1] / shape[1]).astype(int), labels.shape[1] - 1)
    ix = np.minimum((np.arange(shape[2]) * labels.shape[2] / shape[2]).astype(int), labels.shape[2] - 1)
    return labels[:, iy][:, :, ix].astype(np.int32)


# --------------------------------------------------------------------------- cellpose


def run_cellpose(
    channels: list[np.ndarray],
    model_type: str,
    diameter_px: float,
    anisotropy: float,
    gpu: bool,
) -> np.ndarray:
    """Cellpose 3D on a list of ZYX volumes (``[cyto, nucleus]`` or ``[nucleus]``)."""
    from importlib.metadata import version

    from cellpose import models

    major = int(version("cellpose").split(".")[0])
    x = np.stack(channels, axis=0)  # C, Z, Y, X
    kwargs = dict(diameter=diameter_px, do_3D=True, anisotropy=anisotropy, channel_axis=0, z_axis=1)
    if major >= 4:
        model = models.CellposeModel(gpu=gpu)
        masks = model.eval(x, **kwargs)[0]
    else:
        model = models.Cellpose(model_type=model_type, gpu=gpu)
        chans = [1, 2] if len(channels) == 2 else [0, 0]
        masks = model.eval(x, channels=chans, **kwargs)[0]
    return np.asarray(masks, dtype=np.int32)


_SAM_MODELS: dict = {}


def run_cellpose_sam(
    channels: list[np.ndarray],
    diameter_px: float,
    anisotropy: float,
    gpu: bool,
    cellprob: float = 0.0,
    use_bfloat16: bool = False,
    pretrained_model: str = "cpsam_v2",
    flow_threshold: float = 0.4,
    normalize: dict | None = None,
    min_size: int = 15,
    batch_size: int = 8,
) -> np.ndarray:
    """Cellpose-SAM (Cellpose >= 4) 3D on a list of ZYX volumes (``[cyto, nucleus]`` or ``[nucleus]``).

    ``pretrained_model``: weights name, pinned to the trial's ``cpsam_v2`` (the Cellpose 4.2.1.1 default) so a
    future default change cannot silently swap weights. The model is loaded once per process. fp32 by default:
    bf16 is emulated on CPUs and on pre-Ampere
    GPUs (V100), and gave identical masks in the trial. ``normalize`` defaults to per-plane
    (``norm3D: False``), as in the trial. Raises if ``gpu`` is requested but Cellpose fell back to CPU
    (a silent CPU run would blow the PBS walltime).
    """
    from importlib.metadata import version

    from cellpose import models

    if int(version("cellpose").split(".")[0]) < 4:
        raise RuntimeError(f"cell_method cpsam needs Cellpose >= 4, found {version('cellpose')} "
                           "(install the segmentation extras: pip install -e \".[segmentation]\")")
    key = (bool(gpu), bool(use_bfloat16), pretrained_model)
    if key not in _SAM_MODELS:
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model, use_bfloat16=use_bfloat16)
        device = str(getattr(model, "device", "cpu"))
        if gpu and not device.startswith("cuda"):
            raise RuntimeError(f"segmentation.gpu is true but Cellpose is running on {device}")
        log.info("Cellpose-SAM %s loaded on %s (bfloat16=%s)", pretrained_model, device, use_bfloat16)
        _SAM_MODELS[key] = model
    kwargs = dict(
        do_3D=True, anisotropy=anisotropy, diameter=diameter_px, cellprob_threshold=cellprob,
        flow_threshold=flow_threshold, normalize=normalize if normalize is not None else {"norm3D": False},
        min_size=min_size, batch_size=batch_size,
    )
    if len(channels) > 1:
        x = np.stack(channels, axis=0)  # C, Z, Y, X
        kwargs.update(channel_axis=0, z_axis=1)
    else:
        x = channels[0]
        kwargs.update(z_axis=0)
    masks = _SAM_MODELS[key].eval(x, **kwargs)[0]
    return np.asarray(masks, dtype=np.int32)


# --------------------------------------------------------------------------- z range / intensity (cpsam)


def robust_foreground(vol: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    """Otsu foreground with the threshold computed on a percentile-clipped copy.

    Plain Otsu fails on these stacks: one saturating artefact (ConA max 51128 vs p99 919 on a 250806
    full field) drags the threshold to 13802 and leaves 0.01% of voxels as foreground.
    """
    v = vol.astype(np.float32, copy=False)
    return v > filters.threshold_otsu(np.clip(v, None, np.percentile(v, clip_pct)))


def substrate_planes(cona: np.ndarray, min_area_frac: float = 0.5) -> list[int]:
    """Planes where ConA foreground covers most of the field: the stained fibronectin coating, not cells.

    250806 W13: z12 79% coverage (in/out contrast 1.006, 35% of all ConA-positive voxels), z13 31%
    (real cell contrast, kept), deeper planes 5-11%. Absolute criterion: a ratio-to-median rule also
    flagged z13.
    """
    fg = robust_foreground(cona)
    return [int(z) for z in range(fg.shape[0]) if fg[z].mean() > min_area_frac]


def cell_layer_z(dapi: np.ndarray, min_area_frac: float = 0.02) -> tuple[int, int]:
    """(z_start, z_stop) of the planes whose DAPI foreground covers more than ``min_area_frac`` of the field.

    Threshold from clipped Otsu on a subsampled stack (the trial used plain Otsu, which a single saturating
    voxel can break). Falls back to the whole stack if no plane qualifies.
    """
    sub = dapi[::2, ::4, ::4].astype(np.float32)
    th = filters.threshold_otsu(np.clip(sub, None, np.percentile(sub, 99.5)))
    area = np.array([(dapi[z, ::4, ::4] > th).mean() for z in range(dapi.shape[0])])
    nz = np.flatnonzero(area > min_area_frac)
    if nz.size == 0:
        return 0, dapi.shape[0]
    return int(nz.min()), int(nz.max()) + 1


def depth_correct(vol: np.ndarray, p: float = 99.0, smooth_z: int = 5, floor_frac: float = 0.25) -> np.ndarray:
    """Undo confocal depth attenuation by flattening a per-plane high percentile (segmentation input only).

    The per-plane statistic is median-smoothed over z so planes with genuinely little signal are not
    amplified into noise, and the gain is clipped to [floor_frac, 1 / floor_frac].
    """
    vol = vol.astype(np.float32)
    bg = np.percentile(vol, 1)
    sig = np.maximum(np.array([np.percentile(vol[z], p) for z in range(vol.shape[0])]) - bg, 1e-3)
    sm = ndimage.median_filter(sig, size=smooth_z, mode="nearest")
    gain = np.clip(np.median(sm) / sm, floor_frac, 1.0 / floor_frac)
    return ((vol - bg) * gain[:, None, None]).astype(np.float32)


# --------------------------------------------------------------------------- watershed refinement


def n_workers() -> int:
    """Threads for slice-wise filters: PBS ncpus if set, else CPU count (max 8). scipy / skimage
    filters release the GIL, so threads give a near-linear speed-up on the per-slice work."""
    n = os.environ.get("NCPUS") or os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1
    return max(1, min(int(n), 8))


def _map_slices(fn, n_slices: int):
    with ThreadPoolExecutor(max_workers=n_workers()) as ex:
        return list(ex.map(fn, range(n_slices)))


def _shrink_from_boundaries(labels2d: np.ndarray, shrink_px: float) -> np.ndarray:
    """Keep label pixels further than ``shrink_px`` from any label boundary."""
    if shrink_px <= 0:
        return labels2d
    bounds = segmentation.find_boundaries(labels2d, mode="inner")
    far = ndimage.distance_transform_edt(~bounds) > shrink_px
    return np.where(far, labels2d, 0)


def refine_watershed(
    labels: np.ndarray,
    edge_image: np.ndarray,
    seed_shrink_px: float,
    background_shrink_px: float,
) -> np.ndarray:
    """Slice-wise seeded watershed that snaps coarse 3D labels to image edges.

    Seeds: each label shrunk away from its boundaries; background seed: the
    label-free area shrunk away from objects.
    """
    out = np.zeros_like(labels, dtype=np.int32)

    def one(z):
        lab = labels[z]
        if lab.max() == 0:
            return
        seeds = _shrink_from_boundaries(lab, seed_shrink_px).astype(np.int32) + 1
        seeds[lab == 0] = 0
        seeds[seeds == 1] = 0
        background = ndimage.distance_transform_edt(lab == 0) > background_shrink_px
        seeds[background] = 1
        ws = segmentation.watershed(edge_image[z], markers=seeds)
        out[z] = np.where(ws > 1, ws - 1, 0)

    _map_slices(one, labels.shape[0])
    return out


def column_majority(labels: np.ndarray) -> np.ndarray:
    """Make cell ownership z-consistent: every labelled voxel in an XY column goes to the cell that owns
    most of that column. Foreground (cell 3D extent / height) is unchanged; only slice-to-slice owner
    flips of thin sheets between neighbours are removed. Suits a monolayer, where true stacking of two
    cells in one column is rare (a thin sheet under a neighbour's body is given to that neighbour).
    """
    best_n = np.zeros(labels.shape[1:], dtype=np.int32)
    best_l = np.zeros(labels.shape[1:], dtype=np.int32)
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        n = (labels[sl] == lab).sum(0).astype(np.int32)
        bn, bl = best_n[sl[1:]], best_l[sl[1:]]
        better = n > bn
        bn[better], bl[better] = n[better], lab
    return np.where(labels > 0, best_l[None], 0).astype(np.int32)


def nucleus_expansion_cells(
    nuclei: np.ndarray,
    voxel_um,
    max_expand_um: float = 10.0,
    z_glass: int | None = None,
    top_margin_um: float = 1.0,
) -> np.ndarray:
    """Approximate cells by growing each nucleus outward (for experiments where the membrane stain
    fails). In XY, every nucleus footprint expands into the space around it, split between neighbours by
    distance (watershed on the distance transform), up to ``max_expand_um`` from the nuclear edge.
    In z (monolayer), each cell spans from the glass plane ``z_glass`` (or its nucleus bottom) to
    ``top_margin_um`` above its nucleus top. Cell labels equal nucleus labels.
    """
    vz, vy = float(voxel_um[0]), float(voxel_um[1])
    nuc2d = column_majority(nuclei).max(0)
    dist = ndimage.distance_transform_edt(nuc2d == 0) * vy
    foot = segmentation.watershed(dist, markers=nuc2d, mask=dist <= max_expand_um)
    n = int(nuclei.max())
    zmin = np.full(n + 1, nuclei.shape[0], dtype=np.int32)
    zmax = np.full(n + 1, -1, dtype=np.int32)
    for lab, sl in enumerate(ndimage.find_objects(nuclei), start=1):
        if sl is not None:
            zs = np.flatnonzero((nuclei[sl] == lab).any((1, 2))) + sl[0].start
            zmin[lab], zmax[lab] = zs.min(), zs.max()
    top = np.minimum(zmax + int(round(top_margin_um / vz)), nuclei.shape[0] - 1)
    bottom = np.minimum(zmin, z_glass) if z_glass is not None else zmin
    z = np.arange(nuclei.shape[0])[:, None, None]
    inside = (z >= bottom[foot][None]) & (z <= top[foot][None]) & (foot[None] > 0)
    return np.where(inside, foot[None], 0).astype(np.int32)


def ridge_filter(vol: np.ndarray, sigmas_px: list[float], smooth_px: float) -> np.ndarray:
    """Bright-ridge (membrane) enhancement per slice: sato(gaussian(ConA))."""
    out = np.empty(vol.shape, dtype=np.float32)

    def one(z):
        sm = filters.gaussian(vol[z], sigma=smooth_px, preserve_range=True)
        out[z] = filters.sato(sm, sigmas=sigmas_px, black_ridges=False)

    _map_slices(one, vol.shape[0])
    return out


def edge_filter(vol: np.ndarray, smooth_px: float) -> np.ndarray:
    """Edge magnitude per slice (nuclear boundary): roberts(gaussian(DAPI))."""
    out = np.empty(vol.shape, dtype=np.float32)

    def one(z):
        out[z] = filters.sobel(filters.gaussian(vol[z], sigma=smooth_px, preserve_range=True))

    _map_slices(one, vol.shape[0])
    return out


# --------------------------------------------------------------------------- cleanup


def clean_labels(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    """Keep the largest 3D connected component of each label, fill holes per slice,
    drop objects smaller than ``min_voxels`` and relabel 1..N."""
    out = np.zeros_like(labels, dtype=np.int32)
    new_id = 0
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        obj = labels[sl] == lab
        comp, n = ndimage.label(obj)
        if n > 1:
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            obj = comp == sizes.argmax()
        if obj.sum() < min_voxels:
            continue
        obj = np.stack([ndimage.binary_fill_holes(s) for s in obj])
        new_id += 1
        region = out[sl]
        region[obj & (region == 0)] = new_id
    return out


def assign_nuclei_to_cells(nuc_labels: np.ndarray, cell_labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Relabel nuclei with the label of the cell they overlap most, clipped to that cell.

    Returns the relabelled nucleus volume (nucleus label == cell label; several
    nuclei in one cell are merged into that label) and ``{cell_label: n_nuclei}``.
    Nuclei that do not overlap a cell are dropped.
    """
    out = np.zeros_like(cell_labels, dtype=np.int32)
    n_nuclei: dict[int, int] = {}
    for lab, sl in enumerate(ndimage.find_objects(nuc_labels), start=1):
        if sl is None:
            continue
        obj = nuc_labels[sl] == lab
        cells = cell_labels[sl][obj]
        cells = cells[cells > 0]
        if cells.size == 0:
            continue
        cell = int(np.bincount(cells).argmax())
        keep = obj & (cell_labels[sl] == cell)
        out[sl][keep] = cell
        n_nuclei[cell] = n_nuclei.get(cell, 0) + 1
    return out, n_nuclei


# --------------------------------------------------------------------------- segmentation QC


def boundary_ridge_enrichment(labels: np.ndarray, ridge: np.ndarray) -> float:
    """Mean ridge signal on object boundaries / mean inside objects (higher = boundaries on ConA ridges).

    Biased upward by under-segmentation (perinuclear ER is ConA-bright): read with ``signal_coverage``.
    """
    if labels.max() == 0:
        return float("nan")
    b = segmentation.find_boundaries(labels, mode="inner")
    inside = ridge[labels > 0]
    if b.sum() == 0 or inside.size == 0:
        return float("nan")
    return float(ridge[b].mean() / max(inside.mean(), 1e-9))


def signal_coverage(labels: np.ndarray, fg_mask: np.ndarray) -> float:
    """Fraction of stain-positive voxels (independent foreground mask) that fall inside some object."""
    if fg_mask.sum() == 0:
        return float("nan")
    return float((labels[fg_mask] > 0).mean())


# --------------------------------------------------------------------------- driver


def segment_fov(
    dapi: np.ndarray, cona: np.ndarray, voxel_um: tuple, cfg: dict, dapi_seg: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Segment cells and nuclei. Returns (cell_labels, nucleus_labels, info).

    ``dapi_seg``: optional DAPI image used only for nuclear segmentation (e.g. blimp log-z-score
    corrected, which rescues dim / uneven DAPI); ``dapi`` stays the linear image.
    ``segmentation.cell_method``:
      - ``cellpose`` (default): Cellpose 3D on [ConA, DAPI] + watershed snap to ConA ridges.
      - ``nucleus_expansion``: no membrane stain needed; cells grown from nuclei (see
        :func:`nucleus_expansion_cells`). Use when ConA fails (250929). Cell outlines are approximate
        (~85% of spots land in the same cell as with ConA on 250806) and cell volumes are overestimated:
        use nuclear volume as the size covariate.
      - ``cpsam``: Cellpose-SAM (Cellpose >= 4), see :func:`segment_fov_cpsam`.
    """
    sc = cfg["segmentation"]
    if sc.get("cell_method", "cellpose") == "cpsam":
        return segment_fov_cpsam(dapi, cona, voxel_um, sc, dapi_seg=dapi_seg)
    vz, vy, _ = voxel_um
    f = float(sc["xy_downscale"])
    xy_small = vy / f
    anisotropy = vz / xy_small
    shape = dapi.shape
    method = sc.get("cell_method", "cellpose")
    dseg = dapi if dapi_seg is None else dapi_seg
    smooth = sc["smooth_sigma_um"] / vy
    voxel_vol = float(np.prod(voxel_um))

    dapi_s = downscale_xy(dseg, f)
    log.info("Cellpose nuclei: diameter %.1f px, anisotropy %.2f", sc["nucleus_diameter_um"] / xy_small, anisotropy)
    nuclei = upscale_labels_xy(
        run_cellpose([dapi_s], sc["nucleus_model"], sc["nucleus_diameter_um"] / xy_small, anisotropy, sc["gpu"]), shape
    )
    if sc.get("refine_with_watershed", True):
        nuclei = refine_watershed(
            nuclei, edge_filter(dseg, smooth), sc["nucleus_seed_shrink_um"] / vy, _background_shrink_um(sc, "nucleus") / vy
        )
    nuclei = clean_labels(nuclei, int(sc["min_nucleus_volume_um3"] / voxel_vol))

    if method == "nucleus_expansion":
        zprof = np.array([np.percentile(cona[z, ::4, ::4], 99) for z in range(cona.shape[0])])
        z_glass = int(np.argmax(zprof))  # ConA (even when it fails as a membrane stain) peaks at the glass
        cells = nucleus_expansion_cells(nuclei, voxel_um, sc.get("max_expand_um", 10.0), z_glass,
                                        sc.get("expansion_top_margin_um", 1.0))
        n_nuclei = {int(l): 1 for l in np.unique(nuclei[nuclei > 0])}
        extra = {"z_glass": z_glass, "max_expand_um": sc.get("max_expand_um", 10.0)}
    elif method == "cellpose":
        cona_s = downscale_xy(cona, f)
        log.info("Cellpose cells: diameter %.1f px", sc["cell_diameter_um"] / xy_small)
        cells = upscale_labels_xy(
            run_cellpose([cona_s, dapi_s], sc["cell_model"], sc["cell_diameter_um"] / xy_small, anisotropy, sc["gpu"]),
            shape,
        )
        del cona_s
        if sc.get("refine_with_watershed", True):
            ridges = ridge_filter(cona, [s_ / vy for s_ in sc["ridge_sigmas_um"]], smooth)
            cells = refine_watershed(cells, ridges, sc["cell_seed_shrink_um"] / vy, _background_shrink_um(sc, "cell") / vy)
            del ridges
        if sc.get("column_consistent_cells", True):
            cells = column_majority(cells)
        cells = clean_labels(cells, int(sc["min_cell_volume_um3"] / voxel_vol))
        nuclei, n_nuclei = assign_nuclei_to_cells(nuclei, cells)
        extra = {}
    else:
        raise ValueError(f"Unknown segmentation.cell_method {method!r}")
    del dapi_s

    info = {
        "cell_method": method,
        "nucleus_input": sc.get("nucleus_input", "linear"),
        "xy_downscale": f,
        "cellpose_anisotropy": anisotropy,
        "column_consistent_cells": True if method == "nucleus_expansion" else bool(sc.get("column_consistent_cells", True)),
        "n_cells": int(len(np.unique(cells[cells > 0]))),
        "n_cells_with_nucleus": len(n_nuclei),
        "n_nuclei_per_cell": n_nuclei,
        **extra,
    }
    return cells, nuclei, info


def _background_shrink_um(sc: dict, kind: str) -> float:
    """``<kind>_background_shrink_um`` if set, else the shared ``background_seed_shrink_um``."""
    key = f"{kind}_background_shrink_um"
    return float(sc[key] if key in sc else sc["background_seed_shrink_um"])


def segment_fov_cpsam(
    dapi: np.ndarray, cona: np.ndarray, voxel_um: tuple, sc: dict, dapi_seg: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cellpose-SAM segmentation (``cell_method: cpsam``):

    1. z range: the DAPI cell layer (:func:`cell_layer_z`), minus fibronectin substrate planes at its
       bottom (:func:`substrate_planes`; ConA stains the coating, so that plane carries no cell contrast).
    2. Depth-correct DAPI and ConA, downscale XY to near-isotropic voxels (0.25 -> anisotropy ~1.15).
    3. Cellpose-SAM 3D: nuclei on DAPI, cells on [ConA, DAPI] (with ConA alone only 87-98% of nuclei fell
       inside a cell), cellprob -2 for cells. Volume cleanup at the small scale, then upscale.
    4. Watershed snap at full XY: cells to ConA ridges, nuclei to DAPI edges.
    5. Optional column-majority (off in the trial: these cells can genuinely overlap in 3D).
    6. Nuclei take the label of their cell. Labels are padded back to the full z stack (zero outside
       the segmented range; ``fill_dropped_planes`` copies the lowest segmented plane into dropped
       substrate planes instead). Border objects are kept: aggregation drops border cells.

    ``dapi_seg`` (e.g. 250929 log-z-score DAPI) replaces ``dapi`` for every segmentation step.
    """
    from importlib.metadata import version

    vz, vy, _ = voxel_um
    dseg = dapi if dapi_seg is None else dapi_seg
    nz = dapi.shape[0]
    f = float(sc["xy_downscale"])
    xy_small = vy / f
    anisotropy = vz / xy_small
    smooth = sc["smooth_sigma_um"] / vy
    gpu = bool(sc["gpu"])
    bf16 = bool(sc.get("use_bfloat16", False))
    weights = sc.get("cellpose_model", "cpsam_v2")
    batch = int(sc.get("cellpose_batch_size", 8))

    za, zb = cell_layer_z(dseg, sc.get("cell_layer_min_area_frac", 0.02)) if sc.get("crop_cell_layer", True) else (0, nz)
    # DAPI area falls off gradually above the nuclei: on 250806 W13 the trial's top plane (z35) still had
    # 16% cell / 7% nucleus coverage, so cells were cut flat at the top. The margin extends the range up.
    zb = min(nz, zb + int(round(sc.get("cell_layer_top_margin_um", 0.0) / vz)))
    dropped: list[int] = []
    if sc.get("drop_substrate_plane", True):
        sub = substrate_planes(cona[za:zb], sc.get("substrate_min_area_frac", 0.5))
        run = 0
        while run < len(sub) and sub[run] == run:  # only a contiguous run at the bottom is the coverslip
            run += 1
        # The coating is a thin sheet (1-2 planes; 250806 W16 drops 2). A longer run means ConA covers the
        # field through the cell layer (250806 W1, 250929 W2/W11 dropped 28-50 planes and crashed Cellpose),
        # so coverage cannot pick out the substrate in this FOV: drop nothing.
        max_run = int(sc.get("substrate_max_planes", 2))
        if run > max_run:
            log.warning("ConA covers >%.0f%% of the field in %d bottom planes (max %d): no substrate plane dropped",
                        100 * sc.get("substrate_min_area_frac", 0.5), run, max_run)
        else:
            dropped = list(range(za, za + run))
            za += run
    log.info("Cellpose-SAM z range %d:%d of %d (substrate planes dropped: %s)", za, zb, nz, dropped)

    dc, cc = depth_correct(dseg[za:zb]), depth_correct(cona[za:zb])
    shape = dc.shape
    d_s, c_s = downscale_xy(dc, f), downscale_xy(cc, f)
    vvol_s = vz * xy_small * xy_small

    log.info("Cellpose-SAM nuclei: diameter %.1f px, anisotropy %.2f", sc["nucleus_diameter_um"] / xy_small, anisotropy)
    nuc = run_cellpose_sam([d_s], sc["nucleus_diameter_um"] / xy_small, anisotropy, gpu,
                           cellprob=sc.get("cellprob_nuclei", 0.0), use_bfloat16=bf16, pretrained_model=weights,
                           batch_size=batch)
    log.info("Cellpose-SAM cells: diameter %.1f px", sc["cell_diameter_um"] / xy_small)
    cells = run_cellpose_sam([c_s, d_s], sc["cell_diameter_um"] / xy_small, anisotropy, gpu,
                             cellprob=sc.get("cellprob_cells", -2.0), use_bfloat16=bf16, pretrained_model=weights,
                             batch_size=batch)
    del d_s, c_s
    nuc = upscale_labels_xy(clean_labels(nuc, int(sc["min_nucleus_volume_um3"] / vvol_s)), shape)
    cells = upscale_labels_xy(clean_labels(cells, int(sc["min_cell_volume_um3"] / vvol_s)), shape)

    ridges = ridge_filter(cc, [s_ / vy for s_ in sc["ridge_sigmas_um"]], smooth)
    if sc.get("refine_with_watershed", True):
        cells = refine_watershed(cells, ridges, sc["cell_seed_shrink_um"] / vy, _background_shrink_um(sc, "cell") / vy)
        nuc = refine_watershed(nuc, edge_filter(dc, smooth), sc["nucleus_seed_shrink_um"] / vy,
                               _background_shrink_um(sc, "nucleus") / vy)
    column = bool(sc.get("column_consistent_cells", False))
    if column:
        cells = column_majority(cells)
    n_nuc_objects = sum(s is not None for s in ndimage.find_objects(nuc))
    nuc, n_nuclei = assign_nuclei_to_cells(nuc, cells)

    counts = np.array([n_nuclei.get(int(l), 0) for l in np.unique(cells[cells > 0])])
    qc = {
        "boundary_ridge_enrichment": boundary_ridge_enrichment(cells, ridges),
        "cona_signal_coverage": signal_coverage(cells, robust_foreground(cc)),
        "n_nuclei": int(n_nuc_objects),
        "n_orphan_nuclei": int(n_nuc_objects - sum(n_nuclei.values())),
        "frac_cells_mononucleate": float((counts == 1).mean()) if counts.size else float("nan"),
        "frac_cells_no_nucleus": float((counts == 0).mean()) if counts.size else float("nan"),
        # cells reaching the top segmented plane are probably cut off by the z range
        "frac_cells_at_top_plane": float(len(np.unique(cells[-1][cells[-1] > 0])) / counts.size) if counts.size else float("nan"),
    }
    del ridges, dc, cc

    full_cells = np.zeros((nz, *shape[1:]), dtype=np.int32)
    full_nuc = np.zeros_like(full_cells)
    full_cells[za:zb], full_nuc[za:zb] = cells, nuc
    fill = bool(sc.get("fill_dropped_planes", False))
    if fill:
        for z in dropped:
            full_cells[z], full_nuc[z] = cells[0], nuc[0]

    info = {
        "cell_method": "cpsam",
        "nucleus_input": sc.get("nucleus_input", "linear"),
        "cellpose_version": version("cellpose"),
        "cellpose_model": weights,
        "xy_downscale": f,
        "cellpose_anisotropy": anisotropy,
        "column_consistent_cells": column,
        "z_range": [int(za), int(zb)],
        "substrate_planes_dropped": dropped,
        "fill_dropped_planes": fill,
        "n_cells": int(counts.size),
        "n_cells_with_nucleus": len(n_nuclei),
        "n_nuclei_per_cell": n_nuclei,
        **qc,
    }
    return full_cells, full_nuc, info
