"""Per-FOV processing: load -> correct -> segment -> detect -> measure -> assign -> write.

One call processes one field of view and writes, under
``<output_dir>/fov/<fov_id>/``:

- ``spots.parquet``  one row per detected spot, every FISH channel
- ``cells.parquet``  one row per cell (morphology, border flags, nuclear DAPI)
- ``labels.tif``     cell / nucleus label volumes (nucleus label == cell label)
- ``qc.json``        detection thresholds, elbow curves, segmentation and illumination info
- ``overlay.png``    max-projection with outlines and spots
- ``DONE.json`` or ``FAILED.json``
"""

from __future__ import annotations

import json
import logging
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .assign import cell_table, spot_geometry
from .detect import detect_channel, subpixel
from .io import Illumination, load_channels, save_labels, write_table
from .layout import channel_pairs, load_layout, parse_well, well_channels
from .qc import overlay_png
from .spotquant import measure_spots, nearest_neighbour_um, nm_to_px
from .tss import partner_columns

log = logging.getLogger(__name__)


def fov_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def fov_dir(cfg: dict, fov_id: str) -> Path:
    return Path(cfg["paths"]["output_dir"]) / "fov" / fov_id


def run_fov(
    fov_path: str | Path,
    cfg: dict,
    layout: pd.DataFrame | None = None,
    overwrite: bool = False,
    images: dict | None = None,
    voxel_um: tuple | None = None,
    labels: tuple | None = None,
) -> str:
    """Process one FOV. Returns "done" or "skipped"; writes FAILED.json and re-raises on error.

    ``images`` (corrected ZYX float volumes by channel name), ``voxel_um`` and
    ``labels`` (cells, nuclei) can be injected for testing or re-use.
    """
    fov_path = Path(fov_path)
    fov_id = fov_id_from_path(fov_path)
    out = fov_dir(cfg, fov_id)
    if (out / "DONE.json").exists() and not overwrite:
        log.info("%s already done, skipping", fov_id)
        return "skipped"
    out.mkdir(parents=True, exist_ok=True)
    (out / "FAILED.json").unlink(missing_ok=True)
    t0 = time.time()
    try:
        _run(fov_path, fov_id, out, cfg, layout, images, voxel_um, labels)
    except Exception as e:
        (out / "FAILED.json").write_text(
            json.dumps({"fov_id": fov_id, "error": repr(e), "traceback": traceback.format_exc()}, indent=2)
        )
        raise
    done = {
        "fov_id": fov_id,
        "seconds": round(time.time() - t0, 1),
        "smfish_version": __version__,
        "host": platform.node(),
        "config": cfg.get("_config_path"),
    }
    (out / "DONE.json").write_text(json.dumps(done, indent=2))
    return "done"


def _run(fov_path, fov_id, out, cfg, layout, images, voxel_um, labels):
    if layout is None:
        layout = load_layout(cfg)
    well = parse_well(fov_path.name, cfg.get("well_regex", r"Well(\d+)"))
    chans = well_channels(layout, well)
    condition = chans["condition"].iloc[0]
    nuc_ch, cell_ch = cfg["nuclear_channel"], cfg["cell_channel"]
    qc: dict = {"fov_id": fov_id, "well": well, "condition": condition, "channels": {}}

    # ---- load + illumination
    dapi_seg = None
    if images is None:
        raw, voxel_um = load_channels(fov_path, cfg, [nuc_ch, cell_ch, *chans["channel"]])
        dapi_seg = nucleus_seg_image(cfg, raw[nuc_ch])
        illum = Illumination(cfg)
        images = {}
        for name in list(raw):
            images[name] = illum.correct(name, raw.pop(name))
        qc["illumination"] = illum.describe()
    else:
        if voxel_um is None:
            raise ValueError("voxel_um is required when images are injected")
        qc["illumination"] = {"method": "provided"}
    qc["voxel_um"] = list(voxel_um)
    qc["image_shape_zyx"] = list(images[nuc_ch].shape)

    # ---- segmentation
    n_nuclei = None
    if labels is not None:
        cells, nuclei = labels
        qc["segmentation"] = {"source": "provided"}
    else:
        from .segment import segment_fov

        cells, nuclei, seg_info = segment_fov(images[nuc_ch], images[cell_ch], voxel_um, cfg, dapi_seg=dapi_seg)
        n_nuclei = seg_info.pop("n_nuclei_per_cell")
        qc["segmentation"] = {"source": "cellpose", **seg_info}
    if cells.shape != images[nuc_ch].shape:
        raise ValueError(f"Label shape {cells.shape} != image shape {images[nuc_ch].shape}")
    save_labels(out / "labels.tif", cells, nuclei, voxel_um)

    cells_df = cell_table(cells, nuclei, voxel_um, dapi=images[nuc_ch], n_nuclei=n_nuclei)
    cells_df.insert(0, "fov_id", fov_id)
    cells_df.insert(1, "well", well)
    cells_df.insert(2, "condition", condition)
    cells_df["cell_method"] = qc["segmentation"].get("cell_method", qc["segmentation"].get("source"))

    # ---- spots per FISH channel
    det, q = cfg["detection"], cfg["quantification"]
    win = nm_to_px(q["window_radius_nm"], voxel_um)
    bg_in = nm_to_px(q["background_inner_nm"], voxel_um)
    bg_out = nm_to_px(q["background_outer_nm"], voxel_um)
    tables, coords = {}, {}
    for _, ch in chans.iterrows():
        name, role = ch["channel"], ch["role"]
        c, score, dinfo = detect_channel(
            images[name], voxel_um, det["spot_radius_nm"][role], det["permissive_factor"], det.get("elbow_points", 100),
            fixed_threshold=_floor(det.get("permissive_threshold") or {}, name, role),
            exclude_z_edges=det.get("exclude_z_edge_slices", 1),
        )
        qc["channels"][name] = {"probe": ch["probe"], "role": role, "n_detected": int(len(c)), **dinfo}
        log.info("%s %s (%s): %d spots (auto thr %s)", fov_id, name, ch["probe"], len(c), dinfo["auto_threshold"])
        df = pd.DataFrame(c, columns=["z_px", "y_px", "x_px"])
        if det.get("subpixel_fit") and len(c):
            sub = subpixel(images[name], c, voxel_um, det["spot_radius_nm"][role])
            df[["z_sub_px", "y_sub_px", "x_sub_px"]] = sub
        for axis, v in zip("zyx", voxel_um):
            df[f"{axis}_um"] = df[f"{axis}_px"] * v
        df["log_score"] = score
        df["auto_threshold"] = np.nan if dinfo["auto_threshold"] is None else dinfo["auto_threshold"]
        df["nn_dist_um"] = nearest_neighbour_um(c, voxel_um)
        df = pd.concat(
            [
                df,
                measure_spots(images[name], c, win, bg_in, bg_out),
            ],
            axis=1,
        )
        meta = {"fov_id": fov_id, "well": well, "condition": condition, "channel": name, "probe": ch["probe"],
                "gene": ch["gene"], "role": role, "allele": ch["allele"], "expected_absent": bool(ch["expected_absent"])}
        for i, (k, v) in enumerate(meta.items()):
            df.insert(i, k, v)
        df.insert(len(meta), "spot_index", np.arange(len(df)))
        tables[name], coords[name] = df, c

    # ---- cell / compartment / distances: one pass over all channels (per-cell distance transforms are the slow part)
    names = list(tables)
    all_c = np.concatenate([coords[n] for n in names]) if names else np.zeros((0, 3), dtype=np.int64)
    geom = spot_geometry(all_c, cells, nuclei, voxel_um, cfg["compartments"]["perinuclear_shell_um"])
    start = 0
    for n in names:
        k = len(coords[n])
        tables[n] = pd.concat([tables[n], geom.iloc[start : start + k].reset_index(drop=True)], axis=1)
        start += k

    # ---- same-gene partners (intron <-> exon colocalisation, nascent signal at TSS)
    for a, b in channel_pairs(layout, well):
        for src, dst in ((a, b), (b, a)):
            pm = measure_spots(images[dst], coords[src], win, bg_in, bg_out)
            pc = partner_columns(coords[src], coords[dst], dst, pm, voxel_um, cfg["tss"]["colocalisation_distance_nm"])
            tables[src] = pd.concat([tables[src], pc], axis=1)

    spots = pd.concat(tables.values(), ignore_index=True)
    write_table(spots, out / "spots")
    write_table(cells_df, out / "cells")
    qc["n_cells"] = int(len(cells_df))
    (out / "qc.json").write_text(json.dumps(qc, indent=2, default=_json_default))
    overlay_png(out / "overlay.png", images[nuc_ch], cells, nuclei, spots)


def nucleus_seg_image(cfg: dict, raw_dapi: np.ndarray):
    """DAPI image used only for nuclear segmentation (``segmentation.nucleus_input``):
    ``linear`` -> None (use the normal image); ``log_zscore`` -> blimp log10 pixel z-score correction,
    which flattens uneven / dim DAPI (250929)."""
    mode = cfg["segmentation"].get("nucleus_input", "linear")
    if mode == "linear":
        return None
    if mode != "log_zscore":
        raise ValueError(f"Unknown segmentation.nucleus_input {mode!r}")
    import copy

    c2 = copy.deepcopy(cfg)
    c2["illumination"].update(method="zscore", reference_log10=True)
    return Illumination(c2).correct(cfg["nuclear_channel"], raw_dapi)


def _floor(floors: dict, channel: str, role: str):
    """Storage floor for a channel: a number, or a per-role mapping {exon: x, intron: y, neg_control: z}."""
    f = floors.get(channel)
    return f.get(role) if isinstance(f, dict) else f


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
