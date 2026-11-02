"""Figures of smFISH readouts. Each ``fig_*`` returns a Figure and never saves; draw inside
``with style.use("publication"):`` and write with :func:`save`. Used by ``notebooks/1_calibration.py``
(detection steps) and ``notebooks/2_analysis.py`` (size scaling).

Layout follows the size-scaling figures of Sci. Adv. adh5138: per-cell counts against size as open
circles, with a sliding-window Fano factor underneath against a dashed Poisson line at 1. Columns are
cell lines (``conditions`` in the config); thin lines are single wells, the thick line all wells of that line.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import style as S
from .analysis import bootstrap_moments, rolling_moments

# Cell lines in the order and with the labels the figures use: set from the config with
# use_conditions(cfg["conditions"]). Conditions not listed there are drawn after these, under their own name.
CONDITION_ORDER: tuple = ()
CONDITION_LABEL: dict = {}
COMPARTMENT_LABEL = {"cell": "Whole cell", "cytoplasm": "Cytoplasm", "nucleus": "Nucleus"}


def use_conditions(labels: dict) -> None:
    """Order and labels of the cell lines, e.g. ``{"POLR2A-Streaming": "POLR2A-Streaming (knock-in)", ...}``."""
    global CONDITION_ORDER, CONDITION_LABEL
    CONDITION_ORDER, CONDITION_LABEL = tuple(labels), dict(labels)


def label_panels(fig, items: list[tuple]):
    """Bold panel letters at the top-left of each (axes, letter), aligned per column and row. Call last."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = {id(ax): inv.transform(ax.get_tightbbox(r)) for ax, _ in items}
    pos = {id(ax): ax.get_window_extent(r).transformed(inv) for ax, _ in items}
    for ax, letter in items:
        p = pos[id(ax)]
        col = [boxes[id(a)][0, 0] for a, _ in items if abs(pos[id(a)].x0 - p.x0) < 0.01]
        row = [boxes[id(a)][1, 1] for a, _ in items if abs(pos[id(a)].y1 - p.y1) < 0.02]
        fig.text(max(min(col) - 0.005, 0.0), max(row), letter, ha="left", va="bottom", fontsize=9,
                 fontweight="bold")


def save(fig, name: str, out_dir: str | Path, png_dpi: int = 300) -> Path:
    """Write ``name``.pdf and .png; warn if the saved width exceeds the text width."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    dpi=png_dpi if ext == "png" else None)
    width_mm = fig.get_tightbbox(fig.canvas.get_renderer()).width / S.MM
    if width_mm > S.TEXT_WIDTH_MM + 1:
        warnings.warn(f"{name}: saved width {width_mm:.0f} mm > {S.TEXT_WIDTH_MM:.0f} mm text width")
    return out / f"{name}.pdf"


def _conditions(df: pd.DataFrame, conditions):
    present = list(df["condition"].unique())
    order = [c for c in (conditions or CONDITION_ORDER) if c in present]
    return order + [c for c in present if c not in order] if conditions is None else order


def scaling_panel(ax, df: pd.DataFrame, x: str, y: str, colour: str, label: str | None = None,
                  window: float = 0.1, size: float = 5.0):
    """Open-circle scatter of ``y`` against ``x`` with the pooled sliding-window mean on top."""
    ax.scatter(df[x], df[y], s=size, facecolors="none", edgecolors=S.tint(colour, 0.55), linewidths=0.4,
               rasterized=True, label=label, zorder=1)
    r = rolling_moments(df, x, y, window=window)
    if len(r):
        ax.plot(r["x"], r["mean"], color=colour, lw=1.2, zorder=3)


def fano_panel(ax, df: pd.DataFrame, x: str, y: str, colour: str, window: float = 0.1, by_well: bool = True,
               stat: str = "fano", est: pd.DataFrame | None = None, band: bool = False, reference: bool = False,
               n_boot: int = 200):
    """Sliding-window Fano factor (or another :func:`rolling_moments` column, e.g. ``excess_cv2``): one thin
    line per well (``by_well``), a thick line for all wells together. The Poisson line at 1 is drawn for the
    Fano factor only.

    ``band``: 95% ribbon from resampling whole FOVs (:func:`smfish.analysis.bootstrap_moments`; pass ``est`` to
    reuse one bootstrap for several rows). ``reference`` (excess CV^2 only): dashed line for what the excess
    CV^2 would be if the Fano factor stayed at its population value F (median over windows), (F - 1)/mean -
    a trace on that line is the 1/mean fall of a constant Fano factor, departures from it are the information.
    """
    def _pos(v):
        return v.where(v > 0) if stat != "fano" else v  # log axis: drop windows at/below Poisson

    if by_well:
        for _, w in df.groupby("well"):
            r = rolling_moments(w, x, y, window=window, min_cells=30)
            if len(r):
                ax.plot(r["x"], _pos(r[stat]), color=colour, lw=0.6, alpha=S.REPLICATE_ALPHA, zorder=2)
    # all wells together, but only the variance within each well: differences between wells' means
    # (staining, segmentation) are not cell-to-cell noise
    if est is None:
        within = "well" if "well" in df else None
        est = (bootstrap_moments(df, x, y, window=window, within=within, n_boot=n_boot) if band
               else rolling_moments(df, x, y, window=window, within=within))
    if len(est):
        if band and f"{stat}_lo" in est:
            ax.fill_between(est["x"], _pos(est[f"{stat}_lo"]), _pos(est[f"{stat}_hi"]), color=S.tint(colour, 0.6),
                            alpha=0.3, lw=0, zorder=1)
        ax.plot(est["x"], _pos(est[stat]), color=colour, lw=1.2, zorder=3)
        if reference and stat == "excess_cv2":
            f_pop = est["fano"].median()
            ax.plot(est["x"], (f_pop - 1) / est["mean"], color=colour, lw=0.8, ls=(0, (3, 2)), zorder=4)
    if stat == "fano":
        ax.axhline(1.0, **S.REFERENCE_LINE)


def _log_noise_axis(ax, bottom: float = 0.01):
    """Log y-axis with plain tick labels (0.05, 0.1, 0.2, ...) for excess CV^2. ``bottom`` clips the axis so a
    ribbon reaching towards zero (a window whose CI includes Poisson) does not squash the rest of the panel."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.set_ylim(bottom=bottom)
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fig_scaling(counts: pd.DataFrame, compartments=("cell",), y: str = "n_spots", x: str = "cell_volume_pL",
                conditions=None, slopes: pd.DataFrame | None = None, show_fano: bool = True,
                show_cv2: bool = False, window: float = 0.1, ylabel: str = "mRNA per cell",
                xlabel: str = "Cell volume (pL)", height_mm: float | None = None, x_clip: float | None = 99.5,
                band: bool = False, by_well: bool = True, reference: bool = False, n_boot: int = 200):
    """Counts against size, one column per cell line; optional Fano and excess-CV^2 rows underneath.

    ``counts`` is :func:`smfish.analysis.per_cell_counts` output (one row per cell and compartment).
    ``slopes`` (:func:`smfish.analysis.scaling_slope` by condition and compartment) is written into each
    panel when given. ``show_fano=False`` drops the noise row - use it if :func:`fano_window_check`
    says the Fano factor is dominated by size spread or segmentation error. ``show_cv2`` adds the excess
    CV^2 ((var - mean)/mean^2, log axis): needed whenever the curves being compared differ in mean, as
    nucleus and cytoplasm do, since equal Fano factors at different means are not equal noise.
    ``band`` / ``by_well`` / ``reference``: see :func:`fano_panel` (one FOV bootstrap per line and compartment,
    shared by the noise rows).
    """
    import matplotlib.pyplot as plt

    conds = _conditions(counts, conditions)
    # x range to the 99.5th percentile: the few larger "cells" are mostly merged segmentations, and
    # they would stretch the axis over empty space. How many are off-axis is written in each panel.
    x_max = float(np.percentile(counts[x].dropna(), x_clip)) * 1.05 if x_clip else None
    noise_rows = [stat for stat, on in (("fano", show_fano), ("excess_cv2", show_cv2)) if on]
    height = height_mm or 62 + 33 * len(noise_rows)
    fig = plt.figure(figsize=(S.TEXT_WIDTH_MM * S.MM, height * S.MM), layout="constrained")
    gs = fig.add_gridspec(1 + len(noise_rows), len(conds), height_ratios=[2.2] + [1] * len(noise_rows))
    top, rows = [], {stat: [] for stat in noise_rows}
    boots = {}
    for j, cond in enumerate(conds):
        ax = fig.add_subplot(gs[0, j], sharex=top[0] if top else None, sharey=top[0] if top else None)
        top.append(ax)
        d = counts[counts["condition"] == cond]
        for comp in compartments:
            dc = d[d["compartment"] == comp]
            scaling_panel(ax, dc, x, y, S.COMPARTMENTS[comp], COMPARTMENT_LABEL[comp], window)
        ax.set_title(CONDITION_LABEL.get(cond, cond))
        if j == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(labelleft=False)
        if slopes is not None and len(slopes):
            lines = []
            for comp in compartments:
                s = slopes[(slopes["condition"] == cond) & (slopes["compartment"] == comp)]
                if len(s):
                    s = s.iloc[0]
                    name = "" if len(compartments) == 1 else f"{COMPARTMENT_LABEL[comp]}: "
                    lines.append(f"{name}slope {s['slope']:.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}]")
            dn = d[d["compartment"] == compartments[0]]
            lines.append(f"n = {len(dn)} cells, {d['well'].nunique()} wells")
            off = int((dn[x] > x_max).sum()) if x_max else 0
            if off:
                lines.append(f"{off} cells > {x_max:.1f} pL not shown")
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=6,
                    color=S.BLACK, bbox={"facecolor": S.WHITE, "edgecolor": "none", "pad": 1.0})
        above = ax
        for i, stat in enumerate(noise_rows, start=1):
            axn = fig.add_subplot(gs[i, j], sharex=ax, sharey=rows[stat][0] if rows[stat] else None)
            rows[stat].append(axn)
            for comp in compartments:
                dc = d[d["compartment"] == comp]
                key = (cond, comp)
                if key not in boots:
                    within = "well" if "well" in dc else None
                    boots[key] = (bootstrap_moments(dc, x, y, window=window, within=within, n_boot=n_boot) if band
                                  else rolling_moments(dc, x, y, window=window, within=within))
                fano_panel(axn, dc, x, y, S.COMPARTMENTS[comp], window, by_well=by_well, stat=stat,
                           est=boots[key], band=band, reference=reference)
            above.tick_params(labelbottom=False)
            above = axn
            if j == 0:
                axn.set_ylabel("Fano factor" if stat == "fano" else "Excess CV$^2$")
            else:
                axn.tick_params(labelleft=False)
        above.set_xlabel(xlabel)
    top[0].set_ylim(bottom=0)
    top[0].set_xlim(0, x_max)
    if rows.get("fano"):
        rows["fano"][0].set_ylim(bottom=0)
    if rows.get("excess_cv2"):
        _log_noise_axis(rows["excess_cv2"][0])
    if len(compartments) > 1:
        leg = top[-1].legend(loc="upper right", markerscale=2.0, handletextpad=0.2)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
    label_panels(fig, [(ax, chr(65 + j)) for j, ax in enumerate(top)])
    return fig


def fig_total_scaling(counts: pd.DataFrame, slopes: pd.DataFrame | None = None, **kw):
    """Whole-cell POLR2A mRNA against cell volume, knock-in vs parental (Fig A)."""
    return fig_scaling(counts, compartments=("cell",), slopes=slopes, **kw)


def fig_compartment_scaling(counts: pd.DataFrame, slopes: pd.DataFrame | None = None, **kw):
    """Nuclear (red) and cytoplasmic (blue) mRNA against cell volume (Fig B), with an excess-CV^2 row:
    the two compartments differ in mean, so their Fano factors alone do not say which is noisier."""
    kw.setdefault("show_cv2", kw.get("show_fano", True))
    # two overlapping compartments: FOV-bootstrap ribbons instead of per-well lines, and the constant-Fano line
    kw.setdefault("band", True)
    kw.setdefault("by_well", False)
    kw.setdefault("reference", True)
    return fig_scaling(counts, compartments=("cytoplasm", "nucleus"), slopes=slopes, **kw)


def fig_compartment_own_volume(counts: pd.DataFrame, slopes: pd.DataFrame | None = None, conditions=None,
                               y: str = "n_spots", window: float = 0.1, x_clip: float | None = 99.5,
                               volume_slopes: pd.DataFrame | None = None, n_boot: int = 200):
    """Appendix control: each compartment's mRNA against its OWN volume, and the volumes against cell volume.

    Rows: nuclear mRNA against nuclear volume, its excess CV^2; cytoplasmic mRNA against cytoplasmic volume, its
    excess CV^2; nuclear and cytoplasmic volume against cell volume. Columns: cell lines. Against cell volume a
    compartment's slope combines how its mRNA follows its own size with how its size follows cell size; the
    first four rows show the first, the last row the second.

    ``slopes``: :func:`smfish.analysis.scaling_slope` per condition and compartment, each fitted against that
    compartment's own volume. ``volume_slopes``: the same for ``nucleus_volume_pL`` / ``cytoplasm_volume_pL``
    against ``cell_volume_pL`` (``compartment`` = which volume). Excess-CV^2 rows carry the FOV-bootstrap ribbon
    and the constant-Fano line of :func:`fano_panel`.
    """
    import matplotlib.pyplot as plt

    conds = _conditions(counts, conditions)
    comps = (("nucleus", "nucleus_volume_pL", "Nuclear volume (pL)"),
             ("cytoplasm", "cytoplasm_volume_pL", "Cytoplasmic volume (pL)"))
    fig = plt.figure(figsize=(S.TEXT_WIDTH_MM * S.MM, 200 * S.MM), layout="constrained")
    gs = fig.add_gridspec(5, len(conds), height_ratios=[2, 1, 2, 1, 1.8])
    letters = []

    def _note(ax, lines):
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=6,
                color=S.BLACK, bbox={"facecolor": S.WHITE, "edgecolor": "none", "pad": 1.0})

    def _slope_line(tab, cond, comp, prefix=""):
        if tab is None or not len(tab):
            return []
        t = tab[(tab["condition"] == cond) & (tab["compartment"] == comp)]
        if not len(t):
            return []
        t = t.iloc[0]
        return [f"{prefix}slope {t['slope']:.2f} [{t['ci_low']:.2f}, {t['ci_high']:.2f}]"]

    for i, (comp, x, xlabel) in enumerate(comps):
        dc = counts[counts["compartment"] == comp]
        x_max = float(np.percentile(dc[x].dropna(), x_clip)) * 1.05 if x_clip else None
        first = first_n = None
        for j, cond in enumerate(conds):
            d = dc[dc["condition"] == cond]
            ax = fig.add_subplot(gs[2 * i, j], sharex=first, sharey=first)
            axn = fig.add_subplot(gs[2 * i + 1, j], sharex=ax, sharey=first_n)
            first, first_n = first or ax, first_n or axn
            scaling_panel(ax, d, x, y, S.COMPARTMENTS[comp], window=window)
            fano_panel(axn, d, x, y, S.COMPARTMENTS[comp], window, by_well=False, stat="excess_cv2", band=True,
                       reference=True, n_boot=n_boot)
            if i == 0:
                ax.set_title(CONDITION_LABEL.get(cond, cond))
            ax.tick_params(labelbottom=False)
            axn.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(f"{COMPARTMENT_LABEL[comp]} mRNA")
                axn.set_ylabel("Excess CV$^2$")
            else:
                ax.tick_params(labelleft=False)
                axn.tick_params(labelleft=False)
            lines = _slope_line(slopes, cond, comp) + [f"n = {len(d)} cells, {d['well'].nunique()} wells"]
            off = int((d[x] > x_max).sum()) if x_max else 0
            if off:
                lines.append(f"{off} cells > {x_max:.2f} pL not shown")
            _note(ax, lines)
            letters.append((ax, chr(65 + len(letters))))
        first.set_ylim(bottom=0)
        first.set_xlim(0, x_max)
        _log_noise_axis(first_n)

    # compartment volumes against cell volume: how the nucleus/cytoplasm split changes with size
    cells_ = counts[counts["compartment"] == "nucleus"]
    x_max = float(np.percentile(cells_["cell_volume_pL"].dropna(), x_clip)) * 1.05 if x_clip else None
    first = None
    for j, cond in enumerate(conds):
        d = cells_[cells_["condition"] == cond]
        ax = fig.add_subplot(gs[4, j], sharex=first, sharey=first)
        first = first or ax
        lines = []
        for comp, v in (("cytoplasm", "cytoplasm_volume_pL"), ("nucleus", "nucleus_volume_pL")):
            scaling_panel(ax, d, "cell_volume_pL", v, S.COMPARTMENTS[comp], label=COMPARTMENT_LABEL[comp],
                          window=window)
            lines += _slope_line(volume_slopes, cond, comp, prefix=f"{COMPARTMENT_LABEL[comp]}: ")
        if lines:
            _note(ax, lines)
        ax.set_xlabel("Cell volume (pL)")
        if j == 0:
            ax.set_ylabel("Volume (pL)")
        else:
            ax.tick_params(labelleft=False)
        letters.append((ax, chr(65 + len(letters))))
    first.set_xlim(0, x_max)
    y_max = float(np.percentile(cells_["cytoplasm_volume_pL"].dropna(), x_clip)) * 1.1 if x_clip else None
    first.set_ylim(0, y_max)
    leg = ax.legend(loc="lower right", markerscale=2.0, handletextpad=0.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    label_panels(fig, letters)
    return fig


# --------------------------------------------------------------------------- raw-image windows


def pick_example_window(cells: pd.DataFrame, counts: pd.DataFrame, wells, window_um: float = 45.0,
                        voxel_xy_um: float = 0.065, image_px: int = 2304) -> dict:
    """A square window (``window_um``) in one FOV of ``wells`` full of TYPICAL cells.

    Typical = QC-passing, volume within the line's interquartile range and mRNA per pL within its
    interquartile range, so the picture is not a cherry-picked bright cell. The window is centred on a
    typical cell and scored by the typical (then any QC) cells whose centroid lies inside it, away
    from the edge. Deterministic. ``counts`` is whole-cell :func:`per_cell_counts` output.
    """
    c = counts[(counts["compartment"] == "cell") & counts["well"].isin(wells)].copy()
    c["per_pL"] = c["n_spots"] / c["cell_volume_pL"]
    q = lambda s, lo, hi: s.between(s.quantile(lo), s.quantile(hi))  # noqa: E731
    c["typical"] = q(c["cell_volume_pL"], 0.25, 0.75) & q(c["per_pL"], 0.25, 0.75)
    cc = cells[cells["qc_pass"] & cells["well"].isin(wells)].merge(
        c[["fov_id", "cell_id", "typical"]], on=["fov_id", "cell_id"], how="left")
    cc["typical"] = cc["typical"].fillna(False).astype(bool)
    cc["y"], cc["x"] = cc["centroid_y_um"] / voxel_xy_um, cc["centroid_x_um"] / voxel_xy_um
    size = int(round(window_um / voxel_xy_um))
    margin = 0.15 * size
    best, best_score = None, -1.0
    for fov, f in cc.groupby("fov_id"):
        for _, centre in f[f["typical"]].iterrows():
            y0 = int(np.clip(centre["y"] - size / 2, 0, image_px - size))
            x0 = int(np.clip(centre["x"] - size / 2, 0, image_px - size))
            inside = f[f["y"].between(y0 + margin, y0 + size - margin) & f["x"].between(x0 + margin, x0 + size - margin)]
            score = inside["typical"].sum() + 0.5 * len(inside)
            if score > best_score:
                best, best_score = {"fov_id": fov, "y0": y0, "x0": x0, "size": size,
                                    "n_cells": len(inside), "n_typical": int(inside["typical"].sum())}, score
    if best is None:
        raise ValueError(f"No typical QC cells in wells {list(wells)}")
    return best


def raw_path(cfg: dict, fov_id: str) -> Path:
    """The raw OME-TIFF of ``fov_id`` in ``paths.input_dir`` (the FOV id is the file name without its suffix)."""
    hits = sorted(Path(cfg["paths"]["input_dir"]).glob(f"{fov_id}.ome.tif*"))
    if not hits:
        raise FileNotFoundError(f"{fov_id}.ome.tiff not in {cfg['paths']['input_dir']}")
    return hits[0]


def load_window(cfg: dict, fov_id: str, y0: int, x0: int, size: int, channels, labels: bool = True) -> dict:
    """3D crops (ZYX, offset-corrected float) of ``channels`` in one square window of one FOV, plus the
    3D cell / nucleus labels there. Reads the raw OME-TIFF (~0.6 GB per channel) from ``paths.input_dir``
    and, with ``labels``, the FOV's ``labels.tif`` (~2 GB)."""
    from .io import Illumination, load_channels, load_labels
    from .pipeline import fov_dir

    raw, voxel = load_channels(raw_path(cfg, fov_id), cfg, list(channels))
    ill = Illumination(cfg)
    ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
    out = {"fov_id": fov_id, "y0": y0, "x0": x0, "size": size, "voxel_um": voxel,
           "stacks": {c: ill.correct(c, raw[c][:, ys, xs]) for c in channels}}
    if labels:
        cells, nuclei = load_labels(fov_dir(cfg, fov_id) / "labels.tif")
        out["cells3d"], out["nuclei3d"] = cells[:, ys, xs], nuclei[:, ys, xs]
    return out


def window_spots(spots_path, fov_id: str, y0: int, x0: int, size: int, probes, passing_only: bool = True,
                 columns=("probe", "z_px", "y_px", "x_px", "z_um", "y_um", "x_um", "compartment", "cell_id",
                          "in_nucleus", "passes", "log_score")) -> pd.DataFrame:
    """Rows of ``aggregated/spots`` for one FOV and window, with window-relative ``y`` / ``x`` (px)."""
    sp = pd.read_parquet(spots_path, filters=[("fov_id", "==", fov_id), ("probe", "in", list(probes))],
                         columns=list(columns))
    sp = sp[sp["y_px"].between(y0, y0 + size - 1) & sp["x_px"].between(x0, x0 + size - 1)]
    if passing_only:
        sp = sp[sp["passes"]]
    return sp.assign(y=sp["y_px"] - y0, x=sp["x_px"] - x0, z=sp["z_px"])


def _stretch(a, lo, hi):
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def _scale_bar(ax, um: float, voxel_xy_um: float, size_px: int):
    length = um / voxel_xy_um
    x1, y = size_px * 0.95, size_px * 0.93
    ax.plot([x1 - length, x1], [y, y], color=S.WHITE, lw=2, solid_capstyle="butt")
    ax.text(x1 - length / 2, y - size_px * 0.025, f"{um:g} µm", color=S.WHITE, ha="center", va="bottom",
            fontsize=6)


def _image_axes(ax):
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# --------------------------------------------------------------------------- detection explainer


def elbow_curves(cfg: dict, probe: str) -> list[dict]:
    """Per FOV: the LoG-maxima count above each threshold (``qc.json`` elbow curve) and its own automated
    threshold, for every FOV imaging ``probe``."""
    import json

    out = []
    for q in sorted((Path(cfg["paths"]["output_dir"]) / "fov").glob("*/qc.json")):
        d = json.loads(q.read_text())
        for ch, c in d["channels"].items():
            if c.get("probe") == probe and c.get("elbow_thresholds"):
                out.append({"fov_id": d["fov_id"], "well": d["well"], "channel": ch, "auto": c["auto_threshold"],
                            "t": np.asarray(c["elbow_thresholds"]), "n": np.asarray(c["elbow_counts"], float)})
    return out


def load_detection_example(cfg: dict, spots_path, window: dict, probe: str, channel: str, role: str = "exon") -> dict:
    """One channel's 3D crop, its LoG filter (as stage detection runs it) and every STORED candidate of
    ``probe`` in the window (down to the storage floor), with ``passes``."""
    from bigfish import stack

    from .detect import sigma_px

    w = load_window(cfg, window["fov_id"], window["y0"], window["x0"], window["size"], [channel], labels=False)
    img = w["stacks"][channel].astype(np.float32)
    sig = sigma_px(w["voxel_um"], cfg["detection"]["spot_radius_nm"][role])
    log_img = stack.log_filter(img, sig)
    cand = window_spots(spots_path, window["fov_id"], window["y0"], window["x0"], window["size"], [probe],
                        passing_only=False)
    return {"raw": img.max(0), "log": log_img.max(0), "candidates": cand, "voxel_um": w["voxel_um"], **window}


def fig_detection_steps(ex: dict, elbows: list[dict], scan: pd.DataFrame, thresholds: dict, cfg: dict,
                        probe_label: str = "POLR2A odds exon", scale_um: float = 5.0, zoom_um: float = 4.0):
    """Appendix: how a spot becomes a counted mRNA.

    Top: (A) raw smFISH, maximum projection; (B) LoG-filtered; (C) every stored local maximum (grey)
    and those at or above the probe's final threshold (orange); (D) one spot with the integration
    ellipsoid (solid) and the local-background shell (dashed), XY cross-sections at the spot plane.
    Bottom: (E) elbow curves of all FOVs of the probe, each FOV's automated threshold (dots), the
    probe median and the calibrated threshold; (F, G) per channel, false positives per cell in the
    negative controls and the odds/evens detection rate against the threshold, with the value used.
    ``thresholds`` maps channel -> calibrated value, plus ``"probe_median"`` and ``"channel"``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig = plt.figure(figsize=(S.TEXT_WIDTH_MM * S.MM, 100 * S.MM), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95])
    top, bottom = gs[0].subgridspec(1, 4), gs[1].subgridspec(1, 3)
    n = ex["raw"].shape[0]
    vxy = ex["voxel_um"][1]
    ch = thresholds["channel"]
    thr = thresholds[ch]
    raw = _stretch(ex["raw"], *np.percentile(ex["raw"], [1, 99.9]))
    lg = _stretch(ex["log"], *np.percentile(ex["log"], [1, 99.9]))
    c = ex["candidates"]
    axes = [fig.add_subplot(top[0, j]) for j in range(4)]
    axes[0].imshow(raw, cmap=S.IMAGE_CMAP)
    axes[0].set_title("Raw (max. projection)")
    _scale_bar(axes[0], scale_um, vxy, n)
    axes[1].imshow(lg, cmap=S.IMAGE_CMAP)
    axes[1].set_title("LoG filtered")
    axes[2].imshow(raw * 0.5, cmap=S.IMAGE_CMAP, vmin=0, vmax=1)
    rej, ok = c[~c["passes"]], c[c["passes"]]
    axes[2].scatter(rej["x"], rej["y"], s=3, color=S.GREY, linewidths=0, label=f"below {thr:.3g} (n={len(rej)})")
    axes[2].scatter(ok["x"], ok["y"], s=10, facecolors="none", edgecolors=S.ORANGE, linewidths=0.5,
                    label=f"counted (n={len(ok)})")
    axes[2].set_title("Candidates")
    leg = axes[2].legend(loc="lower left", fontsize=5, handletextpad=0.1, borderpad=0.2, markerscale=1.5)
    leg.get_frame().set_alpha(0.85)
    for ax in axes[:3]:
        _image_axes(ax)
        ax.set_xlim(-0.5, n - 0.5), ax.set_ylim(n - 0.5, -0.5)
    # zoom on the brightest isolated counted spot away from the edge
    half = int(round(zoom_um / vxy / 2))
    inner = ok[ok["x"].between(half, n - half - 1) & ok["y"].between(half, n - half - 1)]
    ax = axes[3]
    if len(inner):
        spot = inner.sort_values("log_score", ascending=False).iloc[len(inner) // 2]
        yc, xc = int(spot["y"]), int(spot["x"])
        ax.imshow(raw[yc - half:yc + half + 1, xc - half:xc + half + 1], cmap=S.IMAGE_CMAP,
                  extent=(-half * vxy, half * vxy, half * vxy, -half * vxy))
        q = cfg["quantification"]
        for r_nm, ls in ((q["window_radius_nm"], "-"), (q["background_inner_nm"], "--"), (q["background_outer_nm"], "--")):
            ax.add_patch(Ellipse((0, 0), 2 * r_nm[2] / 1000, 2 * r_nm[1] / 1000, fill=False, ec=S.YELLOW, lw=0.7, ls=ls))
    ax.set_title("Measurement")
    _image_axes(ax)
    letters = [(a, chr(65 + j)) for j, a in enumerate(axes)]

    ax = fig.add_subplot(bottom[0, 0])
    for e in elbows:
        ax.plot(e["t"], np.maximum(e["n"], 0.8), color=S.GREY, lw=0.4, alpha=0.5)
    autos = np.array([e["auto"] for e in elbows if e["auto"] is not None])
    if len(autos):
        y_at = [np.interp(a, e["t"], e["n"]) for e, a in zip([e for e in elbows if e["auto"] is not None], autos)]
        ax.scatter(autos, np.maximum(y_at, 0.8), s=4, color=S.BLACK, zorder=3, label="FOV automated")
    ax.axvline(thresholds["probe_median"], color=S.BLACK, ls="--", lw=0.8, label="median of FOVs")
    ax.axvline(thr, color=S.VERMILLION, lw=1.0, label="calibrated (used)")
    ax.set_yscale("log")
    ax.set_xlim(0, max(thr * 2.2, 1))
    ax.set(xlabel="LoG threshold", ylabel="Local maxima above threshold", title=probe_label)
    ax.legend(loc="upper right", fontsize=5)
    letters.append((ax, "E"))
    for k, chan in enumerate([c for c in ("Atto565", "Atto633") if c in set(scan["channel"])]):
        ax = fig.add_subplot(bottom[0, 1 + k])
        d = scan[scan["channel"] == chan]
        ax.plot(d["threshold"], d["fp_per_cell"].clip(lower=1e-2), color=S.VERMILLION, label="false pos. / cell")
        ax.set_yscale("log")
        ax.set(xlabel="LoG threshold", title=chan)
        if k == 0:
            ax.set_ylabel("False positives per cell")
        ax2 = ax.twinx()
        ax2.plot(d["threshold"], d["detection_rate"], color=S.BLUE, label="odds/evens detection")
        ax2.set_ylim(0, 1.02)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        if k == 1:
            ax2.set_ylabel("Detection rate")
        ax.axvline(thresholds.get(chan, np.nan), color=S.BLACK, lw=0.8, ls="--")
        ax.axhline(1.0, **S.REFERENCE_LINE)
        if k == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=5)
        letters.append((ax, chr(70 + k)))
    label_panels(fig, letters)
    return fig
