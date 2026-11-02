"""Do the conclusions hinge on the exact threshold? Readouts recomputed at 0.6-1.6x the calibrated thresholds.

Every probe's threshold (``calibration/thresholds_applied.csv``) is scaled by the same factor and the
readouts are recounted from the stored candidates (spots are stored down to a permissive floor, so a lower
threshold can be applied after the run). Candidates in clumps of non-specific staining (``in_clump``) are
left out, as in the counts. Used by ``notebooks/1_calibration.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import style as S
from .figures import label_panels

FACTORS = np.round(np.linspace(0.6, 1.6, 11), 2)


def readouts_vs_threshold(agg_dir: str | Path, groups: dict[str, tuple[str, str, list[int]]],
                          factors=FACTORS) -> pd.DataFrame:
    """Per group and threshold factor: mean spots per QC cell, spots per pL and the log-log slope against
    cell volume (cells with >= 1 spot).

    ``groups``: label -> (probe, channel, wells), e.g. ``{"POLR2A, knock-in": ("POLR2A odds exon", "Atto565", [8])}``.
    Each group is counted at ``factor`` x its own calibrated threshold.
    """
    agg = Path(agg_dir)
    applied = pd.read_csv(agg / "calibration" / "thresholds_applied.csv")
    thr = {(r.probe, r.channel): float(r.threshold_median) for r in applied.itertuples()}
    wells = sorted({w for _, _, ws in groups.values() for w in ws})
    spots = pd.read_parquet(agg / "spots.parquet", filters=[("well", "in", wells)],
                            columns=["fov_id", "cell_id", "well", "channel", "probe", "log_score", "in_clump"])
    spots = spots[~spots["in_clump"]]
    cells = pd.read_parquet(agg / "cells.parquet", columns=["fov_id", "cell_id", "well", "qc_pass", "cell_volume_um3"])
    cells = cells[cells["qc_pass"]]

    rows = []
    for label, (probe, channel, ws) in groups.items():
        s = spots[(spots["probe"] == probe) & (spots["channel"] == channel) & spots["well"].isin(ws)]
        base = cells[cells["well"].isin(ws)].set_index(["fov_id", "cell_id"])
        vol = base["cell_volume_um3"]
        for f in factors:
            n = (s[s["log_score"] >= f * thr[(probe, channel)]].groupby(["fov_id", "cell_id"]).size()
                 .reindex(base.index, fill_value=0))
            fit = (n > 0) & (vol > 0)
            slope = np.polyfit(np.log(vol[fit]), np.log(n[fit]), 1)[0] if fit.sum() > 2 else np.nan
            rows.append({"group": label, "probe": probe, "channel": channel, "factor": f,
                         "threshold": f * thr[(probe, channel)], "n_cells": len(n), "spots_per_cell": n.mean(),
                         "spots_per_pL": n.sum() / (vol.sum() / 1000), "slope": slope})
    return pd.DataFrame(rows)


def fig_threshold_robustness(table: pd.DataFrame, readouts: list[str], controls: list[str],
                             ratios: dict[str, tuple[str, str]]):
    """A: spots per cell against the threshold factor, readouts (black) and negative controls (purple);
    B: concentration ratios ``label -> (numerator group, denominator group)``; C: size-scaling slopes of the
    readouts. The calibrated threshold (1.0) and +-20% around it are marked."""
    import matplotlib.pyplot as plt

    wide = {col: table.pivot(index="factor", columns="group", values=col)
            for col in ("spots_per_cell", "spots_per_pL", "slope")}
    styles = ["-", ":", "--", "-."]
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_WIDTH_MM * S.MM, 62 * S.MM), layout="constrained")
    for i, g in enumerate(readouts):
        axes[0].plot(wide["spots_per_cell"].index, wide["spots_per_cell"][g], color=S.BLACK, ls=styles[i % 4],
                     marker="o", ms=2.5, lw=1.1, label=g)
        axes[2].plot(wide["slope"].index, wide["slope"][g], color=S.BLACK, ls=styles[i % 4], marker="o", ms=2.5,
                     lw=1.1, label=g)
    for i, g in enumerate(controls):
        axes[0].plot(wide["spots_per_cell"].index, wide["spots_per_cell"][g], color=S.PURPLE, ls=styles[i % 4],
                     marker="o", ms=2.5, lw=1.1, label=g)
    for i, (label, (num, den)) in enumerate(ratios.items()):
        r = wide["spots_per_pL"][num] / wide["spots_per_pL"][den]
        axes[1].plot(r.index, r, color=S.BLACK, ls=styles[i % 4], marker="o", ms=2.5, lw=1.1, label=label)
    for ax, ylabel in zip(axes, ("Spots per cell", "Ratio of spots per pL", "Log-log slope vs cell volume")):
        ax.axvspan(0.8, 1.2, color=S.tint(S.ORANGE, 0.15), lw=0, zorder=0)
        ax.axvline(1.0, color=S.ORANGE, lw=1.2)
        ax.set(xlabel="Threshold / calibrated threshold", ylabel=ylabel, xlim=(table["factor"].min(), table["factor"].max()))
        ax.legend(fontsize=5)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, max(1.4, 1.1 * np.nanmax(axes[1].get_ylim())))
    axes[2].set_ylim(0, 2)
    label_panels(fig, [(ax, chr(65 + i)) for i, ax in enumerate(axes)])
    return fig
