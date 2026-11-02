"""Quick-look QC overlays (one PNG per FOV)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def overlay_png(path, dapi: np.ndarray, cells: np.ndarray, nuclei: np.ndarray, spots: pd.DataFrame, max_px: int = 1152):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    step = max(1, int(np.ceil(max(dapi.shape[1:]) / max_px)))
    mip = dapi.max(0)[::step, ::step]
    cb = find_boundaries(cells.max(0)[::step, ::step])
    nb = find_boundaries(nuclei.max(0)[::step, ::step])
    lo, hi = np.percentile(mip, [1, 99.5])
    rgb = np.clip((mip - lo) / max(hi - lo, 1e-9), 0, 1)[..., None].repeat(3, axis=2) * 0.8
    rgb[cb] = [1.0, 0.85, 0.0]
    rgb[nb] = [0.0, 0.8, 1.0]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    ax.imshow(rgb, interpolation="nearest")
    colours = ["#ff3b3b", "#4cff4c", "#ff66ff", "#ffffff"]
    if len(spots):
        passing = spots[spots["log_score"] >= spots["auto_threshold"]]
        for colour, (ch, d) in zip(colours, passing.groupby("channel")):
            label = f"{ch}: {d['probe'].iloc[0]} ({len(d)})"
            ax.scatter(d["x_px"] / step, d["y_px"] / step, s=2, c=colour, linewidths=0, label=label)
        ax.legend(loc="upper right", fontsize=8, markerscale=4, framealpha=0.6)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
