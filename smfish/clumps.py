"""Clumps of non-specific staining: extended bright objects the LoG splits into many candidate spots.

On 250806 a few W6 fields carry Atto633 aggregates (5-15 um, 3-7 um above the coverslip) that each yield
20-60 candidates above any plausible threshold. They are not spot-like and must count neither as mRNAs in
the cells they sit on nor as false positives when the threshold is calibrated.

Spacing does not separate them from real signal (candidates are ~0.75 um apart in a clump and in dense
GAPDH cells alike); the local background does: ~100 counts under a clump candidate against 12-22 under a
real spot, dense cells and transcription sites included. A clump is a connected group of high-background
candidates that is larger than any single mRNA or transcription site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

POS = ["z_um", "y_um", "x_um"]
DEFAULTS = {
    "enabled": False,
    "background_factor": 5.0,               # seed: background > factor x the typical background in its well
    "link_um": 1.2,                         # seeds closer than this are linked
    "min_seeds": 4,                         # a clump has at least this many linked seeds ...
    "min_extent_um": 1.5,                   # ... spanning at least this (one mRNA ~0.3 um, a TSS < 1 um)
    "score_floor_factor": 0.75,             # candidates considered: score >= factor x the spot's threshold
    "grow_steps": 2,                        # then grow by link_um over candidates of any background
}


def clump_settings(cfg: dict) -> dict:
    """``aggregation.clumps`` merged over :data:`DEFAULTS` (disabled unless the config enables it)."""
    return {**DEFAULTS, **(cfg.get("aggregation", {}).get("clumps") or {})}


def _factor(setting, channel: str) -> float:
    """``background_factor``: one number, or one per channel (with an optional ``default``)."""
    if isinstance(setting, dict):
        return float(setting.get(channel, setting.get("default", DEFAULTS["background_factor"])))
    return float(setting)


def flag_clumps(spots: pd.DataFrame, cells: pd.DataFrame, kernel_of: dict, settings: dict) -> pd.Series:
    """Clump id per candidate (-1 = not in a clump), unique within the table.

    Needs ``threshold_used`` and ``passes`` (:func:`smfish.aggregate.apply_thresholds`), ``background`` and
    positions in um. Per field, channel and detection kernel:

    1. Seeds: candidates scoring >= ``score_floor_factor`` x their threshold whose local background exceeds
       ``background_factor`` x the median background under such candidates in QC cells of the same well and
       channel. Judging each well against itself matters in dense wells, where real spots (GAPDH, ~900 per
       cell) crowd closely enough to raise each other's background.
    2. Seeds closer than ``link_um`` are linked; a group of >= ``min_seeds`` spanning >= ``min_extent_um``
       is a clump. A group is required, so one bright-background spot is never flagged.
    3. The clump grows ``grow_steps`` times by ``link_um`` over candidates above the floor whatever their
       background, picking up its dimmer edges.
    """
    out = np.full(len(spots), -1, np.int64)
    if not len(spots):
        return pd.Series(out, index=spots.index, name="clump_id")
    good = cells.loc[cells["qc_pass"], ["fov_id", "cell_id"]].assign(_qc=True)
    in_qc = spots[["fov_id", "cell_id"]].merge(good, on=["fov_id", "cell_id"], how="left")["_qc"].notna().to_numpy()
    above = (spots["log_score"] >= settings["score_floor_factor"] * spots["threshold_used"]).to_numpy()
    # reference: median background under candidates in QC cells of the same well and channel, so a dense
    # well (900 GAPDH per cell, which raise each other's background) is judged against itself
    keys = ["well", "channel"]
    bg_ref = spots.loc[above & in_qc].groupby(keys)["background"].median()
    ref = bg_ref.reindex(pd.MultiIndex.from_frame(spots[keys])).to_numpy()
    factor = spots["channel"].map(lambda c: _factor(settings["background_factor"], c)).to_numpy()
    cut = np.where(np.isfinite(ref), ref * factor, np.inf)
    seed = above & (spots["background"].to_numpy() > cut)
    cand = spots.loc[above, ["fov_id", "channel", *POS]].assign(kernel=spots.loc[above, "role"].map(kernel_of),
                                                                seed=seed[above])
    link, next_id = float(settings["link_um"]), 0
    for _, g in cand.groupby(["fov_id", "channel", "kernel"], sort=False):
        s = g[g["seed"]]
        if len(s) < settings["min_seeds"]:
            continue
        xyz = s[POS].to_numpy()
        pairs = cKDTree(xyz).query_pairs(link, output_type="ndarray")
        if not len(pairs):
            continue
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(s), len(s)))
        comp = connected_components(graph, directed=False)[1]
        label = np.full(len(g), -1, np.int64)
        seed_pos = g.index.get_indexer(s.index)
        for c in np.flatnonzero(np.bincount(comp) >= settings["min_seeds"]):
            members = np.flatnonzero(comp == c)
            if np.ptp(xyz[members], axis=0).max() >= settings["min_extent_um"]:
                label[seed_pos[members]] = next_id
                next_id += 1
        if (label < 0).all():
            continue
        all_xyz = g[POS].to_numpy()
        tree = cKDTree(all_xyz)
        for _ in range(int(settings["grow_steps"])):
            inside = np.flatnonzero(label >= 0)
            for i, nbrs in zip(inside, tree.query_ball_point(all_xyz[inside], link)):
                nbrs = np.asarray(nbrs, int)
                label[nbrs[label[nbrs] < 0]] = label[i]
        hit = label >= 0
        out[spots.index.get_indexer(g.index[hit])] = label[hit]
    return pd.Series(out, index=spots.index, name="clump_id")


def clump_report(spots: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Per well, channel and probe: spots above threshold in QC cells, how many sit in clumps, and in how many
    fields. Needs ``in_clump`` and ``above_threshold`` (score >= threshold, before clumps are removed)."""
    good = cells.loc[cells["qc_pass"], ["fov_id", "cell_id"]]
    s = spots[spots["above_threshold"]].merge(good, on=["fov_id", "cell_id"])
    if not len(s):
        return pd.DataFrame(columns=["well", "channel", "probe", "spots", "in_clumps", "fraction", "fields_with_clumps"])
    rep = s.groupby(["well", "channel", "probe"]).agg(spots=("in_clump", "size"), in_clumps=("in_clump", "sum"))
    rep["fraction"] = rep["in_clumps"] / rep["spots"]
    rep["fields_with_clumps"] = s[s["in_clump"]].groupby(["well", "channel", "probe"])["fov_id"].nunique()
    return rep.fillna({"fields_with_clumps": 0}).astype({"fields_with_clumps": int}).reset_index()
