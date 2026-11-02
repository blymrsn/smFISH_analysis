"""Calibration from the plate's control wells.

- Negative control (mrcB, a bacterial gene) -> per-channel spot threshold:
  the lowest LoG threshold at which false-positive spots per cell fall below a target.
- Odds / evens probe sets on the same mRNA -> chromatic offset between channels
  (median displacement of matched pairs) and colocalisation (detection) efficiency.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

POS = ["z_um", "y_um", "x_um"]


def is_negative(spots: pd.DataFrame) -> pd.Series:
    """Spots whose target is absent: neg_control probes (mrcB) and probes in cell lines lacking the
    target (``absent_in`` in probes.yaml, e.g. NeoR / TetO in OsTIR)."""
    neg = spots["role"] == "neg_control"
    if "expected_absent" in spots:
        neg |= spots["expected_absent"].fillna(False).astype(bool)
    return neg


# --------------------------------------------------------------------------- negative control thresholds


def kernel_of_role(cfg: dict) -> dict:
    """Role -> detection-kernel label. Roles detected with the same LoG radius share a label (mrcB, a
    ``neg_control``, uses the exon radius -> ``exon``); LoG scores are only comparable within one."""
    radii = cfg["detection"]["spot_radius_nm"]
    first = {}
    for role, r in radii.items():
        first.setdefault(tuple(r), role)
    return {role: first[tuple(r)] for role, r in radii.items()}


def fp_target(max_fp_per_cell, kernel: str) -> float:
    """``aggregation.max_false_positives_per_cell``: one number, or one per kernel. Intron calls are
    1-2 real sites per nucleus, so they need a far stricter target than mRNA counts."""
    if isinstance(max_fp_per_cell, dict):
        return float(max_fp_per_cell.get(kernel, max_fp_per_cell.get("default", 1.0)))
    return float(max_fp_per_cell)


def negative_control_curve(spots: pd.DataFrame, cells: pd.DataFrame, kernel_of: dict | None = None,
                           n_points: int = 300, statistic: str = "pooled") -> pd.DataFrame:
    """False-positive spots per QC cell vs LoG threshold, per (channel, kernel, negative probe).

    Intron-kernel negatives (e.g. TetO in OsTIR) count nuclear spots only, as TSS calls do.

    Two summaries over the control fields: ``fp_per_cell_pooled`` (all fields' false spots / all their QC
    cells) and ``fp_per_cell_median`` (each field's rate, median over fields; a field with no false spot
    counts as 0). ``fp_per_cell`` is the one named by ``statistic`` and is what thresholds are read from.
    The median keeps a few fields with non-specific staining (one 250806 W6 field carried 4 false spots per
    cell, the typical field 0.3) from setting the threshold for the whole plate.
    """
    if statistic not in ("pooled", "median"):
        raise ValueError(f"statistic must be 'pooled' or 'median', got {statistic!r}")
    neg = spots[is_negative(spots) & (spots["cell_id"] > 0)].copy()
    neg["kernel"] = neg["role"].map(kernel_of) if kernel_of else neg["role"]
    good = cells.loc[cells["qc_pass"], ["fov_id", "cell_id"]]
    rows = []
    for (channel, kernel, probe), d in neg.groupby(["channel", "kernel", "probe"]):
        nuclear = kernel == "intron"
        if nuclear:
            d = d[d["in_nucleus"]]
        wells = sorted(d["well"].unique())
        qc_cells = good.merge(cells.loc[cells["well"].isin(wells), ["fov_id", "cell_id"]])
        d = d.merge(qc_cells, on=["fov_id", "cell_id"])
        n_cells = len(qc_cells)
        if n_cells == 0:
            continue
        hi = float(np.percentile(d["log_score"], 99.9)) if len(d) else 1.0
        grid = np.linspace(0, max(hi, 1e-6), n_points)

        def n_above(scores: np.ndarray) -> np.ndarray:
            scores = np.sort(scores)
            return scores.size - np.searchsorted(scores, grid, side="left")

        pooled = n_above(d["log_score"].to_numpy()) / n_cells
        cells_per_field = qc_cells.groupby("fov_id").size()
        by_field = d.groupby("fov_id")["log_score"]
        per_field = np.stack([n_above(by_field.get_group(f).to_numpy()) if f in by_field.groups
                              else np.zeros(grid.size) for f in cells_per_field.index])
        median = np.median(per_field / cells_per_field.to_numpy()[:, None], axis=0)
        rows.append(pd.DataFrame({"channel": channel, "kernel": kernel, "probe": probe, "threshold": grid,
                                  "fp_per_cell": median if statistic == "median" else pooled,
                                  "fp_per_cell_pooled": pooled, "fp_per_cell_median": median,
                                  "statistic": statistic, "counted": "nucleus" if nuclear else "cell",
                                  "n_cells": n_cells, "n_fields": len(cells_per_field),
                                  "wells": ",".join(map(str, wells))}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def negative_control_thresholds(curve: pd.DataFrame, spots: pd.DataFrame, max_fp_per_cell=1.0) -> pd.DataFrame:
    """Smallest threshold with at most the target false positives per cell, per negative control
    (channel, kernel, probe). See :func:`fp_target` for the target.

    Also reports the median per-FOV automated threshold of that probe for comparison.
    """
    rows = []
    for (channel, kernel, probe), c in curve.groupby(["channel", "kernel", "probe"]):
        target = fp_target(max_fp_per_cell, kernel)
        ok = c[c["fp_per_cell"] <= target]
        thr = float(ok["threshold"].min()) if len(ok) else np.nan
        auto = spots[(spots["channel"] == channel) & (spots["probe"] == probe)].groupby("fov_id")["auto_threshold"].first()
        rows.append({"channel": channel, "kernel": kernel, "probe": probe, "threshold": thr, "max_fp_per_cell": target,
                     "statistic": c["statistic"].iloc[0] if "statistic" in c else "pooled",
                     "counted": c["counted"].iloc[0],
                     "auto_threshold_median": float(auto.median()) if len(auto) else np.nan,
                     "auto_threshold_iqr": float(auto.quantile(0.75) - auto.quantile(0.25)) if len(auto) else np.nan,
                     "n_cells": int(c["n_cells"].iloc[0]), "wells": c["wells"].iloc[0]})
    return pd.DataFrame(rows)


def propose_thresholds(spots: pd.DataFrame, curve: pd.DataFrame, nct: pd.DataFrame, detection: pd.DataFrame,
                       kernel_of: dict, scan: pd.DataFrame | None = None) -> pd.DataFrame:
    """One proposed threshold per (probe, channel), the same in every well, with where it came from.

    1. The probe's own negative control (wells whose cell line lacks the target, e.g. NeoR / TetO in OsTIR).
    2. Otherwise the strictest negative control imaged in the same channel with the same kernel (mrcB,
       NeoR for exon probes): LoG scores are comparable there, so the false-positive rate carries over.
    3. Otherwise (no negative control for that channel and kernel, e.g. 565 introns) the median per-FOV
       automated threshold over every FOV of the probe -- uncalibrated, and flagged as such.

    A value below the probe's stored floor (the highest permissive detection threshold of any FOV) is
    raised to it, since below the floor some FOVs stored nothing. ``fp_per_cell_at_proposed`` is the
    worst false-positive rate among that channel/kernel's negative controls at the proposed value, and
    ``detection_rate_at_proposed`` (exon only) the odds/evens detection rate from :func:`threshold_scan`.
    Applied only through ``aggregation.thresholds`` in the config.
    """
    from .aggregate import probe_median_auto

    medians = probe_median_auto(spots)
    floor = (detection.groupby(["probe", "channel"])["detection_threshold"].max()
             if {"probe", "channel", "detection_threshold"} <= set(detection.columns) else pd.Series(dtype=float))
    rows = []
    for (probe, channel, role), _ in spots.groupby(["probe", "channel", "role"]):
        kernel = kernel_of.get(role, role)
        own = nct[(nct["probe"] == probe) & (nct["channel"] == channel)] if len(nct) else nct
        same = nct[(nct["channel"] == channel) & (nct["kernel"] == kernel)] if len(nct) else nct
        if len(own) and own["threshold"].notna().any():
            value, source = float(own["threshold"].max()), f"own negative control (wells {own['wells'].iloc[0]})"
        elif len(same) and same["threshold"].notna().any():
            best = same.loc[same["threshold"].idxmax()]
            value = float(best["threshold"])
            source = f"negative control {best['probe']} (wells {best['wells']}), same channel and kernel"
        else:
            value, source = float(medians.get((probe, channel), np.nan)), "median per-FOV auto (no negative control)"
        f = float(floor.get((probe, channel), np.nan))
        if np.isfinite(f) and value < f:
            value, source = f, source + "; raised to the stored floor"
        fp = np.nan
        if len(curve):
            fps = [np.interp(value, c["threshold"], c["fp_per_cell"])
                   for _, c in curve[(curve["channel"] == channel) & (curve["kernel"] == kernel)].groupby("probe")]
            fp = float(max(fps)) if fps else np.nan
        det = np.nan
        if scan is not None and len(scan) and kernel == "exon" and "detection_rate" in scan:
            sc = scan[(scan["channel"] == channel) & scan["detection_rate"].notna()]
            det = float(np.interp(value, sc["threshold"], sc["detection_rate"])) if len(sc) else np.nan
        rows.append({"probe": probe, "channel": channel, "kernel": kernel, "proposed": value, "source": source,
                     "probe_median_auto": float(medians.get((probe, channel), np.nan)), "stored_floor": f,
                     "fp_per_cell_at_proposed": fp, "detection_rate_at_proposed": det})
    return pd.DataFrame(rows)


def proposed_thresholds_yaml(proposed: pd.DataFrame) -> str:
    """The ``aggregation.thresholds`` block for :func:`propose_thresholds`, to paste into the config."""
    lines = ["thresholds:", "  default: probe_median"]
    for r in proposed.sort_values(["channel", "probe"]).itertuples():
        lines.append(f'  "{r.probe}@{r.channel}": {r.proposed:.3g}   # {r.source}')
    return "\n".join(lines) + "\n"


def check_against_detection(thresholds: pd.DataFrame, detection: pd.DataFrame) -> pd.DataFrame:
    """FOVs whose permissive detection threshold exceeds the calibrated threshold.

    ``detection`` is ``aggregated/detection_thresholds.csv`` (fov_id, channel, detection_threshold).
    """
    keys = [k for k in ("channel", "probe") if k in thresholds and k in detection]
    m = detection.merge(thresholds[[*keys, "threshold"]], on=keys)
    bad = m[m["detection_threshold"] > m["threshold"]]
    if len(bad):
        log.warning("%d FOV/channel detections were run above the calibrated threshold; lower "
                    "detection.permissive_factor and re-run those FOVs", len(bad))
    return bad


# --------------------------------------------------------------------------- odds / evens


def _passing_positions(spots: pd.DataFrame, fov: str, channel: str, offset=(0.0, 0.0, 0.0)) -> pd.DataFrame:
    d = spots[(spots["fov_id"] == fov) & (spots["channel"] == channel) & spots["passes"]]
    return d.assign(**{p: d[p] + o for p, o in zip(POS, offset)})


def estimate_channel_offset(spots: pd.DataFrame, wells, ch_ref: str, ch_mov: str, max_dist_um: float = 0.8) -> dict:
    """Chromatic offset of ``ch_mov`` relative to ``ch_ref`` from mutual nearest-neighbour pairs.

    Use wells where both channels carry probes on the SAME transcripts (odds / evens).
    Returns the median (dz, dy, dx) in µm to ADD to ``ch_mov`` coordinates, plus spread and n.
    """
    disp = []
    s = spots[spots["well"].isin(wells) & spots["passes"] & (spots["cell_id"] > 0)]
    for fov in s["fov_id"].unique():
        a = _passing_positions(s, fov, ch_ref)[POS].to_numpy()
        b = _passing_positions(s, fov, ch_mov)[POS].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        dab, iab = cKDTree(b).query(a, distance_upper_bound=max_dist_um)
        dba, iba = cKDTree(a).query(b, distance_upper_bound=max_dist_um)
        for i, (d, j) in enumerate(zip(dab, iab)):
            if np.isfinite(d) and iba[j] == i:  # mutual nearest neighbours
                disp.append(a[i] - b[j])
    disp = np.asarray(disp).reshape(-1, 3)
    if len(disp) == 0:
        return {"offset_um": [0.0, 0.0, 0.0], "mad_um": [np.nan] * 3, "n_pairs": 0}
    med = np.median(disp, axis=0)
    mad = np.median(np.abs(disp - med), axis=0) * 1.4826
    return {"offset_um": med.round(4).tolist(), "mad_um": mad.round(4).tolist(), "n_pairs": int(len(disp))}


def colocalisation_efficiency(spots: pd.DataFrame, cells: pd.DataFrame,
                              compartments=("cytoplasm", "perinuclear")) -> pd.DataFrame:
    """Per (well, channel): fraction of passing exon spots with a colocalised partner exon spot.

    For odds / evens wells this estimates the per-probe-set detection efficiency:
    frac(odds with evens) ~ P(evens detected | molecule). Uses partner columns as
    recomputed by :func:`smfish.tss.recompute_partners` (passing spots, offset-corrected).
    """
    good = cells.loc[cells["qc_pass"], ["fov_id", "cell_id"]]
    s = spots[(spots["role"] == "exon") & spots["passes"] & spots["partner_channel"].notna()
              & spots["compartment"].isin(compartments)].merge(good, on=["fov_id", "cell_id"])
    partner_role = spots[["well", "channel", "role", "probe"]].drop_duplicates().rename(
        columns={"channel": "partner_channel", "role": "partner_role", "probe": "partner_probe"})
    s = s.merge(partner_role, on=["well", "partner_channel"])
    s = s[s["partner_role"] == "exon"]
    per_fov = s.groupby(["well", "condition", "channel", "probe", "partner_probe", "fov_id"])["partner_coloc"].mean()
    out = (
        s.groupby(["well", "condition", "channel", "probe", "partner_probe"])
        .agg(n_spots=("partner_coloc", "size"), frac_colocalised=("partner_coloc", "mean"))
        .join(per_fov.groupby(level=[0, 1, 2, 3, 4]).agg(fov_sd="std", n_fov="size"))
        .reset_index()
    )
    return out


# --------------------------------------------------------------------------- combined threshold scan


def threshold_scan(spots: pd.DataFrame, cells: pd.DataFrame, max_dist_nm: float = 400,
                   offsets_um: dict | None = None, confident_quantile: float = 0.5, n_points: int = 120) -> pd.DataFrame:
    """Per channel and LoG threshold: false positives per cell (negative-control wells, e.g. mrcB)
    and detection rate (odds / evens wells: fraction of *confident* partner-channel spots that have a
    spot in this channel within ``max_dist_nm``).

    Confident partner spots = in-cell spots at or above the partner's per-FOV automated threshold and
    in the top ``1 - confident_quantile`` of LoG score within the FOV, i.e. almost certainly real
    molecules. Only stored spots exist, so rates are defined down to the permissive detection floor.
    """
    offsets_um = offsets_um or {}
    s = spots[spots["cell_id"] > 0]
    good = cells.loc[cells["qc_pass"], ["fov_id", "cell_id", "well"]]
    rows = []
    for ch in sorted(s["channel"].unique()):
        hi = float(np.nanpercentile(s.loc[s["channel"] == ch, "log_score"], 99.5))
        grid = np.linspace(0, hi, n_points)
        out = pd.DataFrame({"channel": ch, "threshold": grid})

        # false positives per QC cell in wells where this channel carries the negative control
        # exon-kernel negatives only: the scan's detection side is odds/evens exon, and intron LoG
        # scores (larger kernel) are on another scale
        neg = s[(s["channel"] == ch) & is_negative(s) & (s["role"] != "intron")]
        neg_wells = neg["well"].unique()
        n_cells = int(good["well"].isin(neg_wells).sum())
        if n_cells:
            sc = np.sort(neg.merge(good[["fov_id", "cell_id"]], on=["fov_id", "cell_id"])["log_score"].to_numpy())
            out["fp_per_cell"] = (sc.size - np.searchsorted(sc, grid, side="left")) / n_cells
            out["fp_cells"], out["fp_wells"] = n_cells, ",".join(map(str, neg_wells))

        # detection rate against the partner channel in same-gene exon/exon wells (odds / evens)
        best = []
        for fov, f in s[s["role"] == "exon"].groupby("fov_id"):
            if f["channel"].nunique() != 2 or f["gene"].nunique() != 1 or ch not in set(f["channel"]):
                continue
            mine, other = f[f["channel"] == ch], f[f["channel"] != ch]
            pch = other["channel"].iloc[0]
            conf = other[other["log_score"] >= other["auto_threshold"]]
            conf = conf[conf["log_score"] >= conf["log_score"].quantile(confident_quantile)]
            if conf.empty or mine.empty:
                continue
            a = mine[POS].to_numpy() + np.asarray(offsets_um.get(ch, (0, 0, 0)))
            b = conf[POS].to_numpy() + np.asarray(offsets_um.get(pch, (0, 0, 0)))
            score = mine["log_score"].to_numpy()
            near = cKDTree(a).query_ball_point(b, r=max_dist_nm / 1000)
            best += [max((score[j] for j in js), default=-np.inf) for js in near]
        if best:
            best = np.sort(np.asarray(best))
            out["detection_rate"] = (best.size - np.searchsorted(best, grid, side="left")) / best.size
            out["n_confident_partner"] = best.size
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recommend_thresholds(scan: pd.DataFrame, max_fp_per_cell: float = 1.0) -> pd.DataFrame:
    """Lowest threshold per channel with false positives <= target, and the detection rate there."""
    rows = []
    for ch, d in scan.groupby("channel"):
        row = {"channel": ch, "max_fp_per_cell": max_fp_per_cell}
        if "fp_per_cell" in d and d["fp_per_cell"].notna().any():
            ok = d[d["fp_per_cell"] <= max_fp_per_cell]
            row["threshold"] = float(ok["threshold"].min()) if len(ok) else np.nan
            if "detection_rate" in d and np.isfinite(row["threshold"]):
                row["detection_rate_at_threshold"] = float(np.interp(row["threshold"], d["threshold"], d["detection_rate"]))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reports at the applied thresholds


def negative_control_report(spots: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """False-positive spots per QC cell (by compartment) in every negative-control well, at the
    thresholds actually applied (``passes``). Compare against the real signal in matched wells."""
    good = cells.loc[cells["qc_pass"], ["fov_id", "cell_id", "well"]]
    neg = spots[is_negative(spots) & spots["passes"] & (spots["cell_id"] > 0)]
    neg = neg.merge(good[["fov_id", "cell_id"]], on=["fov_id", "cell_id"])
    rows = []
    for (well, ch), d in spots[is_negative(spots)].groupby(["well", "channel"]):
        n_cells = int((good["well"] == well).sum())
        n = neg[(neg["well"] == well) & (neg["channel"] == ch)]
        row = {"well": well, "condition": d["condition"].iloc[0], "channel": ch, "probe": d["probe"].iloc[0],
               "source": "neg_control probe" if (d["role"] == "neg_control").all() else "target absent in cell line",
               "qc_cells": n_cells, "fp_per_cell": len(n) / n_cells if n_cells else np.nan}
        for comp in ("nucleus", "perinuclear", "cytoplasm"):
            row[f"fp_per_cell_{comp}"] = (n["compartment"] == comp).sum() / n_cells if n_cells else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def single_allele_tss_check(tss_per_cell: pd.DataFrame, cells: pd.DataFrame, layout_alleles: pd.DataFrame) -> pd.DataFrame:
    """For knock-in-only intron probes (one tagged allele): fraction of QC nuclei with >2 active TSSs,
    which should be ~0 (max 1, or 2 sister chromatids after replication). High values = threshold too low."""
    t = tss_per_cell.merge(cells[["fov_id", "cell_id", "qc_pass"]], on=["fov_id", "cell_id"])
    t = t[t["qc_pass"]].merge(layout_alleles, on=["well", "channel"])
    rows = []
    for (well, ch), d in t.groupby(["well", "channel"]):
        rows.append({"well": well, "condition": d["condition"].iloc[0], "channel": ch, "probe": d["probe"].iloc[0],
                     "expected_absent": bool(d["expected_absent"].iloc[0]), "nuclei": len(d),
                     "frac_0": (d["n_tss"] == 0).mean(), "frac_1": (d["n_tss"] == 1).mean(),
                     "frac_2": (d["n_tss"] == 2).mean(), "frac_gt2": (d["n_tss"] > 2).mean()})
    return pd.DataFrame(rows)


def odds_evens_grid(spots: pd.DataFrame, max_dist_nm: float = 400, offsets_um: dict | None = None,
                    n_points: int = 25) -> pd.DataFrame:
    """2025 odds/evens criterion: for every pair of thresholds (channel A, channel B) in wells where both
    channels carry exon probes on the same mRNA, ``combined = frac(A with B partner) x frac(B with A partner)``.

    The best pair maximises agreement between two independent probe sets on the same molecules.
    Distances are physical (``max_dist_nm`` in 3D after chromatic-offset correction), not pixels.
    """
    offsets_um = offsets_um or {}
    s = spots[(spots["cell_id"] > 0) & (spots["role"] == "exon") & ~is_negative(spots)]
    rows = []
    for well, w in s.groupby("well"):
        if w["channel"].nunique() != 2 or w["gene"].nunique() != 1:
            continue
        ch_a, ch_b = sorted(w["channel"].unique())
        acc = {ch_a: ([], []), ch_b: ([], [])}  # per channel: own scores, best partner score within r
        for _, f in w.groupby("fov_id"):
            A, B = f[f["channel"] == ch_a], f[f["channel"] == ch_b]
            if A.empty or B.empty:
                continue
            pa = A[POS].to_numpy() + np.asarray(offsets_um.get(ch_a, (0, 0, 0)))
            pb = B[POS].to_numpy() + np.asarray(offsets_um.get(ch_b, (0, 0, 0)))
            for src, p_src, dst, p_dst, ch in ((A, pa, B, pb, ch_a), (B, pb, A, pa, ch_b)):
                near = cKDTree(p_dst).query_ball_point(p_src, r=max_dist_nm / 1000)
                d_score = dst["log_score"].to_numpy()
                acc[ch][0].extend(src["log_score"].to_numpy())
                acc[ch][1].extend(max((d_score[j] for j in js), default=-np.inf) for js in near)
        sa, ba = (np.asarray(x) for x in acc[ch_a])
        sb, bb = (np.asarray(x) for x in acc[ch_b])
        if sa.size == 0 or sb.size == 0:
            continue
        ga = np.unique(np.quantile(sa, np.linspace(0, 0.98, n_points)))
        gb = np.unique(np.quantile(sb, np.linspace(0, 0.98, n_points)))
        for ta in ga:
            for tb in gb:
                ka, kb = sa >= ta, sb >= tb
                fa = float(np.mean(ba[ka] >= tb)) if ka.any() else np.nan
                fb = float(np.mean(bb[kb] >= ta)) if kb.any() else np.nan
                rows.append({"well": well, "gene": w["gene"].iloc[0], "channel_a": ch_a, "threshold_a": ta,
                             "channel_b": ch_b, "threshold_b": tb, "n_a": int(ka.sum()), "n_b": int(kb.sum()),
                             "frac_a_with_b": fa, "frac_b_with_a": fb, "combined": fa * fb})
    return pd.DataFrame(rows)


def best_odds_evens_pairs(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    return grid.loc[grid.groupby("well")["combined"].idxmax()].reset_index(drop=True)
