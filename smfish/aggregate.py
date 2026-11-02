"""Experiment-level aggregation of per-FOV outputs.

1. Apply the final per-probe spot threshold (``passes``), and drop candidates in clumps of
   non-specific staining (``in_clump``, :mod:`smfish.clumps`).
2. Estimate a single-transcript reference per (well, channel) from isolated,
   in-cell, non-border spots (KDE mode of background-subtracted intensity).
3. Convert intensities to transcript equivalents (continuous, never ceil-binned).
4. Build zero-filled per-cell x channel x compartment counts, and TSS tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .assign import COMPARTMENTS
from .io import read_table, write_table
from .spotquant import single_rna_reference

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- collect


def collect(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Concatenate spots / cells / qc of every FOV that has a DONE marker."""
    fov_root = Path(output_dir) / "fov"
    spots, cells, qcs = [], [], []
    for d in sorted(p for p in fov_root.iterdir() if p.is_dir()):
        if not (d / "DONE.json").exists():
            continue
        spots.append(read_table(d / "spots"))
        cells.append(read_table(d / "cells"))
        qc = json.loads((d / "qc.json").read_text())
        if "image_shape_zyx" not in qc and (d / "labels.tif").exists():  # FOVs processed before this was recorded
            import tifffile

            with tifffile.TiffFile(d / "labels.tif") as t:
                zcyx = t.series[0].shape
            qc["image_shape_zyx"] = [zcyx[0], zcyx[-2], zcyx[-1]]
        qcs.append(qc)
    if not cells:
        raise FileNotFoundError(f"No completed FOVs under {fov_root}")
    return pd.concat(spots, ignore_index=True), pd.concat(cells, ignore_index=True), qcs


# --------------------------------------------------------------------------- thresholds / references


def refresh_probe_metadata(spots: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Take probe metadata that does not affect image processing (gene, allele, expected_absent) from
    the *current* layout / probes.yaml, so corrections apply by re-aggregating instead of re-running.
    The role is stored per FOV (it chose the detection kernel); a mismatch with the config is an error."""
    from .layout import load_layout

    lay = load_layout(cfg)[["well", "channel", "probe", "role", "gene", "allele", "expected_absent"]]
    s = spots.drop(columns=[c for c in ("gene", "allele", "expected_absent") if c in spots], errors="ignore")
    s = s.merge(lay, on=["well", "channel"], how="left", suffixes=("", "_layout"))
    bad = s["probe"] != s["probe_layout"]
    bad |= s["role"] != s["role_layout"]
    if bad.any():
        ex = s.loc[bad, ["well", "channel", "probe", "probe_layout", "role", "role_layout"]].drop_duplicates()
        raise ValueError(f"Spots were processed with a different probe / role than the current layout:\n{ex}")
    return s.drop(columns=["probe_layout", "role_layout"])


def threshold_setting(th: dict, probe: str, channel: str):
    """The configured threshold for one probe in one channel, and the key it came from.

    Lookup order: ``"<probe>@<channel>"``, probe name, channel, then ``default``. The first form is
    needed for probes imaged in both channels (mrcB, GAPDH): LoG scores are not comparable across
    channels.
    """
    for key in (f"{probe}@{channel}", probe, channel):
        if key in th:
            return th[key], key
    return th.get("default", "auto"), "default"


def probe_median_auto(spots: pd.DataFrame) -> pd.Series:
    """Median of the per-FOV automated thresholds over every FOV imaging a probe, per (probe, channel)."""
    per_fov = spots.groupby(["probe", "channel", "fov_id"])["auto_threshold"].first()
    return per_fov.groupby(level=["probe", "channel"]).median()


def apply_thresholds(spots: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Final LoG threshold per spot (``threshold_used``, ``passes``); see :func:`threshold_setting`.

    A setting is a number (LoG units, e.g. calibrated on a negative control), ``"probe_median"``
    (:func:`probe_median_auto`: one value per probe and channel, the same in every well) or ``"auto"``
    (each FOV's own bigfish elbow threshold). ``auto`` moves from FOV to FOV with brightness and spot
    density, so it differs systematically between conditions -- on 250806 the POLR2A intron elbow sits
    at ~1.6-2.1 in Streaming wells and ~3.4 in OsTIR wells -- and counts thresholded that way cannot be
    compared between wells.
    """
    th = cfg["aggregation"]["thresholds"]
    medians = probe_median_auto(spots)
    fixed = {}
    for probe, channel in spots[["probe", "channel"]].drop_duplicates().itertuples(index=False):
        setting, _ = threshold_setting(th, probe, channel)
        if str(setting) == "probe_median":
            fixed[(probe, channel)] = medians.get((probe, channel), np.nan)
        elif str(setting) != "auto":
            fixed[(probe, channel)] = float(setting)
    keys = pd.MultiIndex.from_frame(spots[["probe", "channel"]])
    value = pd.Series(fixed, dtype=float).reindex(keys).to_numpy() if fixed else np.full(len(spots), np.nan)
    spots = spots.copy()
    spots["threshold_used"] = np.where(np.isnan(value), spots["auto_threshold"], value)
    spots["passes"] = spots["log_score"] >= spots["threshold_used"]
    return spots


def remove_clumps(spots: pd.DataFrame, cells: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flag candidates in clumps of non-specific staining (:mod:`smfish.clumps`, ``aggregation.clumps``).

    ``clump_id`` (-1 = none) and ``in_clump``; clump members stop passing, so they are neither counted nor
    used as references. ``above_threshold`` keeps the plain score test for the clump report. With clumps
    disabled nothing is flagged and ``passes`` is unchanged.
    """
    from .calibrate import kernel_of_role
    from .clumps import clump_settings, flag_clumps

    settings = clump_settings(cfg)
    spots = spots.copy()
    spots["above_threshold"] = spots["passes"]
    spots["clump_id"] = flag_clumps(spots, cells, kernel_of_role(cfg), settings) if settings["enabled"] else -1
    spots["in_clump"] = spots["clump_id"] >= 0
    spots["passes"] = spots["above_threshold"] & ~spots["in_clump"]
    return spots


def thresholds_applied(spots: pd.DataFrame, cfg: dict, detection: pd.DataFrame) -> pd.DataFrame:
    """One row per (probe, channel): the threshold actually applied and the config key it came from.

    ``stored_floor`` is the highest LoG level below which some FOV stored no spots (its permissive
    detection threshold); a threshold under it silently loses spots in those FOVs (logged).
    """
    th = cfg["aggregation"]["thresholds"]
    used = spots.groupby(["probe", "channel"])["threshold_used"].agg(["min", "median", "max"])
    floor = (detection.groupby(["probe", "channel"])["detection_threshold"].max()
             if {"probe", "channel", "detection_threshold"} <= set(detection.columns) else pd.Series(dtype=float))
    rows = []
    for (probe, channel), u in used.iterrows():
        setting, key = threshold_setting(th, probe, channel)
        f = float(floor.get((probe, channel), np.nan))
        rows.append({"probe": probe, "channel": channel, "setting": setting, "config_key": key,
                     "threshold_min": u["min"], "threshold_median": u["median"], "threshold_max": u["max"],
                     "stored_floor": f, "below_floor": bool(np.isfinite(f) and u["min"] < f)})
    out = pd.DataFrame(rows)
    for r in out[out["below_floor"]].itertuples() if len(out) else []:
        log.warning("%s @ %s: threshold %.3g is below the stored floor %.3g of some FOVs - spots between the "
                    "two were never stored there", r.probe, r.channel, r.threshold_min, r.stored_floor)
    return out


def flag_cells(cells: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cells = cells.copy()
    agg = cfg["aggregation"]
    ok = cells["has_nucleus"].copy()
    # size filters matter when masks are re-used (legacy masks contain 10-30 µm³ fragments)
    ok &= cells["cell_volume_um3"] >= agg.get("min_cell_volume_um3", 0)
    ok &= cells["nucleus_volume_um3"] >= agg.get("min_nucleus_volume_um3", 0)
    if cfg["aggregation"].get("exclude_border_cells", True):
        ok &= ~cells["border_xy"] & ~cells["border_z"]
    cells["qc_pass"] = ok
    return cells


def reference_spot_mask(spots: pd.DataFrame, cells: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Spots that represent single transcripts: passing, isolated, unclipped window, in QC cells;
    cytoplasmic (incl. perinuclear) for exon / neg_control, nuclear for intron."""
    agg, q = cfg["aggregation"], cfg["quantification"]
    good = pd.MultiIndex.from_frame(cells.loc[cells["qc_pass"], ["fov_id", "cell_id"]])
    in_good = pd.MultiIndex.from_frame(spots[["fov_id", "cell_id"]]).isin(good)
    full_window = spots["n_voxels"] == spots.groupby(["well", "channel"])["n_voxels"].transform("max")
    base = spots["passes"] & in_good & full_window & (spots["nn_dist_um"] >= q["isolation_distance_um"])
    if "expected_absent" in spots:  # target absent in this cell line: these spots are background
        base &= ~spots["expected_absent"].fillna(False).astype(bool)
    cyto = spots["compartment"].isin(["cytoplasm", "perinuclear"])
    if agg.get("reference_compartment", "cytoplasm") == "any":
        cyto = spots["cell_id"] > 0
    return (base & np.where(spots["role"] == "intron", spots["in_nucleus"], cyto)).to_numpy()


def compute_references(spots: pd.DataFrame, cells: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Single-transcript intensity per (well, channel): KDE mode of reference spots (see reference_spot_mask)."""
    agg = cfg["aggregation"]
    s = spots[reference_spot_mask(spots, cells, cfg)]
    rows = []
    for (well, channel), d in s.groupby(["well", "channel"]):
        ref = single_rna_reference(d["intensity_bgsub"])
        rows.append({"well": well, "channel": channel, "probe": d["probe"].iloc[0], "role": d["role"].iloc[0],
                     "ref_mode": ref["mode"], "ref_median": ref["median"], "ref_n": ref["n"]})
    refs = pd.DataFrame(rows, columns=["well", "channel", "probe", "role", "ref_mode", "ref_median", "ref_n"])
    low = refs["ref_n"] < agg.get("min_reference_spots", 50)
    if low.any():
        log.warning("References with too few spots (set to NaN):\n%s", refs.loc[low, ["well", "channel", "probe", "ref_n"]])
        refs.loc[low, "ref_mode"] = np.nan
    return refs


def add_rna_estimates(spots: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    r = refs[["well", "channel", "ref_mode"]]
    s = spots.merge(r, on=["well", "channel"], how="left")
    s["rna_equiv"] = s["intensity_bgsub"] / s["ref_mode"]
    s["n_rna"] = np.where(s["passes"], np.clip(np.round(s["rna_equiv"]), 1, None), 0)
    s.loc[s["passes"] & s["rna_equiv"].isna(), "n_rna"] = np.nan
    if "partner_channel" in s:
        pr = r.rename(columns={"channel": "partner_channel", "ref_mode": "partner_ref_mode"})
        s = s.merge(pr, on=["well", "partner_channel"], how="left")
        s["partner_rna_equiv"] = s["partner_intensity_bgsub"] / s["partner_ref_mode"]
    return s


# --------------------------------------------------------------------------- per-cell tables


def cell_counts_long(spots: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, channel, compartment) - zero-filled, so non-expressing cells are kept."""
    chans = spots[["well", "channel", "probe", "gene", "role"]].drop_duplicates()
    grid = cells[["fov_id", "well", "condition", "cell_id"]].merge(chans, on="well")
    grid = grid.merge(pd.DataFrame({"compartment": COMPARTMENTS}), how="cross")
    passing = spots[spots["passes"] & (spots["cell_id"] > 0)]
    counts = (
        passing.groupby(["fov_id", "cell_id", "channel", "compartment"])
        .agg(n_spots=("spot_index", "size"), n_rna=("n_rna", "sum"))
        .reset_index()
    )
    out = grid.merge(counts, on=["fov_id", "cell_id", "channel", "compartment"], how="left")
    out[["n_spots", "n_rna"]] = out[["n_spots", "n_rna"]].fillna(0)
    # n_rna is undefined (NaN, not 0) where the (well, channel) has no single-transcript reference
    has_ref = spots.groupby(["well", "channel"])["ref_mode"].agg(lambda r: r.notna().any()).rename("_has_ref")
    out = out.merge(has_ref.reset_index(), on=["well", "channel"], how="left")
    out.loc[~out["_has_ref"].fillna(False).astype(bool), "n_rna"] = np.nan
    return out.drop(columns="_has_ref")


def cell_counts_wide(long: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Cells table with columns ``<channel>_<compartment>_n_spots`` / ``_n_rna`` plus ``_cell`` totals."""
    tot = long.groupby(["fov_id", "cell_id", "channel"])[["n_spots", "n_rna"]].sum().reset_index()
    tot["compartment"] = "cell"
    both = pd.concat([long[tot.columns], tot], ignore_index=True)
    wide = both.pivot_table(index=["fov_id", "cell_id"], columns=["channel", "compartment"], values=["n_spots", "n_rna"])
    wide.columns = [f"{ch}_{comp}_{val}" for val, ch, comp in wide.columns]
    wide = wide.reset_index()
    probes = long.drop_duplicates(["fov_id", "channel"]).pivot(index="fov_id", columns="channel", values="probe")
    probes.columns = [f"{c}_probe" for c in probes.columns]
    return cells.merge(wide, on=["fov_id", "cell_id"], how="left").merge(probes.reset_index(), on="fov_id", how="left")


def tss_table(spots: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Transcription-site calls: nuclear intron spots (optionally colocalised with an exon spot).

    ``nascent_rna`` = exon-channel signal at the intron spot / exon single-mRNA reference
    (continuous; NaN if the well has no exon partner channel for this gene).
    """
    require_coloc = cfg["tss"].get("require_exon_colocalisation", False)
    roles = spots[["well", "channel", "role"]].drop_duplicates()
    roles = roles.rename(columns={"channel": "partner_channel", "role": "partner_role"})
    s = spots[(spots["role"] == "intron") & spots["passes"] & spots["in_nucleus"] & (spots["cell_id"] > 0)].copy()
    if "partner_channel" in s:
        s = s.merge(roles, on=["well", "partner_channel"], how="left")
    else:
        s["partner_role"] = np.nan
    has_exon = s["partner_role"] == "exon"
    coloc = s["partner_coloc"].astype("boolean").fillna(False).astype(bool) if "partner_coloc" in s else False
    s["exon_spot_at_tss"] = np.where(has_exon, coloc, False)
    s["is_tss"] = np.where(has_exon & require_coloc, coloc, True)
    s["nascent_rna"] = np.where(has_exon, s.get("partner_rna_equiv", np.nan), np.nan)
    keep = ["fov_id", "well", "condition", "cell_id", "channel", "probe", "gene", "allele", "expected_absent", "spot_index",
            "z_um", "y_um", "x_um", "dist_nucleus_um", "intensity_bgsub", "rna_equiv", "partner_channel",
            "partner_role", "partner_dist_um", "partner_coloc", "exon_spot_at_tss", "partner_intensity_bgsub", "nascent_rna",
            "is_tss"]
    return s[[c for c in keep if c in s.columns]].rename(columns={"rna_equiv": "intron_rna_equiv"})


def tss_per_cell(tss: pd.DataFrame, cells: pd.DataFrame, spots: pd.DataFrame) -> pd.DataFrame:
    """Per cell and intron channel: number of active TSSs and nascent RNA (zero-filled)."""
    intron_ch = spots.loc[spots["role"] == "intron", ["well", "channel", "probe", "allele"]].drop_duplicates()
    grid = cells.loc[cells["has_nucleus"], ["fov_id", "well", "condition", "cell_id"]].merge(intron_ch, on="well")
    active = tss[tss["is_tss"]]
    agg = (
        active.groupby(["fov_id", "cell_id", "channel"])
        .agg(n_tss=("spot_index", "size"), nascent_rna_sum=("nascent_rna", "sum"), nascent_rna_max=("nascent_rna", "max"))
        .reset_index()
    )
    out = grid.merge(agg, on=["fov_id", "cell_id", "channel"], how="left")
    out["n_tss"] = out["n_tss"].fillna(0).astype(int)
    out["any_tss"] = out["n_tss"] > 0
    return out


# --------------------------------------------------------------------------- driver


def run_aggregation(cfg: dict) -> Path:
    from .calibrate import (
        best_odds_evens_pairs,
        check_against_detection,
        colocalisation_efficiency,
        estimate_channel_offset,
        fp_target,
        kernel_of_role,
        negative_control_curve,
        negative_control_report,
        negative_control_thresholds,
        odds_evens_grid,
        propose_thresholds,
        proposed_thresholds_yaml,
        recommend_thresholds,
        single_allele_tss_check,
        threshold_scan,
    )
    from .clumps import clump_report
    from .tss import recompute_partners

    out_dir = Path(cfg["paths"]["output_dir"])
    agg_dir = out_dir / "aggregated"
    cal_dir = agg_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    spots, cells, qcs = collect(out_dir)
    spots = refresh_probe_metadata(spots, cfg)
    cells = flag_cells(cells, cfg)
    spots = apply_thresholds(spots, cfg)
    spots = remove_clumps(spots, cells, cfg)
    spots = recompute_partners(spots, cfg["tss"]["colocalisation_distance_nm"], cfg["tss"].get("channel_offsets_um"))
    if cfg["aggregation"].get("spot_flatfield", True):
        from .flatfield import apply_spot_flatfield, fit_spot_flatfield

        shapes = {tuple(q["image_shape_zyx"][1:]) for q in qcs if "image_shape_zyx" in q}
        if len(shapes) != 1:
            raise ValueError(f"Spot flat-field needs one common image shape, got {shapes}")
        shape_yx = shapes.pop()
        ref_spots = spots[reference_spot_mask(spots, cells, cfg) & (spots["role"] != "intron").to_numpy()]
        ff = fit_spot_flatfield(ref_spots, shape_yx, cfg["aggregation"].get("spot_flatfield_min_spots", 200))
        spots = apply_spot_flatfield(spots, ff, shape_yx)
        (cal_dir / "spot_flatfield.json").write_text(json.dumps(ff, indent=2))
        log.info("Spot flat-field edge/centre, corner/centre: %s",
                 {k: (round(v["edge_over_centre"], 2), round(v["corner_over_centre"], 2)) for k, v in ff.items()})
    refs = compute_references(spots, cells, cfg)
    spots = add_rna_estimates(spots, refs)
    long = cell_counts_long(spots, cells)
    wide = cell_counts_wide(long, cells)
    tss = tss_table(spots, cfg)
    tpc = tss_per_cell(tss, cells, spots)

    write_table(spots, agg_dir / "spots")
    write_table(cells, agg_dir / "cells")
    write_table(long, agg_dir / "cell_counts_long")
    write_table(wide, agg_dir / "cell_counts_wide")
    write_table(tss, agg_dir / "tss")
    write_table(tpc, agg_dir / "tss_per_cell")
    refs.to_csv(agg_dir / "references.csv", index=False)
    det = pd.DataFrame(
        [{"fov_id": q["fov_id"], "well": q["well"], "channel": ch, **{k: v for k, v in info.items() if not k.startswith("elbow")}}
         for q in qcs for ch, info in q["channels"].items()]
    )
    det.to_csv(agg_dir / "detection_thresholds.csv", index=False)
    applied = thresholds_applied(spots, cfg, det)
    applied.to_csv(cal_dir / "thresholds_applied.csv", index=False)
    log.info("Thresholds applied:\n%s", applied.to_string())

    clumps = clump_report(spots, cells)
    clumps.to_csv(cal_dir / "clumps_by_well.csv", index=False)
    if spots["in_clump"].any():
        log.info("Spots in clumps of non-specific staining (removed from counts and calibration):\n%s",
                 clumps[clumps["in_clumps"] > 0].round(4).to_string(index=False))

    # ---- calibration reports from control wells (applied only via the config, never silently)
    # candidates in clumps are neither false spots nor real ones, so calibration leaves them out
    max_fp = cfg["aggregation"].get("max_false_positives_per_cell", 1.0)
    kernel_of = kernel_of_role(cfg)
    calib = spots[~spots["in_clump"]]
    curve = negative_control_curve(calib, cells, kernel_of,
                                   statistic=cfg["aggregation"].get("calibration_statistic", "pooled"))
    nct = pd.DataFrame(columns=["channel", "kernel", "probe", "threshold", "wells"])
    if len(curve):
        curve.to_csv(cal_dir / "negative_control_curve.csv", index=False)
        nct = negative_control_thresholds(curve, calib, max_fp)
        nct.to_csv(cal_dir / "negative_control_thresholds.csv", index=False)
        check_against_detection(nct, det).to_csv(cal_dir / "detection_above_calibrated_threshold.csv", index=False)
        log.info("Negative-control thresholds:\n%s", nct.to_string())
    if "partner_channel" in spots:
        colocalisation_efficiency(spots, cells).to_csv(cal_dir / "colocalisation_efficiency.csv", index=False)
    scan = threshold_scan(calib, cells, cfg["tss"]["colocalisation_distance_nm"], cfg["tss"].get("channel_offsets_um"))
    proposed = propose_thresholds(calib, curve, nct, det, kernel_of, scan)
    proposed.to_csv(cal_dir / "proposed_thresholds.csv", index=False)
    (cal_dir / "proposed_thresholds.yaml").write_text(proposed_thresholds_yaml(proposed))
    log.info("Proposed per-probe thresholds (paste calibration/proposed_thresholds.yaml into "
             "aggregation.thresholds to apply):\n%s", proposed.to_string())
    if len(scan):
        scan.to_csv(cal_dir / "threshold_scan.csv", index=False)
        rec = recommend_thresholds(scan, fp_target(max_fp, "exon"))
        rec.to_csv(cal_dir / "recommended_thresholds.csv", index=False)
        log.info("Recommended thresholds (mrcB false positives + odds/evens detection):\n%s", rec.to_string())
    grid = odds_evens_grid(spots, cfg["tss"]["colocalisation_distance_nm"], cfg["tss"].get("channel_offsets_um"))
    if len(grid):
        grid.to_csv(cal_dir / "odds_evens_grid.csv", index=False)
        best = best_odds_evens_pairs(grid)
        best.to_csv(cal_dir / "odds_evens_best_pairs.csv", index=False)
        log.info("Odds/evens best threshold pairs (2025 criterion, frac_a x frac_b):\n%s", best.round(3).to_string())
    negative_control_report(spots, cells).to_csv(cal_dir / "negative_controls.csv", index=False)
    if "allele" in spots and spots["allele"].notna().any():
        alleles = spots.loc[spots["allele"].notna(), ["well", "channel", "expected_absent"]].drop_duplicates()
        single_allele_tss_check(tpc, cells, alleles).to_csv(cal_dir / "single_allele_tss_check.csv", index=False)
    offsets = []
    for well, d in spots[spots["role"] == "exon"].groupby("well"):
        if d["channel"].nunique() == 2 and d["gene"].nunique() == 1:  # e.g. odds / evens on one mRNA
            ch_ref, ch_mov = sorted(d["channel"].unique())
            offsets.append({"well": well, "gene": d["gene"].iloc[0], "ref_channel": ch_ref, "moving_channel": ch_mov,
                            **estimate_channel_offset(spots, [well], ch_ref, ch_mov)})
    if offsets:
        pd.DataFrame(offsets).to_csv(cal_dir / "channel_offsets.csv", index=False)
        log.info("Chromatic offsets from same-mRNA wells (set tss.channel_offsets_um to apply):\n%s",
                 pd.DataFrame(offsets).to_string())
    log.info("Aggregated %d FOVs, %d cells, %d spots -> %s", len(qcs), len(cells), len(spots), agg_dir)
    return agg_dir
