"""Control-well calibration: odds/evens offset + efficiency, mrcB thresholds."""

import json

import numpy as np
import pandas as pd

from smfish.aggregate import run_aggregation
from smfish.io import read_table
from smfish.layout import load_layout
from smfish.pipeline import run_fov
from synthetic import VOXEL, add_spot, make_fov

SHIFT_PX = np.array([2, 0, 2])  # evens channel displaced by +2 z (0.6 µm) and +2 x (0.13 µm)


def _odds_evens_fov(seed=2, keep_frac=0.7):
    base = make_fov(seed=seed, n_cyto=80, n_tss=0, n_lone_intron=0)
    rng = np.random.default_rng(seed)
    evens = np.zeros_like(base["exon"]) + 20
    kept = base["cyto_spots"][rng.random(len(base["cyto_spots"])) < keep_frac]
    for p in kept:
        add_spot(evens, (p + SHIFT_PX).astype(float), 300.0)
    evens = rng.poisson(evens).astype(np.float32)
    return base, evens, len(kept)


def _run(cfg, tmp_path, well, images, base):
    path = tmp_path / f"20250806_000000_001__Well{well:02d}_synthetic.ome.tiff"
    run_fov(path, cfg, load_layout(cfg), images=images, voxel_um=VOXEL, labels=(base["cells"], base["nuclei"]))


def test_odds_evens_offset_and_efficiency(cfg, tmp_path):
    base, evens, n_kept = _odds_evens_fov()
    images = {"DAPI": base["dapi"], "ConA": base["cona"], "Atto565": base["exon"], "Atto633": evens}
    _run(cfg, tmp_path, 2, images, base)

    agg = run_aggregation(cfg)
    off = pd.read_csv(agg / "calibration" / "channel_offsets.csv").iloc[0]
    est = np.array(json.loads(off["offset_um"]))
    assert np.allclose(est, -SHIFT_PX * np.array(VOXEL), atol=0.03), est

    # without correction a 0.61 µm shift breaks colocalisation at a 0.4 µm radius ...
    raw = pd.read_csv(agg / "calibration" / "colocalisation_efficiency.csv").set_index("channel")
    assert raw.loc["Atto633", "frac_colocalised"] < 0.2
    # ... and correcting it recovers the true efficiencies
    cfg["tss"]["channel_offsets_um"] = {"Atto633": est.tolist()}
    agg = run_aggregation(cfg)
    eff = pd.read_csv(agg / "calibration" / "colocalisation_efficiency.csv").set_index("channel")
    assert abs(eff.loc["Atto565", "frac_colocalised"] - n_kept / len(base["cyto_spots"])) < 0.06
    assert eff.loc["Atto633", "frac_colocalised"] > 0.95


def test_mrcb_negative_control_threshold(cfg, tmp_path):
    base = make_fov(seed=1, n_cyto=60, n_tss=0, n_lone_intron=0)
    rng = np.random.default_rng(5)
    noise = rng.poisson(np.full(base["exon"].shape, 20.0)).astype(np.float32)
    for p in base["cyto_spots"][:3]:  # three dim false positives
        add_spot(noise, p.astype(float), 60.0)
    # well 1: Atto565 = mrcB (negative control), Atto633 = GAPDH exon
    images = {"DAPI": base["dapi"], "ConA": base["cona"], "Atto565": noise, "Atto633": base["exon"]}
    _run(cfg, tmp_path, 1, images, base)
    agg = run_aggregation(cfg)
    nct = pd.read_csv(agg / "calibration" / "negative_control_thresholds.csv").set_index("channel")
    thr = nct.loc["Atto565", "threshold"]
    spots = read_table(agg / "spots")
    neg = spots[(spots["channel"] == "Atto565") & (spots["cell_id"] > 0)]
    assert (neg["log_score"] >= thr).sum() <= 1  # one cell, target <= 1 false positive per cell
    # a channel-level threshold is picked up by apply_thresholds; channels without one fall back to the default
    # (here each FOV's automated threshold; per-probe "<probe>@<channel>" keys would take precedence)
    cfg["aggregation"]["thresholds"] = {"default": "auto", "Atto565": float(thr)}
    spots = read_table(run_aggregation(cfg) / "spots")
    assert (spots.loc[spots["channel"] == "Atto565", "threshold_used"] == thr).all()
    assert spots.loc[spots["channel"] == "Atto633", "threshold_used"].equals(
        spots.loc[spots["channel"] == "Atto633", "auto_threshold"]
    )


def test_threshold_scan_mrcb_plus_odds_evens(cfg, tmp_path):
    """mrcB well (W1, 565 = noise + 3 dim false spots) + odds/evens well (W2) -> scan has both curves."""
    base, evens, n_kept = _odds_evens_fov(seed=4)
    _run(cfg, tmp_path, 2, {"DAPI": base["dapi"], "ConA": base["cona"], "Atto565": base["exon"], "Atto633": evens}, base)
    neg = make_fov(seed=1, n_cyto=60, n_tss=0, n_lone_intron=0)
    rng = np.random.default_rng(5)
    noise = rng.poisson(np.full(neg["exon"].shape, 20.0)).astype(np.float32)
    for p in neg["cyto_spots"][:3]:
        add_spot(noise, p.astype(float), 60.0)
    _run(cfg, tmp_path, 1, {"DAPI": neg["dapi"], "ConA": neg["cona"], "Atto565": noise, "Atto633": neg["exon"]}, neg)
    cfg["tss"]["channel_offsets_um"] = {"Atto633": (-SHIFT_PX * np.array(VOXEL)).tolist()}
    agg = run_aggregation(cfg)
    scan = pd.read_csv(agg / "calibration" / "threshold_scan.csv")
    rec = pd.read_csv(agg / "calibration" / "recommended_thresholds.csv").set_index("channel")
    s565 = scan[scan.channel == "Atto565"]
    assert s565["fp_per_cell"].iloc[0] >= s565["fp_per_cell"].iloc[-1]  # monotone non-increasing
    # 565 (odds) sees every molecule: detection of confident evens spots ~1 at the recommended threshold
    assert rec.loc["Atto565", "detection_rate_at_threshold"] > 0.9
    # 633 (evens) carries only a fraction of the molecules -> its detection rate ~ that fraction at low threshold
    s633 = scan[scan.channel == "Atto633"]
    assert abs(s633["detection_rate"].iloc[0] - n_kept / len(base["cyto_spots"])) < 0.1


def test_absent_target_wells_are_negative_controls(cfg, tmp_path):
    """W9: Atto633 = NeoR in OsTIR (no neo cassette) -> background; W16: one-allele TetO check."""
    base = make_fov(seed=9, n_cyto=60, n_tss=0, n_lone_intron=0)
    rng = np.random.default_rng(9)
    bg = rng.poisson(np.full(base["exon"].shape, 20.0)).astype(np.float32)
    for p in base["cyto_spots"][:4]:  # a few dim cytoplasmic granules, mRNA-like but not NeoR
        add_spot(bg, p.astype(float), 70.0)
    _run(cfg, tmp_path, 9, {"DAPI": base["dapi"], "ConA": base["cona"], "Atto565": base["exon"], "Atto633": bg}, base)
    w16 = make_fov(seed=16, n_tss=4)  # 4 TetO TSS per nucleus: impossible for a single tagged allele
    _run(cfg, tmp_path, 16, {"DAPI": w16["dapi"], "ConA": w16["cona"], "Atto565": w16["exon"], "Atto633": w16["intron"]}, w16)
    agg = run_aggregation(cfg)

    neg = pd.read_csv(agg / "calibration" / "negative_controls.csv").set_index(["well", "channel"])
    assert neg.loc[(9, "Atto633"), "source"] == "target absent in cell line"
    assert neg.loc[(9, "Atto633"), "probe"] == "NeoR"
    scan = pd.read_csv(agg / "calibration" / "threshold_scan.csv")
    assert scan.loc[scan.channel == "Atto633", "fp_per_cell"].notna().any()  # 633 noise floor from W9
    refs = pd.read_csv(agg / "references.csv").set_index(["well", "channel"])
    assert (9, "Atto633") not in refs.index  # background spots never become a single-mRNA reference
    chk = pd.read_csv(agg / "calibration" / "single_allele_tss_check.csv").set_index("well")
    assert chk.loc[16, "frac_gt2"] == 1.0


def test_odds_evens_grid_matches_2025_criterion(cfg, tmp_path):
    base, evens, n_kept = _odds_evens_fov(seed=7)
    _run(cfg, tmp_path, 2, {"DAPI": base["dapi"], "ConA": base["cona"], "Atto565": base["exon"], "Atto633": evens}, base)
    cfg["tss"]["channel_offsets_um"] = {"Atto633": (-SHIFT_PX * np.array(VOXEL)).tolist()}
    agg = run_aggregation(cfg)
    best = pd.read_csv(agg / "calibration" / "odds_evens_best_pairs.csv").iloc[0]
    # expected: odds spots whose (shifted) evens partner exists AND still lies inside the cell
    rng = np.random.default_rng(7)
    kept = base["cyto_spots"][rng.random(len(base["cyto_spots"])) < 0.7] + SHIFT_PX
    kept = kept[(kept < np.array(base["cells"].shape)).all(1)]
    keep = (base["cells"][tuple(kept.T)] > 0).sum() / len(base["cyto_spots"])
    assert best["channel_a"] == "Atto565" and best["channel_b"] == "Atto633"
    assert abs(best["frac_a_with_b"] - keep) < 0.1  # odds with evens partner = evens detection efficiency
    assert best["frac_b_with_a"] > 0.9
    assert abs(best["combined"] - keep) < 0.1
