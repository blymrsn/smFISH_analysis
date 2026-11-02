"""Synthetic end-to-end: detection -> measurement -> assignment -> TSS -> aggregation."""

import numpy as np
import pandas as pd
import pytest

from smfish.aggregate import run_aggregation
from smfish.io import read_table
from smfish.layout import load_layout
from smfish.manifest import check_manifest
from smfish.pipeline import run_fov
from synthetic import VOXEL, make_fov


@pytest.mark.parametrize(
    "well, exon_ch, intron_ch",
    [
        (16, "Atto565", "Atto633"),  # TetO intron in 633, POLR2A exon in 565
        (10, "Atto633", "Atto565"),  # swapped orientation (the legacy pipeline got this wrong)
    ],
)
def test_end_to_end(cfg, tmp_path, well, exon_ch, intron_ch):
    fov = make_fov(seed=well)
    images = {"DAPI": fov["dapi"], "ConA": fov["cona"], exon_ch: fov["exon"], intron_ch: fov["intron"]}
    path = tmp_path / f"20250806_000000_001__Well{well:02d}_synthetic.ome.tiff"
    layout = load_layout(cfg)
    assert run_fov(path, cfg, layout, images=images, voxel_um=VOXEL, labels=(fov["cells"], fov["nuclei"])) == "done"
    assert run_fov(path, cfg, layout, images=images, voxel_um=VOXEL, labels=(fov["cells"], fov["nuclei"])) == "skipped"

    agg = run_aggregation(cfg)
    spots = read_table(agg / "spots")
    exon = spots[(spots["channel"] == exon_ch) & spots["passes"]]
    n_true = len(fov["cyto_spots"]) + len(fov["tss_spots"])
    assert abs(len(exon) - n_true) <= 2, (len(exon), n_true)

    # single-mRNA reference -> cytoplasmic spots ~1 transcript each
    cyto = exon[exon["compartment"].isin(["cytoplasm", "perinuclear"])]
    assert abs(np.median(cyto["rna_equiv"]) - 1) < 0.15

    # per-cell counts by compartment, from the intended channel
    long = read_table(agg / "cell_counts_long")
    nuc_count = long[(long["channel"] == exon_ch) & (long["compartment"] == "nucleus")]["n_spots"].item()
    assert nuc_count == len(fov["tss_spots"])
    cell_rna = long[long["channel"] == exon_ch]["n_rna"].sum()
    expected_rna = len(fov["cyto_spots"]) + 4 * len(fov["tss_spots"])
    assert abs(cell_rna - expected_rna) <= 0.1 * expected_rna
    # too few isolated intron spots for an intron reference -> n_rna undefined, not 0
    assert long[long["channel"] == intron_ch]["n_rna"].isna().all()
    n_intron = len(fov["tss_spots"]) + len(fov["lone_intron_spots"])
    assert long[long["channel"] == intron_ch]["n_spots"].sum() == n_intron

    # TSS (2025 definition): every passing nuclear intron spot is an active TSS; nascent RNA = exon-channel
    # signal measured at it. The lone intron spot (no exon signal) is active with ~0 nascent RNA.
    tss = read_table(agg / "tss")
    n_tss, n_lone = len(fov["tss_spots"]), len(fov["lone_intron_spots"])
    assert len(tss) == n_tss + n_lone and tss["is_tss"].all()
    assert (tss["channel"] == intron_ch).all() and (tss["partner_channel"] == exon_ch).all()
    with_exon = tss[tss["exon_spot_at_tss"]]
    assert len(with_exon) == n_tss
    print("nascent:", with_exon["nascent_rna"].round(2).tolist())
    assert np.allclose(with_exon["nascent_rna"], 4.0, atol=0.8), with_exon["nascent_rna"].tolist()
    assert (tss.loc[~tss["exon_spot_at_tss"], "nascent_rna"].abs() < 0.5).all()
    tpc = read_table(agg / "tss_per_cell")
    assert tpc["n_tss"].item() == n_tss + n_lone

    # stricter definition: require a detected exon spot at the intron spot
    cfg["tss"]["require_exon_colocalisation"] = True
    agg = run_aggregation(cfg)
    assert read_table(agg / "tss")["is_tss"].sum() == n_tss
    assert read_table(agg / "tss_per_cell")["n_tss"].item() == n_tss

    status = check_manifest(cfg, pd.DataFrame([{"index": 0, "fov_id": path.name[:-9], "well": well, "path": str(path)}]))
    assert status["status"].item() == "done"

