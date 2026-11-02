import numpy as np

from smfish.segment import assign_nuclei_to_cells, clean_labels, refine_watershed, upscale_labels_xy


def test_upscale_labels_nearest():
    small = np.array([[[1, 2], [3, 0]]], dtype=np.int32)
    big = upscale_labels_xy(small, (1, 4, 4))
    assert big.shape == (1, 4, 4)
    assert (big[0, :2, :2] == 1).all() and (big[0, 2:, 2:] == 0).all()


def test_clean_labels_keeps_largest_component_and_fills_holes():
    lab = np.zeros((3, 30, 30), np.int32)
    lab[:, 2:20, 2:20] = 1
    lab[1, 8:12, 8:12] = 0  # hole
    lab[:, 25:27, 25:27] = 1  # small detached fragment of the same label
    lab[:, 22:24, 2:4] = 2  # tiny object
    out = clean_labels(lab, min_voxels=50)
    assert out.max() == 1
    assert out[1, 10, 10] == 1  # hole filled
    assert out[:, 25:27, 25:27].max() == 0  # fragment removed


def test_nuclei_take_cell_label():
    cells = np.zeros((2, 20, 40), np.int32)
    cells[:, :, :20] = 7
    cells[:, :, 20:] = 9
    nuc = np.zeros_like(cells)
    nuc[:, 5:10, 5:12] = 1
    nuc[:, 5:10, 25:30] = 2
    nuc[:, 12:15, 18:23] = 3  # straddles the boundary, mostly in cell 9
    out, n = assign_nuclei_to_cells(nuc, cells)
    assert set(np.unique(out[nuc == 1])) == {7}
    assert set(np.unique(out[nuc == 2])) == {9}
    assert (out[(nuc == 3) & (cells == 7)] == 0).all()  # clipped to its cell
    assert n == {7: 1, 9: 2}


def test_watershed_refinement_preserves_labels():
    lab = np.zeros((1, 60, 60), np.int32)
    lab[0, 5:28, 5:55] = 1
    lab[0, 32:55, 5:55] = 2
    edge = np.zeros((1, 60, 60), np.float32)
    edge[0, 30, :] = 1.0
    out = refine_watershed(lab, edge, seed_shrink_px=3, background_shrink_px=3)
    assert set(np.unique(out)) == {0, 1, 2}
    assert out[0, 15, 30] == 1 and out[0, 45, 30] == 2


def test_column_majority_removes_owner_flips_keeps_foreground():
    from smfish.segment import column_majority

    cells = np.zeros((6, 20, 40), np.int32)
    cells[:, 2:18, 2:20] = 1
    cells[:, 2:18, 20:38] = 2
    cells[1:3, 2:18, 16:24] = 2   # thin sheet of cell 2 slides over cell 1 in two slices
    out = column_majority(cells)
    assert ((out > 0) == (cells > 0)).all()                      # foreground unchanged
    assert (out[:, 2:18, 2:20] == 1).all() and (out[:, 2:18, 20:38] == 2).all()


def test_nucleus_expansion_cells():
    from smfish.segment import nucleus_expansion_cells

    nuc = np.zeros((12, 80, 120), np.int32)
    nuc[4:8, 30:50, 20:40] = 3
    nuc[4:9, 30:50, 80:100] = 7
    cells = nucleus_expansion_cells(nuc, (0.3, 0.065, 0.065), max_expand_um=1.0, z_glass=2, top_margin_um=0.6)
    assert set(np.unique(cells)) == {0, 3, 7}
    assert (cells[nuc > 0] == nuc[nuc > 0]).all()                   # nuclei inside their own cell
    assert cells[2, 40, 30] == 3 and cells[1, 40, 30] == 0           # from the glass plane ...
    assert cells[9, 40, 30] == 3 and cells[10, 40, 30] == 0          # ... to 2 slices above the nucleus
    assert cells[6, 40, 50] == 3 and cells[6, 40, 70] == 7           # expanded, split between neighbours
    assert cells[6, 40, 60] == 0 and cells[6, 5, 30] == 0            # beyond 1 µm from any nucleus: none
    wide = nucleus_expansion_cells(nuc, (0.3, 0.065, 0.065), max_expand_um=2.0)
    assert wide[6, 40, 58] == 3 and wide[6, 40, 62] == 7            # neighbours meet halfway


# --------------------------------------------------------------------------- Cellpose-SAM (cpsam) path


def test_depth_correct_flattens_decay():
    from smfish.segment import depth_correct

    rng = np.random.default_rng(0)
    vol = np.zeros((12, 64, 64), np.float32)
    vol[:, 20:40, 20:40] = 1000
    vol *= np.linspace(1.0, 0.4, 12)[:, None, None]            # signal decays with depth
    vol += rng.normal(100, 5, vol.shape).astype(np.float32)
    out = depth_correct(vol)
    peaks = out[:, 20:40, 20:40].mean((1, 2))
    assert peaks.max() / peaks.min() < 1.15                       # vs 2.5 before


def test_substrate_planes_flags_only_a_sheet():
    from smfish.segment import substrate_planes

    cona = np.full((6, 80, 80), 100.0, np.float32)
    cona[0] = 900                                                  # coated glass: bright everywhere
    cona[1:, 10:30, 10:30] = 900                                   # cells: small fraction of the field
    cona[3, 50, 50] = 60000                                        # saturating artefact must not matter
    assert substrate_planes(cona) == [0]


def test_cell_layer_z_robust_to_saturated_voxel():
    from smfish.segment import cell_layer_z

    dapi = np.full((20, 128, 128), 100.0, np.float32)
    dapi[6:14, 30:90, 30:90] = 800
    dapi[2, 0, 0] = 65000                                          # plain Otsu would sit near this
    assert cell_layer_z(dapi) == (6, 14)


def test_background_shrink_fallback():
    from smfish.segment import _background_shrink_um

    assert _background_shrink_um({"background_seed_shrink_um": 1.3}, "cell") == 1.3
    assert _background_shrink_um({"background_seed_shrink_um": 1.3, "cell_background_shrink_um": 3.0}, "cell") == 3.0


def _cpsam_cfg(**over):
    sc = dict(cell_method="cpsam", gpu=False, xy_downscale=0.5, nucleus_diameter_um=3.0, cell_diameter_um=6.0,
              min_nucleus_volume_um3=0.5, min_cell_volume_um3=1.0, cell_seed_shrink_um=0.2,
              nucleus_seed_shrink_um=0.2, background_seed_shrink_um=0.4, ridge_sigmas_um=[0.15, 0.3],
              smooth_sigma_um=0.1, column_consistent_cells=False)
    sc.update(over)
    return {"segmentation": sc}


def _fake_cellpose_sam(channels, diameter_px, anisotropy, gpu, cellprob=0.0, **kw):
    """Stand-in for Cellpose-SAM: two cells (left / right halves of the bright area), one nucleus each."""
    shape = channels[0].shape
    out = np.zeros(shape, np.int32)
    zz = slice(0, shape[0] - 1)
    h, w = shape[1], shape[2]
    if len(channels) == 2:   # cells
        out[zz, h // 4: 3 * h // 4, w // 8: w // 2] = 1
        out[zz, h // 4: 3 * h // 4, w // 2: 7 * w // 8] = 2
    else:                    # nuclei
        out[zz, 3 * h // 8: 5 * h // 8, w // 4: 3 * w // 8] = 1
        out[zz, 3 * h // 8: 5 * h // 8, 5 * w // 8: 3 * w // 4] = 2
    return out


def _cpsam_stack():
    """14 planes: z0-1 empty, z2 coated glass (ConA everywhere), z3-11 two cells, z12-13 empty."""
    rng = np.random.default_rng(1)
    shape = (14, 96, 128)
    dapi = rng.normal(100, 3, shape).astype(np.float32)
    cona = rng.normal(100, 3, shape).astype(np.float32)
    dapi[2:12, 36:60, 32:48] = 900
    dapi[2:12, 36:60, 80:96] = 900
    cona[3:12, 24:72, 16:112] = 500
    cona[2] = 900
    return dapi, cona


def test_segment_fov_cpsam_pads_to_full_z_and_pairs_nuclei(monkeypatch):
    from smfish import segment

    monkeypatch.setattr(segment, "run_cellpose_sam", _fake_cellpose_sam)
    monkeypatch.setattr("importlib.metadata.version", lambda pkg: "4.0.0")
    dapi, cona = _cpsam_stack()
    cells, nuclei, info = segment.segment_fov(dapi, cona, (0.3, 0.065, 0.065), _cpsam_cfg())
    assert cells.shape == dapi.shape and nuclei.shape == dapi.shape
    za, zb = info["z_range"]
    assert info["substrate_planes_dropped"] == [2] and (za, zb) == (3, 12)   # glass plane dropped
    assert cells[:za].max() == 0 and cells[zb:].max() == 0          # zero outside the segmented range
    assert info["n_cells"] == 2 and info["n_orphan_nuclei"] == 0 and info["frac_cells_mononucleate"] == 1.0
    assert ((nuclei == 0) | (nuclei == cells)).all()                 # nucleus label == cell label
    assert info["cell_method"] == "cpsam" and not info["column_consistent_cells"]
    for k in ("boundary_ridge_enrichment", "cona_signal_coverage"):
        assert np.isfinite(info[k])

    filled, _, _ = segment.segment_fov(dapi, cona, (0.3, 0.065, 0.065), _cpsam_cfg(fill_dropped_planes=True))
    assert (filled[2] == filled[3]).all() and filled[2].max() > 0

    _, _, info3 = segment.segment_fov(dapi, cona, (0.3, 0.065, 0.065), _cpsam_cfg(cell_layer_top_margin_um=1.5))
    assert info3["z_range"] == [3, 14]                               # +5 planes, capped at the stack top
    assert 0 <= info3["frac_cells_at_top_plane"] <= 1

    dense = cona.copy()
    dense[2:12] = 900                                                # ConA over the whole field in every plane
    _, _, info4 = segment.segment_fov(dapi, dense, (0.3, 0.065, 0.065), _cpsam_cfg())
    assert info4["substrate_planes_dropped"] == [] and info4["z_range"] == [2, 12]


def test_build_manifest_subset(tmp_path, monkeypatch):
    import pandas as pd

    from smfish import manifest

    for w in (1, 2, 3):
        for k in range(5):
            (tmp_path / f"x_Well{w:02d}_Seq{k:04d}.ome.tiff").touch()
    monkeypatch.setattr(manifest, "load_layout", lambda cfg: pd.DataFrame({"well": [1, 2, 3]}))
    cfg = {"paths": {"input_dir": str(tmp_path)}}
    assert len(manifest.build_manifest(cfg)) == 15
    a = manifest.build_manifest(cfg, wells=[1, 2], per_well=2, seed=0, include=["x_Well03_Seq0004"])
    b = manifest.build_manifest(cfg, wells=[1, 2], per_well=2, seed=0, include=["x_Well03_Seq0004"])
    assert a.groupby("well").size().to_dict() == {1: 2, 2: 2, 3: 1}
    assert a["fov_id"].tolist() == b["fov_id"].tolist()            # reproducible
    assert a["index"].tolist() == list(range(len(a)))
    assert manifest.build_manifest(cfg, wells=[1]).shape[0] == 5   # no subset options: unchanged behaviour
