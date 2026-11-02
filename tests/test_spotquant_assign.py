import numpy as np

from smfish.assign import cell_table, spot_geometry
from smfish.spotquant import measure_spots, nm_to_px, single_rna_reference
from smfish.tss import nearest_partner
from synthetic import VOXEL, add_spot, ellipsoid_mask

WIN = nm_to_px([600, 250, 250], VOXEL)
BG_IN = nm_to_px([900, 400, 400], VOXEL)
BG_OUT = nm_to_px([1500, 650, 650], VOXEL)


def test_local_background_removes_offset_and_scales_linearly():
    rng = np.random.default_rng(1)
    vol = np.full((20, 100, 100), 150.0, np.float32)  # flat background, unlike legacy constant
    vol += np.linspace(0, 80, 100)[None, None, :]  # plus a gradient
    pos = np.array([[10, 30, 30], [10, 30, 70], [10, 70, 50]])
    for p, a in zip(pos, [300, 600, 1200]):
        add_spot(vol, p.astype(float), a)
    vol = rng.normal(vol, 3).astype(np.float32)
    m = measure_spots(vol, pos, WIN, BG_IN, BG_OUT)
    ratio = m["intensity_bgsub"].to_numpy() / m["intensity_bgsub"].iloc[0]
    assert np.allclose(ratio, [1, 2, 4], rtol=0.06)
    empty = measure_spots(vol, np.array([[10, 50, 15]]), WIN, BG_IN, BG_OUT)
    assert abs(empty["intensity_bgsub"].iloc[0]) < 0.1 * m["intensity_bgsub"].iloc[0]


def test_reference_mode_ignores_bright_tail():
    rng = np.random.default_rng(0)
    singles = rng.normal(1000, 150, 800)
    clusters = rng.normal(1000, 150, 200) * rng.integers(2, 6, 200)
    ref = single_rna_reference(np.concatenate([singles, clusters]))
    assert abs(ref["mode"] - 1000) < 80
    assert ref["median"] > ref["mode"]


def test_compartments_and_distances():
    shape = (30, 200, 200)
    centre = (15, 100, 100)
    cells = ellipsoid_mask(shape, centre, (12, 90, 90)).astype(np.int32)
    nuclei = (ellipsoid_mask(shape, centre, (6, 40, 40)) & (cells > 0)).astype(np.int32)
    pts = np.array(
        [
            [15, 100, 100],  # nucleus centre
            [15, 100, 148],  # 8 px = 0.52 µm outside nuclear edge -> perinuclear
            [15, 100, 180],  # far cytoplasm
            [15, 5, 5],  # outside the cell
        ]
    )
    g = spot_geometry(pts, cells, nuclei, VOXEL, perinuclear_um=1.0)
    assert g["compartment"].tolist() == ["nucleus", "perinuclear", "cytoplasm", "extracellular"]
    assert g.loc[0, "dist_nucleus_um"] < -1.5  # deep inside (nuclear z-radius 1.8 µm)
    assert 0.4 < g.loc[1, "dist_nucleus_um"] < 0.65
    assert 0.5 < g.loc[2, "dist_membrane_um"] < 0.8  # 10 px to the membrane in x
    assert g.loc[3, "cell_id"] == 0

    ct = cell_table(cells, nuclei, VOXEL, dapi=nuclei.astype(float))
    assert len(ct) == 1 and not ct.loc[0, "border_xy"] and not ct.loc[0, "border_z"]
    expected_nuc = 4 / 3 * np.pi * (6 * 0.3) * (40 * 0.065) ** 2
    assert abs(ct.loc[0, "nucleus_volume_um3"] / expected_nuc - 1) < 0.05


def test_nearest_partner_in_microns():
    a = np.array([[10, 50, 50], [10, 50, 90]])
    b = np.array([[11, 50, 52]])  # 0.3 µm in z + 0.13 µm in x
    d, i = nearest_partner(a, b, VOXEL)
    assert abs(d[0] - np.hypot(0.3, 0.13)) < 1e-6 and i[0] == 0
    assert d[1] > 2.0


def test_edge_slice_artifact_ignored_by_detection():
    from smfish.detect import detect_channel
    from synthetic import add_spot

    rng = np.random.default_rng(0)
    vol = rng.poisson(np.full((20, 120, 120), 20.0)).astype(np.float32)
    vol[0] = rng.poisson(np.full((120, 120), 60.0))  # noisy first slice, like Atto633 slice 0 on 250806
    pts = [(z, y, x) for z, y, x in zip(rng.integers(4, 16, 25), rng.integers(10, 110, 25), rng.integers(10, 110, 25))]
    for p in pts:
        add_spot(vol, np.array(p, float), 300.0)
    c, score, info = detect_channel(vol, VOXEL, [350, 150, 150], 0.6, exclude_z_edges=1)
    passing = c[score >= info["auto_threshold"]]
    assert (c[:, 0] > 0).all() and (c[:, 0] < 19).all()
    assert abs(len(passing) - len(set(pts))) <= 2


def test_role_specific_storage_floor_and_elbow_tail():
    from smfish.detect import elbow_curve
    from smfish.pipeline import _floor

    floors = {"Atto565": {"exon": 2.5, "intron": 1.0}, "Atto633": 2.0}
    assert _floor(floors, "Atto565", "intron") == 1.0 and _floor(floors, "Atto565", "exon") == 2.5
    assert _floor(floors, "Atto633", "intron") == 2.0 and _floor(floors, "Missing", "exon") is None
    values = np.concatenate([np.random.default_rng(0).uniform(0, 1, 100000), [5.0, 7.0, 9.0]])  # bulk + 3 bright TSSs
    t, k = elbow_curve(values, 100)
    assert t.max() == 9.0 and k[np.searchsorted(t, 5.0)] >= 2  # the bright tail is in the curve
