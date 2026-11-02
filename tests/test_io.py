import numpy as np
import pytest
import tifffile

from smfish.io import Illumination, load_labels, save_labels

pytest.importorskip("aicsimageio")


def _blimp_pixel_z_score(original, mean_image, std_image, mean_mean, mean_std, log_transform=True):
    """Formula of current blimp pixel_z_score (float path, log10 by default)."""
    original = original.astype(float)
    if log_transform:
        original = np.log10(original)
    z = (original - mean_image) / std_image
    corrected = mean_mean + (mean_std * z)
    return 10**corrected if log_transform else corrected


@pytest.fixture
def illum_cfg(cfg, tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:64, :64]
    vignette = 1.0 - 0.4 * (((yy - 32) / 32) ** 2 + ((xx - 32) / 32) ** 2) / 2  # 0.6 .. 1.0
    n_ch = len(cfg["channels"])
    mean = np.stack([100 + 400 * vignette for _ in range(n_ch)]).astype(np.float32)
    std = np.stack([5 + 40 * vignette + rng.uniform(0, 1, (64, 64)) for _ in range(n_ch)]).astype(np.float32)
    pkl = tmp_path / "IllumCorr.pkl"
    tifffile.imwrite(tmp_path / "IllumCorr_mean_image.ome.tiff", mean, metadata={"axes": "CYX"}, ome=True)
    tifffile.imwrite(tmp_path / "IllumCorr_std_image.ome.tiff", std, metadata={"axes": "CYX"}, ome=True)
    cfg["paths"]["illumination_correction"] = str(pkl)
    return cfg, mean, std, vignette


def test_native_zscore_matches_blimp_formula(illum_cfg):
    cfg, mean, std, _ = illum_cfg
    cfg["illumination"]["method"] = "zscore"
    cfg["illumination"]["reference_log10"] = False
    ill = Illumination(cfg)
    raw = np.random.default_rng(1).integers(90, 2000, (3, 64, 64)).astype(np.uint16)
    c = 4  # Atto565
    expected = _blimp_pixel_z_score(raw, mean[c], std[c], mean[c].mean(), std[c].mean(), log_transform=False) - 100
    assert np.allclose(ill.correct("Atto565", raw), expected, atol=1e-2)


def test_native_zscore_matches_blimp_log10(illum_cfg):
    cfg, mean, std, _ = illum_cfg
    lmean, lstd = np.log10(mean), std / mean / np.log(10)  # log10-space references, as blimp fits them
    cfg["illumination"]["method"] = "zscore"
    cfg["illumination"]["reference_log10"] = True
    ill = Illumination(cfg)
    ill.mean, ill.std = lmean.astype(np.float32), lstd.astype(np.float32)
    raw = np.random.default_rng(2).integers(90, 2000, (3, 64, 64)).astype(np.uint16)
    c = 4
    expected = _blimp_pixel_z_score(raw, lmean[c], lstd[c], lmean[c].mean(), lstd[c].mean()) - 100
    assert np.allclose(ill.correct("Atto565", raw), expected, rtol=1e-3)


def test_flatfield_from_log10_reference(illum_cfg):
    cfg, mean, _, vignette = illum_cfg
    cfg["illumination"]["method"] = "flatfield"
    cfg["illumination"]["reference_log10"] = True
    ill = Illumination(cfg)
    ill.mean = np.log10(mean).astype(np.float32)
    raw = (100 + 500.0 * vignette)[None].astype(np.float32)
    out = ill.correct("Atto565", raw)
    assert out.std() / out.mean() < 1e-3


def test_flatfield_removes_vignette(illum_cfg):
    cfg, _, _, vignette = illum_cfg
    cfg["illumination"]["method"] = "flatfield"
    cfg["illumination"]["reference_log10"] = False
    ill = Illumination(cfg)
    true_signal = 500.0
    raw = (100 + true_signal * vignette)[None].astype(np.float32)  # offset + vignetted uniform signal
    out = ill.correct("Atto565", raw)
    assert out.std() / out.mean() < 1e-3  # flat after correction


def test_per_channel_method_flatfields_stains_only(illum_cfg):
    cfg, _, _, vignette = illum_cfg
    cfg["illumination"]["method"] = {"DAPI": "flatfield", "ConA": "flatfield", "default": "none"}
    cfg["illumination"]["reference_log10"] = False
    ill = Illumination(cfg)
    raw = (100 + 500.0 * vignette)[None].astype(np.float32)
    assert ill.method_for("DAPI") == "flatfield" and ill.method_for("Atto565") == "none"
    assert ill.correct("DAPI", raw).std() / ill.correct("DAPI", raw).mean() < 1e-3   # flat
    assert np.allclose(ill.correct("Atto565", raw), raw - 100)                       # FISH: offset only


def test_labels_round_trip_new_and_legacy_layout(tmp_path):
    cells = np.zeros((5, 20, 20), np.int32)
    cells[1:4, 2:10, 2:10] = 3
    nuc = (cells > 0).astype(np.int32) * 3
    nuc[:, :5] = 0
    save_labels(tmp_path / "labels.tif", cells, nuc, (0.3, 0.065, 0.065))
    c2, n2 = load_labels(tmp_path / "labels.tif")
    assert (c2 == cells).all() and (n2 == nuc).all()
    # legacy blimp files: T, C, Z, Y, X
    tifffile.imwrite(tmp_path / "legacy.tif", np.stack([cells, nuc])[None])
    c3, n3 = load_labels(tmp_path / "legacy.tif")
    assert (c3 == cells).all() and (n3 == nuc).all()
