"""Size-scaling helpers and figures, on synthetic per-cell tables."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smfish import style as S  # noqa: E402
from smfish.analysis import (bootstrap_moments, fano_window_check, noise_by_bins, per_cell_counts,  # noqa: E402
                             rolling_moments, scaling_slope)
from smfish.figures import fig_compartment_own_volume, fig_compartment_scaling, fig_total_scaling  # noqa: E402


def _tables(n_per_fov=80, n_fov=6, slope=1.0, dispersion=None, seed=0):
    """cells + cell_counts_long for two cell lines: counts ~ 20 x volume(pL)^slope, Poisson or NB."""
    rng = np.random.default_rng(seed)
    cells, long = [], []
    for cond, wells in (("POLR2A-Streaming", (2, 8)), ("OsTIRF47G", (3, 9))):
        for f in range(n_fov):
            well = wells[f % 2]
            fov = f"{cond}_{f}"
            vol = rng.uniform(1.0, 4.0, n_per_fov)  # pL
            mean = 20 * vol ** slope
            if dispersion is None:
                total = rng.poisson(mean)
            else:  # negative binomial with variance mean + mean^2 / dispersion
                total = rng.negative_binomial(dispersion, dispersion / (dispersion + mean))
            nuc = rng.binomial(total, 0.2)
            for i in range(n_per_fov):
                cells.append({"fov_id": fov, "well": well, "condition": cond, "cell_id": i + 1, "qc_pass": True,
                              "cell_volume_um3": vol[i] * 1000, "nucleus_volume_um3": vol[i] * 300,
                              "cytoplasm_volume_um3": vol[i] * 700, "nucleus_dapi_sum": 1.0})
                for comp, k in (("nucleus", nuc[i]), ("perinuclear", 0), ("cytoplasm", total[i] - nuc[i])):
                    long.append({"fov_id": fov, "well": well, "condition": cond, "cell_id": i + 1,
                                 "channel": "Atto565", "probe": "POLR2A odds exon", "gene": "POLR2A", "role": "exon",
                                 "compartment": comp, "n_spots": k, "n_rna": float(k)})
    return pd.DataFrame(cells), pd.DataFrame(long)


def test_per_cell_counts_keeps_zeros_and_compartments():
    cells, long = _tables()
    long.loc[long["cell_id"] == 1, ["n_spots", "n_rna"]] = 0
    c = per_cell_counts(long, cells, "POLR2A odds exon", compartments=("cell", "nucleus", "cytoplasm"))
    assert set(c["compartment"]) == {"cell", "nucleus", "cytoplasm"}
    whole = c[c["compartment"] == "cell"]
    assert len(whole) == len(cells)  # non-expressing cells kept
    assert (whole.loc[whole["cell_id"] == 1, "n_spots"] == 0).all()
    assert np.isclose(whole["cell_volume_pL"], whole["cell_volume_um3"] / 1000).all()


def test_rolling_fano_poisson_vs_overdispersed():
    cells, long = _tables(n_fov=10, seed=1)
    c = per_cell_counts(long, cells, "POLR2A odds exon")
    # narrow window: the size trend inside a window is small, so Poisson counts give Fano ~1
    r = rolling_moments(c, "cell_volume_pL", "n_spots", window=0.03, min_cells=40)
    assert 0.8 < r["fano"].median() < 1.4
    cells, long = _tables(n_fov=10, dispersion=2.0, seed=1)
    r = rolling_moments(per_cell_counts(long, cells, "POLR2A odds exon"), "cell_volume_pL", "n_spots",
                        window=0.03, min_cells=40)
    assert r["fano"].median() > 5


def test_excess_cv2_poisson_vs_overdispersed():
    # Poisson: no CV^2 above counting noise; negative binomial (variance mean + mean^2/k): excess CV^2 = 1/k,
    # whatever the mean - while its Fano factor (1 + mean/k) grows with the mean
    cells, long = _tables(n_fov=10, seed=5)
    r = rolling_moments(per_cell_counts(long, cells, "POLR2A odds exon"), "cell_volume_pL", "n_spots",
                        window=0.03, min_cells=40)
    assert abs(r["excess_cv2"].median()) < 0.02
    assert np.allclose(r["cv2"] - r["excess_cv2"], 1 / r["mean"])
    cells, long = _tables(n_fov=10, dispersion=4.0, seed=5)
    r = rolling_moments(per_cell_counts(long, cells, "POLR2A odds exon"), "cell_volume_pL", "n_spots",
                        window=0.03, min_cells=40)
    assert 0.18 < r["excess_cv2"].median() < 0.32


def test_bootstrap_band_brackets_estimate():
    cells, long = _tables(n_fov=10, dispersion=4.0, seed=6)
    c = per_cell_counts(long, cells, "POLR2A odds exon")
    r = bootstrap_moments(c, "cell_volume_pL", "n_spots", window=0.1, n_boot=50)
    ok = r["excess_cv2_lo"].notna()
    assert ok.mean() > 0.8
    inside = (r.loc[ok, "excess_cv2_lo"] <= r.loc[ok, "excess_cv2"]) & (r.loc[ok, "excess_cv2"] <= r.loc[ok, "excess_cv2_hi"])
    assert inside.mean() > 0.9


def _constant_fano_tables(fano=3.0, n_cells=3000, seed=7):
    """counts = fano x Poisson(mean/fano): Fano factor exactly ``fano`` at every mean, mean ~ 20 x volume."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(1.0, 4.0, n_cells)
    n = fano * rng.poisson(20 * vol / fano)
    return pd.DataFrame({"fov_id": rng.integers(0, 20, n_cells), "well": 1, "cell_volume_pL": vol, "n_spots": n})


def test_noise_by_bins_constant_fano_reference():
    # constant Fano: the excess CV^2 falls as 1/mean and sits on the constant-Fano line in every bin
    d = _constant_fano_tables()
    t = noise_by_bins(d, "cell_volume_pL", "n_spots", [1, 2, 3, 4], n_boot=30)
    assert np.allclose(t["excess_cv2"], t["excess_cv2_const_fano"], rtol=0.25)
    assert t["excess_cv2"].iloc[0] > 2 * t["excess_cv2"].iloc[2]
    assert (t["excess_cv2_lo"] <= t["excess_cv2"]).all() and (t["excess_cv2"] <= t["excess_cv2_hi"]).all()
    # negative binomial (fixed k): excess CV^2 = 1/k at every mean, so above the constant-Fano line at high mean
    cells, long = _tables(n_fov=10, dispersion=4.0, seed=8)
    c = per_cell_counts(long, cells, "POLR2A odds exon")
    t = noise_by_bins(c, "cell_volume_pL", "n_spots", [1, 2, 3, 4], n_boot=0)
    assert t["excess_cv2"].iloc[2] > 1.3 * t["excess_cv2_const_fano"].iloc[2]


def test_fano_window_check_grows_with_window():
    cells, long = _tables(n_fov=10, seed=2)
    chk = fano_window_check(per_cell_counts(long, cells, "POLR2A odds exon"), "cell_volume_pL", "n_spots",
                            windows=(0.03, 0.5))
    # a wide window mixes small and large cells: the size spread inflates the Fano factor
    assert chk["median_fano"].iloc[1] > chk["median_fano"].iloc[0]


def test_scaling_slope_recovers_exponent():
    cells, long = _tables(slope=0.7, seed=3)
    c = per_cell_counts(long, cells, "POLR2A odds exon")
    s = scaling_slope(c, "cell_volume_pL", "n_spots", ["condition"], n_boot=200).set_index("condition")
    for cond in ("POLR2A-Streaming", "OsTIRF47G"):
        assert abs(s.loc[cond, "slope"] - 0.7) < 0.08
        assert s.loc[cond, "ci_low"] < s.loc[cond, "slope"] < s.loc[cond, "ci_high"]


def test_figures_build_at_text_width():
    cells, long = _tables()
    c = per_cell_counts(long, cells, "POLR2A odds exon", compartments=("cell", "nucleus", "cytoplasm"))
    slopes = scaling_slope(c, "cell_volume_pL", "n_spots", ["condition", "compartment"], n_boot=50)
    with S.use("publication"):
        # two columns (cell lines) x (counts, Fano) - or counts only without the Fano row; the compartment
        # figure adds an excess-CV^2 row; the appendix control is (nucleus, cytoplasm) x cell lines
        own = c[c["compartment"] != "cell"]
        # appendix control: (nucleus, its noise, cytoplasm, its noise, volumes) x cell lines
        cases = [(fig_total_scaling(c, slopes), 4), (fig_total_scaling(c, slopes, show_fano=False), 2),
                 (fig_compartment_scaling(c, slopes, n_boot=20), 6),
                 (fig_compartment_scaling(c, slopes, show_fano=False), 2),
                 (fig_compartment_own_volume(own, n_boot=20), 10)]
        for fig, n_axes in cases:
            fig.canvas.draw()
            width_mm = fig.get_tightbbox(fig.canvas.get_renderer()).width / S.MM
            assert width_mm <= S.TEXT_WIDTH_MM + 1
            assert len(fig.axes) == n_axes


def test_within_well_fano_ignores_between_well_offsets():
    cells, long = _tables(n_fov=10, seed=4)
    c = per_cell_counts(long, cells, "POLR2A odds exon")
    c.loc[c["well"].isin([2, 3]), "n_spots"] *= 2  # one well per line twice as bright: not cell noise
    pooled = rolling_moments(c, "cell_volume_pL", "n_spots", window=0.03, min_cells=40)["fano"].median()
    within = rolling_moments(c, "cell_volume_pL", "n_spots", window=0.03, min_cells=40, within="well")["fano"].median()
    assert within < pooled
    # doubling a Poisson count doubles its Fano, so within-well stays ~1.5, while pooling adds the offset
    assert within < 2.5 < pooled


def test_pick_example_window_prefers_typical_cells():
    cells, long = _tables(n_per_fov=40, n_fov=4, seed=5)
    rng = np.random.default_rng(5)
    cells["centroid_y_um"] = rng.uniform(5, 145, len(cells))
    cells["centroid_x_um"] = rng.uniform(5, 145, len(cells))
    counts = per_cell_counts(long, cells, "POLR2A odds exon")
    from smfish.figures import pick_example_window

    w = pick_example_window(cells, counts, wells=[2, 8], window_um=45, image_px=2304)
    assert w["fov_id"].startswith("POLR2A-Streaming")
    assert w["n_typical"] >= 1 and w["size"] == round(45 / 0.065)
    assert 0 <= w["y0"] <= 2304 - w["size"] and 0 <= w["x0"] <= 2304 - w["size"]
