"""Clumps of non-specific staining, and the median-over-fields negative-control calibration."""

import numpy as np
import pandas as pd

from smfish.aggregate import remove_clumps
from smfish.calibrate import negative_control_curve, negative_control_thresholds
from smfish.clumps import DEFAULTS, flag_clumps

KERNEL_OF = {"exon": "exon", "neg_control": "exon", "intron": "intron"}


def _spots(rows):
    s = pd.DataFrame(rows, columns=["fov_id", "z_um", "y_um", "x_um", "log_score", "background"])
    s = s.assign(channel="Atto633", role="exon", probe="NeoR", well=8, cell_id=1, in_nucleus=False,
                 threshold_used=6.0, auto_threshold=4.0, expected_absent=False)
    s["passes"] = s["log_score"] >= s["threshold_used"]
    return s


def _cells(fovs):
    return pd.DataFrame({"fov_id": fovs, "cell_id": 1, "well": 8, "qc_pass": True})


def _field(rng, fov, n_real=200, clump=False, dense_real=False):
    """Real spots (background ~15) spread through a cell; optionally a 6 um aggregate (background ~100,
    candidates 0.75 um apart) or a tight cluster of real spots (normal background)."""
    rows = [(fov, *rng.uniform(0, [5, 40, 40]), rng.uniform(8, 30), rng.normal(15, 3)) for _ in range(n_real)]
    if clump:
        rows += [(fov, 4.0, 10 + 0.75 * i, 10 + 0.4 * j, 20.0, 100.0) for i in range(8) for j in range(2)]
    if dense_real:
        rows += [(fov, 2.0 + 0.3 * (i % 2), 30 + 0.6 * (i // 3), 30 + 0.6 * (i % 3), 15.0, 18.0) for i in range(15)]
    return rows


def test_flags_an_aggregate_but_not_real_spots():
    rng = np.random.default_rng(0)
    s = _spots(_field(rng, "f1", clump=True, dense_real=True))
    ids = flag_clumps(s, _cells(["f1"]), KERNEL_OF, {**DEFAULTS, "enabled": True})
    in_clump = ids >= 0
    aggregate = s["background"] == 100.0
    assert in_clump[aggregate].all()                          # every aggregate candidate
    assert ids[in_clump].nunique() == 1
    assert not in_clump[s["background"] == 18.0].any()        # crowded real spots (normal background)
    # growing takes in real spots touching the aggregate, but none further than two growth steps away
    xyz, agg_xyz = s[["z_um", "y_um", "x_um"]].to_numpy(), s.loc[aggregate, ["z_um", "y_um", "x_um"]].to_numpy()
    far = np.linalg.norm(xyz[:, None] - agg_xyz[None], axis=2).min(1) > 2 * DEFAULTS["link_um"]
    assert not in_clump[far].any()


def test_a_single_bright_background_spot_is_not_a_clump():
    rng = np.random.default_rng(1)
    rows = _field(rng, "f1") + [("f1", 2.0, 20.0, 20.0, 25.0, 150.0)]
    s = _spots(rows)
    assert (flag_clumps(s, _cells(["f1"]), KERNEL_OF, {**DEFAULTS, "enabled": True}) < 0).all()


def test_remove_clumps_stops_clump_members_passing(cfg):
    rng = np.random.default_rng(2)
    s = _spots(_field(rng, "f1", clump=True))
    cfg["aggregation"]["clumps"] = {"enabled": True}
    out = remove_clumps(s, _cells(["f1"]), cfg)
    assert out.loc[out["in_clump"], "passes"].eq(False).all()
    assert out.loc[out["in_clump"], "above_threshold"].all()
    cfg["aggregation"]["clumps"] = {"enabled": False}
    off = remove_clumps(s, _cells(["f1"]), cfg)
    assert not off["in_clump"].any() and off["passes"].equals(s["passes"])


def test_median_calibration_ignores_one_bad_field():
    """Nine control fields with ~0.5 false spots per cell above 5, one with 20 bright ones: the pooled rate
    is pushed over target and its threshold rises; the median over fields is not."""
    rng = np.random.default_rng(3)
    fovs = [f"f{i}" for i in range(10)]
    rows = []
    for f in fovs:
        rows += [(f, 1.0, 1.0, 1.0, x, 15.0) for x in rng.uniform(2, 5.5, 4)]
    rows += [("f9", 1.0, 1.0, 1.0, x, 15.0) for x in rng.uniform(8, 12, 20)]
    s = _spots(rows).assign(probe="mrcB", role="neg_control")
    cells = _cells(fovs)
    thr = {}
    for stat in ("pooled", "median"):
        curve = negative_control_curve(s, cells, KERNEL_OF, statistic=stat)
        thr[stat] = negative_control_thresholds(curve, s, 1.0)["threshold"].iloc[0]
        assert {"fp_per_cell_pooled", "fp_per_cell_median"} <= set(curve.columns)
    assert thr["median"] < 6 < thr["pooled"]


def test_a_dense_well_is_judged_against_itself():
    """A well packed with real spots on raised background (crowding), next to a sparse well: judged against a
    plate-wide reference the dense well would look like one big clump; against its own well it is clean."""
    rng = np.random.default_rng(4)
    sparse = _spots(_field(rng, "f1"))
    grid = np.stack(np.meshgrid(np.arange(0, 3, 0.7), np.arange(0, 12, 0.7), np.arange(0, 12, 0.7)), -1).reshape(-1, 3)
    dense = _spots([("f2", *p, 15.0, rng.normal(80, 8)) for p in grid]).assign(well=1)
    s = pd.concat([sparse, dense], ignore_index=True)
    cells = pd.concat([_cells(["f1"]), _cells(["f2"]).assign(well=1)], ignore_index=True)
    assert (flag_clumps(s, cells, KERNEL_OF, {**DEFAULTS, "enabled": True}) < 0).all()
