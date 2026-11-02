"""Readouts recounted at scaled thresholds, on a synthetic aggregated/ folder."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("pyarrow")

from smfish import style as S  # noqa: E402
from smfish.robustness import fig_threshold_robustness, readouts_vs_threshold  # noqa: E402


def _aggregated(tmp_path):
    """Two wells of one probe: well 8 has twice the spots of well 9 at every score. Each cell has a fixed set of
    candidate scores, so the count above a threshold is known exactly. One clump candidate must never count."""
    rng = np.random.default_rng(0)
    cells, spots = [], []
    for well, per_cell in ((8, 20), (9, 10)):
        for cid in range(1, 31):
            vol = rng.uniform(1000, 3000)
            cells.append({"fov_id": f"f{well}", "cell_id": cid, "well": well, "qc_pass": True, "cell_volume_um3": vol})
            for score in np.linspace(5.0, 14.5, per_cell):
                spots.append({"fov_id": f"f{well}", "cell_id": cid, "well": well, "channel": "Atto565",
                              "probe": "POLR2A odds exon", "log_score": score, "in_clump": False})
    spots.append({**spots[0], "log_score": 50.0, "in_clump": True})
    agg = tmp_path / "aggregated"
    (agg / "calibration").mkdir(parents=True)
    pd.DataFrame(cells).to_parquet(agg / "cells.parquet")
    pd.DataFrame(spots).to_parquet(agg / "spots.parquet")
    pd.DataFrame([{"probe": "POLR2A odds exon", "channel": "Atto565", "threshold_median": 10.0}]).to_csv(
        agg / "calibration" / "thresholds_applied.csv", index=False)
    return agg


def test_readouts_vs_threshold(tmp_path):
    agg = _aggregated(tmp_path)
    groups = {"knock-in": ("POLR2A odds exon", "Atto565", [8]), "parental": ("POLR2A odds exon", "Atto565", [9])}
    t = readouts_vs_threshold(agg, groups, factors=[0.5, 1.0]).set_index(["group", "factor"])
    assert t.loc[("knock-in", 0.5), "spots_per_cell"] == 20      # every candidate >= 5
    assert t.loc[("parental", 0.5), "spots_per_cell"] == 10
    expected_ki = (np.linspace(5.0, 14.5, 20) >= 10.0).sum()     # the clump candidate (50) is never counted
    assert t.loc[("knock-in", 1.0), "spots_per_cell"] == expected_ki
    assert t.loc[("knock-in", 1.0), "threshold"] == 10.0
    assert (t["n_cells"] == 30).all()
    with S.use("publication"):
        fig = fig_threshold_robustness(t.reset_index(), readouts=["knock-in"], controls=["parental"],
                                       ratios={"knock-in / parental": ("knock-in", "parental")})
        assert len(fig.axes) == 3
