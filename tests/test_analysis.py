"""Replicate-aware statistics helpers."""

import numpy as np
import pandas as pd


def test_bootstrap_over_fovs_on_synthetic_counts():
    from smfish.analysis import bootstrap_over_fovs

    rng = np.random.default_rng(0)
    rows = []
    for cond, rate in (("A", 20.0), ("B", 40.0)):
        for f in range(6):
            fov_effect = rng.normal(1, 0.1)
            vol = rng.uniform(800, 1600, 60)
            counts = rng.poisson(rate * fov_effect * vol / 1000)
            rows.append(pd.DataFrame({"condition": cond, "fov_id": f"{cond}{f}", "cell_volume_um3": vol, "n_spots": counts}))
    df = pd.concat(rows, ignore_index=True)
    boot = bootstrap_over_fovs(df, "n_spots", ["condition"], n_boot=300).set_index("condition")
    assert boot.loc["A", "ci_low"] < boot.loc["A", "estimate"] < boot.loc["A", "ci_high"]
    assert boot.loc["A", "ci_high"] < boot.loc["B", "ci_low"]
    assert (boot["n_fov"] == 6).all()
