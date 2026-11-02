import numpy as np
import pandas as pd

from smfish.flatfield import apply_spot_flatfield, fit_spot_flatfield

SHAPE = (2304, 2304)


def _vignetted_spots(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y, x = rng.uniform(0, 2303, n), rng.uniform(0, 2303, n)
    u, v = 2 * y / 2303 - 1, 2 * x / 2303 - 1
    true_gain = np.exp(-0.25 * (u**2 + v**2))  # corners ~40% dimmer than centre
    well = rng.choice([2, 3], n)
    level = np.where(well == 2, 2000.0, 6000.0)  # wells differ in brightness
    inten = level * true_gain * rng.lognormal(0, 0.25, n)
    inten[:200] *= 4  # some clusters, must not bias the fit
    return pd.DataFrame({"channel": "Atto565", "well": well, "y_px": y, "x_px": x, "intensity_bgsub": inten}), true_gain


def test_spot_flatfield_recovers_vignetting():
    spots, true_gain = _vignetted_spots()
    models = fit_spot_flatfield(spots, SHAPE)
    m = models["Atto565"]
    assert abs(m["corner_over_centre"] - np.exp(-0.5)) < 0.05
    assert abs(m["edge_over_centre"] - np.exp(-0.25)) < 0.05
    out = apply_spot_flatfield(spots, models, SHAPE)
    r = np.hypot(spots.y_px - 1151.5, spots.x_px - 1151.5)
    for w, level in ((2, 2000.0), (3, 6000.0)):
        sel = (spots.well == w).to_numpy()
        centre = np.median(out.intensity_bgsub[sel & (r < 400)])
        corner = np.median(out.intensity_bgsub[sel & (r > 1300)])
        assert abs(corner / centre - 1) < 0.06, (corner, centre)
    assert "intensity_bgsub_raw" in out
