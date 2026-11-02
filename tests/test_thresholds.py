"""One threshold per probe and channel: lookup order, probe_median, and the proposal sources."""

import numpy as np
import pandas as pd

from smfish.aggregate import apply_thresholds, probe_median_auto, threshold_setting
from smfish.calibrate import fp_target, kernel_of_role, propose_thresholds


def _spots():
    """Two FOVs of one probe whose per-FOV elbow differs (as between a dim and a bright well)."""
    rows = []
    for fov, well, auto in (("f1", 4, 1.5), ("f2", 5, 3.5), ("f3", 6, 2.5)):
        for score in (1.0, 2.0, 3.0, 4.0):
            rows.append({"fov_id": fov, "well": well, "probe": "POLR2A intron", "channel": "Atto565",
                         "role": "intron", "log_score": score, "auto_threshold": auto})
    for score in (2.0, 5.0):
        rows.append({"fov_id": "f1", "well": 4, "probe": "mrcB", "channel": "Atto565", "role": "neg_control",
                     "log_score": score, "auto_threshold": 4.0})
        rows.append({"fov_id": "f2", "well": 5, "probe": "mrcB", "channel": "Atto633", "role": "neg_control",
                     "log_score": score, "auto_threshold": 3.0})
    return pd.DataFrame(rows)


def test_lookup_order():
    th = {"default": "auto", "Atto565": 9.0, "mrcB": 7.0, "mrcB@Atto633": 6.0}
    assert threshold_setting(th, "mrcB", "Atto633") == (6.0, "mrcB@Atto633")
    assert threshold_setting(th, "mrcB", "Atto565") == (7.0, "mrcB")
    assert threshold_setting(th, "POLR2A intron", "Atto565") == (9.0, "Atto565")
    assert threshold_setting(th, "POLR2A intron", "Atto633") == ("auto", "default")


def test_probe_median_is_one_value_across_wells():
    s = apply_thresholds(_spots(), {"aggregation": {"thresholds": {"default": "probe_median"}}})
    intron = s[s["probe"] == "POLR2A intron"]
    assert (intron["threshold_used"] == 2.5).all()  # median of 1.5, 3.5, 2.5 - not each FOV's own
    # mrcB is imaged in two channels: one value per channel, not pooled across them
    assert set(s.loc[s["probe"] == "mrcB", "threshold_used"]) == {4.0, 3.0}
    assert s["passes"].equals(s["log_score"] >= s["threshold_used"])


def test_auto_and_fixed_still_work():
    cfg = {"aggregation": {"thresholds": {"default": "auto", "mrcB@Atto633": 4.5}}}
    s = apply_thresholds(_spots(), cfg)
    intron = s[s["probe"] == "POLR2A intron"]
    assert intron["threshold_used"].equals(intron["auto_threshold"])
    assert (s.loc[(s["probe"] == "mrcB") & (s["channel"] == "Atto633"), "threshold_used"] == 4.5).all()


def test_fp_target_per_kernel():
    assert fp_target(1.0, "intron") == 1.0
    assert fp_target({"exon": 1.0, "intron": 0.1}, "intron") == 0.1
    assert fp_target({"exon": 1.0, "intron": 0.1}, "exon") == 1.0


def test_proposal_sources(cfg):
    kernel_of = kernel_of_role(cfg)
    assert kernel_of["neg_control"] == "exon" and kernel_of["intron"] == "intron"
    spots = _spots()
    nct = pd.DataFrame([{"channel": "Atto565", "kernel": "exon", "probe": "mrcB", "threshold": 4.2, "wells": "1"}])
    curve = pd.DataFrame({"channel": "Atto565", "kernel": "exon", "probe": "mrcB",
                          "threshold": [0.0, 4.2, 8.0], "fp_per_cell": [10.0, 1.0, 0.0]})
    det = pd.DataFrame({"fov_id": ["f1", "f2", "f3"], "probe": "POLR2A intron", "channel": "Atto565",
                        "detection_threshold": [1.0, 1.0, 1.0]})
    p = propose_thresholds(spots, curve, nct, det, kernel_of).set_index(["probe", "channel"])
    # the 565 intron probe has no intron-kernel negative control: median auto, flagged as such
    assert p.loc[("POLR2A intron", "Atto565"), "proposed"] == 2.5
    assert p.loc[("POLR2A intron", "Atto565"), "source"].startswith("median per-FOV auto")
    # mrcB calibrates itself; in 633 it has no curve here, so it falls back to its median
    assert p.loc[("mrcB", "Atto565"), "proposed"] == 4.2
    assert np.isclose(p.loc[("mrcB", "Atto565"), "fp_per_cell_at_proposed"], 1.0)
    assert p.loc[("mrcB", "Atto633"), "source"].startswith("median per-FOV auto")


def test_proposal_raised_to_stored_floor(cfg):
    spots = _spots()
    det = pd.DataFrame({"fov_id": ["f1", "f2", "f3"], "probe": "POLR2A intron", "channel": "Atto565",
                        "detection_threshold": [1.0, 3.0, 1.0]})
    p = propose_thresholds(spots, pd.DataFrame(), pd.DataFrame(columns=["channel", "kernel", "probe", "threshold",
                                                                          "wells"]),
                           det, kernel_of_role(cfg)).set_index(["probe", "channel"])
    assert p.loc[("POLR2A intron", "Atto565"), "proposed"] == 3.0
    assert p.loc[("POLR2A intron", "Atto565"), "source"].endswith("raised to the stored floor")
