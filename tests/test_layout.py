import pandas as pd
import pytest

from smfish.layout import channel_pairs, load_layout, matched_pairs, parse_well, well_channels


def test_roles_follow_layout_not_channel_index(cfg):
    lay = load_layout(cfg)
    w10 = well_channels(lay, 10).set_index("channel")
    assert w10.loc["Atto565", "role"] == "intron"
    assert w10.loc["Atto633", "role"] == "exon"
    w16 = well_channels(lay, 16).set_index("channel")
    assert w16.loc["Atto565", "role"] == "exon"
    assert w16.loc["Atto633", "role"] == "intron"
    assert w16.loc["Atto633", "allele"] == "KI"


def test_same_gene_pairs(cfg):
    lay = load_layout(cfg)
    assert channel_pairs(lay, 10) == [("Atto565", "Atto633")]
    assert channel_pairs(lay, 16) == [("Atto565", "Atto633")]  # TetO intron is on POLR2A
    assert channel_pairs(lay, 17) == []  # CTCF intron vs POLR2A exon
    assert channel_pairs(lay, 8) == []  # POLR2A exon vs NeoR


def test_matched_pairs(cfg):
    pairs = matched_pairs(load_layout(cfg))
    groups = pairs.groupby("layout_key")["well"].apply(sorted).tolist()
    assert sorted(groups) == [[2, 3], [4, 5], [8, 9], [10, 11], [14, 15]]


def test_unknown_probe_raises(cfg, tmp_path):
    bad = tmp_path / "layout.csv"
    pd.DataFrame({"Well": [1], "ATTO565 probe": ["Mystery exon"], "ATTO633 probe": ["NeoR"], "Condition": ["x"]}).to_csv(
        bad, index=False
    )
    cfg["paths"]["plate_layout"] = str(bad)
    with pytest.raises(KeyError, match="Mystery exon"):
        load_layout(cfg)


def test_parse_well():
    assert parse_well("20250806_210619_006__Well02_ChannelX.ome.tiff") == 2


def test_targets_absent_in_background_line(cfg):
    lay = load_layout(cfg).set_index(["well", "channel"])
    assert lay.loc[(9, "Atto633"), "expected_absent"]  # NeoR in OsTIR
    assert not lay.loc[(8, "Atto633"), "expected_absent"]  # NeoR in Streaming
    assert lay.loc[(15, "Atto633"), "expected_absent"]  # TetO in OsTIR
    assert not lay.loc[(16, "Atto633"), "expected_absent"]
    assert not lay.loc[(9, "Atto565"), "expected_absent"]  # POLR2A exon present in OsTIR


def test_replicate_plate_has_the_same_design(cfg):
    import copy
    from pathlib import Path

    root = Path(cfg["_config_path"]).parents[2]
    w29 = copy.deepcopy(cfg)
    w29["paths"]["plate_layout"] = str(root / "config" / "250929" / "plate_layout.csv")
    a = load_layout(cfg).sort_values(["well", "channel"]).reset_index(drop=True)
    c = load_layout(w29).sort_values(["well", "channel"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, c)  # 250929 is a replicate of the 250806 plate design


def test_relative_paths_resolve_against_the_repository_root(cfg):
    from pathlib import Path

    root = Path(cfg["_config_path"]).parents[2]
    assert (root / "pyproject.toml").exists()
    assert Path(cfg["paths"]["plate_layout"]) == root / "config" / "250806" / "plate_layout.csv"
