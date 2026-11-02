"""Plate layout -> per-well, per-channel probe roles.

This is the single source of truth for which probe (and therefore which role:
exon / intron / neg_control) is imaged in which channel of which well.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

ROLES = {"exon", "intron", "neg_control"}


def load_probe_dictionary(path: str | Path) -> dict:
    with open(path) as f:
        probes = yaml.safe_load(f)
    for name, info in probes.items():
        if info.get("role") not in ROLES:
            raise ValueError(f"Probe {name!r}: role must be one of {sorted(ROLES)}, got {info.get('role')!r}")
        if not info.get("gene"):
            raise ValueError(f"Probe {name!r} has no gene")
    return probes


def load_layout(cfg: dict) -> pd.DataFrame:
    """Return a long table: one row per (well, FISH channel) with probe metadata.

    Columns: well, condition, channel, probe, gene, role, probe_set, allele, expected_absent, layout_key.
    ``expected_absent`` is True where the cell line lacks the target (``absent_in`` in probes.yaml):
    such wells act as negative controls for that channel.
    ``layout_key`` identifies wells with the same probe/channel design, so
    wells sharing a key but differing in condition are matched pairs.
    """
    raw = pd.read_csv(cfg["paths"]["plate_layout"], encoding="utf-8-sig")  # Excel CSVs start with a BOM
    well_col = cfg.get("layout_well_column", "Well")
    cond_col = cfg.get("layout_condition_column", "Condition")
    raw = raw.dropna(subset=[well_col])
    raw[well_col] = raw[well_col].astype(int)
    probes = load_probe_dictionary(cfg["paths"]["probes"])

    rows = []
    for _, r in raw.iterrows():
        for col, channel in cfg["layout_probe_columns"].items():
            probe = r.get(col)
            if pd.isna(probe) or str(probe).strip() == "":
                continue
            probe = str(probe).strip()
            if probe not in probes:
                raise KeyError(f"Well {r[well_col]}: probe {probe!r} not in probe dictionary {cfg['paths']['probes']}")
            info = probes[probe]
            rows.append(
                dict(
                    well=int(r[well_col]),
                    condition=r.get(cond_col),
                    channel=channel,
                    probe=probe,
                    gene=info["gene"],
                    role=info["role"],
                    probe_set=info.get("probe_set"),
                    allele=info.get("allele"),
                    expected_absent=str(r.get(cond_col)) in set(info.get("absent_in") or []),
                )
            )
    long = pd.DataFrame(rows)
    design = (
        (long["channel"] + ":" + long["probe"])
        .groupby(long["well"])
        .agg(lambda s: " | ".join(sorted(s)))
        .rename("layout_key")
    )
    return long.merge(design, left_on="well", right_index=True)


def well_channels(layout: pd.DataFrame, well: int) -> pd.DataFrame:
    sub = layout[layout["well"] == well]
    if sub.empty:
        raise KeyError(f"Well {well} not in plate layout")
    return sub.reset_index(drop=True)


def channel_pairs(layout: pd.DataFrame, well: int) -> list[tuple[str, str]]:
    """Channel pairs in a well whose probes target the same gene (colocalisation partners)."""
    sub = well_channels(layout, well)
    pairs = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            if sub.loc[i, "gene"] == sub.loc[j, "gene"]:
                pairs.append((sub.loc[i, "channel"], sub.loc[j, "channel"]))
    return pairs


def matched_pairs(layout: pd.DataFrame) -> pd.DataFrame:
    """Groups of wells with identical probe design but different conditions."""
    w = layout.drop_duplicates("well")[["well", "condition", "layout_key"]]
    groups = w.groupby("layout_key").filter(lambda d: d["condition"].nunique() > 1)
    return groups.sort_values(["layout_key", "condition"]).reset_index(drop=True)


def parse_well(name: str, regex: str = r"Well(\d+)") -> int:
    m = re.search(regex, str(name))
    if not m:
        raise ValueError(f"Cannot parse well from {name!r} with {regex!r}")
    return int(m.group(1))
