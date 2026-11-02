"""FOV manifest (what should be processed) and completeness check (what was)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .layout import load_layout, parse_well
from .pipeline import fov_dir, fov_id_from_path


def build_manifest(
    cfg: dict,
    wells: list[int] | None = None,
    per_well: int | None = None,
    seed: int = 0,
    include: list[str] | None = None,
) -> pd.DataFrame:
    """One row per input OME-TIFF in a well that has at least one FISH probe in the layout.

    ``per_well``: keep a reproducible random subset of N FOVs per well (``seed``), e.g. for a trial.
    ``include``: FOV ids always kept, whatever ``wells`` / ``per_well`` say (e.g. a reference FOV).
    """
    layout = load_layout(cfg)
    files = sorted(Path(cfg["paths"]["input_dir"]).glob(cfg.get("file_glob", "*.ome.tiff")))
    rows = []
    for f in files:
        well = parse_well(f.name, cfg.get("well_regex", r"Well(\d+)"))
        if well not in set(layout["well"]):
            continue
        rows.append({"fov_id": fov_id_from_path(f), "well": well, "path": str(f)})
    m = pd.DataFrame(rows, columns=["fov_id", "well", "path"])
    keep = m["well"].isin(wells) if wells is not None else pd.Series(True, index=m.index)
    if per_well is not None:
        picked = [i for _, g in m[keep].groupby("well")
                  for i in g.sample(n=min(per_well, len(g)), random_state=seed).index]
        keep = m.index.isin(picked)
    if include:
        missing = set(include) - set(m["fov_id"])
        if missing:
            raise ValueError(f"--include FOVs not found in {cfg['paths']['input_dir']}: {sorted(missing)}")
        keep = keep | m["fov_id"].isin(include)
    m = m[keep].reset_index(drop=True)
    m.insert(0, "index", range(len(m)))
    return m


def manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["output_dir"]) / "manifest.csv"


def check_manifest(cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    """Status per FOV: done / failed / missing (+ error message for failures)."""
    rows = []
    for _, r in manifest.iterrows():
        d = fov_dir(cfg, r["fov_id"])
        if (d / "DONE.json").exists():
            status, err = "done", ""
        elif (d / "FAILED.json").exists():
            status, err = "failed", json.loads((d / "FAILED.json").read_text()).get("error", "")
        else:
            status, err = "missing", ""
        rows.append({**r.to_dict(), "status": status, "error": err})
    return pd.DataFrame(rows)


def summarise_status(status: pd.DataFrame) -> pd.DataFrame:
    return status.pivot_table(index="well", columns="status", values="fov_id", aggfunc="count", fill_value=0)
