"""Shared set-up for the notebooks: one plate's config and folders, and the cell-line labels for figures."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .figures import use_conditions

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Plate:
    name: str
    cfg: dict
    out: Path        # outputs/<plate>: the pipeline's output_dir
    agg: Path        # outputs/<plate>/aggregated: step 2
    figures: Path    # outputs/<plate>/figures/<notebook>
    results: Path    # results/<plate>: the small tables kept in the repository

    def rel(self, path) -> str:
        """``path`` relative to the repository root, for printing."""
        return os.path.relpath(path, ROOT)


def setup(plate: str, notebook: str) -> Plate:
    """Load ``config/<plate>/experiment.yaml`` and check that steps 1-2 have produced ``aggregated/``."""
    cfg = load_config(ROOT / "config" / plate / "experiment.yaml")
    out = Path(cfg["paths"]["output_dir"])
    agg = out / "aggregated"
    if not (agg / "cells.parquet").exists():
        raise FileNotFoundError(f"{os.path.relpath(agg, ROOT)} not found: run steps 1-2 (pbs/) for plate {plate}, "
                                "or copy / link its output folder to outputs/<plate>")
    use_conditions(cfg.get("conditions", {}))
    results = ROOT / "results" / plate
    results.mkdir(parents=True, exist_ok=True)
    return Plate(plate, cfg, out, agg, out / "figures" / notebook, results)
