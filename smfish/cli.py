"""Command line interface.

    python -m smfish.cli check-env [--download]
    python -m smfish.cli manifest  -c config/250806/experiment.yaml [--wells 10,11,16] [--per-well 3 --seed 0]
                                   [--include <fov_id> [--include ...]]
    python -m smfish.cli run       -c config/250806/experiment.yaml --index $PBS_ARRAY_INDEX
    python -m smfish.cli check     -c config/250806/experiment.yaml
    python -m smfish.cli aggregate -c config/250806/experiment.yaml

In normal use the pbs/ scripts call these (see README.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="smfish")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check-env", "manifest", "run", "check", "aggregate"):
        s = sub.add_parser(name)
        s.add_argument("-v", "--verbose", action="store_true")
        if name == "check-env":
            s.add_argument("--download", action="store_true", help="fetch the Cellpose-SAM weights (needs internet)")
            s.add_argument("--no-gpu-test", action="store_true", help="skip the small Cellpose-SAM run")
            continue
        s.add_argument("-c", "--config", required=True)
        if name in ("manifest", "run", "check"):
            s.add_argument("--manifest", default=None, help="manifest CSV (default: <output_dir>/manifest.csv)")
        if name == "manifest":
            s.add_argument("--wells", default=None, help="comma-separated well numbers (default: all in layout)")
            s.add_argument("--per-well", type=int, default=None, help="random subset of N FOVs per well (trials)")
            s.add_argument("--seed", type=int, default=0, help="random seed for --per-well")
            # repeatable, not comma-separated: 250806 FOV ids contain commas (ChannelXXX,405,633,...)
            s.add_argument("--include", action="append", default=None, help="FOV id always included (repeatable)")
        if name == "run":
            s.add_argument("--index", type=int, help="row of manifest.csv (PBS array index)")
            s.add_argument("--fov", help="path to a single OME-TIFF instead of --index")
            s.add_argument("--overwrite", action="store_true")
        if name == "check":
            s.add_argument("--write-pending", default=None, help="write manifest indices not yet done, one per line")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if a.cmd == "check-env":
        from .check_env import check_env

        return check_env(download_weights=a.download, gpu_test=not a.no_gpu_test)
    cfg = load_config(a.config)

    from .manifest import build_manifest, check_manifest, manifest_path, summarise_status

    if a.cmd == "manifest":
        wells = [int(w) for w in a.wells.split(",")] if a.wells else None
        m = build_manifest(cfg, wells, per_well=a.per_well, seed=a.seed, include=a.include)
        path = Path(a.manifest) if a.manifest else manifest_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(path, index=False)
        hint = f"MANIFEST={path} " if a.manifest else ""
        print(f"{len(m)} FOVs -> {path}  (submit with: {hint}bash pbs/1_process_fovs.sh <config> all)")
        print(m.groupby("well").size().rename("n_fov").to_string())
        return 0

    if a.cmd == "run":
        from .pipeline import run_fov

        if a.fov:
            path = a.fov
        elif a.index is not None:
            m = pd.read_csv(a.manifest or manifest_path(cfg))
            path = m.loc[m["index"] == a.index, "path"].item()
        else:
            p.error("run needs --index or --fov")
        print(run_fov(path, cfg, overwrite=a.overwrite))
        return 0

    if a.cmd == "check":
        path = Path(a.manifest) if a.manifest else manifest_path(cfg)
        if not path.exists():
            print(f"No manifest at {path}: step 1 has not run yet (bash pbs/1_process_fovs.sh {a.config})")
            return 1
        status = check_manifest(cfg, pd.read_csv(path))
        print(summarise_status(status).to_string())
        bad = status[status["status"] != "done"]
        if a.write_pending:
            bad["index"].to_csv(a.write_pending, index=False, header=False)
        if len(bad):
            print("\nNot done (resubmit with: bash pbs/1_process_fovs.sh <config> pending):")
            print(bad[["index", "fov_id", "status", "error"]].to_string(index=False))
            return 1
        print("\nAll FOVs done.")
        return 0

    if a.cmd == "aggregate":
        from .aggregate import run_aggregation

        print(run_aggregation(cfg))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
