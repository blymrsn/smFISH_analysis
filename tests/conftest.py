import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from smfish.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config" / "250806" / "experiment.yaml")
    c["paths"]["output_dir"] = str(tmp_path / "out")
    c["aggregation"]["min_reference_spots"] = 20
    c["aggregation"]["min_cell_volume_um3"] = 0  # synthetic cells are smaller than real ones
    c["aggregation"]["min_nucleus_volume_um3"] = 0
    return c
