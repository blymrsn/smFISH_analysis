"""Check the environment before any Cellpose-SAM jobs are submitted (``smfish check-env``).

    # login node (has internet): download the ~1.15 GB weights once, into CELLPOSE_LOCAL_MODELS_PATH
    export CELLPOSE_LOCAL_MODELS_PATH=<scratch>/cellpose_models
    python -m smfish.cli check-env --download

    # interactive GPU node: package versions, CUDA, then Cellpose-SAM on a small synthetic 3D stack
    python -m smfish.cli check-env

Returns 0 when every check passes, 1 on the first failure.
"""

from __future__ import annotations

import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

CELLPOSE_VERSION = "4.2.1.1"   # the version the published runs used
WEIGHTS = "cpsam_v2"


class CheckFailed(RuntimeError):
    pass


def versions() -> None:
    for pkg in ("numpy", "scipy", "scikit-image", "tifffile", "big-fish", "torch", "cellpose"):
        try:
            print(f"  {pkg:16s} {version(pkg)}")
        except PackageNotFoundError:
            print(f"  {pkg:16s} -")
    if int(version("cellpose").split(".")[0]) < 4:
        raise CheckFailed("Cellpose >= 4 is needed for Cellpose-SAM")
    if version("cellpose") != CELLPOSE_VERSION:
        print(f"  WARNING: cellpose {version('cellpose')} is not {CELLPOSE_VERSION}, used for the published runs")
    if int(version("numpy").split(".")[0]) >= 2:
        raise CheckFailed("numpy 2 was installed; big-fish needs numpy 1.26")


def model_dir() -> Path:
    from cellpose import models

    return Path(getattr(models, "MODEL_DIR", os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", "~/.cellpose/models")))


def download() -> None:
    from cellpose import models

    models.CellposeModel(gpu=False, pretrained_model=WEIGHTS)
    if not (model_dir() / WEIGHTS).exists():
        raise CheckFailed(f"no {WEIGHTS} weights in {model_dir()} after loading the model")
    print(f"  {model_dir() / WEIGHTS}")


def gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise CheckFailed(f"no CUDA GPU visible to torch {torch.__version__}")
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}, {p.total_memory / 1e9:.1f} GB, torch CUDA {torch.version.cuda}")
    if not (model_dir() / WEIGHTS).exists():
        raise CheckFailed(f"no {WEIGHTS} weights in {model_dir()}: run check-env --download on a node with internet")


def synthetic() -> None:
    """Six blurred ellipsoids in noise at the pipeline's near-isotropic voxels (anisotropy ~1.15)."""
    from scipy import ndimage

    from .segment import run_cellpose_sam

    rng = np.random.default_rng(0)
    z, y, x = np.mgrid[:24, :256, :256]
    vol = np.zeros((24, 256, 256), np.float32)
    for cy, cx in [(60, 60), (60, 190), (128, 128), (190, 60), (190, 190), (128, 40)]:
        vol[((z - 12) / 8) ** 2 + ((y - cy) / 25) ** 2 + ((x - cx) / 25) ** 2 < 1] = 1.0
    vol = ndimage.gaussian_filter(vol, 2) * 1000 + rng.normal(100, 10, vol.shape).astype(np.float32)
    t = time.time()
    n = len(np.unique(run_cellpose_sam([vol], diameter_px=50, anisotropy=1.15, gpu=True))) - 1
    print(f"  {n} objects (6 expected) in {time.time() - t:.1f} s")
    if n == 0:
        raise CheckFailed("Cellpose-SAM found no objects in the synthetic stack")


def check_env(download_weights: bool = False, gpu_test: bool = True) -> int:
    steps = [("versions", versions)]
    steps += [("download weights", download)] if download_weights else [("GPU", gpu)]
    if gpu_test and not download_weights:
        steps.append(("Cellpose-SAM on a synthetic stack", synthetic))
    try:
        for name, step in steps:
            print(name)
            step()
    except CheckFailed as e:
        print(f"FAIL: {e}")
        return 1
    print("OK")
    return 0
