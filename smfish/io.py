"""Image loading, voxel-size checks, illumination correction and table I/O."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import channel_index

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- images


def _open(path: str | Path):
    from aicsimageio import AICSImage

    return AICSImage(str(path))


def read_voxel_size_um(img, cfg: dict) -> tuple[float, float, float]:
    """Voxel size (z, y, x) in µm from OME metadata, checked against the config."""
    pps = img.physical_pixel_sizes
    expected = np.asarray(cfg["voxel_size_um"]["expected"], dtype=float)
    measured = np.array([pps.Z, pps.Y, pps.X], dtype=object)
    if any(v is None for v in measured):
        raise ValueError(f"OME metadata lacks physical pixel sizes {tuple(measured)}; refusing to guess")
    measured = measured.astype(float)
    rel = np.abs(measured - expected) / expected
    if np.any(rel > cfg["voxel_size_um"]["rel_tolerance"]):
        raise ValueError(f"Voxel size {tuple(measured)} µm deviates from expected {tuple(expected)} µm")
    return tuple(float(v) for v in measured)


def load_channels(path: str | Path, cfg: dict, names: list[str]) -> tuple[dict[str, np.ndarray], tuple]:
    """Load the requested channels (ZYX, raw dtype) and the voxel size in µm."""
    img = _open(path)
    voxel = read_voxel_size_um(img, cfg)
    if img.dims.C != len(cfg["channels"]):
        raise ValueError(f"{path}: {img.dims.C} channels, config defines {len(cfg['channels'])}")
    out = {}
    for name in names:
        out[name] = np.asarray(img.get_image_data("ZYX", C=channel_index(cfg, name), T=0))
    return out, voxel


# --------------------------------------------------------------------------- illumination


class Illumination:
    """Per-channel illumination correction.

    blimp (current berrygroup/blimp) fits the mean / std images on **log10** intensities and
    ``pixel_z_score`` works in log10 space; set ``illumination.reference_log10: true`` for
    references made that way (250806: mean image ~2 = log10(~100 counts)).

    Methods (``illumination.method`` in the config: one for every channel, or one per channel as a mapping with
an optional ``default``, e.g. ``{DAPI: flatfield, ConA: flatfield, default: none}`` to flat-field the stains
used for segmentation and leave the FISH channels linear and uncorrected):

    - ``flatfield``: ``(x - offset) / F`` with ``F = (M - offset) / mean(M - offset)``, where
      ``M`` is the mean image in linear units (``10**mean`` for log10 references).
      Linear in photon counts, so spot intensities stay comparable. Recommended for quantification.
    - ``zscore``: identical to blimp ``pixel_z_score``: in log10 space for log10 references
      (``10**(mean_mean + mean_std * (log10 x - mean) / std)``). NOTE: this is non-linear per pixel
      (a pixel-dependent power of x), fine for segmentation but not for intensity-based RNA counting.
    - ``blimp``: as ``zscore`` but mean/std images read through ``blimp`` from the pickle.
    - ``none``: ``x - offset``.

    Output is always float32 and offset-subtracted.
    """

    METHODS = ("none", "flatfield", "zscore", "blimp")

    def __init__(self, cfg: dict):
        ic = cfg["illumination"]
        self.method = ic["method"]
        self.offset = {k: float(v) for k, v in ic.get("camera_offset", {}).items()}
        self.cfg = cfg
        self.log10 = bool(ic.get("reference_log10", False))
        self.mean = self.std = None
        used = set(self.method.values()) if isinstance(self.method, dict) else {self.method}
        unknown = used - set(self.METHODS)
        if unknown:
            raise ValueError(f"Unknown illumination method(s) {sorted(unknown)}")
        needs = used & {"flatfield", "zscore", "blimp"}
        if needs:
            if len(needs) > 1:
                raise ValueError(f"One reference-based method per run, got {sorted(needs)}")
            self.mean, self.std = _load_reference_images(cfg["paths"]["illumination_correction"], needs.pop())

    def method_for(self, name: str) -> str:
        """The correction applied to channel ``name``."""
        if isinstance(self.method, dict):
            return self.method.get(name, self.method.get("default", "none"))
        return self.method

    def correct(self, name: str, vol: np.ndarray) -> np.ndarray:
        if name not in self.offset:
            raise KeyError(f"No camera_offset configured for channel {name!r}")
        off = np.float32(self.offset[name])
        method = self.method_for(name)
        if method == "none":
            return vol.astype(np.float32) - off
        c = channel_index(self.cfg, name)
        mean = self.mean[c].astype(np.float32)
        if mean.shape != vol.shape[-2:]:
            raise ValueError(f"Illumination image {mean.shape} does not match image {vol.shape[-2:]}")
        if method == "flatfield":
            flat = (np.power(np.float32(10), mean) if self.log10 else mean) - off
            if np.any(flat <= 0):
                raise ValueError(f"Flat-field for {name} has non-positive pixels; check camera_offset / reference_log10")
            flat = flat / flat.mean()
            if flat.min() < 0.2:  # would amplify some pixels > 5x: reference is offset-dominated, not illumination
                raise ValueError(
                    f"Flat-field for {name} would amplify pixels {1 / flat.min():.0f}x (min normalised flat "
                    f"{flat.min():.3f}); the reference is dominated by camera offset. Use method: none and the "
                    "spot-based flat-field (aggregation.spot_flatfield)."
                )
            return (vol.astype(np.float32) - off) / flat[None]
        std = self.std[c].astype(np.float32)
        x = vol.astype(np.float32)
        if self.log10:
            z = (np.log10(np.maximum(x, 1)) - mean[None]) / std[None]
            return np.power(np.float32(10), np.float32(mean.mean()) + np.float32(std.mean()) * z) - off
        z = (x - mean[None]) / std[None]
        return np.float32(mean.mean()) + np.float32(std.mean()) * z - off

    def describe(self) -> dict:
        return {"method": self.method, "camera_offset": self.offset, "reference_log10": self.log10}


def _load_reference_images(pkl_path: str | Path, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std images (C, Y, X) saved by blimp beside ``IllumCorr.pkl``."""
    pkl_path = Path(pkl_path)
    if method in ("flatfield", "zscore"):
        mean_p = pkl_path.with_name(pkl_path.stem + "_mean_image.ome.tiff")
        std_p = pkl_path.with_name(pkl_path.stem + "_std_image.ome.tiff")
        if mean_p.exists() and std_p.exists():
            mean = np.asarray(_open(mean_p).get_image_data("CYX", Z=0, T=0))
            std = np.asarray(_open(std_p).get_image_data("CYX", Z=0, T=0))
            return mean, std
        log.warning("%s / %s not found; falling back to loading the pickle via blimp", mean_p.name, std_p.name)
    try:
        from blimp.preprocessing.illumination_correction import IlluminationCorrection
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "Illumination reference images not found as TIFFs and blimp is not installed. "
            "Install blimp or set illumination.method: none"
        ) from e
    ic = IlluminationCorrection(from_file=pkl_path)
    if ic.method != "pixel_z_score":
        raise NotImplementedError(f"IllumCorr method {ic.method!r}: only pixel_z_score is supported natively")
    mean = np.asarray(ic.mean_image.get_image_data("CYX", Z=0, T=0))
    std = np.asarray(ic.std_image.get_image_data("CYX", Z=0, T=0))
    return mean, std


# --------------------------------------------------------------------------- tables


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet when pyarrow is available, else gzipped CSV. Returns the path written."""
    path = Path(path)
    try:
        import pyarrow  # noqa: F401

        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except ImportError:
        out = path.with_suffix(".csv.gz")
        df.to_csv(out, index=False)
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a table written by :func:`write_table` (stem given with or without suffix)."""
    path = Path(path)
    stem = path.with_suffix("") if path.suffix in (".parquet", ".gz") else path
    stem = Path(str(stem).removesuffix(".csv"))
    for cand, reader in ((stem.with_suffix(".parquet"), pd.read_parquet), (Path(str(stem) + ".csv.gz"), pd.read_csv)):
        if cand.exists():
            return reader(cand)
    raise FileNotFoundError(f"No table at {stem}.parquet or {stem}.csv.gz")


def save_labels(path: str | Path, cell_labels: np.ndarray, nuc_labels: np.ndarray, voxel_um) -> None:
    """Save labels as a 2-channel ZCYX ImageJ-hyperstack TIFF (opens in napari / Fiji)."""
    import tifffile

    arr = np.stack([cell_labels, nuc_labels], axis=1).astype(np.int32)  # Z, C, Y, X
    tifffile.imwrite(
        str(path),
        arr,
        compression="zlib",
        imagej=False,
        metadata={"axes": "ZCYX", "channels": ["cells", "nuclei"], "voxel_size_um_zyx": list(voxel_um)},
    )


def load_labels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (cells, nuclei) label volumes from a file written by :func:`save_labels` or the legacy pipeline."""
    import tifffile

    arr = np.squeeze(tifffile.imread(str(path)))
    if arr.ndim != 4:
        raise ValueError(f"{path}: expected 4D labels after squeeze, got {arr.shape}")
    # new files: Z, C, Y, X ; legacy blimp files: C, Z, Y, X
    if arr.shape[1] == 2 and arr.shape[0] != 2:
        arr = np.moveaxis(arr, 1, 0)
    return arr[0].astype(np.int32), arr[1].astype(np.int32)
