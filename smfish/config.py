"""Load and validate the experiment YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_SECTIONS = [
    "paths",
    "channels",
    "nuclear_channel",
    "cell_channel",
    "layout_probe_columns",
    "voxel_size_um",
    "illumination",
    "segmentation",
    "detection",
    "quantification",
    "compartments",
    "tss",
    "aggregation",
]

PATH_KEYS = ["plate_layout", "probes", "input_dir", "output_dir", "illumination_correction"]


# Keys replaced wholesale by an inheriting config instead of merged (e.g. a 4-channel experiment
# must not inherit channel 4/5 from a 6-channel base).
REPLACE_KEYS = {"channels", "layout_probe_columns"}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        merge = isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS
        out[k] = _deep_merge(out[k], v) if merge else v
    return out


def _read(path: Path) -> dict:
    """Read a YAML file; ``base: other.yaml`` (relative to this file) is loaded first and overridden."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = _deep_merge(_read((path.parent / base).resolve()), cfg)
    return cfg


def load_config(path: str | Path) -> dict:
    """Read the YAML config, check required sections and resolve relative paths.

    A config may start with ``base: experiment.yaml`` and override only what differs
    (e.g. local paths). Relative paths are resolved against the project root (the nearest
    folder above the YAML holding ``pyproject.toml``), so the same config works from any working dir.
    """
    path = Path(path).resolve()
    cfg = _read(path)

    missing = [k for k in REQUIRED_SECTIONS if k not in cfg]
    if missing:
        raise KeyError(f"Config {path} is missing sections: {missing}")

    root = project_root(path)
    for key in PATH_KEYS:
        value = cfg["paths"].get(key)
        if value is not None:
            value = Path(value).expanduser()
            cfg["paths"][key] = str(value if value.is_absolute() else root / value)

    cfg["channels"] = {int(k): v for k, v in cfg["channels"].items()}
    names = list(cfg["channels"].values())
    for ch in [cfg["nuclear_channel"], cfg["cell_channel"], *cfg["layout_probe_columns"].values()]:
        if ch not in names:
            raise KeyError(f"Channel {ch!r} not in config 'channels' {names}")

    cfg["_config_path"] = str(path)
    return cfg


def project_root(config_path: Path) -> Path:
    """Nearest folder above ``config_path`` holding ``pyproject.toml``; else the config folder's parent."""
    for parent in config_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return config_path.parent.parent


def channel_index(cfg: dict, name: str) -> int:
    for idx, ch in cfg["channels"].items():
        if ch == name:
            return idx
    raise KeyError(name)
