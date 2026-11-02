"""Figure style: palette, roles, rcParams and sizing.

Colour-blind-safe Okabe-Ito palette; serif type at 7 pt, sized for a 150 mm text width.
Call sites use roles (``NUCLEUS``, ``CYTOPLASM``), not colours. Importing this module does not touch
``rcParams``; draw inside ``with style.use("publication"):`` instead. It must not import pyplot.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib
import numpy as np

#: Wong 2011, Nature Methods 8:441.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
NEUTRAL = {"grey": "#9E9E9E", "grey_faint": "#D9D9D9", "white": "#FFFFFF"}

BLACK, BLUE, VERMILLION, ORANGE, SKY, GREEN, YELLOW, PURPLE = (
    OKABE_ITO[k] for k in ("black", "blue", "vermillion", "orange", "sky", "green", "yellow", "purple"))
GREY, GREY_FAINT, WHITE = NEUTRAL["grey"], NEUTRAL["grey_faint"], NEUTRAL["white"]


def tint(colour, fraction):
    """Mix ``fraction`` of ``colour`` with white, as an OPAQUE hex (figures are saved transparent)."""
    rgb = np.asarray(matplotlib.colors.to_rgb(colour), dtype=float)
    return matplotlib.colors.to_hex(np.clip(1.0 - fraction * (1.0 - rgb), 0, 1)).upper()


# ---- roles
#: Compartments, as in the size-scaling figures: cytoplasm blue, nucleus red, whole cell black.
CYTOPLASM = BLUE
NUCLEUS = VERMILLION
CELL = BLACK
COMPARTMENTS = {"cell": CELL, "cytoplasm": CYTOPLASM, "nucleus": NUCLEUS}
#: Poisson reference (Fano = 1) and other reference lines.
REFERENCE_LINE = {"color": GREY, "linestyle": "--", "linewidth": 0.8, "zorder": 0}
#: Per-replicate (well) lines under the pooled line.
REPLICATE_ALPHA = 0.45
#: Image overlays: cell outline, nucleus outline, spots.
OUTLINE_CELL = YELLOW
OUTLINE_NUCLEUS = SKY
SPOT = VERMILLION
IMAGE_CMAP = "gray"

# ---- sizing
MM = 1 / 25.4
TEXT_WIDTH_MM = 150.0
MAX_HEIGHT_MM = 220.0

# ---- rcParams
_SERIF_STACK = ["STIXGeneral", "DejaVu Serif"]
BASE_RC = {
    "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILLION, YELLOW, BLACK]),
    "axes.unicode_minus": True,
    "axes.formatter.use_mathtext": True,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
}
PUBLICATION_RC = {
    **BASE_RC,
    "font.family": _SERIF_STACK,
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "figure.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": WHITE,
    "legend.edgecolor": "none",
    "legend.fancybox": False,
    "legend.borderpad": 0.3,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
    "grid.linestyle": ":",
    "axes.titlepad": 4.0,
    "figure.dpi": 100,
}
STYLES = {"publication": PUBLICATION_RC, "none": {}}


@contextmanager
def use(name: str = "publication"):
    """Draw inside a style. Wrap figure *creation*: rcParams are read as artists are made."""
    if name not in STYLES:
        raise ValueError(f"Unknown style {name!r}. Known: {sorted(STYLES)}")
    with matplotlib.rc_context(STYLES[name]):
        yield
