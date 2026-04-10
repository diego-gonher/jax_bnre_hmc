"""Default matplotlib style: SciencePlots-like look without a LaTeX installation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.style

_STYLE_FILE = Path(__file__).resolve().parent / "styles" / "science_nompl.mplstyle"
_applied: bool = False


def apply_plot_style() -> None:
    """Apply the bundled style sheet (idempotent; safe to call many times).

    Uses ``science_nompl.mplstyle`` (SciencePlots-inspired axes/ticks/colors with
    ``text.usetex: False`` and mathtext fonts).
    """
    global _applied
    if _applied:
        return
    if not _STYLE_FILE.is_file():
        raise FileNotFoundError(f"Missing matplotlib style file: {_STYLE_FILE}")
    matplotlib.style.use(str(_STYLE_FILE))
    _applied = True
