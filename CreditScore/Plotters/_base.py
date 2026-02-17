# CreditScore/plotters/_base.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class PlotBase:
    """
    Shared mix-in for all plotters.
    """

    _DPI: int = 300
    _EXT: str = ".png"

    # ------------------------------------------------------------------ #
    # static helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def label(fig: plt.Figure, name: str) -> plt.Figure:
        """Attach *name* to *fig* and return the same figure (fluent-style)."""
        fig.set_label(name)
        return fig

    @staticmethod
    def style_axes(ax: plt.Axes, title=None, xlabel=None, ylabel=None) -> None:
        """Apply consistent styling for plot titles and labels."""
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

    # ------------------------------------------------------------------ #
    # saving                                                             #
    # ------------------------------------------------------------------ #
    def _save(self, fig: plt.Figure, name: str, directory: Path | str) -> None:
        """
        Save *fig* into *directory* using *name* + extension.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}{self._EXT}", dpi=self._DPI, bbox_inches="tight")

    def _ensure_dir(self, path: Path | str) -> Path:
        """Create *path* if it does not yet exist and return it as Path."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
