from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from CreditScore.Plotters._base import PlotBase


# --------------------------------------------------------------------------- #
#                        ──  F E A T U R E   P L O T S  ──                    #
# --------------------------------------------------------------------------- #
class FeatureImportancePlotter(PlotBase):
    """Plots horizontal bar charts of feature importances or coefficients."""

    def plot_importance(
        self,
        features: Iterable[str],
        magnitudes: np.ndarray,
        save_dir: Optional[Path | str] = None,
    ) -> Figure:
        """
        Parameters
        ----------
        features
            Display labels for the bars.
        magnitudes
            Importance magnitude (positive numbers, same order as *features*).
        save_dir
            Optional directory in which the figure will be stored.
        """
        features = list(features)
        fig_h = 0.35 * len(features) + 1
        fig, ax = plt.subplots(figsize=(6, fig_h))

        bars = ax.barh(features, magnitudes, color="#1f77b4")
        ax.invert_yaxis()
        self.style_axes(ax, title="Feature importance", xlabel="|β|")
        ax.spines[["top", "right"]].set_visible(False)

        for bar, value in zip(bars, magnitudes):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )

        plt.tight_layout()
        fig = self.label(fig, "feature_importance")

        if save_dir:
            self._save(fig, fig.get_label(), save_dir)
        return fig

    def plot_coefficients(
        self,
        coef_series: pd.Series,
        save_dir: Optional[Path | str] = None,
    ) -> Figure:
        """
        Bar plot of the absolute values of a coefficient vector.

        Parameters
        ----------
        coef_series
            pd.Series where the index contains feature names and the values are
            (signed) coefficients.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(coef_series.index, coef_series.abs().values)
        self.style_axes(ax, title="Absolute value of coefficients", xlabel="Feature", ylabel="|β|")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fig = self.label(fig, "feature_coefficients")
        if save_dir:
            self._save(fig, fig.get_label(), save_dir)
        return fig
