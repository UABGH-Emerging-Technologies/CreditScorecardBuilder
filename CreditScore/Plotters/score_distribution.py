# CreditScore/Plotters/shap.py
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib.figure import Figure

from config.config import logger
from CreditScore.Plotters._base import PlotBase


class ScoreDistributionPlotter(PlotBase):
    """Plot score distributions for event vs non-event groups."""

    def plot_score_distribution(
        self, X: pd.DataFrame, y: pd.Series, save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot score distribution for events vs non-events.

        Args:
            X: Feature data
            y: Labels
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        # Get scores
        scores = self.model.score(X)

        # Create mask for events
        mask_event = y == 1

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms
        ax.hist(
            scores[~mask_event],
            bins=30,
            label="Non-event",
            color="blue",
            alpha=0.6,
            density=True,
            edgecolor="black",
        )
        ax.hist(
            scores[mask_event],
            bins=30,
            label="Event",
            color="red",
            alpha=0.6,
            density=True,
            edgecolor="black",
        )

        # Formatting
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution - Events vs Non-Events")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.set_label("Score_Distributions")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Score distribution saved to {save_path}")

        return fig
