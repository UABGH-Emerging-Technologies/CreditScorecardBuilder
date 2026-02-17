from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from CreditScore.Plotters._base import PlotBase


# --------------------------------------------------------------------------- #
#                       ──  T H R E S H O L D   P L O T ──                    #
# --------------------------------------------------------------------------- #
class ThresholdMetricPlotter(PlotBase):
    """
    Plots Precision, Recall, KS and CAP over all possible thresholds given
    true labels and predicted probabilities.
    """

    def plot(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: pd.Series | np.ndarray,
        optimal_threshold: float,
        save_dir: Optional[Path | str] = None,
    ) -> Figure:
        """
        Parameters
        ----------
        y_true
            Array-like of binary ground truth.
        y_proba
            Predicted positive class probabilities.
        optimal_threshold
            Vertical helper line will be drawn at this value.
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        ks_stats, cap_scores = [], []
        for t in thresholds:
            preds = (y_proba >= t).astype(int)

            tpr = ((preds == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
            fpr = ((preds == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
            ks_stats.append(abs(tpr - fpr))

            sorted_idx = np.argsort(-y_proba)
            sorted_labels = y_true[sorted_idx]
            captured = sorted_labels[: preds.sum()].sum() / y_true.sum()
            cap_scores.append(captured)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precision[:-1], "--", label="Precision")
        ax.plot(thresholds, recall[:-1], "-", label="Recall")
        ax.plot(thresholds, ks_stats, ":", label="KS statistic")
        ax.plot(thresholds, cap_scores, "-.", label="CAP score")
        ax.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal = {optimal_threshold:.4f}",
        )

        self.style_axes(ax, title="Threshold metrics", xlabel="Threshold", ylabel="Score")
        ax.legend()
        fig.tight_layout()

        fig = self.label(fig, "threshold_metrics")
        if save_dir:
            self._save(fig, fig.get_label(), save_dir)
        return fig
