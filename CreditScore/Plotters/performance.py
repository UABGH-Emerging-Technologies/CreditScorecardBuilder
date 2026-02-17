# CreditScore/Plotters/performance.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from CreditScore.evaluate import ModelEvaluator, PrcMetrics, RocMetrics
from CreditScore.Plotters._base import PlotBase


class PerformancePlotter(PlotBase):
    """
    Create ROC, PRC, CAP and KS plots given *metrics* or a *ModelEvaluator*.

    This class has ZERO knowledge of OptBinning, Streamlit, etc.
    """

    # ----------------------------------------------------------------— #
    # construction helpers                                               #
    # ----------------------------------------------------------------— #
    @staticmethod
    def _ensure_1d(arr):
        """Normalize a 2D predict_proba output to a 1D probability array."""
        return arr[:, 1] if arr.ndim == 2 else arr

    # ----------------------------------------------------------------— #
    # public API                                                         #
    # ----------------------------------------------------------------— #
    def from_evaluator(
        self,
        evaluator: ModelEvaluator,
        *,
        X_train,
        y_train,
        X_test,
        y_test,
        save_dir: Optional[Path | str] = None,
        use_baseline_rate: bool = False,
    ) -> Dict[str, Figure]:
        """
        Convenience wrapper: obtain probabilities from *evaluator.model* and
        immediately draw all four performance plots.
        """
        p_tr = self._ensure_1d(evaluator.model.predict_proba(X_train))
        p_te = self._ensure_1d(evaluator.model.predict_proba(X_test))

        roc_tr = evaluator.roc_auc(y_train, p_tr)
        roc_te = evaluator.roc_auc(y_test, p_te)
        prc_tr = evaluator.prc_auc(y_train, p_tr)
        prc_te = evaluator.prc_auc(y_test, p_te)
        event_rate = float(y_test.mean())

        figs = {
            "roc": self._plot_roc(roc_tr, roc_te),
            "prc": self._plot_prc(prc_tr, prc_te, event_rate, use_baseline_rate),
            "cap": self._plot_cap(y_test, p_te, event_rate, use_baseline_rate),
            "ks": self._plot_ks(y_test, p_te),
        }

        if save_dir:
            for f in figs.values():
                self._save(f, f.get_label(), save_dir)

        return figs

    # ----------------------------------------------------------------— #
    # individual plot creators                                           #
    # ----------------------------------------------------------------— #
    def _plot_roc(self, train: RocMetrics, test: RocMetrics) -> Figure:
        """Plot ROC curves for train/test metrics."""
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(train.fpr, train.tpr, color="steelblue", label=f"Train  AUC = {train.auc:.3f}")
        ax.plot(test.fpr, test.tpr, color="crimson", label=f"Test   AUC = {test.auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.4)

        self.style_axes(ax, "ROC – train vs test", "False positive rate", "True positive rate")
        ax.legend(loc="lower right")
        fig = self.label(fig, "roc_curve")
        return fig

    def _plot_prc(
        self, train: PrcMetrics, test: PrcMetrics, event_rate=float, use_baseline_rate: bool = False
    ) -> Figure:
        """Plot precision-recall curves for train/test metrics."""
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(
            train.recall,
            train.precision,
            color="steelblue",
            label=f"Train AUC-PR = {train.auc:.2f}",
        )
        ax.plot(
            test.recall, test.precision, color="crimson", label=f"Test  AUC-PR = {test.auc:.2f}"
        )
        if use_baseline_rate:
            # horizontal line at event rate
            ax.hlines(
                event_rate,
                xmin=0,
                xmax=1,
                colors="purple",
                linestyles=":",
                label=f"Baseline (rate={event_rate:.2f})",
            )

        self.style_axes(ax, "Precision-Recall – train vs test", "Recall", "Precision")
        ax.legend()
        fig = self.label(fig, "pr_curve")
        return fig

    def _plot_cap(self, y_true, y_proba, event_rate=float, use_baseline_rate=False) -> Figure:
        """Plot cumulative accuracy profile (CAP) curve."""
        # identical logic as your previous function but shortened
        order = np.argsort(-y_proba)
        cum_pos = np.cumsum(y_true.iloc[order])
        pct_pos = cum_pos / y_true.sum()

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(np.linspace(0, 1, len(pct_pos)), pct_pos, label="CAP", color="steelblue")
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
        if use_baseline_rate:
            # constant classifier (event_rate) captured over population
            ax.hlines(
                event_rate,
                xmin=0,
                xmax=1,
                colors="purple",
                linestyles=":",
                label=f"Baseline (rate={event_rate:.2f})",
            )

        self.style_axes(ax, "CAP curve", "Fraction of population", "Fraction of positives captured")
        ax.legend()
        fig = self.label(fig, "cap_curve")
        return fig

    def _plot_ks(self, y_true, y_proba) -> Figure:
        """Plot the Kolmogorov-Smirnov (KS) curve."""
        pos = y_proba[y_true == 1]
        neg = y_proba[y_true == 0]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(np.sort(pos), np.linspace(0, 1, len(pos)), label="Positive", color="steelblue")
        ax.plot(np.sort(neg), np.linspace(0, 1, len(neg)), label="Negative", color="crimson")
        self.style_axes(ax, "KS curve", "Predicted probability", "Cumulative distribution")
        ax.legend()
        fig = self.label(fig, "ks_curve")
        return fig
