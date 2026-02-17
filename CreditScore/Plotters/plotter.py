"""Core plotting helpers for model evaluation and interpretation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve


# --------------------------------------------------------------------------- #
#                               ──  B A S E  ──                               #
# --------------------------------------------------------------------------- #
class PlotBase:
    """
    Mixin that provides:
        • consistent labelling
        • basic styling helpers
        • pathlib-based saving
    Every subclass should only worry about the *data* it wants to visualise.
    """

    _FIG_DPI: int = 300  # default export quality
    _FIG_EXT: str = ".png"  # default export format

    # ---------- public helpers ------------------------------------------------
    @staticmethod
    def label(fig: plt.Figure, name: str) -> plt.Figure:
        """
        Assign a deterministic label to *fig* and return the same figure so
        callers can keep using it in one expression.
        """
        fig.set_label(name)
        return fig

    @staticmethod
    def style_axes(
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:
        """Apply a minimal, common styling to *ax*."""
        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.3)

    # ---------- protected helpers --------------------------------------------
    def _ensure_dir(self, path: Path | str) -> Path:
        """Create *path* if it does not yet exist and return it as Path."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _save(
        self,
        fig: plt.Figure,
        name: str,
        directory: Path | str,
        dpi: int | None = None,
    ) -> None:
        """
        Save *fig* under *directory/name + self._FIG_EXT* with *dpi*.
        The directory is created if necessary.
        """
        d = self._ensure_dir(directory)
        dpi = dpi or self._FIG_DPI
        fig.savefig(d / f"{name}{self._FIG_EXT}", dpi=dpi, bbox_inches="tight")


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
    ) -> plt.Figure:
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
    ) -> plt.Figure:
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


# --------------------------------------------------------------------------- #
#                  ──  P E R F O R M A N C E   P L O T S  ──                  #
# --------------------------------------------------------------------------- #
class ModelPerformancePlotter(PlotBase):
    """
    Convenience wrapper around a *ModelEvaluator* implementation that already
    provides individual .plot_X() routines (ROC, PR, …).
    """

    def __init__(self, evaluator) -> None:
        """
        Parameters
        ----------
        evaluator
            An object exposing the four methods:
                plot_roc_curve(...)
                plot_pr_curve(...)
                plot_cumulative_accuracy_profile(...)
                plot_ks_statistic(...)
            Each method must accept *show=False* and return a matplotlib.figure.Figure.
        """
        self.evaluator = evaluator

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def plot_all(
        self,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        save_dir: Optional[Path | str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Produce ROC, PR, CAP and KS plots.

        Returns
        -------
        dict
            Mapping of a short, stable key → figure.
        """
        figs: Dict[str, plt.Figure] = {}

        figs["roc"] = self.label(
            self.evaluator.plot_roc_curve(y_train, y_test, y_train_pred, y_test_pred, show=False),
            "roc_curve",
        )
        figs["pr"] = self.label(
            self.evaluator.plot_pr_curve(y_train, y_test, y_train_pred, y_test_pred, show=False),
            "pr_curve",
        )
        figs["cap"] = self.label(
            self.evaluator.plot_cumulative_accuracy_profile(
                y_train, y_test, y_train_pred, y_test_pred, show=False
            ),
            "cap_curve",
        )
        figs["ks"] = self.label(
            self.evaluator.plot_ks_statistic(y_test, y_test_pred, show=False),
            "ks_curve",
        )

        if save_dir:
            for f in figs.values():
                self._save(f, f.get_label(), save_dir)
        return figs


# --------------------------------------------------------------------------- #
#                           ──  S H A P   P L O T S  ──                       #
# --------------------------------------------------------------------------- #
class SHAPPlotter(PlotBase):
    """
    Generates SHAP decision plots for a (binning→logistic) score-card type
    pipeline.  All Streamlit-specific display logic has been removed – this
    class is framework agnostic and returns plain matplotlib figures.
    """

    def __init__(self, scorecard_model, logistic_wrapper) -> None:
        """
        Parameters
        ----------
        scorecard_model
            Object that provides two attributes
                • .binning_process
                • .scorecard.estimator_
        logistic_wrapper
            Callable that *wraps* estimator_ (e.g. from scorecardpy)
            into something SHAP understands.
        """
        self.sc_model = scorecard_model
        self.wrapper = logistic_wrapper

    # ------------------------------------------------------------------ #
    # private helpers                                                    #
    # ------------------------------------------------------------------ #
    def _explainer(self, X_train_binned: pd.DataFrame) -> shap.Explainer | None:
        """
        Build a shap.LinearExplainer around the wrapped estimator.
        Returns None and warns on failure.
        """
        try:
            wrapped = self.wrapper(self.sc_model.scorecard.estimator_)
            return shap.LinearExplainer(
                wrapped,
                X_train_binned,
                feature_perturbation="interventional",
                model_output="log_odds",
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Unable to create SHAP explainer: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def decision_plots(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        thresholds: Iterable[float] = (0.1, 0.2),
        save_dir: Optional[Path | str] = None,
        use_baseline_rate: bool = False,
    ) -> Dict[str, plt.Figure]:
        """
        Produce SHAP decision plots for each threshold.

        Returns
        -------
        dict
            key = e.g. "shap_decision_10_all"
            value = matplotlib.figure.Figure
        """
        save_path: Optional[Path] = Path(save_dir) if save_dir else None
        if save_path:
            self._ensure_dir(save_path)

        # Binning transform
        X_train_bin = pd.DataFrame(
            self.sc_model.binning_process.transform(X_train),
            columns=X_train.columns,
        )
        X_test_bin = pd.DataFrame(
            self.sc_model.binning_process.transform(X_test),
            columns=X_test.columns,
        )

        # Fit estimator if needed
        estimator = self.sc_model.scorecard.estimator_
        if not hasattr(estimator, "coef_"):
            estimator.fit(X_train_bin, y_train)

        explainer = self._explainer(X_train_bin)
        if explainer is None:
            return {}

        expected = explainer.expected_value
        y_pred_test = estimator.predict_proba(X_test_bin)[:, 1]

        figures: Dict[str, plt.Figure] = {}

        for thr in thresholds:
            mask = y_pred_test >= thr
            if mask.sum() == 0:
                continue

            X_sel = X_test_bin[mask]
            shap_vals = explainer.shap_values(X_sel)

            # ---- full feature set -------------------------------------------------
            fig_all, ax_all = plt.subplots(figsize=(10, 7))
            shap.decision_plot(
                # expected,
                shap_vals,
                X_sel,
                feature_order="hclust",
                show=False,
                ignore_warnings=True,
            )
            if use_baseline_rate:
                ax_all.axvline(
                    expected, color="purple", linestyle=":", label=f"Baseline (E={expected:.2f})"
                )
                ax_all.legend()
            self.style_axes(ax_all, title=f"SHAP decision – P̂ ≥ {thr:.0%} (all features)")
            tag = f"shap_decision_{int(thr * 100)}_all"
            figures[tag] = self.label(fig_all, tag)
            if save_path:
                self._save(fig_all, tag, save_path)

            # ---- top-10 only -------------------------------------------------------
            fig_top, ax_top = plt.subplots(figsize=(10, 7))
            shap.decision_plot(
                expected,
                shap_vals,
                X_sel,
                feature_display_range=slice(None, -11, -1),
                link="logit",
                show=False,
                ignore_warnings=True,
            )
            if use_baseline_rate:
                ax_all.axvline(
                    expected, color="purple", linestyle=":", label=f"Baseline (E={expected:.2f})"
                )
                ax_all.legend()
            self.style_axes(ax_top, title=f"SHAP decision – P̂ ≥ {thr:.0%} (top-10)")
            tag_top = f"shap_decision_{int(thr * 100)}_top10"
            figures[tag_top] = self.label(fig_top, tag_top)
            if save_path:
                self._save(fig_top, tag_top, save_path)

        return figures
