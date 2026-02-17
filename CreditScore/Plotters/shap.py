# CreditScore/Plotters/shap.py
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib.figure import Figure

from CreditScore.Plotters._base import PlotBase


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
    ) -> Dict[str, Figure]:
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

        figures: Dict[str, Figure] = {}

        for thr in thresholds:
            mask = y_pred_test >= thr
            if mask.sum() == 0:
                continue

            X_sel = X_test_bin[mask]
            shap_vals = explainer.shap_values(X_sel)

            # ---- full feature set -------------------------------------------------
            fig_all, ax_all = plt.subplots(figsize=(10, 7))
            shap.decision_plot(
                expected,
                shap_vals,
                X_sel,
                feature_order="hclust",
                show=False,
                ignore_warnings=True,
            )
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
            self.style_axes(ax_top, title=f"SHAP decision – P̂ ≥ {thr:.0%} (top-10)")
            tag_top = f"shap_decision_{int(thr * 100)}_top10"
            figures[tag_top] = self.label(fig_top, tag_top)
            if save_path:
                self._save(fig_top, tag_top, save_path)

        return figures
