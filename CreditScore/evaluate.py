"""Evaluation metrics, monitoring, and power analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from optbinning.scorecard import ScorecardMonitoring
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)

from config.config import logger


# ───────────────────────────────────────── dataclasses ───────────────────────
@dataclass(frozen=True)
class RocMetrics:
    """Container for ROC curve arrays and AUC."""

    fpr: np.ndarray
    tpr: np.ndarray
    auc: float


@dataclass(frozen=True)
class PrcMetrics:
    """Container for precision-recall arrays and AUC."""

    precision: np.ndarray
    recall: np.ndarray
    auc: float


# ──────────────────────────────────── evaluator base ─────────────────────────
class ModelEvaluator:
    """
    Thin layer that ONLY computes metrics – no plotting takes place here.
    """

    def __init__(self, model) -> None:
        """
        Parameters
        ----------
        model
            A fitted classifier exposing .predict_proba() or .score().
            (Type is not enforced to stay framework agnostic.)
        """
        self.model = model

    # --------------------------------------------------------------------- #
    # ROC / AUC                                                             #
    # --------------------------------------------------------------------- #
    @staticmethod
    def roc_auc(
        y_true: pd.Series | np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> RocMetrics:
        """
        Compute ROC curve and AUC.

        Parameters
        ----------
        y_true          : ground-truth labels (binary)
        y_pred_proba    : positive-class probabilities (N or Nx2)

        Returns
        -------
        RocMetrics
        """
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return RocMetrics(fpr=fpr, tpr=tpr, auc=auc(fpr, tpr))

    # --------------------------------------------------------------------- #
    # Precision-Recall / AUC-PR                                             #
    # --------------------------------------------------------------------- #
    @staticmethod
    def prc_auc(
        y_true: pd.Series | np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> PrcMetrics:
        """
        Compute Precision-Recall curve and AUC-PR.
        """
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return PrcMetrics(
            precision=precision,
            recall=recall,
            auc=auc(recall, precision),
        )

    # --------------------------------------------------------------------- #
    # Convenience wrappers that call the model                              #
    # --------------------------------------------------------------------- #
    def predict_proba(self, X) -> np.ndarray:
        """
        Helper so UI code can obtain probabilities in a single line even if
        the wrapped model returns a shape (N,) instead of (N,2).
        """
        proba = self.model.predict_proba(X)
        return proba if proba.ndim == 2 else np.column_stack([1 - proba, proba])

    def compute_precision_recall(
        self, y_true: pd.Series, y_pred_proba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute precision, recall and AUC-PR.

        Returns
        -------
        precision : ndarray
        recall    : ndarray
        auc_pr    : float
        """
        y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return precision, recall, auc(recall, precision)


class ScorecardEvaluator(ModelEvaluator):
    """Evaluator specific to scorecard models."""

    def __init__(self, model):
        super().__init__(model)
        self.monitoring = None
        self._monitoring_fitted = False

    def initialize_monitoring(
        self,
        psi_method: str = "quantile",
        psi_n_bins: int = 5,
        psi_min_bin_size: float = 0.05,
        show_digits: int = 2,
    ):
        """
        Initialize scorecard monitoring.

        Args:
            psi_method: Method for PSI calculation
            psi_n_bins: Number of bins for PSI
            psi_min_bin_size: Minimum bin size
            show_digits: Number of digits to show
        """
        # Check if model is OptbinningScorecardModel
        if hasattr(self.model, "scorecard"):
            # Our wrapper class
            scorecard = self.model.scorecard
        elif hasattr(self.model, "table"):
            # Direct scorecard object
            scorecard = self.model
        else:
            raise ValueError("Model must be a scorecard model for monitoring")

        self.monitoring = ScorecardMonitoring(
            scorecard,
            psi_method=psi_method,
            psi_n_bins=psi_n_bins,
            psi_min_bin_size=psi_min_bin_size,
            show_digits=show_digits,
            verbose=False,
        )

    def fit_monitoring(
        self, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series
    ):
        """
        Fit the monitoring system.

        Args:
            X_test: Test features
            y_test: Test labels
            X_train: Training features
            y_train: Training labels
        """
        if self.monitoring is None:
            self.initialize_monitoring()

        # Fit monitoring with error handling
        try:
            self.monitoring.fit(X_test, y_test, X_train, y_train)
            self._monitoring_fitted = True
        except ValueError as e:
            # Handle statistical test failures (e.g., chi-square with zero frequencies)
            import warnings

            warnings.warn(f"Monitoring fit failed: {str(e)}. PSI monitoring will not be available.")
            self._monitoring_fitted = False
            self.monitoring = None
        except Exception as e:
            # Handle other unexpected errors
            import warnings

            warnings.warn(
                f"Unexpected error in monitoring fit: {str(e)}. PSI monitoring will not be available."
            )
            self._monitoring_fitted = False
            self.monitoring = None

    def get_psi_report(self) -> pd.DataFrame:
        """Get PSI report table."""
        if self.monitoring is None or not getattr(self, "_monitoring_fitted", False):
            # Return empty DataFrame with expected structure
            return pd.DataFrame({"Variable": [], "PSI": [], "Test": []})

        return self.monitoring.tests_table()

    def plot_psi(self) -> plt.Figure:
        """Plot PSI values."""
        if self.monitoring is None or not getattr(self, "_monitoring_fitted", False):
            # Create a placeholder figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "PSI monitoring not available\n(insufficient data for statistical tests)",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
            ax.set_title("Population Stability Index (PSI)")
            ax.axis("off")
            return fig

        try:
            fig = self.monitoring.psi_plot()
            if isinstance(fig, plt.Figure):
                return fig
            else:
                fig = plt.gcf()
                fig.set_figheight(3)
                return fig

        except Exception as e:
            # If plotting fails, return a placeholder
            import warnings

            warnings.warn(f"PSI plot failed: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "PSI plot generation failed",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
            ax.set_title("Population Stability Index (PSI)")
            ax.axis("off")
            return fig


class StatsmodelsEvaluator(ModelEvaluator):
    """Evaluator specific to statsmodels logistic regression."""

    def __init__(self, model):
        super().__init__(model)

    def power_analysis_wald(
        self, coeff_name: str, alpha: float = 0.05, power: float = 0.80, tails: str = "Two"
    ) -> int:
        """
        Perform Wald-based power analysis.

        Args:
            coeff_name: Name of coefficient to test
            alpha: Significance level
            power: Desired power
            tails: "One" or "Two" tailed test

        Returns:
            Required sample size
        """
        result = self.model.result

        # Get coefficient value
        beta_j = result.params[coeff_name]

        # Design matrix and fitted probabilities
        X = result.model.exog
        p = result.predict()
        W = p * (1 - p)  # n-vector of weights

        # Average Fisher information per subject
        Ibar = (X.T * W) @ X / len(X)
        vper = np.linalg.inv(Ibar)[
            result.model.exog_names.index(coeff_name), result.model.exog_names.index(coeff_name)
        ]

        # Critical values
        z_alpha = norm.isf(alpha if tails == "One" else alpha / 2)
        z_beta = norm.isf(1 - power)

        # Required sample size
        n_req = (z_alpha + z_beta) ** 2 * vper / beta_j**2

        return int(np.ceil(n_req))

    def power_analysis_simulation(
        self,
        coeff_name: str,
        sample_sizes: List[int],
        n_simulations: int = 1000,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Perform simulation-based power analysis.

        Args:
            coeff_name: Name of coefficient to test
            sample_sizes: List of sample sizes to test
            n_simulations: Number of simulations per size
            alpha: Significance level

        Returns:
            DataFrame with sample sizes and power estimates
        """
        result = self.model.result

        # Get model parameters
        beta_hat = result.params.to_numpy(dtype=float)
        X_full = result.model.exog.astype(float)
        mask = ~np.isnan(X_full).any(axis=1)
        X_full = X_full[mask, :]

        # Column index of parameter to test
        param_idx = list(result.params.index).index(coeff_name)

        def simulate_once(n):
            """Run one simulation for sample size n."""
            # Draw with replacement
            draw_idx = np.random.choice(len(X_full), n, replace=True)
            Xs = X_full[draw_idx, :]

            # Generate outcomes
            linpred = Xs @ beta_hat
            p = expit(linpred)
            y_star = np.random.binomial(1, p)

            try:
                # Fit model with increased max iterations
                rs = sm.Logit(y_star, Xs).fit(disp=False, maxiter=10000)
                # Check if significant
                return int(rs.pvalues[param_idx] < alpha)
            except Exception:
                # Model failed to converge
                return 0

        # Run simulations for each sample size
        power_results = []
        for n in sample_sizes:
            successes = sum(simulate_once(n) for _ in range(n_simulations))
            power = successes / n_simulations
            power_results.append({"n": n, "power": power})
            logger.info(f"n={n}: power={power:.3f}")

        return pd.DataFrame(power_results)

    def plot_power_curve(
        self, power_df: pd.DataFrame, target_power: float = 0.8, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot power curve from simulation results.

        Args:
            power_df: DataFrame with 'n' and 'power' columns
            target_power: Target power level to highlight
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot power curve
        ax.plot(power_df["n"], power_df["power"], "b-o", markersize=6)

        # Add target power line
        ax.axhline(
            y=target_power, color="red", linestyle="--", label=f"Target Power = {target_power}"
        )

        # Find where we hit target power
        hit = power_df[power_df["power"] >= target_power]
        if not hit.empty:
            n_required = hit.iloc[0]["n"]
            ax.axvline(x=n_required, color="green", linestyle=":", label=f"n = {n_required}")

        # Formatting
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Statistical Power")
        ax.set_title("Power Analysis - Sample Size vs Power")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.set_label("Power_Curve")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Power curve saved to {save_path}")

        return fig

    def get_gpower_parameters(
        self, coeff_name: str, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate parameters for G*Power replication.

        Args:
            coeff_name: Name of coefficient
            X: Feature data
            y: Target data

        Returns:
            Dictionary of G*Power parameters
        """
        result = self.model.result

        # Odds ratio for the coefficient
        or_value = float(np.exp(result.params[coeff_name]))

        # Check if binary or continuous predictor
        x_col = X[coeff_name]

        if set(x_col.unique()) <= {0, 1}:
            # Binary predictor
            p0 = y[x_col == 0].mean() if (x_col == 0).any() else y.mean()
            prop_x1 = (x_col == 1).mean()
            x_distribution = "Binomial"
            x_param_1 = prop_x1
            x_param_2 = None
        else:
            # Continuous predictor
            p0 = y.mean()
            prop_x1 = np.nan
            x_distribution = "Normal"
            x_param_1 = x_col.mean()  # mu
            x_param_2 = x_col.std()  # sigma

        # Calculate R² other X
        other_cols = [c for c in X.columns if c != coeff_name]
        r2_other = 0.0

        if other_cols:
            # Build design matrix for other predictors
            exog = pd.get_dummies(X[other_cols], drop_first=True, dtype=float)
            exog = sm.add_constant(exog)
            endog = pd.to_numeric(x_col, errors="coerce")
            good = ~(exog.isna().any(axis=1) | endog.isna())
            exog, endog = exog.loc[good], endog.loc[good]

            if len(exog) > 0:
                if set(x_col.unique()) <= {0, 1}:
                    # Binary outcome - use McFadden R²
                    log_mod = sm.Logit(endog.astype(float), exog.astype(float))
                    log_res = log_mod.fit(disp=False, maxiter=10000)
                    r2_other = 1.0 - log_res.llf / log_res.llnull
                else:
                    # Continuous outcome - use ordinary R²
                    ols_res = sm.OLS(endog.astype(float), exog.astype(float)).fit()
                    r2_other = float(ols_res.rsquared)

        return {
            "odds_ratio": or_value,
            "pr_y1_x1_h0": p0,
            "alpha": 0.05,
            "power": 0.80,
            "r2_other_x": r2_other,
            "x_distribution": x_distribution,
            "x_param_1": x_param_1,
            "x_param_2": x_param_2,
            "n_tested_predictors": 1,
            "n_total_predictors": len(X.columns),
        }


class ModelComparison:
    """Compare multiple models."""

    @staticmethod
    def compare_roc_curves(
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare ROC curves for multiple models.

        Args:
            models: Dictionary of model_name -> model instance
            X_test: Test features
            y_test: Test labels
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        for (name, model), color in zip(models.items(), colors):
            # Get predictions
            y_pred = model.predict_proba(X_test)
            if len(y_pred.shape) > 1:
                y_pred = y_pred[:, 1]

            # Compute ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_score = auc(fpr, tpr)

            # Plot
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=color, linewidth=2)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)

        # Formatting
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Model Comparison - ROC Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.set_label("Compare_ROC_Curves")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison saved to {save_path}")

        return fig
