# CreditScore/variable_selection.py

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VariableSelection:
    """
    Automated variable selection for scorecard models.

    This class implements multiple stages of variable selection:
    1. Post-binning: Remove variables with insufficient variability after binning
    2. Post-regularization: Remove variables shrunk to zero by Elastic Net
    """

    def __init__(self):
        self.selection_report = {
            "original_variables": [],
            "post_missing_removed": [],
            "post_binning_removed": [],
            "post_binning_remaining": [],
            "post_regularization_removed": [],
            "final_selected": [],
            "removal_reasons": {},
        }

    def filter_missing_value_variables(
        self,
        X: pd.DataFrame,
        variable_names: List[str],
        missing_threshold: float = 0.6,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Remove variables with too many missing values."""
        selected_variables = []
        removal_reasons = {}

        for var in variable_names:
            if var in X.columns:
                missing_ratio = X[var].isna().mean()
                if missing_ratio < missing_threshold:
                    selected_variables.append(var)
                else:
                    reason = f"{missing_ratio:.1%} missing values"
                    removal_reasons[var] = reason
                    self.selection_report["removal_reasons"][var] = reason
                    logger.info(f"Removing variable '{var}': {reason}")
            else:
                reason = "Variable not found in input data"
                removal_reasons[var] = reason
                self.selection_report["removal_reasons"][var] = reason
                logger.info(f"Removing variable '{var}': {reason}")

        return selected_variables, removal_reasons

    def filter_days_before_variables(
        self,
        variable_names: List[str],
        keep_variable: str = "PreopAlbumin_Days_Before",
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Remove variables ending in '_Days_Before' except for the specified one to keep.

        Args:
            variable_names: List of variable names to filter
            keep_variable: Variable to retain even if it matches the pattern

        Returns:
            Tuple of (filtered_variables, removal_reasons)
        """
        selected_variables = []
        removal_reasons = {}

        for var in variable_names:
            if var.endswith("_Days_Before") and var != keep_variable:
                reason = "Excluded based on '_Days_Before' pattern"
                removal_reasons[var] = reason
                self.selection_report["removal_reasons"][var] = reason
                self.selection_report.setdefault("pattern_removed", []).append(var)
                logger.info(f"Removing variable '{var}': {reason}")
            else:
                selected_variables.append(var)

        return selected_variables, removal_reasons

    def filter_post_binning_variables(
        self,
        X_binned: pd.DataFrame,
        variable_names: List[str],
        min_unique_values: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Remove variables that have insufficient unique values after binning.

        Args:
            X_binned: Binned feature data
            variable_names: List of variable names to check
            min_unique_values: Minimum number of unique values required (from config if None)

        Returns:
            Tuple of (selected_variables, removal_reasons)
        """
        # Get config if not provided
        if min_unique_values is None:
            from CreditScore.utils import ConfigurationManager

            config = ConfigurationManager.variable_selection_args()
            min_unique_values = config.get("post_binning", {}).get("min_unique_values", 2)

        selected_variables = []
        removal_reasons = {}

        self.selection_report["original_variables"] = variable_names.copy()

        for var in variable_names:
            if var in X_binned.columns:
                unique_count = X_binned[var].nunique()

                if unique_count >= min_unique_values:
                    selected_variables.append(var)
                else:
                    reason = f"Only {unique_count} unique value(s) after binning"
                    removal_reasons[var] = reason
                    self.selection_report["post_binning_removed"].append(var)
                    self.selection_report["removal_reasons"][var] = reason
                    logger.info(f"Removing variable '{var}': {reason}")
            else:
                reason = "Variable not found in binned data"
                removal_reasons[var] = reason
                self.selection_report["post_binning_removed"].append(var)
                self.selection_report["removal_reasons"][var] = reason

        self.selection_report["post_binning_remaining"] = selected_variables.copy()

        logger.info(
            f"Post-binning selection: {len(selected_variables)}/{len(variable_names)} variables retained"
        )

        return selected_variables, removal_reasons

    def filter_post_regularization_variables(
        self,
        coefficients: pd.DataFrame,
        variable_names: List[str],
        coefficient_threshold: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Remove variables that were shrunk to zero by regularization.

        Args:
            coefficients: DataFrame with 'Feature' and 'Coefficient' columns
            variable_names: List of variable names to check
            coefficient_threshold: Threshold below which coefficients are considered zero (from config if None)

        Returns:
            Tuple of (selected_variables, removal_reasons)
        """
        # Get config if not provided
        if coefficient_threshold is None:
            from CreditScore.utils import ConfigurationManager

            config = ConfigurationManager.variable_selection_args()
            coefficient_threshold = config.get("post_regularization", {}).get(
                "coefficient_threshold", 1e-6
            )

        selected_variables = []
        removal_reasons = {}

        for var in variable_names:
            # Find coefficient for this variable
            var_coef = coefficients[coefficients["Feature"] == var]

            if var_coef.empty:
                reason = "Variable not found in model coefficients"
                removal_reasons[var] = reason
                self.selection_report["post_regularization_removed"].append(var)
                self.selection_report["removal_reasons"][var] = reason
                logger.info(f"Removing variable '{var}': {reason}")
            else:
                coef_value = abs(var_coef["Coefficient"].iloc[0])

                if coef_value > coefficient_threshold:
                    selected_variables.append(var)
                else:
                    reason = (
                        f"Coefficient shrunk to zero by regularization (|coef| = {coef_value:.2e})"
                    )
                    removal_reasons[var] = reason
                    self.selection_report["post_regularization_removed"].append(var)
                    self.selection_report["removal_reasons"][var] = reason
                    logger.info(f"Removing variable '{var}': {reason}")

        self.selection_report["final_selected"] = selected_variables.copy()

        logger.info(
            f"Post-regularization selection: {len(selected_variables)}/{len(variable_names)} variables retained"
        )

        return selected_variables, removal_reasons

    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the variable selection process.

        Returns:
            Dictionary with selection statistics and details
        """
        n_original = len(self.selection_report["original_variables"])
        n_post_binning = len(self.selection_report["post_binning_remaining"])
        n_final = len(self.selection_report["final_selected"])

        summary = {
            "original_count": n_original,
            "post_binning_count": n_post_binning,
            "final_count": n_final,
            "total_removed": n_original - n_final,
            "removal_rate": (n_original - n_final) / n_original if n_original > 0 else 0,
            "stages": {
                "post_binning_removed": len(self.selection_report["post_binning_removed"]),
                "post_regularization_removed": len(
                    self.selection_report["post_regularization_removed"]
                ),
            },
            "removed_variables": {
                "post_binning": self.selection_report["post_binning_removed"],
                "post_regularization": self.selection_report["post_regularization_removed"],
            },
            "final_variables": self.selection_report["final_selected"],
            "removal_reasons": self.selection_report["removal_reasons"],
        }

        return summary

    def generate_selection_report(self) -> str:
        """
        Generate a human-readable selection report.

        Returns:
            Formatted string report
        """
        summary = self.get_selection_summary()

        report = ["🎯 Automated Variable Selection Report", "=" * 50, ""]

        # Overview
        report.append(f"Original variables: {summary['original_count']}")
        report.append(f"Final selected variables: {summary['final_count']}")
        report.append(
            f"Variables removed: {summary['total_removed']} ({summary['removal_rate']:.1%})"
        )
        report.append("")

        # Stage-by-stage breakdown
        report.append("📊 Selection Stages:")
        report.append(
            f"  1. Post-binning filter: {summary['stages']['post_binning_removed']} variables removed"
        )
        report.append(
            f"  2. Post-regularization filter: {summary['stages']['post_regularization_removed']} variables removed"
        )
        report.append("")

        # Removed variables by stage
        if summary["removed_variables"]["post_binning"]:
            report.append("🔍 Variables removed after binning (insufficient variability):")
            for var in summary["removed_variables"]["post_binning"]:
                reason = summary["removal_reasons"].get(var, "Unknown reason")
                report.append(f"  - {var}: {reason}")
            report.append("")

        if summary["removed_variables"]["post_regularization"]:
            report.append("⚖️ Variables removed after regularization (zero coefficients):")
            for var in summary["removed_variables"]["post_regularization"]:
                reason = summary["removal_reasons"].get(var, "Unknown reason")
                report.append(f"  - {var}: {reason}")
            report.append("")

        # Final selected variables
        report.append("✅ Final selected variables:")
        for var in summary["final_variables"]:
            report.append(f"  - {var}")

        if not summary["final_variables"]:
            report.append("  ⚠️ No variables selected!")

        report.append("")
        report.append("💡 Benefits of automated selection:")
        report.append("  • Improved model interpretability")
        report.append("  • Reduced overfitting risk")
        report.append("  • Better scorecard stability")
        report.append("  • Focused on most predictive features")

        return "\n".join(report)


class ScorecardVariableSelector:
    """
    Specialized variable selector for scorecard models that integrates with the training process.
    """

    def __init__(self):
        self.selector = VariableSelection()
        self.initial_variables = None
        self.binning_selected_variables = None
        self.final_selected_variables = None

    def select_variables_full_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scorecard_model_class,
        initial_binning_config: Dict[str, Any],
        scorecard_config: Dict[str, Any],
        fit_config: Dict[str, Any],
    ) -> Tuple[Any, List[str], Dict[str, Any]]:
        """
        Run the full variable selection pipeline integrated with model training.

        Args:
            X_train: Training features
            y_train: Training target
            scorecard_model_class: The scorecard model class to instantiate
            initial_binning_config: Configuration for binning
            scorecard_config: Configuration for scorecard
            fit_config: Configuration for model fitting

        Returns:
            Tuple of (final_model, selected_variables, selection_report)
        """
        self.initial_variables = X_train.columns.tolist()
        logger.info(f"Starting variable selection with {len(self.initial_variables)} variables")

        # Stage 1: Initial model training with all variables
        logger.info("Stage 1: Training initial model with all variables...")
        initial_model = scorecard_model_class(**scorecard_config)

        # Filter variables based on missing value ratio
        missing_filtered_variables, _ = self.selector.filter_missing_value_variables(
            X_train, self.initial_variables
        )

        if not missing_filtered_variables:
            raise ValueError("No variables remain after missing value filtering!")

        pattern_filtered_variables, _ = self.selector.filter_days_before_variables(
            missing_filtered_variables
        )

        if not pattern_filtered_variables:
            raise ValueError("No variables remain after pattern-based filtering!")

        # Proceed with binning only on missing-filtered variables
        binning_process, X_train_binned = initial_model._perform_binning(
            X_train[pattern_filtered_variables],
            y_train,
            pattern_filtered_variables,
            initial_binning_config,
        )

        # Filter variables based on post-binning variability
        self.binning_selected_variables, _ = self.selector.filter_post_binning_variables(
            X_train_binned, pattern_filtered_variables
        )

        if not self.binning_selected_variables:
            raise ValueError("No variables remain after post-binning filtering!")

        # Stage 2: Train model with binning-selected variables
        logger.info(
            f"Stage 2: Training model with {len(self.binning_selected_variables)} post-binning variables..."
        )
        X_train_selected = X_train[self.binning_selected_variables]

        intermediate_model = scorecard_model_class(**scorecard_config)
        intermediate_model.fit(X_train_selected, y_train, **fit_config)

        # Get coefficients and filter based on regularization
        coefficients = intermediate_model.get_coefficients()
        self.final_selected_variables, _ = self.selector.filter_post_regularization_variables(
            coefficients, self.binning_selected_variables
        )

        if not self.final_selected_variables:
            logger.warning(
                "No variables remain after regularization filtering! Using all post-binning variables."
            )
            self.final_selected_variables = self.binning_selected_variables

        # Stage 3: Train final model with selected variables
        logger.info(
            f"Stage 3: Training final model with {len(self.final_selected_variables)} selected variables..."
        )
        X_train_final = X_train[self.final_selected_variables]

        final_model = scorecard_model_class(**scorecard_config)
        final_model.fit(X_train_final, y_train, **fit_config)

        # Generate selection report
        selection_report = self.selector.get_selection_summary()

        logger.info(
            f"Variable selection complete: {len(self.initial_variables)} → {len(self.final_selected_variables)} variables"
        )

        return final_model, self.final_selected_variables, selection_report
