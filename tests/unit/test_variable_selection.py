# tests/unit/test_variable_selection.py

"""
Tests for the automated variable selection pipeline.

This functionality was critical for solving the "only 2 unique scores" problem
caused by extreme class imbalance. The variable selection removes:
1. Variables with insufficient variability after binning
2. Variables with zero coefficients after regularization
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from CreditScore.train import OptbinningScorecardModel
from CreditScore.variable_selection import (
    ScorecardVariableSelector,
    VariableSelection,
)


@pytest.mark.variable_selection
@pytest.mark.unit
class TestVariableSelection:
    """Test the core variable selection functionality."""

    def test_filter_post_binning_variables_basic(self):
        """Test basic post-binning variable filtering."""
        # Create sample binned data with different levels of variability
        X_binned = pd.DataFrame(
            {
                "constant_var": [0] * 100,  # No variability
                "low_var": [0] * 95 + [1] * 5,  # Very low variability
                "good_var": [0] * 70 + [1] * 20 + [2] * 10,  # Good variability
                "excellent_var": [0] * 50 + [1] * 30 + [2] * 20,  # Excellent variability
            }
        )

        variable_names = list(X_binned.columns)

        selector = VariableSelection()

        # Test with min_unique_values=2
        filtered_vars, removal_reasons = selector.filter_post_binning_variables(
            X_binned, variable_names, min_unique_values=2
        )

        # Should remove constant_var (only 1 unique value)
        assert "constant_var" not in filtered_vars
        assert "low_var" in filtered_vars
        assert "good_var" in filtered_vars
        assert "excellent_var" in filtered_vars
        assert len(filtered_vars) == 3

        # Check removal reasons
        assert "constant_var" in removal_reasons
        assert "Only 1 unique value(s) after binning" in removal_reasons["constant_var"]

    def test_filter_post_binning_variables_custom_threshold(self):
        """Test post-binning filtering with custom threshold."""
        X_binned = pd.DataFrame(
            {
                "var1": [0] * 100,  # 1 unique
                "var2": [0] * 95 + [1] * 5,  # 2 unique
                "var3": [0] * 80 + [1] * 15 + [2] * 5,  # 3 unique
                "var4": [0] * 60 + [1] * 20 + [2] * 15 + [3] * 5,  # 4 unique
            }
        )

        variable_names = list(X_binned.columns)
        selector = VariableSelection()

        # Test with min_unique_values=3
        filtered_vars, removal_reasons = selector.filter_post_binning_variables(
            X_binned, variable_names, min_unique_values=3
        )

        assert "var1" not in filtered_vars
        assert "var2" not in filtered_vars
        assert "var3" in filtered_vars
        assert "var4" in filtered_vars
        assert len(filtered_vars) == 2

    def test_filter_post_binning_variables_with_missing_data(self):
        """Test post-binning filtering handles missing data correctly."""
        X_binned = pd.DataFrame(
            {
                "var_with_missing": [0, 0, 0, np.nan, np.nan],
                "var_constant_with_missing": [1, 1, 1, np.nan, np.nan],
                "var_good": [0, 1, 2, 0, 1],
            }
        )

        variable_names = list(X_binned.columns)
        selector = VariableSelection()

        filtered_vars, removal_reasons = selector.filter_post_binning_variables(
            X_binned, variable_names, min_unique_values=2
        )

        # var_with_missing has only 1 unique non-missing value
        assert "var_with_missing" not in filtered_vars
        # var_constant_with_missing has only 1 unique non-missing value
        assert "var_constant_with_missing" not in filtered_vars
        # var_good has multiple unique values
        assert "var_good" in filtered_vars

    def test_filter_post_regularization_variables_basic(self, sample_medical_data):
        """Test basic post-regularization variable filtering."""
        # Create mock coefficients DataFrame (the method expects coefficients directly, not model)
        mock_coef_df = pd.DataFrame(
            {
                "Feature": ["Intercept", "Age", "BMI", "ASAStatus", "ZeroCoef"],
                "Coefficient": [0.5, 0.1, 0.05, 0.02, 0.0],  # ZeroCoef has zero coefficient
            }
        )

        variable_names = ["Age", "BMI", "ASAStatus", "ZeroCoef"]
        selector = VariableSelection()

        filtered_vars, removal_reasons = selector.filter_post_regularization_variables(
            mock_coef_df, variable_names, coefficient_threshold=1e-6
        )

        # Should remove ZeroCoef (coefficient = 0.0)
        assert "Age" in filtered_vars
        assert "BMI" in filtered_vars
        assert "ASAStatus" in filtered_vars
        assert "ZeroCoef" not in filtered_vars
        assert len(filtered_vars) == 3

        # Check removal reasons
        assert "ZeroCoef" in removal_reasons
        assert "shrunk to zero" in removal_reasons["ZeroCoef"]

    def test_filter_post_regularization_variables_custom_threshold(self):
        """Test post-regularization filtering with custom threshold."""
        mock_coef_df = pd.DataFrame(
            {
                "Feature": ["Intercept", "BigCoef", "MediumCoef", "SmallCoef", "TinyCoef"],
                "Coefficient": [0.5, 0.1, 0.05, 0.001, 0.0001],
            }
        )

        variable_names = ["BigCoef", "MediumCoef", "SmallCoef", "TinyCoef"]
        selector = VariableSelection()

        # Test with threshold=0.01
        filtered_vars, removal_reasons = selector.filter_post_regularization_variables(
            mock_coef_df, variable_names, coefficient_threshold=0.01
        )

        # Should keep BigCoef (0.1) and MediumCoef (0.05)
        # Should remove SmallCoef (0.001) and TinyCoef (0.0001)
        assert "BigCoef" in filtered_vars
        assert "MediumCoef" in filtered_vars
        assert "SmallCoef" not in filtered_vars
        assert "TinyCoef" not in filtered_vars
        assert len(filtered_vars) == 2

    def test_filter_post_regularization_handles_missing_features(self):
        """Test that post-regularization filtering handles missing features gracefully."""
        mock_coef_df = pd.DataFrame(
            {
                "Feature": ["Intercept", "Age", "BMI"],  # Missing ASAStatus
                "Coefficient": [0.5, 0.1, 0.05],
            }
        )

        variable_names = ["Age", "BMI", "ASAStatus"]  # ASAStatus not in coefficients
        selector = VariableSelection()

        filtered_vars, removal_reasons = selector.filter_post_regularization_variables(
            mock_coef_df, variable_names, coefficient_threshold=1e-6
        )

        # Should only include features that exist in coefficients
        assert "Age" in filtered_vars
        assert "BMI" in filtered_vars
        assert "ASAStatus" not in filtered_vars
        assert len(filtered_vars) == 2

        # Check removal reason for missing feature
        assert "ASAStatus" in removal_reasons
        assert "not found in model coefficients" in removal_reasons["ASAStatus"]

    def test_generate_selection_report(self):
        """Test selection report generation."""
        selector = VariableSelection()

        # Simulate the selection process by setting up the selection_report dictionary
        selector.selection_report = {
            "original_variables": ["A", "B", "C", "D", "E"],
            "post_binning_removed": ["D"],
            "post_binning_remaining": ["A", "B", "C", "E"],
            "post_regularization_removed": ["B"],
            "final_selected": ["A", "C", "E"],
            "removal_reasons": {
                "D": "Only 1 unique value(s) after binning",
                "B": "Coefficient shrunk to zero by regularization",
            },
        }

        report = selector.generate_selection_report()

        # Check key elements are in the report
        assert "🎯 Automated Variable Selection Report" in report
        assert "Original variables: 5" in report
        assert "Final selected variables: 3" in report
        assert "Variables removed: 2" in report
        assert "Post-binning filter: 1 variables removed" in report
        assert "Post-regularization filter: 1 variables removed" in report
        assert "- D: Only 1 unique value(s) after binning" in report
        assert "- B: Coefficient shrunk to zero by regularization" in report
        assert "- A" in report  # Should be in final selected
        assert "- C" in report  # Should be in final selected
        assert "- E" in report  # Should be in final selected


@pytest.mark.variable_selection
@pytest.mark.integration
@pytest.mark.regression
class TestScorecardVariableSelector:
    """Test the complete scorecard variable selection pipeline."""

    def test_select_variables_full_pipeline_integration(self, sample_medical_data):
        """Test the full variable selection pipeline with real data."""
        data = sample_medical_data

        # Include some variables that should be filtered out
        features = ["Age", "BMI", "ASAStatus", "EBL", "AlbuminLevel"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Add a constant feature that should be removed
        X["constant_feature"] = 1

        selector = ScorecardVariableSelector()

        model, selected_vars, report = selector.select_variables_full_pipeline(
            X_train=X,
            y_train=y,
            scorecard_model_class=OptbinningScorecardModel,
            initial_binning_config={},  # Empty dict for defaults
            scorecard_config={},  # Empty dict for defaults
            fit_config={"hyperparameter_search": False},  # Speed up test
        )

        # Should return a trained model
        assert model is not None
        assert model.is_fitted

        # Should have selected some variables (but probably not all)
        assert len(selected_vars) > 0
        assert len(selected_vars) <= len(features) + 1  # +1 for constant_feature

        # constant_feature should be removed
        assert "constant_feature" not in selected_vars

        # Report should contain useful information
        assert "original_count" in report
        assert "final_count" in report
        assert "total_removed" in report
        assert report["original_count"] == len(features) + 1  # +1 for constant_feature
        assert report["final_count"] == len(selected_vars)

    def test_select_variables_with_extreme_imbalance(self):
        """Test variable selection with extreme class imbalance (like the original bug)."""
        np.random.seed(42)
        n_samples = 1000

        # Create data with extreme imbalance (1% positive class)
        X = pd.DataFrame(
            {
                "good_predictor": np.random.normal(0, 1, n_samples),
                "weak_predictor": np.random.normal(0, 0.1, n_samples),  # Very weak signal
                "random_noise": np.random.normal(0, 1, n_samples),
                "constant_feature": [1] * n_samples,
                "almost_constant": [0] * 990 + [1] * 10,
            }
        )

        # Create target with extreme imbalance
        y = pd.Series([0] * 990 + [1] * 10)

        selector = ScorecardVariableSelector()

        model, selected_vars, report = selector.select_variables_full_pipeline(
            X_train=X,
            y_train=y,
            scorecard_model_class=OptbinningScorecardModel,
            initial_binning_config={},
            scorecard_config={},
            fit_config={"hyperparameter_search": False},
        )

        # Should successfully train despite extreme imbalance
        assert model.is_fitted

        # Should filter out problematic variables
        assert "constant_feature" not in selected_vars

        # Should have removed some variables
        assert report["total_removed"] > 0
        assert report["final_count"] < report["original_count"]

    def test_select_variables_preserves_good_predictors(self, sample_medical_data):
        """Test that variable selection preserves genuinely predictive variables."""
        data = sample_medical_data

        # Use variables that should be predictive of mortality
        features = ["Age", "ASAStatus", "EMERGENCY"]  # These should be kept
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        selector = ScorecardVariableSelector()

        model, selected_vars, report = selector.select_variables_full_pipeline(
            X_train=X,
            y_train=y,
            scorecard_model_class=OptbinningScorecardModel,
            initial_binning_config={},
            scorecard_config={},
            fit_config={"hyperparameter_search": False},
        )

        # Should keep most or all of the good predictors
        good_predictors_kept = sum(1 for var in features if var in selected_vars)
        assert good_predictors_kept >= 2, "Should keep most good predictors"

    def test_variable_selection_improves_score_diversity(self, sample_data_with_issues):
        """
        REGRESSION TEST: Ensure variable selection improves score diversity.

        This addresses the original "only 2 unique scores" problem.
        """
        data = sample_data_with_issues
        features = [col for col in data.columns if col != "target"]
        target = "target"

        X = data[features].copy()
        y = data[target].copy()

        # Train model WITHOUT variable selection
        model_no_selection = OptbinningScorecardModel()
        model_no_selection.fit(X, y, hyperparameter_search=False)
        scores_no_selection = model_no_selection.score(X)
        unique_scores_no_selection = len(np.unique(scores_no_selection))

        # Train model WITH variable selection
        selector = ScorecardVariableSelector()
        model_with_selection, selected_vars, report = selector.select_variables_full_pipeline(
            X_train=X,
            y_train=y,
            scorecard_model_class=OptbinningScorecardModel,
            initial_binning_config={},
            scorecard_config={},
            fit_config={"hyperparameter_search": False},
        )

        X_selected = X[selected_vars]
        scores_with_selection = model_with_selection.score(X_selected)
        unique_scores_with_selection = len(np.unique(scores_with_selection))

        # Variable selection should improve score diversity
        # (or at least not make it worse)
        assert (
            unique_scores_with_selection >= unique_scores_no_selection
        ), f"Variable selection should not reduce score diversity: {unique_scores_with_selection} vs {unique_scores_no_selection}"

        # Should have removed some problematic variables
        assert len(selected_vars) < len(features), "Should have removed some variables"
        assert "constant_feature" not in selected_vars, "Should remove constant features"


@pytest.mark.variable_selection
@pytest.mark.unit
class TestVariableSelectionErrorHandling:
    """Test error handling and edge cases in variable selection."""

    def test_no_variables_after_binning_filter(self):
        """Test behavior when all variables are filtered out in binning stage."""
        # Create data where all variables are constant
        X_binned = pd.DataFrame({"const1": [0] * 100, "const2": [1] * 100, "const3": [2] * 100})

        variable_names = list(X_binned.columns)
        selector = VariableSelection()

        filtered_vars, removal_reasons = selector.filter_post_binning_variables(
            X_binned, variable_names, min_unique_values=2
        )

        # Should return empty list
        assert len(filtered_vars) == 0
        # All variables should be removed
        assert len(removal_reasons) == 3
        assert all(
            "Only 1 unique value(s) after binning" in reason for reason in removal_reasons.values()
        )

    def test_no_variables_after_regularization_filter(self):
        """Test behavior when all variables are filtered out in regularization stage."""
        mock_coef_df = pd.DataFrame(
            {
                "Feature": ["Intercept", "ZeroCoef1", "ZeroCoef2"],
                "Coefficient": [0.5, 0.0, 0.0],  # All non-intercept coefficients are zero
            }
        )

        variable_names = ["ZeroCoef1", "ZeroCoef2"]
        selector = VariableSelection()

        filtered_vars, removal_reasons = selector.filter_post_regularization_variables(
            mock_coef_df, variable_names, coefficient_threshold=1e-6
        )

        # Should return empty list
        assert len(filtered_vars) == 0
        # Both variables should be removed for zero coefficients
        assert len(removal_reasons) == 2
        assert all("shrunk to zero" in reason for reason in removal_reasons.values())

    def test_empty_variable_list_input(self):
        """Test behavior with empty variable list input."""
        X_binned = pd.DataFrame({"dummy": [1, 2, 3]})
        variable_names = []  # Empty list

        selector = VariableSelection()

        filtered_vars, removal_reasons = selector.filter_post_binning_variables(
            X_binned, variable_names, min_unique_values=2
        )

        assert len(filtered_vars) == 0
        assert len(removal_reasons) == 0  # No variables to process

    def test_single_variable_selection(self):
        """Test variable selection with only one variable."""
        X = pd.DataFrame({"single_var": np.random.normal(0, 1, 100)})
        y = pd.Series(np.random.choice([0, 1], 100))

        selector = ScorecardVariableSelector()

        model, selected_vars, report = selector.select_variables_full_pipeline(
            X_train=X,
            y_train=y,
            scorecard_model_class=OptbinningScorecardModel,
            initial_binning_config={},
            scorecard_config={},
            fit_config={"hyperparameter_search": False},
        )

        # Should handle single variable case
        assert len(selected_vars) <= 1
        if len(selected_vars) == 1:
            assert selected_vars[0] == "single_var"
