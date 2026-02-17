# tests/unit/test_missing_data_handling.py

"""
Tests for missing data handling differences between OptBinning and other models.

Key issues tested:
1. OptBinning scorecard models handle missing values automatically (drop_missing=False)
2. Other models typically require missing values to be dropped (drop_missing=True)
3. Missing target values should always be removed regardless of model type
4. OptBinning assigns missing values to appropriate bins
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from CreditScore.data import DataPreprocessor
from CreditScore.train import (
    OptbinningScorecardModel,
    StatsmodelsLogisticModel,
)


@pytest.mark.missing_data
@pytest.mark.regression
@pytest.mark.optbinning
class TestMissingDataHandling:
    """Test missing data handling across different model types."""

    def test_optbinning_preserves_missing_values(self, sample_medical_data):
        """
        REGRESSION TEST: OptBinning models should preserve missing values in features.

        Previously, missing values were being dropped for all models, but OptBinning
        handles missing values automatically and assigns them to bins.
        """
        data = sample_medical_data.copy()

        # Add more missing values for testing
        data.loc[0:100, "AlbuminLevel"] = np.nan
        data.loc[50:150, "BMI"] = np.nan
        data.loc[200:250, "Age"] = np.nan

        features = ["Age", "BMI", "AlbuminLevel", "ASAStatus"]
        target = "thirty_day_mortality"

        preprocessor = DataPreprocessor()

        # For OptBinning, preserve missing values
        X, y = preprocessor.prepare_modeling_data(
            data,
            target,
            features,
            drop_missing=False,  # CRITICAL: OptBinning handles missing
            validate_ranges=True,
        )

        # Should preserve rows with missing feature values
        assert len(X) > 0, "Should have data after preprocessing"

        # Should have missing values in features
        total_missing = X.isnull().sum().sum()
        assert total_missing > 0, "Should preserve missing values for OptBinning"

        # Should be able to train OptBinning model with missing data
        y_encoded = preprocessor.encode_target(y)

        # Remove only missing target values
        valid_target_mask = y_encoded.notna()
        X_valid = X[valid_target_mask]
        y_valid = y_encoded[valid_target_mask]

        model = OptbinningScorecardModel()
        model.fit(X_valid, y_valid, hyperparameter_search=False)

        # Should successfully train and predict with missing data
        assert model.is_fitted
        scores = model.score(X_valid)
        assert len(scores) == len(X_valid)
        assert not np.isnan(scores).any(), "Scores should not contain NaN"

    def test_statsmodels_requires_complete_cases(self, sample_medical_data):
        """
        Test that statsmodels logistic regression typically requires complete cases.

        Unlike OptBinning, statsmodels doesn't handle missing values automatically.
        """
        data = sample_medical_data.copy()

        # Add missing values
        data.loc[0:50, "AlbuminLevel"] = np.nan
        data.loc[100:150, "BMI"] = np.nan

        features = ["Age", "BMI", "AlbuminLevel", "ASAStatus"]
        target = "thirty_day_mortality"

        preprocessor = DataPreprocessor()

        # For statsmodels, typically need to drop missing
        X, y = preprocessor.prepare_modeling_data(
            data,
            target,
            features,
            drop_missing=True,  # Drop missing for statsmodels
            validate_ranges=True,
        )

        # Should have fewer rows after dropping missing
        assert len(X) < len(data), "Should have fewer rows after dropping missing"

        # Should have no missing values
        total_missing = X.isnull().sum().sum()
        assert total_missing == 0, "Should have no missing values after drop_missing=True"

        # Should be able to train statsmodels with complete cases
        y_encoded = preprocessor.encode_target(y)

        model = StatsmodelsLogisticModel()
        model.fit(X, y_encoded, check_separation=False)  # Skip separation check for speed

        assert model.is_fitted

    def test_missing_target_values_always_removed(self, sample_medical_data):
        """
        CRITICAL TEST: Missing target values should be removed regardless of model type.

        Even OptBinning can't handle missing target values.
        """
        data = sample_medical_data.copy()

        # Add missing target values
        data.loc[0:50, "thirty_day_mortality"] = np.nan

        features = ["Age", "BMI", "ASAStatus"]
        target = "thirty_day_mortality"

        preprocessor = DataPreprocessor()

        # Test with OptBinning (drop_missing=False for features)
        X_opt, y_opt = preprocessor.prepare_modeling_data(
            data, target, features, drop_missing=False, validate_ranges=True
        )

        y_opt_encoded = preprocessor.encode_target(y_opt)

        # Should have missing target values before filtering
        assert y_opt_encoded.isnull().any(), "Should have missing target values"

        # Remove missing targets
        valid_mask = y_opt_encoded.notna()
        X_opt_valid = X_opt[valid_mask]
        y_opt_valid = y_opt_encoded[valid_mask]

        # After filtering, should have no missing targets
        assert not y_opt_valid.isnull().any(), "Should have no missing targets after filtering"
        assert len(X_opt_valid) == len(y_opt_valid), "X and y should have same length"
        assert len(X_opt_valid) < len(data), "Should have fewer rows after removing missing targets"

    def test_data_preprocessor_drop_missing_parameter(self, sample_medical_data):
        """Test that drop_missing parameter works correctly in DataPreprocessor."""
        data = sample_medical_data.copy()

        # Add missing values in features
        data.loc[0:30, "BMI"] = np.nan
        data.loc[50:80, "AlbuminLevel"] = np.nan

        features = ["Age", "BMI", "AlbuminLevel", "ASAStatus"]
        target = "thirty_day_mortality"

        preprocessor = DataPreprocessor()

        # Test drop_missing=True
        X_drop, y_drop = preprocessor.prepare_modeling_data(
            data, target, features, drop_missing=True, validate_ranges=True
        )

        # Should have no missing values
        assert X_drop.isnull().sum().sum() == 0, "drop_missing=True should remove all missing"

        # Test drop_missing=False
        X_keep, y_keep = preprocessor.prepare_modeling_data(
            data, target, features, drop_missing=False, validate_ranges=True
        )

        # Should preserve missing values
        assert X_keep.isnull().sum().sum() > 0, "drop_missing=False should preserve missing"

        # drop_missing=False should have more rows
        assert len(X_keep) > len(X_drop), "drop_missing=False should preserve more rows"

    def test_optbinning_handles_high_missingness(self):
        """Test OptBinning with very high missingness rates."""
        np.random.seed(42)
        n_samples = 500

        # Create data with high missingness
        data = pd.DataFrame(
            {
                "mostly_missing": [1.0] * 50 + [np.nan] * 450,  # 90% missing
                "half_missing": [1.0] * 250 + [np.nan] * 250,  # 50% missing
                "low_missing": [1.0] * 450 + [np.nan] * 50,  # 10% missing
                "complete": np.random.normal(0, 1, n_samples),  # No missing
                "target": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            }
        )

        features = ["mostly_missing", "half_missing", "low_missing", "complete"]
        target = "target"

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_modeling_data(
            data,
            target,
            features,
            drop_missing=False,  # OptBinning should handle high missingness
            validate_ranges=True,
        )

        y_encoded = preprocessor.encode_target(y)

        # Should preserve all rows (no missing targets in this test)
        assert len(X) == n_samples

        # Should train successfully despite high missingness
        model = OptbinningScorecardModel()
        model.fit(X, y_encoded, hyperparameter_search=False)

        assert model.is_fitted
        scores = model.score(X)
        assert len(scores) == n_samples
        assert not np.isnan(scores).any()

    def test_missing_value_encoding_consistency(self, sample_medical_data):
        """Test that missing values are encoded consistently throughout pipeline."""
        data = sample_medical_data.copy()

        # Add various forms of missing data
        data.loc[0:20, "AlbuminLevel"] = np.nan
        data.loc[21:40, "AlbuminLevel"] = None
        data.loc[41:50, "BMI"] = "_"  # String representation of missing
        data.loc[51:60, "BMI"] = "NA"  # Another string representation
        data.loc[61:70, "BMI"] = ""  # Empty string

        features = ["Age", "BMI", "AlbuminLevel"]
        target = "thirty_day_mortality"

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_modeling_data(
            data, target, features, drop_missing=False, validate_ranges=True
        )

        # DataPreprocessor should convert various missing representations to NaN
        assert X["AlbuminLevel"].isnull().sum() >= 40  # At least np.nan and None
        assert X["BMI"].isnull().sum() >= 30  # At least _, NA, and empty string

        # All missing values should be pandas NaN
        missing_albumin = X["AlbuminLevel"].isnull()
        missing_bmi = X["BMI"].isnull()

        # Check that all missing values are actually NaN (not other representations)
        albumin_missing_values = X.loc[missing_albumin, "AlbuminLevel"].unique()
        bmi_missing_values = X.loc[missing_bmi, "BMI"].unique()

        # Should only have NaN in missing positions
        assert len(albumin_missing_values) <= 1  # Should be just NaN or empty
        assert len(bmi_missing_values) <= 1  # Should be just NaN or empty

        if len(albumin_missing_values) == 1:
            assert pd.isna(albumin_missing_values[0])
        if len(bmi_missing_values) == 1:
            assert pd.isna(bmi_missing_values[0])


@pytest.mark.missing_data
@pytest.mark.integration
class TestMissingDataIntegration:
    """Integration tests for missing data handling in the full pipeline."""

    def test_scorecard_builder_missing_data_workflow(self, sample_medical_data):
        """Test the complete workflow as used in scorecard_builder.py with missing data."""
        data = sample_medical_data.copy()

        # Add realistic missing data patterns
        data.loc[0:100, "AlbuminLevel"] = np.nan  # Lab value missing
        data.loc[50:120, "PreOpHemoglobin"] = np.nan  # Another lab value missing
        data.loc[200:220, "thirty_day_mortality"] = np.nan  # Missing outcomes

        features = ["Age", "BMI", "ASAStatus", "AlbuminLevel", "PreOpHemoglobin"]
        target = "thirty_day_mortality"

        # Simulate the exact workflow from scorecard_builder.py
        preprocessor = DataPreprocessor()

        # Step 1: Prepare data with drop_missing=False (OptBinning workflow)
        X, y = preprocessor.prepare_modeling_data(
            data,
            target,
            features,
            drop_missing=False,  # OptBinning handles missing values
            validate_ranges=True,
        )

        # Step 2: Encode target
        y = preprocessor.encode_target(y)

        # Step 3: Remove rows where target is missing (critical step)
        if y.isnull().any():
            valid_mask = y.notna()
            X = X[valid_mask]
            y = y[valid_mask]

        # Step 4: Train model
        model = OptbinningScorecardModel()
        model.fit(X, y, hyperparameter_search=False)

        # Step 5: Generate predictions for all valid cases
        scores = model.score(X)
        probas = model.predict_proba(X)

        # Verify workflow completed successfully
        assert model.is_fitted
        assert len(scores) == len(X) == len(y)
        assert not np.isnan(scores).any()
        assert not np.isnan(probas).any()

        # Verify missing feature values were preserved and handled
        feature_missing_count = X.isnull().sum().sum()
        assert feature_missing_count > 0, "Should have preserved missing feature values"

        # Verify no missing target values remain
        assert not y.isnull().any(), "Should have no missing target values"

    def test_comparison_optbinning_vs_statsmodels_missing_data(self, sample_medical_data):
        """Compare how OptBinning and statsmodels handle the same missing data."""
        data = sample_medical_data.copy()

        # Add missing data
        data.loc[0:50, "BMI"] = np.nan
        data.loc[100:150, "AlbuminLevel"] = np.nan

        features = ["Age", "BMI", "AlbuminLevel", "ASAStatus"]
        target = "thirty_day_mortality"

        preprocessor = DataPreprocessor()

        # OptBinning workflow (preserve missing)
        X_opt, y_opt = preprocessor.prepare_modeling_data(
            data, target, features, drop_missing=False, validate_ranges=True
        )
        y_opt = preprocessor.encode_target(y_opt)

        # Statsmodels workflow (drop missing)
        X_stats, y_stats = preprocessor.prepare_modeling_data(
            data, target, features, drop_missing=True, validate_ranges=True
        )
        y_stats = preprocessor.encode_target(y_stats)

        # OptBinning should have more data points
        assert len(X_opt) > len(X_stats), "OptBinning should preserve more data"

        # OptBinning should have missing values, statsmodels should not
        assert X_opt.isnull().sum().sum() > 0, "OptBinning should have missing values"
        assert X_stats.isnull().sum().sum() == 0, "Statsmodels should have no missing values"

        # Both should train successfully on their respective datasets
        opt_model = OptbinningScorecardModel()
        opt_model.fit(X_opt, y_opt, hyperparameter_search=False)

        stats_model = StatsmodelsLogisticModel()
        stats_model.fit(X_stats, y_stats, check_separation=False)

        assert opt_model.is_fitted
        assert stats_model.is_fitted

        # Both should make predictions on their training data
        opt_scores = opt_model.score(X_opt)
        stats_probas = stats_model.predict_proba(X_stats)

        assert len(opt_scores) == len(X_opt)
        assert len(stats_probas) == len(X_stats)

    def test_missing_data_edge_cases(self):
        """Test edge cases in missing data handling."""
        # Case 1: All feature values missing for one variable
        data_all_missing = pd.DataFrame(
            {
                "all_missing": [np.nan] * 100,
                "good_var": np.random.normal(0, 1, 100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_modeling_data(
            data_all_missing, "target", ["all_missing", "good_var"], drop_missing=False
        )

        # Should preserve the all-missing variable
        assert "all_missing" in X.columns
        assert X["all_missing"].isnull().all()

        # OptBinning cannot handle variables with ALL missing values
        y_encoded = preprocessor.encode_target(y)
        model = OptbinningScorecardModel()

        # This should fail because all_missing has no values to bin
        with pytest.raises(ValueError):
            model.fit(X, y_encoded, hyperparameter_search=False)

        # Case 2: Only one non-missing value
        data_one_value = pd.DataFrame(
            {
                "one_value": [1.0] + [np.nan] * 99,
                "good_var": np.random.normal(0, 1, 100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        X2, y2 = preprocessor.prepare_modeling_data(
            data_one_value, "target", ["one_value", "good_var"], drop_missing=False
        )

        # Should preserve the variable with one value
        assert "one_value" in X2.columns
        assert X2["one_value"].count() == 1  # Only one non-missing value

        # Should still train
        y2_encoded = preprocessor.encode_target(y2)
        model2 = OptbinningScorecardModel()
        model2.fit(X2, y2_encoded, hyperparameter_search=False)
        assert model2.is_fitted
