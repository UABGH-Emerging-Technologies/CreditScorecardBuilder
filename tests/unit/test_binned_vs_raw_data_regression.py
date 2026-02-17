# tests/unit/test_binned_vs_raw_data_regression.py

"""
Regression tests for the critical issue where OptBinning scorecard methods
were receiving binned data instead of raw data, causing incorrect results.

This was a major issue discovered during development where:
1. Scorecard.fit() was receiving binned data instead of raw data
2. Scorecard.score() and predict_proba() were receiving binned data
3. This caused models to only produce 2 unique scores (54 and 55)
4. The issue was fixed by ensuring scorecard methods receive unbinned data
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from optbinning import BinningProcess, Scorecard

from CreditScore.train import OptbinningScorecardModel


@pytest.mark.regression
@pytest.mark.optbinning
class TestBinnedVsRawDataRegression:
    """Test to ensure scorecard methods receive raw data, not binned data."""

    def test_scorecard_fit_receives_raw_data(self, sample_medical_data):
        """
        REGRESSION TEST: Ensure Scorecard.fit() receives raw (unbinned) data.

        Previously, fit() was receiving binned data which caused poor model performance.
        """
        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus", "EBL"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        model = OptbinningScorecardModel()

        # Mock the Scorecard.fit method to capture what data it receives
        with patch.object(Scorecard, "fit") as mock_fit:
            model.fit(X, y, hyperparameter_search=False)

            # Verify fit was called
            assert mock_fit.called, "Scorecard.fit() should have been called"

            # Get the arguments passed to fit
            args, kwargs = mock_fit.call_args
            fit_X, fit_y = args[0], args[1]

            # CRITICAL: The data passed to fit should be RAW data, not binned
            # Check that we're passing the original feature values
            assert isinstance(fit_X, pd.DataFrame), "X should be a DataFrame"
            assert len(fit_X.columns) == len(features), "Should have all original features"

            # Verify data types are preserved (not all converted to integers like binned data)
            assert fit_X["Age"].dtype in ["float64", "int64"], "Age should be numeric"
            assert fit_X["BMI"].dtype in ["float64"], "BMI should be float"

            # Verify we have reasonable value ranges (not bin indices)
            assert fit_X["Age"].min() >= 18, "Age should have realistic minimum"
            assert fit_X["Age"].max() <= 100, "Age should have realistic maximum"
            assert fit_X["BMI"].min() > 10, "BMI should have realistic minimum"
            assert fit_X["BMI"].max() < 60, "BMI should have realistic maximum"

    def test_scorecard_score_receives_raw_data(self, trained_scorecard_model):
        """
        REGRESSION TEST: Ensure Scorecard.score() receives raw (unbinned) data.

        Previously, score() was receiving binned data which limited unique scores.
        """
        model, X, y = trained_scorecard_model

        # Mock the scorecard.score method to capture input data
        with patch.object(model.scorecard, "score", wraps=model.scorecard.score) as mock_score:
            scores = model.score(X)

            # Verify score was called
            assert mock_score.called, "scorecard.score() should have been called"

            # Get the arguments passed to score
            args, kwargs = mock_score.call_args
            score_X = args[0]

            # CRITICAL: Verify raw data characteristics
            assert isinstance(score_X, pd.DataFrame), "Input should be DataFrame"

            # Check for realistic value ranges (not bin indices 0, 1, 2, etc.)
            if "Age" in score_X.columns:
                assert score_X["Age"].min() >= 18, "Age should be realistic, not bin index"
                assert score_X["Age"].max() <= 100, "Age should be realistic, not bin index"

            if "BMI" in score_X.columns:
                assert score_X["BMI"].min() > 10, "BMI should be realistic, not bin index"
                assert score_X["BMI"].max() < 60, "BMI should be realistic, not bin index"

            # Verify we get more than 2 unique scores (the original bug)
            assert (
                len(np.unique(scores)) > 2
            ), f"Should have >2 unique scores, got {len(np.unique(scores))}"

    def test_scorecard_predict_proba_receives_raw_data(self, trained_scorecard_model):
        """
        REGRESSION TEST: Ensure predict_proba() receives raw data.
        """
        model, X, y = trained_scorecard_model

        with patch.object(
            model.scorecard, "predict_proba", wraps=model.scorecard.predict_proba
        ) as mock_predict:
            probas = model.predict_proba(X)

            assert mock_predict.called, "scorecard.predict_proba() should have been called"

            # Get input data
            args, kwargs = mock_predict.call_args
            predict_X = args[0]

            # Verify raw data characteristics
            assert isinstance(predict_X, pd.DataFrame), "Input should be DataFrame"

            # Check for realistic ranges
            if "Age" in predict_X.columns:
                assert predict_X["Age"].min() >= 18, "Age should be realistic"
                assert predict_X["Age"].max() <= 100, "Age should be realistic"

    def _test_binning_process_vs_scorecard_data_flow_removed(self, sample_medical_data):
        """
        REGRESSION TEST: Verify correct data flow between binning and scorecard.

        This test ensures that:
        1. BinningProcess.fit() receives raw data
        2. BinningProcess.transform() produces binned data
        3. Scorecard.fit() receives raw data (NOT binned data)
        4. Scorecard scoring methods receive raw data
        """
        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        model = OptbinningScorecardModel()

        # Track what data flows through the system
        binning_fit_data = None
        binning_transform_input = None
        binning_transform_output = None
        scorecard_fit_data = None

        original_binning_fit = BinningProcess.fit
        original_binning_transform = BinningProcess.transform
        original_scorecard_fit = Scorecard.fit

        def capture_binning_fit(self, X_input, y_input):
            nonlocal binning_fit_data
            binning_fit_data = X_input.copy()
            return original_binning_fit(self, X_input, y_input)

        def capture_binning_transform(self, X_input):
            nonlocal binning_transform_input, binning_transform_output
            binning_transform_input = X_input.copy()
            result = original_binning_transform(self, X_input)
            binning_transform_output = result.copy() if hasattr(result, "copy") else result
            return result

        def capture_scorecard_fit(self, X_input, y_input, **kwargs):
            nonlocal scorecard_fit_data
            scorecard_fit_data = X_input.copy()
            return original_scorecard_fit(self, X_input, y_input, **kwargs)

        with patch.object(BinningProcess, "fit", capture_binning_fit), patch.object(
            BinningProcess, "transform", capture_binning_transform
        ), patch.object(Scorecard, "fit", capture_scorecard_fit):
            model.fit(X, y, hyperparameter_search=False)

        # Verify data flow
        assert binning_fit_data is not None, "BinningProcess.fit should have been called"
        assert scorecard_fit_data is not None, "Scorecard.fit should have been called"

        # CRITICAL: Scorecard should receive the SAME data as BinningProcess
        # (both should get raw data)
        pd.testing.assert_frame_equal(
            binning_fit_data.sort_index(),
            scorecard_fit_data.sort_index(),
            "BinningProcess.fit and Scorecard.fit should receive identical raw data",
        )

        # Verify binning actually transforms data (produces different output)
        if binning_transform_input is not None and binning_transform_output is not None:
            # Transform output should be different from input (binned vs raw)
            if isinstance(binning_transform_output, np.ndarray):
                binned_df = pd.DataFrame(binning_transform_output, columns=features)
            else:
                binned_df = binning_transform_output

            # Binned data should have different characteristics than raw data
            # (usually integers representing bin assignments)
            for col in features:
                if col in X.columns and X[col].dtype in ["float64", "int64"]:
                    raw_unique = len(X[col].dropna().unique())
                    binned_unique = (
                        len(binned_df[col].dropna().unique())
                        if col in binned_df.columns
                        else raw_unique
                    )

                    # Binning should reduce the number of unique values
                    # (unless the feature already had very few unique values)
                    if raw_unique > 10:
                        assert (
                            binned_unique < raw_unique
                        ), f"Binning should reduce unique values for {col}"

    def test_regression_unique_scores_count(self, trained_scorecard_model):
        """
        REGRESSION TEST: Ensure we get more than 2 unique scores.

        The original bug caused models to only produce 2 unique scores (54 and 55).
        This test ensures we have sufficient score granularity.
        """
        model, X, y = trained_scorecard_model

        scores = model.score(X)
        unique_scores = np.unique(scores)

        # CRITICAL: We should have more than 2 unique scores
        assert (
            len(unique_scores) > 2
        ), f"Expected >2 unique scores, got {len(unique_scores)}: {unique_scores}"

        # Should have reasonable score range
        assert scores.min() >= 0, "Scores should be non-negative"
        assert scores.max() <= 100, "Scores should not exceed 100"

        # Should have reasonable score distribution
        score_std = scores.std()
        assert (
            score_std > 1
        ), f"Score standard deviation too low: {score_std}, indicates limited variability"

    def test_probabilities_distribution(self, trained_scorecard_model):
        """
        REGRESSION TEST: Ensure probability predictions have reasonable distribution.

        The binned data bug also affected probability predictions.
        """
        model, X, y = trained_scorecard_model

        probas = model.predict_proba(X)

        # Extract positive class probabilities
        if len(probas.shape) > 1 and probas.shape[1] > 1:
            pos_probas = probas[:, 1]
        else:
            pos_probas = probas.flatten()

        unique_probas = np.unique(pos_probas)

        # Should have more than 2 unique probability values
        assert len(unique_probas) > 2, f"Expected >2 unique probabilities, got {len(unique_probas)}"

        # Probabilities should be in valid range
        assert np.all(pos_probas >= 0), "Probabilities should be >= 0"
        assert np.all(pos_probas <= 1), "Probabilities should be <= 1"

        # Should have reasonable spread
        proba_std = pos_probas.std()
        assert proba_std > 0.001, f"Probability standard deviation too low: {proba_std}"


@pytest.mark.regression
@pytest.mark.optbinning
class TestDataTypeConsistency:
    """Test that data types are preserved correctly through the pipeline."""

    def test_data_types_preserved_through_pipeline(self, sample_medical_data):
        """Ensure data types don't get corrupted in the binning/scoring pipeline."""
        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Record original dtypes
        original_dtypes = X.dtypes.copy()

        model = OptbinningScorecardModel()
        model.fit(X, y, hyperparameter_search=False)

        # Test that scoring preserves input data
        X_before_scoring = X.copy()
        scores = model.score(X)

        # Input DataFrame should be unchanged
        pd.testing.assert_frame_equal(X, X_before_scoring)
        assert X.dtypes.equals(original_dtypes), "Input data types should be preserved"

    def test_missing_values_handled_correctly(self, sample_medical_data):
        """Test that missing values are preserved for OptBinning but handled appropriately."""
        data = sample_medical_data.copy()

        # Add more missing values for testing
        data.loc[0:50, "AlbuminLevel"] = np.nan
        data.loc[100:150, "BMI"] = np.nan

        features = ["Age", "BMI", "AlbuminLevel", "ASAStatus"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets only
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Count missing values before
        missing_before = X.isnull().sum().sum()
        assert missing_before > 0, "Should have missing values for this test"

        model = OptbinningScorecardModel()
        model.fit(X, y, hyperparameter_search=False)

        # Model should handle missing values and produce scores for all rows
        scores = model.score(X)
        assert len(scores) == len(
            X
        ), "Should produce scores for all rows including those with missing values"
        assert not np.isnan(scores).any(), "Scores should not contain NaN values"
