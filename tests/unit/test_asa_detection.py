# tests/unit/test_asa_detection.py

"""
Tests for ASA (American Society of Anesthesiologists) status variable auto-detection.

This functionality automatically detects variables that contain "asa" in their name
and have ≤6 unique values, then treats them as categorical rather than numerical.
"""

import numpy as np
import pandas as pd
import pytest

from CreditScore.asa_detection import ASAVariableDetector


@pytest.mark.asa
@pytest.mark.unit
class TestASAVariableDetector:
    """Test the ASA variable detection functionality."""

    def test_detect_asa_variables_basic(self):
        """Test basic ASA variable detection."""
        df = pd.DataFrame(
            {
                "ASAStatus": [1, 2, 3, 4, 5, 1, 2, 3],
                "asa_physical_status": [1, 2, 3, 1, 2, 3, 1, 2],
                "Age": [65, 70, 45, 80, 55, 60, 75, 50],
                "BMI": [25.5, 30.2, 22.1, 28.7, 26.3, 24.8, 29.1, 23.4],
                "NotASA": [1, 2, 3, 4, 5, 6, 7, 8],  # Contains >6 unique values
            }
        )

        config = {
            "fuzz_catch_asa_category": True,
            "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
        }

        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)

        # Should detect both ASAStatus and asa_physical_status
        assert "ASAStatus" in asa_vars
        assert "asa_physical_status" in asa_vars
        assert "Age" not in asa_vars
        assert "BMI" not in asa_vars
        assert "NotASA" not in asa_vars
        assert len(asa_vars) == 2

    def test_detect_asa_variables_case_insensitive(self):
        """Test that ASA detection is case-insensitive."""
        df = pd.DataFrame(
            {
                "ASAStatus": [1, 2, 3, 4, 5],
                "asa_status": [1, 2, 3, 4, 5],
                "Asa_Physical": [1, 2, 3, 4, 5],
                "my_ASA_score": [1, 2, 3, 4, 5],
                "preop_asa": [1, 2, 3, 4, 5],
            }
        )

        config = {
            "fuzz_catch_asa_category": True,
            "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
        }

        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)

        # All should be detected
        expected_vars = ["ASAStatus", "asa_status", "Asa_Physical", "my_ASA_score", "preop_asa"]
        for var in expected_vars:
            assert var in asa_vars
        assert len(asa_vars) == 5

    def test_detect_asa_variables_max_unique_threshold(self):
        """Test the maximum unique values threshold."""
        df = pd.DataFrame(
            {
                "asa_low_unique": [1, 2, 3, 1, 2, 3, 1, 2],  # 3 unique values
                "asa_medium_unique": [1, 2, 3, 4, 5, 6, 1, 2],  # 6 unique values (at threshold)
                "asa_high_unique": [1, 2, 3, 4, 5, 6, 7, 8],  # 8 unique values (above threshold)
            }
        )

        config = {
            "fuzz_catch_asa_category": True,
            "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
        }

        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)

        assert "asa_low_unique" in asa_vars
        assert "asa_medium_unique" in asa_vars
        assert "asa_high_unique" not in asa_vars
        assert len(asa_vars) == 2

    def test_detect_asa_variables_disabled(self):
        """Test that detection is disabled when flag is False."""
        df = pd.DataFrame({"ASAStatus": [1, 2, 3, 4, 5], "asa_physical_status": [1, 2, 3, 4, 5]})

        config = {
            "fuzz_catch_asa_category": False,  # Disabled
            "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
        }

        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)

        assert len(asa_vars) == 0

    def test_detect_asa_variables_only_numeric(self):
        """Test that only numeric columns are considered for ASA detection."""
        df = pd.DataFrame(
            {
                "ASAStatus": [1, 2, 3, 4, 5],  # Numeric - should be detected
                "asa_text": ["I", "II", "III", "IV", "V"],  # String - should not be detected
                "asa_mixed": [1, 2, "III", 4, 5],  # Mixed - should not be detected
            }
        )

        config = {
            "fuzz_catch_asa_category": True,
            "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
        }

        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)

        assert "ASAStatus" in asa_vars
        assert "asa_text" not in asa_vars
        assert "asa_mixed" not in asa_vars
        assert len(asa_vars) == 1

    def test_update_categorical_lists(self):
        """Test updating categorical and numerical feature lists."""
        categorical_features = ["Gender", "Surgery_Type"]
        numerical_features = ["Age", "BMI", "ASAStatus", "EBL"]
        asa_variables = ["ASAStatus"]

        updated_cat, updated_num = ASAVariableDetector.update_categorical_lists(
            categorical_features, numerical_features, asa_variables
        )

        # ASAStatus should move from numerical to categorical
        assert "ASAStatus" in updated_cat
        assert "ASAStatus" not in updated_num

        # Other features should remain unchanged
        assert "Gender" in updated_cat
        assert "Surgery_Type" in updated_cat
        assert "Age" in updated_num
        assert "BMI" in updated_num
        assert "EBL" in updated_num

    def test_update_categorical_lists_multiple_asa_vars(self):
        """Test updating lists with multiple ASA variables."""
        categorical_features = ["Gender"]
        numerical_features = ["Age", "ASAStatus", "asa_physical_status", "BMI"]
        asa_variables = ["ASAStatus", "asa_physical_status"]

        updated_cat, updated_num = ASAVariableDetector.update_categorical_lists(
            categorical_features, numerical_features, asa_variables
        )

        # Both ASA variables should move to categorical
        assert "ASAStatus" in updated_cat
        assert "asa_physical_status" in updated_cat
        assert "ASAStatus" not in updated_num
        assert "asa_physical_status" not in updated_num

        # Non-ASA features should remain in their original lists
        assert "Gender" in updated_cat
        assert "Age" in updated_num
        assert "BMI" in updated_num

    def test_get_asa_info_summary(self):
        """Test getting summary information about ASA variables."""
        df = pd.DataFrame(
            {
                "ASAStatus": [1, 2, 3, 4, 5, 1, 2, np.nan],
                "asa_physical_status": [1, 2, 3, 1, 2, 3, 1, 2],
            }
        )

        asa_variables = ["ASAStatus", "asa_physical_status"]
        info = ASAVariableDetector.get_asa_info_summary(asa_variables, df)

        # Check ASAStatus info
        assert "ASAStatus" in info
        asa_info = info["ASAStatus"]
        assert asa_info["unique_values"] == [1, 2, 3, 4, 5]  # Should be sorted
        assert asa_info["n_unique"] == 5
        assert asa_info["missing_count"] == 1
        assert "1" in str(asa_info["value_counts"]) or 1 in asa_info["value_counts"]

        # Check asa_physical_status info
        assert "asa_physical_status" in info
        asa_phys_info = info["asa_physical_status"]
        assert asa_phys_info["unique_values"] == [1, 2, 3]
        assert asa_phys_info["n_unique"] == 3
        assert asa_phys_info["missing_count"] == 0

    def test_validate_asa_values_valid_numeric(self):
        """Test validation of valid ASA numeric values."""
        df = pd.DataFrame(
            {
                "ASAStatus": [1, 2, 3, 4, 5, 6],  # Valid ASA values
                "asa_clean": [1, 2, 3, 1, 2, 3],  # Valid subset
            }
        )

        asa_variables = ["ASAStatus", "asa_clean"]
        warnings = ASAVariableDetector.validate_asa_values(df, asa_variables)

        # Should have no warnings for valid values
        assert len(warnings) == 0

    def test_validate_asa_values_invalid_numeric(self):
        """Test validation detects invalid ASA numeric values."""
        df = pd.DataFrame(
            {
                "ASAStatus": [0, 1, 2, 3, 7, 8],  # Contains 0, 7, 8 (invalid)
                "asa_weird": [1, 2, 3, 10, 15, 1],  # Contains 10, 15 (invalid)
            }
        )

        asa_variables = ["ASAStatus", "asa_weird"]
        warnings = ASAVariableDetector.validate_asa_values(df, asa_variables)

        # Should detect warnings
        assert "ASAStatus" in warnings
        assert "asa_weird" in warnings

        # Check specific warning types
        asa_warnings = warnings["ASAStatus"]
        assert any("0" in warning for warning in asa_warnings)  # Should warn about 0
        assert any(
            "7" in warning or "8" in warning for warning in asa_warnings
        )  # Should warn about >6

        weird_warnings = warnings["asa_weird"]
        assert any("10" in warning or "15" in warning for warning in weird_warnings)

    def test_validate_asa_values_string_values(self):
        """Test validation of string ASA values."""
        df = pd.DataFrame(
            {
                "asa_roman": ["I", "II", "III", "IV", "V", "VI"],  # Valid Roman numerals
                "asa_invalid_strings": [
                    "I",
                    "II",
                    "Bad",
                    "Value",
                    "VII",
                    "Extra",
                ],  # Invalid strings - need 6 items to match
            }
        )

        asa_variables = ["asa_roman", "asa_invalid_strings"]
        warnings = ASAVariableDetector.validate_asa_values(df, asa_variables)

        # Valid Roman numerals should have no warnings
        assert "asa_roman" not in warnings

        # Invalid strings should have warnings
        assert "asa_invalid_strings" in warnings
        invalid_warnings = warnings["asa_invalid_strings"]
        assert any(
            "Bad" in warning or "Value" in warning or "VII" in warning
            for warning in invalid_warnings
        )

    def test_missing_asa_values_handling(self):
        """Test that missing ASA values are handled properly."""
        df = pd.DataFrame(
            {
                "ASAStatus": [1, 2, np.nan, 4, 5, np.nan],
                "asa_mostly_missing": [np.nan, np.nan, np.nan, 1, np.nan, np.nan],
            }
        )

        config = {
            "fuzz_catch_asa_category": True,
            "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
        }

        # Should still detect despite missing values
        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)
        assert "ASAStatus" in asa_vars
        assert "asa_mostly_missing" in asa_vars

        # Info summary should handle missing values
        info = ASAVariableDetector.get_asa_info_summary(asa_vars, df)
        assert info["ASAStatus"]["missing_count"] == 2
        assert info["asa_mostly_missing"]["missing_count"] == 5


@pytest.mark.asa
@pytest.mark.integration
class TestASADetectionIntegration:
    """Integration tests for ASA detection with the training pipeline."""

    def test_asa_detection_in_binning_process(self, sample_medical_data):
        """Test that ASA detection works in the actual binning process."""
        from CreditScore.train import OptbinningScorecardModel

        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus", "asa_physical_status"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Check initial data types
        assert X["ASAStatus"].dtype in ["int64", "float64"], "ASAStatus should start as numeric"
        assert X["asa_physical_status"].dtype in [
            "int64",
            "float64",
        ], "asa_physical_status should start as numeric"

        # Create model with ASA detection enabled
        model = OptbinningScorecardModel()

        # Test that the model can be trained successfully with ASA detection
        model.fit(X, y, hyperparameter_search=False)

        # Model should be fitted successfully
        assert model.is_fitted
        assert model.binning_process is not None
        assert model.scorecard is not None

        # Should be able to generate scores
        scores = model.score(X)
        assert len(scores) == len(X)
        assert not np.isnan(scores).any()

    def test_config_integration_with_asa_detection(self, mock_config_loader):
        """Test that ASA detection configuration is properly loaded and used."""
        # This uses the mock config loader with ASA detection enabled
        config = mock_config_loader.binning_args()

        assert config["fuzz_catch_asa_category"] is True
        assert "asa_detection" in config
        assert config["asa_detection"]["name_contains"] == "asa"
        assert config["asa_detection"]["max_unique_values"] == 6

        # Test with sample data
        df = pd.DataFrame(
            {"ASAStatus": [1, 2, 3, 4, 5], "age": [65, 70, 45, 80, 55], "target": [0, 1, 0, 1, 0]}
        )

        asa_vars = ASAVariableDetector.detect_asa_variables(df, config)
        assert "ASAStatus" in asa_vars
        assert "age" not in asa_vars
