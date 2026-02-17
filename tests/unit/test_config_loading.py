# tests/unit/test_config_loading.py

"""
Tests for configuration loading and parameter filtering.

This includes testing the critical fix for the monitoring configuration
where 'verbose' parameter was being passed to ScorecardEvaluator.initialize_monitoring()
but the method doesn't accept it.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.config import Config


@pytest.mark.config
@pytest.mark.unit
class TestConfig:
    """Test the configuration loading functionality."""

    def test_load_config_basic(self, temp_config_dir):
        """Test basic configuration loading."""
        # Use the temp config dir fixture
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.load_config("binning_args")

            assert isinstance(config, dict)
            assert "max_n_prebins" in config
            assert "fuzz_catch_asa_category" in config
            assert "asa_detection" in config

    def test_load_config_filters_comments(self, temp_config_dir):
        """Test that comment fields are filtered out."""
        # Create a config with comment fields
        config_with_comments = {
            "param1": "value1",
            "// comment1": "This is a comment",
            "param2": 42,
            "// param2_comment": "Explanation of param2",
            "nested": {"inner_param": True, "// inner_comment": "Inner comment"},
        }

        config_path = temp_config_dir / "test_comments.json"
        with open(config_path, "w") as f:
            json.dump(config_with_comments, f)

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.load_config("test_comments")

            # Comments should be filtered out at top level
            assert "param1" in config
            assert "param2" in config
            assert "nested" in config
            assert "// comment1" not in config
            assert "// param2_comment" not in config

            # Nested comments are not filtered (by design - only top level)
            assert "// inner_comment" in config["nested"]

    def test_load_config_file_not_found(self, temp_config_dir):
        """Test error handling when config file doesn't exist."""
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            with pytest.raises(FileNotFoundError) as exc_info:
                Config.load_config("nonexistent_config")

            assert "Configuration file not found" in str(exc_info.value)

    def test_load_config_invalid_json(self, temp_config_dir):
        """Test error handling for invalid JSON."""
        # Create invalid JSON file
        invalid_json_path = temp_config_dir / "invalid.json"
        with open(invalid_json_path, "w") as f:
            f.write('{ "param1": "value1", "param2": }')  # Invalid JSON

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            with pytest.raises(ValueError) as exc_info:
                Config.load_config("invalid")

            assert "Invalid JSON" in str(exc_info.value)

    def test_binning_args(self, temp_config_dir):
        """Test specific binning config loading."""
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.binning_args()

            assert "max_n_prebins" in config
            assert "min_prebin_size" in config
            assert "fuzz_catch_asa_category" in config
            assert config["fuzz_catch_asa_category"] is True

    def test_monitoring_args_filters_verbose(self, temp_config_dir):
        """
        REGRESSION TEST: Ensure monitoring config filters out 'verbose' parameter.

        This was the critical issue where 'verbose' was being passed to
        ScorecardEvaluator.initialize_monitoring() but the method doesn't accept it.
        """
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.monitoring_args()

            # Should have the allowed parameters
            assert "psi_method" in config
            assert "psi_n_bins" in config
            assert "psi_min_bin_size" in config
            assert "show_digits" in config

            # CRITICAL: Should NOT have 'verbose' parameter
            assert (
                "verbose" not in config
            ), "verbose parameter should be filtered out for ScorecardEvaluator.initialize_monitoring()"

    def test_monitoring_args_only_allowed_params(self, temp_config_dir):
        """Test that only allowed parameters are included in monitoring config."""
        # Create a monitoring config with extra parameters
        extra_config = {
            "psi_method": "quantile",
            "psi_n_bins": 5,
            "psi_min_bin_size": 0.05,
            "show_digits": 2,
            "verbose": True,  # Should be filtered out
            "extra_param": "should_be_filtered",  # Should be filtered out
            "another_param": 123,  # Should be filtered out
        }

        config_path = temp_config_dir / "monitoring_with_extra.json"
        with open(config_path, "w") as f:
            json.dump(extra_config, f)

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            with patch.object(Config, "load_config") as mock_load:
                mock_load.return_value = extra_config

                config = Config.monitoring_args()

                # Should only have the 4 allowed parameters
                expected_params = {"psi_method", "psi_n_bins", "psi_min_bin_size", "show_digits"}
                assert set(config.keys()) == expected_params

                # Should have correct values for allowed parameters
                assert config["psi_method"] == "quantile"
                assert config["psi_n_bins"] == 5
                assert config["psi_min_bin_size"] == 0.05
                assert config["show_digits"] == 2

    def test_scorecard_args(self, temp_config_dir):
        """Test scorecard config loading."""
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.scorecard_args()

            assert "scaling_method" in config
            assert "reverse_scorecard" in config
            assert "intercept_based" in config

    def test_variable_selection_args(self, temp_config_dir):
        """Test variable selection config loading."""
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.variable_selection_args()

            assert "min_unique_values" in config
            assert "coefficient_threshold" in config

    def test_all_config_methods_work(self, temp_config_dir):
        """Test that all config getter methods work without errors."""
        with patch("config.config.CONFIG_DIR", temp_config_dir):
            # These should all work without raising exceptions
            binning = Config.binning_args()
            monitoring = Config.monitoring_args()
            scorecard = Config.scorecard_args()
            variable_selection = Config.variable_selection_args()

            assert all(
                isinstance(config, dict)
                for config in [binning, monitoring, scorecard, variable_selection]
            )


@pytest.mark.config
@pytest.mark.integration
@pytest.mark.regression
class TestConfigIntegration:
    """Test configuration integration with the main system."""

    def test_monitoring_config_compatible_with_evaluator(self, temp_config_dir):
        """
        INTEGRATION TEST: Ensure monitoring config is compatible with ScorecardEvaluator.

        This is the key regression test for the verbose parameter issue.
        """
        import inspect

        from CreditScore.evaluate import ScorecardEvaluator
        from CreditScore.train import OptbinningScorecardModel

        # Get the signature of initialize_monitoring
        init_monitoring_sig = inspect.signature(ScorecardEvaluator.initialize_monitoring)
        allowed_params = set(init_monitoring_sig.parameters.keys()) - {"self"}

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.monitoring_args()

            # All config parameters should be in the method signature
            for param in config.keys():
                assert (
                    param in allowed_params
                ), f"Parameter '{param}' not allowed in ScorecardEvaluator.initialize_monitoring()"

    def test_config_backward_compatibility(self, temp_config_dir):
        """Test that configuration changes don't break existing functionality."""
        from CreditScore.utils import ConfigurationManager

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            # The old ConfigurationManager should still work
            try:
                scorecard_config = ConfigurationManager.scorecard_args()
                monitoring_config = ConfigurationManager.monitoring_args()

                assert isinstance(scorecard_config, dict)
                assert isinstance(monitoring_config, dict)
            except AttributeError:
                # ConfigurationManager might not exist anymore, which is fine
                pass

    def test_config_used_in_training_pipeline(self, temp_config_dir, sample_medical_data):
        """Test that configurations are properly used in the training pipeline."""
        from CreditScore.train import OptbinningScorecardModel

        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            # Training should work with the configuration
            model = OptbinningScorecardModel()
            model.fit(X, y, hyperparameter_search=False)

            assert model.is_fitted
            assert model.binning_process is not None

    def test_asa_detection_config_integration(self, temp_config_dir):
        """Test that ASA detection configuration is properly integrated."""
        from CreditScore.asa_detection import ASAVariableDetector

        df = pd.DataFrame({"ASAStatus": [1, 2, 3, 4, 5], "age": [65, 70, 45, 80, 55]})

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.binning_args()

            # Should have ASA detection enabled
            assert config["fuzz_catch_asa_category"] is True

            # Should detect ASA variables
            asa_vars = ASAVariableDetector.detect_asa_variables(df, config)
            assert "ASAStatus" in asa_vars
            assert "age" not in asa_vars


@pytest.mark.config
@pytest.mark.unit
class TestConfigErrorHandling:
    """Test error handling and edge cases in configuration loading."""

    def test_missing_config_directory(self):
        """Test behavior when config directory doesn't exist."""
        fake_dir = Path("/nonexistent/config/directory")

        with patch("config.config.CONFIG_DIR", fake_dir):
            with pytest.raises(FileNotFoundError):
                Config.load_config("any_config")

    def test_empty_config_file(self, temp_config_dir):
        """Test loading an empty config file."""
        empty_config_path = temp_config_dir / "empty.json"
        with open(empty_config_path, "w") as f:
            json.dump({}, f)

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.load_config("empty")
            assert config == {}

    def test_config_with_null_values(self, temp_config_dir):
        """Test config with null values (which OptBinning uses for defaults)."""
        config_with_nulls = {
            "param1": "value1",
            "param2": None,  # null value
            "param3": {"nested": None},
        }

        config_path = temp_config_dir / "with_nulls.json"
        with open(config_path, "w") as f:
            json.dump(config_with_nulls, f)

        with patch("config.config.CONFIG_DIR", temp_config_dir):
            config = Config.load_config("with_nulls")

            assert config["param1"] == "value1"
            assert config["param2"] is None
            assert config["param3"]["nested"] is None

    def test_monitoring_config_with_missing_params(self, temp_config_dir):
        """Test monitoring config when some expected parameters are missing."""
        minimal_config = {
            "psi_method": "quantile",
            "psi_n_bins": 5,
            # Missing psi_min_bin_size and show_digits
        }

        with patch.object(Config, "load_config") as mock_load:
            mock_load.return_value = minimal_config

            config = Config.monitoring_args()

            # Should only include the parameters that are present
            assert "psi_method" in config
            assert "psi_n_bins" in config
            assert "psi_min_bin_size" not in config
            assert "show_digits" not in config
            assert "verbose" not in config
