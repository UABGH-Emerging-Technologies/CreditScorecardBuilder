# tests/conftest.py

import shutil

# Add the project root to the Python path
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from config.config import Config
from CreditScore.data import DataLoader, DataPreprocessor
from CreditScore.train import OptbinningScorecardModel

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_medical_data() -> pd.DataFrame:
    """Create sample medical data for testing."""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "patient_id": range(n_samples),
            "Age": np.random.normal(65, 15, n_samples).clip(18, 100),
            "BMI": np.random.normal(28, 5, n_samples).clip(15, 50),
            "Gender": np.random.choice(["M", "F"], n_samples),
            "ASAStatus": np.random.choice(
                [1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]
            ),
            "asa_physical_status": np.random.choice(
                [1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1]
            ),
            "EBL": np.random.lognormal(4, 1.5, n_samples).clip(0, 5000),
            "AlbuminLevel": np.random.normal(3.5, 0.8, n_samples).clip(1.0, 5.0),
            "PreOpHemoglobin": np.random.normal(12, 2, n_samples).clip(6, 18),
            "CrystalloidsTotal": np.random.lognormal(6, 1, n_samples).clip(0, 10000),
            "EMERGENCY": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "CVL": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            "thirty_day_mortality": np.random.choice(
                [0, 1], n_samples, p=[0.987, 0.013]
            ),  # 1.3% positive rate
        }
    )

    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data.loc[missing_indices, "AlbuminLevel"] = None

    missing_indices = np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    data.loc[missing_indices, "PreOpHemoglobin"] = None

    # Add some edge cases for ASA detection
    data.loc[data.index[:50], "ASAStatus"] = np.nan  # Missing ASA values

    return data


@pytest.fixture
def sample_data_with_issues() -> pd.DataFrame:
    """Create sample data with common data quality issues."""
    np.random.seed(123)
    n_samples = 500

    data = pd.DataFrame(
        {
            "constant_feature": [1] * n_samples,  # Constant feature (should be removed)
            "almost_constant": [1] * (n_samples - 5) + [0] * 5,  # Almost constant
            "high_missing": [1.0] * 50 + [np.nan] * (n_samples - 50),  # High missing rate
            "normal_feature": np.random.normal(0, 1, n_samples),
            "categorical_feature": np.random.choice(["A", "B", "C"], n_samples),
            "binary_feature": np.random.choice([0, 1], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        }
    )

    return data


@pytest.fixture
def temp_config_dir() -> Path:
    """Create temporary config directory with test configurations."""
    temp_dir = Path(tempfile.mkdtemp())
    config_dir = temp_dir / "config"
    config_dir.mkdir()

    # Create test configuration files
    binning_config = {
        "max_n_prebins": 20,
        "min_prebin_size": 0.05,
        "min_n_bins": 2,
        "max_n_bins": 5,
        "min_bin_size": 0.01,
        "verbose": True,
        "fuzz_catch_asa_category": True,
        "asa_detection": {"name_contains": "asa", "max_unique_values": 6},
    }

    monitoring_config = {
        "psi_method": "quantile",
        "psi_n_bins": 5,
        "psi_min_bin_size": 0.05,
        "show_digits": 2,
        "verbose": False,  # This should be filtered out
    }

    scorecard_config = {
        "scaling_method": "min_max",
        "scaling_method_params": {"min": 0, "max": 100},
        "reverse_scorecard": True,
        "intercept_based": True,
        "rounding": True,
    }

    variable_selection_config = {"min_unique_values": 2, "coefficient_threshold": 1e-6}

    # Write config files
    import json

    with open(config_dir / "binning_args.json", "w") as f:
        json.dump(binning_config, f, indent=2)

    with open(config_dir / "monitoring_args.json", "w") as f:
        json.dump(monitoring_config, f, indent=2)

    with open(config_dir / "scorecard_args.json", "w") as f:
        json.dump(scorecard_config, f, indent=2)

    with open(config_dir / "variable_selection_args.json", "w") as f:
        json.dump(variable_selection_config, f, indent=2)

    yield config_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def trained_scorecard_model(
    sample_medical_data,
) -> Tuple[OptbinningScorecardModel, pd.DataFrame, pd.Series]:
    """Create a trained scorecard model for testing."""
    data = sample_medical_data

    # Prepare data
    features = ["Age", "BMI", "ASAStatus", "EBL", "AlbuminLevel", "PreOpHemoglobin"]
    target = "thirty_day_mortality"

    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_modeling_data(
        data,
        target,
        features,
        drop_missing=False,  # OptBinning handles missing
        validate_ranges=True,
    )
    y = preprocessor.encode_target(y)

    # Remove rows with missing target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # Create and train model
    model = OptbinningScorecardModel()
    model.fit(X, y, hyperparameter_search=False)  # Skip hyperparameter search for speed

    return model, X, y


@pytest.fixture
def mock_config_loader(temp_config_dir, monkeypatch):
    """Mock the config loader to use temporary config directory."""

    def mock_load_config(config_name: str) -> Dict[str, Any]:
        import json

        config_path = temp_config_dir / f"{config_name}.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        # Remove comment fields
        return {k: v for k, v in config.items() if not k.startswith("//")}

    monkeypatch.setattr(Config, "load_config", mock_load_config)
    return Config
