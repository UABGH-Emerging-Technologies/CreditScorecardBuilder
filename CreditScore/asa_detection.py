# CreditScore/asa_detection.py

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class ASAVariableDetector:
    """
    Utility class for automatically detecting and handling ASA status variables.

    ASA (American Society of Anesthesiologists) status is a categorical variable
    that rates the physical condition of patients before surgery (ASA I-VI).
    It's often encoded numerically but should be treated as categorical.
    """

    @staticmethod
    def detect_asa_variables(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
        """
        Detect ASA status variables in a DataFrame.

        Args:
            df: DataFrame to check
            config: Configuration dictionary with ASA detection settings

        Returns:
            List of column names that appear to be ASA variables
        """
        if not config.get("fuzz_catch_asa_category", False):
            return []

        asa_detection = config.get("asa_detection", {})
        name_contains = asa_detection.get("name_contains", "asa").lower()
        max_unique = asa_detection.get("max_unique_values", 6)

        asa_variables = []

        for col in df.columns:
            col_lower = col.lower()
            if name_contains in col_lower:
                # Check if it's numeric (could be ASA encoded as numbers)
                if df[col].dtype in ["int64", "float64"]:
                    unique_count = df[col].nunique()
                    if unique_count <= max_unique:
                        asa_variables.append(col)
                        logger.info(
                            f"Detected ASA variable: '{col}' ({unique_count} unique values)"
                        )

        return asa_variables

    @staticmethod
    def update_categorical_lists(
        categorical_features: List[str], numerical_features: List[str], asa_variables: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Update categorical and numerical feature lists to treat ASA variables as categorical.

        Args:
            categorical_features: Current list of categorical features
            numerical_features: Current list of numerical features
            asa_variables: List of detected ASA variables

        Returns:
            Tuple of (updated_categorical_features, updated_numerical_features)
        """
        updated_categorical = categorical_features.copy()
        updated_numerical = numerical_features.copy()

        for asa_var in asa_variables:
            if asa_var in updated_numerical:
                updated_numerical.remove(asa_var)
                updated_categorical.append(asa_var)
                logger.info(f"Moved '{asa_var}' from numerical to categorical features")

        return updated_categorical, updated_numerical

    @staticmethod
    def get_asa_info_summary(asa_variables: List[str], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary information about detected ASA variables.

        Args:
            asa_variables: List of ASA variable names
            df: DataFrame containing the variables

        Returns:
            Dictionary with ASA variable information
        """
        asa_info = {}

        for var in asa_variables:
            if var in df.columns:
                unique_vals = sorted(df[var].dropna().unique())
                value_counts = df[var].value_counts().to_dict()

                asa_info[var] = {
                    "unique_values": unique_vals,
                    "n_unique": len(unique_vals),
                    "value_counts": value_counts,
                    "missing_count": df[var].isnull().sum(),
                    "dtype": str(df[var].dtype),
                }

        return asa_info

    @staticmethod
    def validate_asa_values(df: pd.DataFrame, asa_variables: List[str]) -> Dict[str, List[str]]:
        """
        Validate that ASA variables contain reasonable values.

        Args:
            df: DataFrame containing ASA variables
            asa_variables: List of ASA variable names

        Returns:
            Dictionary of warnings for each variable
        """
        warnings = {}

        # Expected ASA values (can be 1-6 or I-VI)
        expected_numeric = {1, 2, 3, 4, 5, 6}
        expected_roman = {"I", "II", "III", "IV", "V", "VI"}

        for var in asa_variables:
            var_warnings = []

            if var not in df.columns:
                continue

            unique_vals = set(df[var].dropna().unique())

            # Check for unexpected values
            if df[var].dtype in ["int64", "float64"]:
                unexpected = unique_vals - expected_numeric
                if unexpected:
                    var_warnings.append(f"Unexpected numeric values: {unexpected}")

                # Check for 0 (sometimes used for missing)
                if 0 in unique_vals:
                    var_warnings.append("Contains 0 - consider if this represents missing data")

                # Check for values > 6
                high_vals = [v for v in unique_vals if isinstance(v, (int, float)) and v > 6]
                if high_vals:
                    var_warnings.append(
                        f"Values > 6 detected: {high_vals} - unusual for ASA status"
                    )

            else:
                # String/object type
                unexpected = unique_vals - expected_roman
                if unexpected:
                    var_warnings.append(f"Unexpected string values: {unexpected}")

            if var_warnings:
                warnings[var] = var_warnings

        return warnings
