"""Data loading, preprocessing, splitting, and validation utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from various file formats."""

    @staticmethod
    def load_file(file_path: Union[str, Path], file_obj=None) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.

        Args:
            file_path: Path to the file or filename if using file_obj
            file_obj: Optional file object (for Streamlit uploads)

        Returns:
            pd.DataFrame: Loaded data
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Handle Streamlit file upload case
        if file_obj is not None:
            extension = Path(file_path).suffix.lower()
        else:
            extension = file_path.suffix.lower()

        try:
            if extension == ".csv":
                if file_obj:
                    file_obj.seek(0)
                    return pd.read_csv(file_obj)
                return pd.read_csv(file_path)
            elif extension in [".xlsx", ".xls"]:
                if file_obj:
                    file_obj.seek(0)
                    return pd.read_excel(file_obj)
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {extension}")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise


class DataPreprocessor:
    """Handles common data preprocessing tasks."""

    def __init__(self):
        self.missing_values = ["_", "NA", "N/A", "n/a", "", " ", "null", "NULL"]
        self.label_encoders = {}

    def handle_missing_values(self, df: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Args:
            df: Input dataframe
            drop_missing: Whether to drop rows with missing values

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        initial_rows = len(df)
        logger.info(f"Starting with {initial_rows} rows")

        # Replace common missing value representations with NaN
        df = df.replace(self.missing_values, np.nan)

        # Log missing value counts
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.info(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")

        # Drop or handle missing values
        if drop_missing:
            df = df.dropna()
            logger.info(
                f"Dropped {initial_rows - len(df)} rows with missing values. Remaining rows: {len(df)}"
            )
        else:
            logger.info(f"Preserving missing values. Total rows: {len(df)}")

        return df

    def prepare_modeling_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        drop_missing: bool = False,
        validate_ranges: bool = True,
        clip_negative: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling by selecting columns and handling missing values.

        Args:
            df: Input dataframe
            target_column: Name of the target column
            feature_columns: List of feature column names
            drop_missing: Whether to drop rows with missing values
            validate_ranges: Whether to validate data ranges
            clip_negative: Whether to clip negative values to 0 for age/duration columns

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        """
        logger.info(f"prepare_modeling_data: Starting with {len(df)} rows")
        logger.info(f"drop_missing parameter: {drop_missing}")

        # Select relevant columns
        modeling_df = df[feature_columns + [target_column]].copy()
        logger.info(f"After column selection: {len(modeling_df)} rows")

        # Handle missing values
        modeling_df = self.handle_missing_values(modeling_df, drop_missing)
        logger.info(f"After handle_missing_values: {len(modeling_df)} rows")

        # Validate data ranges if requested
        if validate_ranges:
            self._validate_data_ranges(modeling_df, feature_columns)

        # Clip negative values if requested
        if clip_negative:
            modeling_df = self._clip_negative_values(modeling_df, feature_columns)

        # Separate features and target
        X = modeling_df[feature_columns]
        y = modeling_df[target_column]

        logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def _clip_negative_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Clip negative values to 0 for columns that shouldn't have negative values.

        Args:
            df: Dataframe to process
            feature_columns: List of feature columns

        Returns:
            pd.DataFrame: Dataframe with clipped values
        """
        df = df.copy()
        non_negative_patterns = ["age", "duration", "time", "count", "number"]

        for col in feature_columns:
            if df[col].dtype in ["int64", "float64"]:
                col_lower = col.lower()

                # Check if this column should be non-negative
                if any(pattern in col_lower for pattern in non_negative_patterns):
                    if (df[col] < 0).any():
                        n_clipped = (df[col] < 0).sum()
                        logger.info(f"Clipping {n_clipped} negative values in column '{col}' to 0")
                        df[col] = df[col].clip(lower=0)

        return df

    def _validate_data_ranges(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Validate that data values are in reasonable ranges.

        Args:
            df: Dataframe to validate
            feature_columns: List of feature columns to check
        """
        # Define columns that should not have negative values
        non_negative_patterns = ["age", "duration", "time", "count", "number"]

        for col in feature_columns:
            if df[col].dtype in ["int64", "float64"]:
                col_lower = col.lower()

                # Check if this column should be non-negative
                if any(pattern in col_lower for pattern in non_negative_patterns):
                    min_val = df[col].min()
                    if min_val < 0:
                        negative_count = (df[col] < 0).sum()
                        logger.warning(
                            f"Column '{col}' contains {negative_count} negative values "
                            f"(min: {min_val:.2f}). This may indicate data quality issues."
                        )

                        # Optionally show value distribution
                        if negative_count > 0:
                            logger.info(
                                f"Value range for '{col}': [{min_val:.2f}, {df[col].max():.2f}]"
                            )

    def encode_categorical_features(
        self,
        X: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        method: str = "label",
    ) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            X: Feature dataframe
            categorical_features: List of categorical feature names (auto-detected if None)
            method: Encoding method ('label' or 'onehot')

        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()

        # Auto-detect categorical features if not provided
        if categorical_features is None:
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

            num_cols = X.select_dtypes(include=["number"]).columns
            low_card_cols = [
                col for col in num_cols if X[col].nunique() <= Config.UNIQUE_CATEGORY_THRESHOLD
            ]
            # merge + dedupe
            categorical_features = []
            if low_card_cols:
                cat_cols = cat_cols + low_card_cols

            for c in cat_cols:
                if c not in categorical_features:
                    categorical_features.append(c)

        if method == "label":
            for col in categorical_features:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.label_encoders[col].fit_transform(X[col])
                else:
                    # Handle unseen categories
                    X_encoded[col] = X[col].apply(
                        lambda x: (
                            self.label_encoders[col].transform([x])[0]
                            if x in self.label_encoders[col].classes_
                            else -1
                        )
                    )

        elif method == "onehot":
            for col in categorical_features:
                n_unique = X[col].nunique()
                if n_unique > 2:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
                else:
                    # Binary categorical - just label encode
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col])

        return X_encoded

    def encode_target(self, y: pd.Series, binary: bool = True) -> pd.Series:
        """
        Encode target variable.

        Args:
            y: Target series
            binary: Whether this is a binary classification task

        Returns:
            pd.Series: Encoded target
        """
        logger.info(f"encode_target: Starting with {len(y)} values")
        logger.info(f"Target unique values: {y.unique()}")
        logger.info(f"Target null count: {y.isnull().sum()}")

        if binary:
            # For scorecard models, we need to handle missing target values
            if y.isnull().any():
                logger.warning(
                    f"Target variable has {y.isnull().sum()} missing values. These rows will be excluded."
                )
                # Create a mask for non-null values
                valid_mask = y.notna()
                # Only encode non-null values
                y_valid = y[valid_mask]
                encoded_valid = pd.Series(
                    pd.Categorical(y_valid).codes, index=y_valid.index, name=y.name
                )
                # Create full series with NaN for missing values
                encoded = pd.Series(index=y.index, dtype="float64", name=y.name)
                encoded[valid_mask] = encoded_valid
                logger.info(
                    f"After encoding, {encoded.notna().sum()} valid values out of {len(encoded)}"
                )
            else:
                # No missing values, encode normally
                encoded = pd.Series(pd.Categorical(y).codes, index=y.index, name=y.name)

            logger.info(f"After encoding, unique values: {encoded.dropna().unique()}")
            logger.info(f"Value counts:\n{encoded.value_counts()}")
            return encoded
        else:
            # For multi-class, use label encoding
            le = LabelEncoder()
            return pd.Series(le.fit_transform(y), index=y.index, name=y.name)


class DataSplitter:
    """Handles train/test splitting with various strategies."""

    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            X: Feature dataframe
            y: Target series
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by target variable

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        logger.info(f"Data split completed: Train size={len(X_train)}, Test size={len(X_test)}")

        return X_train, X_test, y_train, y_test


class DataPipeline:
    """High-level pipeline for credit scoring data preparation."""

    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter()

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        encode_categoricals: bool = True,
        encoding_method: str = "label",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete pipeline for data preparation.

        Args:
            target_column: Name of target column
            feature_columns: List of feature columns
            test_size: Test set proportion
            random_state: Random seed
            encode_categoricals: Whether to encode categorical features
            encoding_method: Method for encoding ('label' or 'onehot')
            file_obj: Optional file object for Streamlit

        Returns:
            Dictionary with X_train, X_test, y_train, y_test, and metadata
        """

        # Prepare modeling data
        X, y = self.preprocessor.prepare_modeling_data(df, target_column, feature_columns)

        # Encode target
        y = self.preprocessor.encode_target(y)

        # Encode categorical features if requested
        if encode_categoricals:
            X = self.preprocessor.encode_categorical_features(X, method=encoding_method)

        # Store metadata
        metadata = {
            "n_samples": len(df),
            "n_features": len(feature_columns),
            "target_distribution": y.value_counts().to_dict(),
            "categorical_features": X.select_dtypes(include=["object"]).columns.tolist(),
            "numerical_features": X.select_dtypes(include=["number"]).columns.tolist(),
        }

        return X, y


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_binary_target(y: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate that target is binary.

        Args:
            y: Target series or array

        Returns:
            True if binary, False otherwise
        """
        if isinstance(y, pd.Series):
            unique_values = y.nunique()
        else:
            unique_values = len(np.unique(y))
        return unique_values == 2

    @staticmethod
    def check_data_quality(
        df: pd.DataFrame, target_column: str, feature_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Check data quality issues.

        Args:
            df: DataFrame to check
            target_column: Name of target column
            feature_columns: List of feature column names

        Returns:
            Dictionary with quality check results
        """
        issues = {
            "missing_values": {},
            "constant_features": [],
            "high_cardinality": [],
            "target_imbalance": None,
            "n_samples": len(df),
            "n_features": len(feature_columns),
        }

        # Check missing values
        for col in feature_columns + [target_column]:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 0:
                issues["missing_values"][col] = missing_pct

        # Check constant features
        for col in feature_columns:
            if df[col].nunique() == 1:
                issues["constant_features"].append(col)

        # Check high cardinality for categorical features
        for col in feature_columns:
            if df[col].dtype == "object":
                n_unique = df[col].nunique()
                if n_unique > 0.5 * len(df):
                    issues["high_cardinality"].append((col, n_unique))

        # Check target imbalance
        target_dist = df[target_column].value_counts(normalize=True)
        min_class_pct = target_dist.min() * 100
        issues["target_imbalance"] = min_class_pct

        return issues

    @staticmethod
    def report_data_issues(issues: Dict[str, Any]) -> str:
        """
        Create readable report from data quality issues.

        Args:
            issues: Dictionary of issues from check_data_quality

        Returns:
            Formatted report string
        """
        report = []

        report.append(f"Data Summary:")
        report.append(f"- Samples: {issues['n_samples']}")
        report.append(f"- Features: {issues['n_features']}")
        report.append("")

        if issues["missing_values"]:
            report.append("Missing Values:")
            for col, pct in issues["missing_values"].items():
                report.append(f"  - {col}: {pct:.1f}%")
            report.append("")

        if issues["constant_features"]:
            report.append("Constant Features (will be removed):")
            for col in issues["constant_features"]:
                report.append(f"  - {col}")
            report.append("")

        if issues["high_cardinality"]:
            report.append("High Cardinality Features:")
            for col, n_unique in issues["high_cardinality"]:
                report.append(f"  - {col}: {n_unique} unique values")
            report.append("")

        if issues["target_imbalance"] is not None:
            report.append(f"Target Imbalance: Minority class = {issues['target_imbalance']:.1f}%")
            if issues["target_imbalance"] < 10:
                report.append("  ⚠️ Severe class imbalance detected")

        return "\n".join(report)


def _filter_features_by_category(data: pd.DataFrame, category: str, target: str) -> pd.DataFrame:
    """Returns filtered dataset excluding features outside the desired category."""
    if category == "All":
        return data  # No filtering at all

    excluded_tags = {
        "Preop": ["postop", "intraop"],
        "Intraop": ["preop", "postop"],
        "Postop": ["preop", "intraop"],
    }

    excluded = excluded_tags.get(category, [])
    keep_cols = [target] + [
        col
        for col in data.columns
        if col == target or not any(tag in col.lower() for tag in excluded)
    ]
    return data[keep_cols]
