"""Prediction helpers for trained scorecard and regression models."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Base class for making predictions with trained models."""

    def __init__(self, model):
        """
        Initialize predictor with a trained model.

        Args:
            model: Trained model instance
        """
        self.model = model

    def predict_single(
        self, features: Dict[str, Any], return_proba: bool = True, threshold: Optional[float] = None
    ) -> Union[float, int]:
        """
        Make prediction for a single observation.

        Args:
            features: Dictionary of feature_name -> value
            return_proba: Whether to return probability or class

        Returns:
            Prediction (probability or class)
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])

        if return_proba:
            proba = self.model.predict_proba(X)
            return float(proba[0, 1])
        else:
            if threshold is None:
                threshold = getattr(self.model, "optimal_threshold", 0.1)
            proba = self.model.predict_proba(X)
            return int(proba[0, 1] >= threshold)

    def predict_batch(
        self, X: pd.DataFrame, return_proba: bool = True, threshold: float = 0.1
    ) -> np.ndarray:
        """
        Make predictions for multiple observations.

        Args:
            X: Features DataFrame
            return_proba: Whether to return probabilities or classes
            threshold: Classification threshold (if return_proba=False)

        Returns:
            Array of predictions
        """
        proba = self.model.predict_proba(X)

        if return_proba:
            # Return positive class probabilities
            if len(proba.shape) > 1:
                return proba[:, 1]
            return proba
        else:
            # Use model's optimized threshold if none provided
            if threshold is None:
                threshold = getattr(self.model, "optimal_threshold", 0.1)

            # Return binary predictions
            if len(proba.shape) > 1:
                return (proba[:, 1] >= threshold).astype(int)
            return (proba >= threshold).astype(int)

    def predict_with_confidence(
        self, X: pd.DataFrame, n_iterations: int = 100, sample_fraction: float = 0.8
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals using bootstrap.

        Args:
            X: Features DataFrame
            n_iterations: Number of bootstrap iterations
            sample_fraction: Fraction of training data to sample

        Returns:
            Dictionary with mean predictions and confidence intervals
        """
        predictions = []

        for _ in range(n_iterations):
            # Note: This is a simplified version
            # In practice, you'd retrain on bootstrap samples
            proba = self.model.predict_proba(X)
            if len(proba.shape) > 1:
                predictions.append(proba[:, 1])
            else:
                predictions.append(proba)

        predictions = np.array(predictions)

        return {
            "mean": predictions.mean(axis=0),
            "std": predictions.std(axis=0),
            "lower_95": np.percentile(predictions, 2.5, axis=0),
            "upper_95": np.percentile(predictions, 97.5, axis=0),
        }


class ScorecardPredictor(ModelPredictor):
    """Predictor specific to scorecard models."""

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get scorecard scores for observations.

        Args:
            X: Features DataFrame

        Returns:
            Array of scores
        """
        return self.model.score(X)

    def predict_with_details(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with detailed scorecard breakdown.

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            Dictionary with score, probability, and feature contributions
        """
        X = pd.DataFrame([features])

        # Get score and probability
        score = self.model.score(X)[0]
        proba = self.model.predict_proba(X)[0, 1]

        # Get scorecard table for feature contributions
        scorecard_table = self.model.get_scorecard_table()

        # Calculate feature contributions
        contributions = {}
        for feature, value in features.items():
            # Find the bin for this feature and value
            feature_rows = scorecard_table[scorecard_table["Variable"] == feature]
            # This is simplified - actual implementation would need proper bin matching
            if not feature_rows.empty:
                contributions[feature] = {
                    "value": value,
                    "points": 0,  # Placeholder for actual point calculation
                }

        return {
            "score": float(score),
            "probability": float(proba),
            "risk_category": self._get_risk_category(score),
            "feature_contributions": contributions,
        }

    def _get_risk_category(self, score: float) -> str:
        """
        Categorize risk based on score.

        Args:
            score: Scorecard score

        Returns:
            Risk category string
        """
        # These thresholds would typically be calibrated
        # With reverse_scorecard=True, higher scores mean higher risk
        if score >= 80:
            return "Very High Risk"
        elif score >= 60:
            return "High Risk"
        elif score >= 40:
            return "Medium Risk"
        elif score >= 20:
            return "Low Risk"
        else:
            return "Very Low Risk"


class BatchPredictor:
    """Handle batch predictions with data preprocessing."""

    def __init__(self, model, preprocessor=None, feature_columns: Optional[List[str]] = None):
        """
        Initialize batch predictor.

        Args:
            model: Trained model instance
            preprocessor: Data preprocessor instance
            feature_columns: List of expected feature columns
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.predictor = ModelPredictor(model)

    def predict_file(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """
        Make predictions for a file of data.

        Args:
            file_path: Path to input file (CSV or Excel)
            output_path: Optional path to save predictions
            return_proba: Whether to return probabilities

        Returns:
            DataFrame with original data and predictions
        """
        from CreditScore.data import DataLoader

        # Load data
        loader = DataLoader()
        df = loader.load_file(file_path)

        # Validate columns
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Select only required columns
            X = df[self.feature_columns]
        else:
            X = df

        # Preprocess if preprocessor available
        if self.preprocessor:
            X = self.preprocessor.handle_missing_values(X, drop_missing=False)
            X = self.preprocessor.encode_categorical_features(X)

        # Make predictions
        predictions = self.predictor.predict_batch(X, return_proba=return_proba)

        # Add predictions to dataframe
        result_df = df.copy()
        if return_proba:
            result_df["predicted_probability"] = predictions
        else:
            result_df["predicted_class"] = predictions

        # Save if output path provided
        if output_path:
            if str(output_path).endswith(".csv"):
                result_df.to_csv(output_path, index=False)
            else:
                result_df.to_excel(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")

        return result_df


class ModelSerializer:
    """Save and load trained models."""

    @staticmethod
    def save_model(model, save_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None):
        """
        Save model to disk.

        Args:
            model: Trained model instance
            save_path: Path to save model
            metadata: Optional metadata to save with model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata if provided
        if metadata:
            metadata_path = save_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @staticmethod
    def load_model(load_path: Union[str, Path], load_metadata: bool = True) -> Union[Any, tuple]:
        """
        Load model from disk.

        Args:
            load_path: Path to saved model
            load_metadata: Whether to load metadata

        Returns:
            Model instance or (model, metadata) tuple
        """
        load_path = Path(load_path)

        # Load model
        with open(load_path, "rb") as f:
            model = pickle.load(f)

        # Load metadata if requested
        if load_metadata:
            metadata_path = load_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return model, metadata
            else:
                return model, {}

        return model


class PredictionExplainer:
    """Explain individual predictions."""

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer.

        Args:
            model: Trained model instance
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

    def explain_prediction(
        self, features: Dict[str, Any], method: str = "coefficients"
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            features: Dictionary of feature values
            method: Explanation method ('coefficients' or 'shap')

        Returns:
            Dictionary with explanation details
        """
        X = pd.DataFrame([features])

        if method == "coefficients":
            # Get coefficients
            coef_df = self.model.get_coefficients()

            # Calculate feature impacts
            impacts = {}
            for _, row in coef_df.iterrows():
                feature = row["Feature"]
                if feature != "Intercept" and feature in features:
                    coef = row["Coefficient"]
                    value = features[feature]
                    # Simplified - actual calculation depends on preprocessing
                    impact = coef * value
                    impacts[feature] = {"value": value, "coefficient": coef, "impact": impact}

            # Sort by absolute impact
            sorted_impacts = dict(
                sorted(impacts.items(), key=lambda x: abs(x[1]["impact"]), reverse=True)
            )

            return {
                "method": "coefficients",
                "prediction": float(self.model.predict_proba(X)[0, 1]),
                "feature_impacts": sorted_impacts,
            }

        elif method == "shap":
            # Placeholder for SHAP implementation
            raise NotImplementedError("SHAP explanation not yet implemented")

        else:
            raise ValueError(f"Unknown explanation method: {method}")
