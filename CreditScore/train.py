"""Model training utilities for scorecards and logistic regression."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Optbinning imports
from optbinning import BinningProcess, Scorecard
from pandas.api.types import is_numeric_dtype
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression

# Statsmodels imports
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV

from config.config import Config

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self):
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Fit the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        pass

    @abstractmethod
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients."""
        pass


class OptbinningScorecardModel(BaseModel):
    """
    Scorecard model using optbinning package.
    This maintains the exact approach from streamlit-app.py.
    """

    def __init__(
        self,
        scaling_method: str = "min_max",
        scaling_method_params: Dict[str, Any] = None,
        reverse_scorecard: bool = False,
        intercept_based: bool = True,
        rounding: bool = True,
    ):
        super().__init__()
        self.scaling_method = scaling_method
        self.scaling_method_params = scaling_method_params or {"min": 0, "max": 100}
        self.reverse_scorecard = reverse_scorecard
        self.intercept_based = intercept_based
        self.rounding = rounding

        self.binning_process = None
        self.scorecard = None
        self.best_params = None
        self.optimal_threshold = None
        self.monotonic_trend_map = {
            "PreopEGFR60DayMostRecent_Value": None,
            "PreopEGFR60DayLowest_Value": None,
            "PreopHgbA1c_Value": "descending",
            "PreopHematocrit_Value": None,  # no options working here
            "UrineOutput_Value": None,
            "Weight_Value": "valley",
            "BPFirstInRoom_BP_Sys": "peak",
            "PreopPTT_Value": None,
            "ProcedureTypeHipArthroplasty_Value": None,
            # "AsaStatusClassification_Value_Code": "auto", #no options working here
        }

    def _perform_binning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        binning_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BinningProcess, pd.DataFrame]:
        """
        Perform binning on the training data.

        Args:
            X_train: Training features
            y_train: Training target
            feature_names: List of feature names to bin
            binning_config: Optional binning configuration parameters

        Returns:
            Tuple of (binning_process, X_train_binned)
        """
        # Get default binning config if not provided
        if binning_config is None:
            binning_config = Config.binning_args()

        # Step 1 – Initial type-based classification
        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
        numerical_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

        print(f"Initial numerical features: {numerical_features}")
        print(f"Initial categorical features: {categorical_features}")

        manual_categoricals = [
            "SurgicalServiceGroup_Value_Code",
            "Race_Value_Code",
            "ComplicationMpogAcuteKidneyInjury_Value",
        ]
        for col in manual_categoricals:
            if col in numerical_features:
                print(f"Manually classifying column '{col}' as categorical (override)")
                numerical_features.remove(col)
                if col not in categorical_features:
                    categorical_features.append(col)

        # Step 2 – Reclassify boolean-like and low-cardinality numerics
        for col in numerical_features.copy():
            col_data = X_train[col]
            unique_vals = set(col_data.dropna().unique())
            n_unique = len(unique_vals)

            is_boolean_like = (
                col_data.dtype == "bool"
                or unique_vals.issubset({0, 1})
                or unique_vals.issubset({"True", "False"})
            )

            if is_boolean_like:
                print(f"Classifying column '{col}' as categorical (boolean-like)")
                numerical_features.remove(col)
                if col not in categorical_features:
                    categorical_features.append(col)
                # Optional: normalize string booleans to actual bools
                if col_data.dtype == "object":
                    X_train[col] = col_data.map({"True": True, "False": False})
                continue

            if n_unique <= Config.UNIQUE_CATEGORY_THRESHOLD:
                print(f"Reclassifying column '{col}' as categorical (unique count = {n_unique})")
                numerical_features.remove(col)
                if col not in categorical_features:
                    categorical_features.append(col)

        print(f"Final numerical features: {numerical_features}")
        print(f"Final categorical features: {categorical_features}")

        # Auto-detect ASA status variables and treat as categorical
        if binning_config.get("fuzz_catch_asa_category", False):
            asa_detection = binning_config.get("asa_detection", {})
            name_contains = asa_detection.get("name_contains", "asa").lower()
            max_unique = asa_detection.get("max_unique_values", 6)

            # Check numerical columns for ASA-like variables
            for col in numerical_features.copy():
                col_lower = col.lower()
                if name_contains in col_lower:
                    unique_count = X_train[col].nunique()
                    if unique_count <= max_unique:
                        # Move from numerical to categorical
                        numerical_features.remove(col)
                        categorical_features.append(col)
                        logger.info(
                            f"Auto-detected ASA variable '{col}' ({unique_count} unique values) - treating as categorical"
                        )

        binning_fit_params = {}

        for var in feature_names:
            config = {}

            if var == "AsaStatusClassification_Value_Code":
                config["user_splits"] = [[1, 2], [3], [4], [5]]
            else:
                if var in self.monotonic_trend_map:
                    config["monotonic_trend"] = self.monotonic_trend_map[var]

            binning_fit_params[var] = config

        # Initialize and fit BinningProcess with custom parameters
        # Only using parameters that BinningProcess actually accepts
        binning_process = BinningProcess(
            variable_names=feature_names,
            categorical_variables=categorical_features,
            binning_fit_params=binning_fit_params,
            max_n_prebins=binning_config.get("max_n_prebins", 20),
            min_prebin_size=binning_config.get("min_prebin_size", 0.05),
            min_n_bins=binning_config.get("min_n_bins", None),
            max_n_bins=binning_config.get("max_n_bins", None),
            min_bin_size=binning_config.get("min_bin_size", None),
            max_bin_size=binning_config.get("max_bin_size", None),
            max_pvalue=binning_config.get("max_pvalue", None),
            max_pvalue_policy=binning_config.get("max_pvalue_policy", "consecutive"),
            selection_criteria=binning_config.get("selection_criteria", None),
            verbose=binning_config.get("verbose", False),
        )

        logger.info(
            f"Binning with config: min_bin_size={binning_config.get('min_bin_size', 0.05)}, "
            f"min_n_bins={binning_config.get('min_n_bins', 2)}"
        )

        binning_process.fit(X_train, y_train)

        # Transform data
        X_train_binned = binning_process.transform(X_train)

        # Convert back to DataFrame
        X_train_binned = pd.DataFrame(X_train_binned, columns=feature_names)

        return binning_process, X_train_binned

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameter_search: bool = True,
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 5,
        **kwargs,
    ):
        """
        Fit the scorecard model.

        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_search: Whether to perform hyperparameter search
            param_grid: Grid of parameters for search
            cv: Number of cross-validation folds
        """
        feature_names = X_train.columns.tolist()

        # Get binning configuration for medical data
        from config.config import Config

        binning_config = Config.binning_args()

        # Perform binning
        self.binning_process, X_train_binned = self._perform_binning(
            X_train, y_train, feature_names, binning_config
        )
        asa_binning = self.binning_process.get_binned_variable("AsaStatusClassification_Value_Code")

        for i, bin_group in enumerate(asa_binning.user_splits):
            clean_labels = [int(cat) if hasattr(cat, "__int__") else str(cat) for cat in bin_group]
            print(f"Bin {i+1}: {clean_labels}")


        # Hyperparameter search for logistic regression
        if hyperparameter_search:
            if param_grid is None:
                param_grid = {"C": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]}

            # Initialize GridSearchCV for elastic net
            en_cv = GridSearchCV(
                LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000),
                param_grid,
                cv=cv,
                scoring="average_precision",
                n_jobs=-1,
            )
            en_cv.fit(X_train_binned, y_train)

            self.best_params = en_cv.best_params_
            logger.info(f"Best parameters found: {self.best_params}")

            # Initialize model with best parameters
            estimator = LogisticRegression(
                penalty="elasticnet",
                C=self.best_params["C"],
                l1_ratio=self.best_params["l1_ratio"],
                solver="saga",
                max_iter=10000,
                n_jobs=-1,
            )
        else:
            # Use default logistic regression
            estimator = LogisticRegression(max_iter=10000, solver="lbfgs")

        # Initialize and fit Scorecard
        self.scorecard = Scorecard(
            binning_process=self.binning_process,
            estimator=estimator,
            scaling_method=self.scaling_method,
            rounding=self.rounding,
            scaling_method_params=self.scaling_method_params,
            reverse_scorecard=self.reverse_scorecard,
            intercept_based=self.intercept_based,
            # monotonic_trend=self.monotonic_trend_map
        )

        self.scorecard.fit(X_train, y_train, metric_missing="empirical", metric_special="empirical")
        self.is_fitted = True
        logger.info("Scorecard model fitted successfully")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the scorecard."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        probas = self.scorecard.predict_proba(X)
        logger.info(f"predict_proba: Input shape {X.shape}, Output shape {probas.shape}")
        return probas

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate scorecard scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        return self.scorecard.score(X)

    def get_coefficients(self) -> pd.DataFrame:
        """Get scorecard coefficients from the underlying estimator."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Access the estimator within the scorecard
        estimator = self.scorecard.estimator_
        feature_names = self.binning_process.variable_names

        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": estimator.coef_[0]})

        # Add intercept
        intercept_row = pd.DataFrame(
            {"Feature": ["Intercept"], "Coefficient": [estimator.intercept_[0]]}
        )

        coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)

        return coef_df

    def get_scorecard_table(self, style: str = "detailed") -> pd.DataFrame:
        """Get the detailed scorecard table."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get the scorecard table
        table = self.scorecard.table(style=style).round(3)

        # Convert any list/array columns to strings for display
        for col in table.columns:
            if table[col].dtype == "object":
                # Check if any values are lists or arrays
                first_non_null = (
                    table[col].dropna().iloc[0] if len(table[col].dropna()) > 0 else None
                )
                if first_non_null is not None and (isinstance(first_non_null, (list, np.ndarray))):
                    # Convert lists/arrays to string representation
                    table[col] = table[col].apply(lambda x: str(x) if x is not None else x)

        return table

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_event_rate: bool = False,
        y_train: Optional[pd.Series] = None,
    ) -> float:
        """Choose a probability threshold using PR/KS/CAP scoring."""
        from CreditScore.utils import calculate_ranking_based_cap_score

        if use_event_rate:
            if y_train is None:
                raise ValueError("y_train is required to compute event rate threshold.")
            event_rate = y_train.mean()
            self.optimal_threshold = event_rate
            self.threshold_strategy_label = "Event rate"
            return self.optimal_threshold

        probas = self.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probas)

        best_score = -np.inf
        optimal_threshold = 0.5  # fallback

        for i, t in enumerate(thresholds):
            preds = (probas >= t).astype(int)
            prec = precision[i]
            rec = recall[i]
            fpr = sum((preds == 1) & (y_val == 0)) / sum(y_val == 0)
            tpr = sum((preds == 1) & (y_val == 1)) / sum(y_val == 1)
            ks_stat = abs(tpr - fpr)

            cap_score = calculate_ranking_based_cap_score(y_val, probas, preds)
            score = 0.4 * prec + 0.3 * rec + 0.2 * ks_stat + 0.1 * cap_score

            if score > best_score:
                best_score = score
                optimal_threshold = t

        self.optimal_threshold = optimal_threshold
        return self.optimal_threshold

    def predict_class(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return hard class predictions (0/1).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float | None
            Cut-off probability.  If *None*, the model’s
            ``self.optimal_threshold`` is used (must be set beforehand by
            calling ``optimize_threshold``).

        Returns
        -------
        np.ndarray
            Array of 0/1 predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # Decide which threshold to use ---------------------------------
        if threshold is None:
            if self.optimal_threshold is None:
                raise ValueError(
                    "No threshold provided and 'optimal_threshold' is not set. "
                    "Call 'optimize_threshold' first or pass a threshold."
                )
            threshold = self.optimal_threshold
            strategy = getattr(self, "threshold_strategy_label", "stored threshold")
            print(f"[Info] Using model’s stored {strategy}: {threshold}")
        else:
            print(f"[Info] Using custom threshold: {threshold}")

        # Predict classes -----------------------------------------------
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)


class StatsmodelsLogisticModel(BaseModel):
    """
    Logistic regression model using statsmodels.
    This maintains the exact approach from coefficients.py.
    """

    def __init__(self):
        super().__init__()
        self.result = None
        self.formula = None
        self.feature_columns = None

    def _check_separation(self, X: pd.DataFrame, y: pd.Series):
        """
        Check for perfect or quasi-complete separation.

        Args:
            X: Features
            y: Target
        """
        # Check each feature for separation
        for col in X.columns:
            if X[col].dtype in ["int64", "float64"]:
                # For continuous variables, check if perfectly separated
                y0_vals = X.loc[y == 0, col]
                y1_vals = X.loc[y == 1, col]

                if len(y0_vals) > 0 and len(y1_vals) > 0:
                    if y0_vals.max() < y1_vals.min() or y1_vals.max() < y0_vals.min():
                        logger.warning(f"Perfect separation detected in feature '{col}'")
            else:
                # For categorical, check if any category perfectly predicts outcome
                crosstab = pd.crosstab(X[col], y)
                if (crosstab == 0).any().any():
                    logger.warning(f"Quasi-complete separation detected in feature '{col}'")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_name: str = "target",
        check_separation: bool = True,
        **kwargs,
    ):
        """
        Fit the statsmodels logistic regression.

        Args:
            X_train: Training features
            y_train: Training target
            target_name: Name for the target variable in formula
            check_separation: Whether to check for perfect separation
        """
        # Check for perfect/quasi-complete separation
        if check_separation:
            self._check_separation(X_train, y_train)

        # Create combined dataframe for formula API
        data = X_train.copy()
        data[target_name] = y_train

        # Store feature columns
        self.feature_columns = X_train.columns.tolist()

        # Build formula: numeric vars as-is, categorical vars wrapped in C()
        rhs_terms = [
            f"C({col})" if not is_numeric_dtype(X_train[col]) else col
            for col in self.feature_columns
        ]
        self.formula = f"{target_name} ~ " + " + ".join(rhs_terms)

        # Fit logistic regression with robust strategy
        model = smf.logit(formula=self.formula, data=data)

        # Try different fitting strategies
        fit_success = False
        strategies = [
            # Strategy 1: Standard Newton with relaxed tolerance
            {"method": "newton", "maxiter": 10000, "tol": 1e-6},
            # Strategy 2: Newton with ridge factor for stability
            {"method": "newton", "maxiter": 10000, "tol": 1e-6, "ridge_factor": 1e-8},
            # Strategy 3: BFGS optimizer (often more stable)
            {"method": "bfgs", "maxiter": 10000, "gtol": 1e-5},
            # Strategy 4: L-BFGS optimizer
            {"method": "lbfgs", "maxiter": 10000, "gtol": 1e-5},
            # Strategy 5: Regularized fit as last resort
            {"regularized": True, "method": "l1", "alpha": 0.01, "maxiter": 10000},
        ]

        for i, strategy in enumerate(strategies):
            try:
                if strategy.get("regularized", False):
                    # Use regularized fit
                    logger.info(f"Trying strategy {i+1}: Regularized fit with L1")
                    self.result = model.fit_regularized(
                        method=strategy["method"],
                        alpha=strategy["alpha"],
                        disp=False,
                        maxiter=strategy["maxiter"],
                    )
                else:
                    # Use standard fit
                    logger.info(f"Trying strategy {i+1}: {strategy['method']} optimizer")
                    self.result = model.fit(
                        disp=False, **{k: v for k, v in strategy.items() if k != "regularized"}
                    )

                # Check if converged
                if hasattr(self.result, "mle_retvals") and self.result.mle_retvals["converged"]:
                    fit_success = True
                    logger.info(f"Successfully converged using strategy {i+1}")
                    break

            except Exception as e:
                logger.warning(f"Strategy {i+1} failed: {str(e)}")
                continue

        if not fit_success:
            logger.warning("All strategies failed to converge properly. Using last result.")

        # Check if we used regularization
        self.used_regularization = False
        if hasattr(self.result, "method") and self.result.method in ["l1", "l2", "elastic"]:
            self.used_regularization = True
            logger.warning(
                "Regularized fit was used. P-values and confidence intervals are not available."
            )

        self.is_fitted = True
        logger.info("Statsmodels logistic regression fitting completed")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Statsmodels returns 1D array of probabilities for positive class
        probs_pos = self.result.predict(X)

        # Convert to 2D array with both classes
        probs_neg = 1 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients with standard errors and p-values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Extract coefficients
        coef_series = self.result.params.rename("Coefficient")

        # Check if we have inference statistics (not available for regularized models)
        if hasattr(self, "used_regularization") and self.used_regularization:
            # For regularized models, only coefficients are available
            summary_df = coef_series.reset_index().rename(columns={"index": "Feature"})
            summary_df["Std_Error"] = np.nan
            summary_df["P_value"] = np.nan
            summary_df["CI_Lower"] = np.nan
            summary_df["CI_Upper"] = np.nan
            summary_df["Note"] = "Regularized model - inference not available"
        else:
            # Extract all statistics for non-regularized models
            std_err = self.result.bse.rename("Std_Error")
            pval_series = self.result.pvalues.rename("P_value")

            # Get confidence intervals
            conf_int = self.result.conf_int()
            conf_int.columns = ["CI_Lower", "CI_Upper"]

            # Combine all statistics
            summary_df = (
                pd.concat([coef_series, std_err, pval_series, conf_int], axis=1)
                .reset_index()
                .rename(columns={"index": "Feature"})
            )

        return summary_df

    def get_summary(self) -> str:
        """Get full model summary."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return str(self.result.summary())

    def get_model_stats(self) -> Dict[str, float]:
        """Get model-level statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        stats = {
            "aic": self.result.aic,
            "bic": self.result.bic,
            "log_likelihood": self.result.llf,
            "null_log_likelihood": self.result.llnull,
            "pseudo_r2": 1 - self.result.llf / self.result.llnull,
        }

        # Add convergence info if available
        if hasattr(self.result, "mle_retvals"):
            stats["n_iterations"] = self.result.mle_retvals.get("iterations", "N/A")
            stats["converged"] = self.result.mle_retvals.get("converged", False)

        return stats

    def diagnose_convergence(self) -> Dict[str, Any]:
        """
        Diagnose potential convergence issues.

        Returns:
            Dictionary with diagnostic information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        diagnostics = {
            "converged": False,
            "condition_number": None,
            "eigenvalues": None,
            "high_vif_features": [],
            "warnings": [],
        }

        # Check convergence status
        if hasattr(self.result, "mle_retvals"):
            diagnostics["converged"] = self.result.mle_retvals.get("converged", False)

        # Check condition number
        try:
            from numpy.linalg import cond

            hess = self.result.model.hessian(self.result.params)
            diagnostics["condition_number"] = cond(hess)
            if diagnostics["condition_number"] > 1000:
                diagnostics["warnings"].append(
                    f"High condition number ({diagnostics['condition_number']:.2e}) indicates multicollinearity"
                )
        except BaseException:
            pass

        # Check for large standard errors
        large_se = self.result.bse[self.result.bse > 10]
        if len(large_se) > 0:
            diagnostics["warnings"].append(
                f"Large standard errors detected for: {list(large_se.index)}"
            )

        return diagnostics


class ModelFactory:
    """Factory class for creating different model types."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance based on type.

        Args:
            model_type: Type of model ('scorecard' or 'statsmodels')
            **kwargs: Additional parameters for model initialization

        Returns:
            BaseModel instance
        """
        if model_type == "scorecard":
            return OptbinningScorecardModel(**kwargs)
        elif model_type == "statsmodels":
            return StatsmodelsLogisticModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class ScorecardBinningHelper:
    """Helper class for binning operations used by scorecard models."""

    @staticmethod
    def perform_binning(
        X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, feature_names: List[str]
    ) -> Tuple[BinningProcess, pd.DataFrame, pd.DataFrame]:
        """
        Perform binning on train and test data.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            feature_names: List of feature names

        Returns:
            Tuple of (binning_process, X_train_binned, X_test_binned)
        """
        # Identify categorical features
        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

        # Initialize and fit BinningProcess
        binning_process = BinningProcess(
            variable_names=feature_names, categorical_variables=categorical_features
        )
        binning_process.fit(X_train, y_train)

        # Transform data
        X_train_binned = binning_process.transform(X_train)
        X_test_binned = binning_process.transform(X_test)

        # Convert to DataFrames
        X_train_binned = pd.DataFrame(X_train_binned, columns=feature_names)
        X_test_binned = pd.DataFrame(X_test_binned, columns=feature_names)

        return binning_process, X_train_binned, X_test_binned
