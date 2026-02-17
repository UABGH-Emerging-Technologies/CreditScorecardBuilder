# CreditScore.train

Model training utilities for scorecards and logistic regression.

## Classes

### `BaseModel`

Abstract base class for all models.

#### Methods

- `fit(self, X_train, y_train, **kwargs)`
  - Fit the model.

- `predict_proba(self, X)`
  - Predict probabilities.

- `get_coefficients(self)`
  - Get model coefficients.

### `OptbinningScorecardModel`

Scorecard model using optbinning package.
This maintains the exact approach from streamlit-app.py.

#### Methods

- `fit(self, X_train, y_train, hyperparameter_search, param_grid, cv, **kwargs)`
  - Fit the scorecard model.

Args:
    X_train: Training features
    y_train: Training target
    hyperparameter_search: Whether to perform hyperparameter search
    param_grid: Grid of parameters for search
    cv: Number of cross-validation folds

- `predict_proba(self, X)`
  - Predict probabilities using the scorecard.

- `score(self, X)`
  - Calculate scorecard scores.

- `get_coefficients(self)`
  - Get scorecard coefficients from the underlying estimator.

- `get_scorecard_table(self, style)`
  - Get the detailed scorecard table.

- `optimize_threshold(self, X_val, y_val, use_event_rate, y_train)`
  - Choose a probability threshold using PR/KS/CAP scoring.

- `predict_class(self, X, threshold)`
  - Return hard class predictions (0/1).

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

### `StatsmodelsLogisticModel`

Logistic regression model using statsmodels.
This maintains the exact approach from coefficients.py.

#### Methods

- `fit(self, X_train, y_train, target_name, check_separation, **kwargs)`
  - Fit the statsmodels logistic regression.

Args:
    X_train: Training features
    y_train: Training target
    target_name: Name for the target variable in formula
    check_separation: Whether to check for perfect separation

- `predict_proba(self, X)`
  - Predict probabilities.

- `get_coefficients(self)`
  - Get model coefficients with standard errors and p-values.

- `get_summary(self)`
  - Get full model summary.

- `get_model_stats(self)`
  - Get model-level statistics.

- `diagnose_convergence(self)`
  - Diagnose potential convergence issues.

Returns:
    Dictionary with diagnostic information

### `ModelFactory`

Factory class for creating different model types.

#### Methods

- `create_model(model_type, **kwargs)`
  - Create a model instance based on type.

Args:
    model_type: Type of model ('scorecard' or 'statsmodels')
    **kwargs: Additional parameters for model initialization

Returns:
    BaseModel instance

### `ScorecardBinningHelper`

Helper class for binning operations used by scorecard models.

#### Methods

- `perform_binning(X_train, X_test, y_train, feature_names)`
  - Perform binning on train and test data.

Args:
    X_train: Training features
    X_test: Test features
    y_train: Training target
    feature_names: List of feature names

Returns:
    Tuple of (binning_process, X_train_binned, X_test_binned)
