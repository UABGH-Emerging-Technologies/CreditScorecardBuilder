# CreditScore.predict

Prediction helpers for trained scorecard and regression models.

## Classes

### `ModelPredictor`

Base class for making predictions with trained models.

#### Methods

- `predict_single(self, features, return_proba, threshold)`
  - Make prediction for a single observation.

Args:
    features: Dictionary of feature_name -> value
    return_proba: Whether to return probability or class

Returns:
    Prediction (probability or class)

- `predict_batch(self, X, return_proba, threshold)`
  - Make predictions for multiple observations.

Args:
    X: Features DataFrame
    return_proba: Whether to return probabilities or classes
    threshold: Classification threshold (if return_proba=False)

Returns:
    Array of predictions

- `predict_with_confidence(self, X, n_iterations, sample_fraction)`
  - Make predictions with confidence intervals using bootstrap.

Args:
    X: Features DataFrame
    n_iterations: Number of bootstrap iterations
    sample_fraction: Fraction of training data to sample

Returns:
    Dictionary with mean predictions and confidence intervals

### `ScorecardPredictor`

Predictor specific to scorecard models.

#### Methods

- `predict_score(self, X)`
  - Get scorecard scores for observations.

Args:
    X: Features DataFrame

Returns:
    Array of scores

- `predict_with_details(self, features)`
  - Make prediction with detailed scorecard breakdown.

Args:
    features: Dictionary of feature_name -> value

Returns:
    Dictionary with score, probability, and feature contributions

### `BatchPredictor`

Handle batch predictions with data preprocessing.

#### Methods

- `predict_file(self, file_path, output_path, return_proba)`
  - Make predictions for a file of data.

Args:
    file_path: Path to input file (CSV or Excel)
    output_path: Optional path to save predictions
    return_proba: Whether to return probabilities

Returns:
    DataFrame with original data and predictions

### `ModelSerializer`

Save and load trained models.

#### Methods

- `save_model(model, save_path, metadata)`
  - Save model to disk.

Args:
    model: Trained model instance
    save_path: Path to save model
    metadata: Optional metadata to save with model

- `load_model(load_path, load_metadata)`
  - Load model from disk.

Args:
    load_path: Path to saved model
    load_metadata: Whether to load metadata

Returns:
    Model instance or (model, metadata) tuple

### `PredictionExplainer`

Explain individual predictions.

#### Methods

- `explain_prediction(self, features, method)`
  - Explain a single prediction.

Args:
    features: Dictionary of feature values
    method: Explanation method ('coefficients' or 'shap')

Returns:
    Dictionary with explanation details
