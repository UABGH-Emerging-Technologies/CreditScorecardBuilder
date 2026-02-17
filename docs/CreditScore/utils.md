# CreditScore.utils

Shared helpers for metrics, reporting, and data cleaning.

## Classes

### `ModelMetrics`

Common model metrics calculations.

#### Methods

- `calculate_cap_curve(y_true, y_pred_proba)`
  - Compute the CAP (Cumulative Accuracy Profile) score.

Args:
    y_true: True labels
    y_pred_proba: Predicted probabilities

Returns:
    CAP score (area under the CAP curve)

- `calculate_metrics(y_true, y_pred_proba, y_pred_class)`
  - Calculate both probability-based and classification-based metrics.

- `create_metrics_comparison_df(metrics_dict)`
  - Create comparison DataFrame from metrics dictionary.

Args:
    metrics_dict: Dictionary of model_name -> metrics

Returns:
    DataFrame with metrics comparison

### `VisualizationHelper`

Common visualization utilities.

#### Methods

- `setup_plot_style()`
  - Set up consistent plot styling.

- `save_figure(fig, save_path, dpi)`
  - Save figure with consistent settings.

Args:
    fig: Matplotlib figure
    save_path: Path to save figure
    dpi: DPI for saved figure

- `create_subplots(n_plots, n_cols, figsize)`
  - Create subplots with consistent sizing.

Args:
    n_plots: Number of plots needed
    n_cols: Number of columns
    figsize: Figure size (auto-calculated if None)

Returns:
    Tuple of (figure, axes)

### `ConfigurationManager`

Manage model configurations using JSON config files.

#### Methods

- `scorecard_args()`
  - Get scorecard configuration for model initialization.

- `get_scorecard_fit_config()`
  - Get cross-validation configuration for model fitting.

- `binning_args()`
  - Get binning configuration optimized for medical data with class imbalance.

- `monitoring_args()`
  - Get monitoring configuration.

- `power_analysis_args()`
  - Get power analysis configuration.

- `variable_selection_args()`
  - Get variable selection configuration.

### `CacheMixin`

Cached helpers for model fitting and diagnostics.

#### Methods

- `fit_logit_model(X, y, target_column)`
  - Fit a statsmodels logistic model and return coefficients.

- `compute_vif(df)`
  - Compute variance inflation factors for numeric columns.

- `to_csv_bytes(df)`
  - Serialize a DataFrame to UTF-8 CSV bytes.

### `LogisticWrapper`

Expose scikit-learn coefficients for SHAP compatibility.

### `DefinitionMatcher`

Match feature names to phenotype definitions.

#### Methods

- `get_definition(self, feature)`
  - Returns a single string containing the matched phenotype's return_columns entries
relevant to `feature`. Uses multiple strategies:
  1) Exact return_column key match
  2) Exact phenotype name match
  3) Exact composite "phenotype|return_key" match
  4) Token-sort fuzzy match across all candidates
If no match above threshold, returns "No definition available".

## Functions

### `generate_model_summary(y_train, y_test, y_train_pred, y_test_pred, scorecard_model, threshold_summary, feature_filter)`

Generate a structured summary of model accuracy, top risk factors, and dataset distribution.

### `clean_add_results_to_docx(self, doc, results)`

Docx table renderer with custom column sizing.

### `calculate_ranking_based_cap_score(y_true, y_pred_proba, preds)`

Score a CAP curve based on ranked predictions.

### `run_integrity_checks(df)`

Render a Streamlit data integrity summary.

### `dict_to_markdown(d, indent)`

Recursively format dictionary as Markdown

### `ensure_dataframe(obj, col_name)`

Ensure an object is represented as a DataFrame.
