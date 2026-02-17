# CreditScore.evaluate

Evaluation metrics, monitoring, and power analysis helpers.

## Classes

### `RocMetrics`

Container for ROC curve arrays and AUC.

### `PrcMetrics`

Container for precision-recall arrays and AUC.

### `ModelEvaluator`

Thin layer that ONLY computes metrics – no plotting takes place here.

#### Methods

- `roc_auc(y_true, y_pred_proba)`
  - Compute ROC curve and AUC.

Parameters
----------
y_true          : ground-truth labels (binary)
y_pred_proba    : positive-class probabilities (N or Nx2)

Returns
-------
RocMetrics

- `prc_auc(y_true, y_pred_proba)`
  - Compute Precision-Recall curve and AUC-PR.

- `predict_proba(self, X)`
  - Helper so UI code can obtain probabilities in a single line even if
the wrapped model returns a shape (N,) instead of (N,2).

- `compute_precision_recall(self, y_true, y_pred_proba)`
  - Compute precision, recall and AUC-PR.

Returns
-------
precision : ndarray
recall    : ndarray
auc_pr    : float

### `ScorecardEvaluator`

Evaluator specific to scorecard models.

#### Methods

- `initialize_monitoring(self, psi_method, psi_n_bins, psi_min_bin_size, show_digits)`
  - Initialize scorecard monitoring.

Args:
    psi_method: Method for PSI calculation
    psi_n_bins: Number of bins for PSI
    psi_min_bin_size: Minimum bin size
    show_digits: Number of digits to show

- `fit_monitoring(self, X_test, y_test, X_train, y_train)`
  - Fit the monitoring system.

Args:
    X_test: Test features
    y_test: Test labels
    X_train: Training features
    y_train: Training labels

- `get_psi_report(self)`
  - Get PSI report table.

- `plot_psi(self)`
  - Plot PSI values.

### `StatsmodelsEvaluator`

Evaluator specific to statsmodels logistic regression.

#### Methods

- `power_analysis_wald(self, coeff_name, alpha, power, tails)`
  - Perform Wald-based power analysis.

Args:
    coeff_name: Name of coefficient to test
    alpha: Significance level
    power: Desired power
    tails: "One" or "Two" tailed test

Returns:
    Required sample size

- `power_analysis_simulation(self, coeff_name, sample_sizes, n_simulations, alpha)`
  - Perform simulation-based power analysis.

Args:
    coeff_name: Name of coefficient to test
    sample_sizes: List of sample sizes to test
    n_simulations: Number of simulations per size
    alpha: Significance level

Returns:
    DataFrame with sample sizes and power estimates

- `plot_power_curve(self, power_df, target_power, save_path)`
  - Plot power curve from simulation results.

Args:
    power_df: DataFrame with 'n' and 'power' columns
    target_power: Target power level to highlight
    save_path: Optional path to save figure

Returns:
    matplotlib figure

- `get_gpower_parameters(self, coeff_name, X, y)`
  - Calculate parameters for G*Power replication.

Args:
    coeff_name: Name of coefficient
    X: Feature data
    y: Target data

Returns:
    Dictionary of G*Power parameters

### `ModelComparison`

Compare multiple models.

#### Methods

- `compare_roc_curves(models, X_test, y_test, save_path)`
  - Compare ROC curves for multiple models.

Args:
    models: Dictionary of model_name -> model instance
    X_test: Test features
    y_test: Test labels
    save_path: Optional path to save figure

Returns:
    matplotlib figure
