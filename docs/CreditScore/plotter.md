# CreditScore.plotter

Core plotting helpers for model evaluation and interpretation.

## Classes

### `PlotBase`

Mixin that provides:
    • consistent labelling
    • basic styling helpers
    • pathlib-based saving
Every subclass should only worry about the *data* it wants to visualise.

#### Methods

- `label(fig, name)`
  - Assign a deterministic label to *fig* and return the same figure so
callers can keep using it in one expression.

- `style_axes(ax, title, xlabel, ylabel)`
  - Apply a minimal, common styling to *ax*.

### `FeatureImportancePlotter`

Plots horizontal bar charts of feature importances or coefficients.

#### Methods

- `plot_importance(self, features, magnitudes, save_dir)`
  - Parameters
----------
features
    Display labels for the bars.
magnitudes
    Importance magnitude (positive numbers, same order as *features*).
save_dir
    Optional directory in which the figure will be stored.

- `plot_coefficients(self, coef_series, save_dir)`
  - Bar plot of the absolute values of a coefficient vector.

Parameters
----------
coef_series
    pd.Series where the index contains feature names and the values are
    (signed) coefficients.

### `ModelPerformancePlotter`

Convenience wrapper around a *ModelEvaluator* implementation that already
provides individual .plot_X() routines (ROC, PR, …).

#### Methods

- `plot_all(self, y_train, y_test, y_train_pred, y_test_pred, save_dir)`
  - Produce ROC, PR, CAP and KS plots.

Returns
-------
dict
    Mapping of a short, stable key → figure.

### `SHAPPlotter`

Generates SHAP decision plots for a (binning→logistic) score-card type
pipeline.  All Streamlit-specific display logic has been removed – this
class is framework agnostic and returns plain matplotlib figures.

#### Methods

- `decision_plots(self, X_train, X_test, y_train, thresholds, save_dir, use_baseline_rate)`
  - Produce SHAP decision plots for each threshold.

Returns
-------
dict
    key = e.g. "shap_decision_10_all"
    value = matplotlib.figure.Figure
