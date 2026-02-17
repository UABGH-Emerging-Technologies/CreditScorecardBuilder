# Configuration Files

This directory contains JSON configuration files for different components of the CreditScore system.

## Configuration Files

### `binning_args.json`
Configuration for OptBinning BinningProcess:
- Controls how variables are binned before scorecard creation
- Key parameters: `min_bin_size`, `max_n_bins`, `min_n_bins`
- Use `null` for package default values

### `scorecard_args.json`
Configuration for OptBinning Scorecard initialization:
- Controls scorecard scaling and formatting
- Key parameters: `scaling_method`, `reverse_scorecard`, `intercept_based`

### `cross_validation_args.json`
Configuration for hyperparameter search and cross-validation:
- Controls ElasticNet parameter grid search
- Key parameters: `param_grid`, `cv`, `scoring`
- Set `hyperparameter_search: false` to disable search

### `monitoring_args.json`
Configuration for scorecard monitoring and PSI calculation:
- Controls Population Stability Index (PSI) monitoring
- Key parameters: `psi_method`, `psi_n_bins`, `psi_min_bin_size`

### `variable_selection_args.json`
Configuration for automated variable selection:
- Controls thresholds for removing variables
- Key parameters: `min_unique_values`, `coefficient_threshold`

### `power_analysis_args.json`
Configuration for statistical power analysis:
- Controls sample size calculations and simulations
- Key parameters: `default_alpha`, `default_power`, simulation settings

## Usage

Configurations are automatically loaded by the `Config` class in `config.py`. 

To modify settings:
1. Edit the JSON files directly
2. Use `null` for package default values
3. Restart the application to pick up changes

## JSON Comments

Since JSON doesn't support comments, we use comment fields:
- Keys starting with `//` are ignored by the config loader
- These provide documentation for each parameter
- Example: `"// min_bin_size_comment": "Explanation of parameter"`

## Example Usage

```python
from config.config import Config

# Load specific configurations
binning_config = Config.binning_args()
scorecard_config = Config.scorecard_args()

# Or use through ConfigurationManager for backward compatibility
from CreditScore.utils import ConfigurationManager
config = ConfigurationManager.binning_args()
```