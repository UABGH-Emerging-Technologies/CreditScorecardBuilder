# CreditScore Test Suite

This comprehensive test suite covers critical functionality and regression testing for the CreditScore project, with special focus on the key issues discovered during development.

We may decide this is too many or some are irrelevant, but the missing data and binned-data usage tests are important, because those are subtle problems that are made hard to spot by default arguments.

## Overview

The test suite is organized into several categories addressing specific areas of functionality:

### 🔴 Critical Regression Tests

**Binned vs Raw Data Issue** (`test_binned_vs_raw_data_regression.py`)
- **Problem**: OptBinning scorecard methods were receiving binned data instead of raw data
- **Impact**: Models only produced 2 unique scores (54 and 55), severe performance degradation
- **Tests**: Verify scorecard methods (fit, score, predict_proba) receive unbinned data
- **Markers**: `@pytest.mark.regression`, `@pytest.mark.optbinning`

**Configuration Parameter Filtering** (`test_config_loading.py`)
- **Problem**: `verbose` parameter passed to `ScorecardEvaluator.initialize_monitoring()` but method doesn't accept it
- **Impact**: TypeError crashes in monitoring initialization
- **Tests**: Verify configuration loading filters unsupported parameters
- **Markers**: `@pytest.mark.config`, `@pytest.mark.regression`

### 🎯 Feature-Specific Tests

**ASA Detection** (`test_asa_detection.py`)
- Auto-detection of ASA status variables based on naming patterns
- Case-insensitive detection with configurable thresholds
- Integration with binning process for categorical treatment
- **Markers**: `@pytest.mark.asa`

**Variable Selection** (`test_variable_selection.py`)
- Two-stage automated variable selection pipeline
- Post-binning filter (removes variables with insufficient variability)
- Post-regularization filter (removes zero-coefficient variables)
- Addresses extreme class imbalance issues
- **Markers**: `@pytest.mark.variable_selection`

**Missing Data Handling** (`test_missing_data_handling.py`)
- OptBinning vs statsmodels differences in missing value handling
- Proper preservation of missing values for OptBinning models
- Target variable missing value removal
- **Markers**: `@pytest.mark.missing_data`

## Test Categories and Markers

### Markers
- `regression`: Critical regression tests for major bugs
- `unit`: Unit tests for individual components
- `integration`: Integration tests across multiple components
- `config`: Configuration loading and management tests
- `asa`: ASA variable detection tests
- `missing_data`: Missing data handling tests
- `variable_selection`: Automated variable selection tests
- `optbinning`: OptBinning-specific functionality tests
- `slow`: Tests that take longer to run

### Running Tests

#### Quick Commands
```bash
# Run critical regression tests only
python run_tests.py regression

# Run quick test suite (excludes slow tests)
python run_tests.py quick

# Run full test suite with coverage
python run_tests.py full
```

#### Category-Specific Testing
```bash
# Run all tests
python run_tests.py --category all

# Run only regression tests
python run_tests.py --category regression

# Run configuration tests
python run_tests.py --category config

# Run ASA detection tests
python run_tests.py --category asa

# Run variable selection tests
python run_tests.py --category variable_selection

# Run missing data tests
python run_tests.py --category missing_data
```

#### Advanced Options
```bash
# Run with coverage reporting
python run_tests.py --coverage

# Include slow tests
python run_tests.py --slow

# Run specific test file
python run_tests.py --file tests/unit/test_asa_detection.py

# Run specific test function
python run_tests.py --test test_scorecard_fit_receives_raw_data

# Verbose output
python run_tests.py --verbose

# Quiet output
python run_tests.py --quiet
```

#### Direct pytest Commands
```bash
# Run all tests
pytest

# Run regression tests only
pytest -m regression

# Run fast tests only
pytest -m "not slow"

# Run with coverage
pytest --cov=CreditScore --cov=config --cov-report=html

# Run specific test
pytest tests/unit/test_binned_vs_raw_data_regression.py::TestBinnedVsRawDataRegression::test_scorecard_fit_receives_raw_data
```

## Key Test Files

### `test_binned_vs_raw_data_regression.py`
**Purpose**: Prevent regression of the critical binned vs raw data bug

**Critical Tests**:
- `test_scorecard_fit_receives_raw_data()`: Ensures Scorecard.fit() gets raw data
- `test_scorecard_score_receives_raw_data()`: Ensures score() gets raw data
- `test_binning_process_vs_scorecard_data_flow()`: Verifies complete data flow
- `test_regression_unique_scores_count()`: Ensures >2 unique scores

### `test_config_loading.py`
**Purpose**: Ensure configuration system works correctly

**Critical Tests**:
- `test_monitoring_args_filters_verbose()`: Regression test for verbose parameter
- `test_monitoring_config_compatible_with_evaluator()`: Integration test
- `test_load_config_filters_comments()`: Comment filtering functionality

### `test_asa_detection.py`
**Purpose**: Verify ASA variable auto-detection

**Key Tests**:
- `test_detect_asa_variables_basic()`: Core detection functionality
- `test_detect_asa_variables_case_insensitive()`: Case handling
- `test_asa_detection_in_binning_process()`: Integration test

### `test_variable_selection.py`
**Purpose**: Test automated variable selection pipeline

**Key Tests**:
- `test_select_variables_full_pipeline_integration()`: End-to-end pipeline
- `test_variable_selection_improves_score_diversity()`: Regression test for score diversity
- `test_select_variables_with_extreme_imbalance()`: Handles edge cases

### `test_missing_data_handling.py`
**Purpose**: Verify missing data handling differences

**Key Tests**:
- `test_optbinning_preserves_missing_values()`: OptBinning behavior
- `test_statsmodels_requires_complete_cases()`: Statsmodels behavior
- `test_missing_target_values_always_removed()`: Critical safety check

## Test Data and Fixtures

### `conftest.py` Fixtures
- `sample_medical_data()`: Realistic medical data with proper class imbalance
- `sample_data_with_issues()`: Data with quality issues for testing edge cases
- `temp_config_dir()`: Temporary configuration directory for testing
- `trained_scorecard_model()`: Pre-trained model for testing
- `mock_config_loader()`: Mocked configuration system

## Coverage Requirements

The test suite aims for:
- **Critical paths**: 100% coverage for regression-prone areas
- **Overall coverage**: >70% for main codebase
- **Configuration**: 100% coverage for config loading
- **ASA detection**: 100% coverage for detection logic
- **Variable selection**: >90% coverage for selection pipeline

## Running in CI/CD

```yaml
# Example GitHub Actions configuration
- name: Run Critical Regression Tests
  run: python run_tests.py regression

- name: Run Full Test Suite
  run: python run_tests.py --coverage

- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

## Test Development Guidelines

1. **Regression Tests**: Always write regression tests for critical bugs
2. **Test Data**: Use realistic medical data patterns in fixtures
3. **Markers**: Apply appropriate pytest markers for categorization
4. **Mocking**: Mock external dependencies, test real integration paths
5. **Documentation**: Include clear docstrings explaining test purpose
6. **Performance**: Mark slow tests with `@pytest.mark.slow`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Missing Dependencies**: Run `pip install -e ".[dev]"` 
3. **Test Data Issues**: Check that fixtures generate valid data
4. **OptBinning Warnings**: Warnings are filtered in pytest.ini
5. **Slow Tests**: Use `--category regression` for quick validation

### Debug Commands
```bash
# Run single test with full output
pytest -vvs tests/unit/test_binned_vs_raw_data_regression.py::TestBinnedVsRawDataRegression::test_scorecard_fit_receives_raw_data

# Debug with pdb
pytest --pdb tests/unit/test_config_loading.py

# Show test durations
pytest --durations=10
```

This test suite provides comprehensive coverage of the CreditScore functionality with special attention to the critical bugs and edge cases discovered during development.