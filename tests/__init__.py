# tests/__init__.py

"""
Test suite for CreditScore project.

This test suite includes comprehensive tests for:
- Regression testing for critical bugs (binned vs raw data)
- ASA variable auto-detection functionality
- Configuration loading and parameter filtering
- Automated variable selection pipeline
- Missing data handling across different model types

Key regression tests:
1. OptBinning scorecard methods receive raw data, not binned data
2. Configuration loading properly filters unsupported parameters
3. Variable selection improves model performance with extreme class imbalance
4. Missing data is handled appropriately for different model types
"""

__version__ = "1.0.0"
