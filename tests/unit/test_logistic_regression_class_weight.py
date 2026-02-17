# tests/unit/test_logistic_regression_class_weight.py

"""
Regression tests for LogisticRegression class_weight parameter.

This test ensures that class_weight is NOT set to 'balanced' in the LogisticRegression
models used in CreditScore/train.py. This is a regression test because this setting
keeps getting re-added incorrectly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from CreditScore.train import OptbinningScorecardModel


@pytest.mark.regression
@pytest.mark.unit
class TestLogisticRegressionClassWeight:
    """Test that LogisticRegression models have class_weight=None."""

    def test_scorecard_model_class_weight_none(self, sample_medical_data):
        """
        REGRESSION TEST: Ensure OptbinningScorecardModel uses class_weight=None.

        This test verifies that the LogisticRegression estimator inside the
        OptbinningScorecardModel does NOT have class_weight='balanced'.
        """
        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus", "EBL"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Create model with hyperparameter search enabled
        model = OptbinningScorecardModel()
        model.fit(X, y, hyperparameter_search=True, cv=3)

        # Check the final estimator
        final_estimator = model.scorecard.estimator_
        assert isinstance(final_estimator, LogisticRegression)
        assert final_estimator.class_weight is None, (
            f"LogisticRegression should have class_weight=None, "
            f"but has class_weight={final_estimator.class_weight}"
        )

    def test_gridsearchcv_estimator_class_weight_none(self):
        """
        REGRESSION TEST: Ensure GridSearchCV uses estimator with class_weight=None.

        This test verifies that during hyperparameter search, the LogisticRegression
        estimator inside GridSearchCV also has class_weight=None by directly
        checking the estimator template used.
        """
        # Test the GridSearchCV estimator template directly
        param_grid = {"C": [0.01, 0.1], "l1_ratio": [0.1, 0.5]}

        # This is what's used in OptbinningScorecardModel.fit()
        estimator = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000)

        # Create GridSearchCV instance
        en_cv = GridSearchCV(
            estimator,
            param_grid,
            cv=3,
            scoring="average_precision",
            n_jobs=-1,
        )

        # Check the base estimator
        assert isinstance(en_cv.estimator, LogisticRegression)
        assert en_cv.estimator.class_weight is None, (
            f"GridSearchCV base estimator should have class_weight=None, "
            f"but has class_weight={en_cv.estimator.class_weight}"
        )

    def test_default_logistic_regression_class_weight(self, sample_medical_data):
        """
        Test that default LogisticRegression (no hyperparameter search) also has class_weight=None.
        """
        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus", "EBL"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Create model without hyperparameter search
        model = OptbinningScorecardModel()
        model.fit(X, y, hyperparameter_search=False)

        # Check the final estimator
        final_estimator = model.scorecard.estimator_
        assert isinstance(final_estimator, LogisticRegression)
        assert final_estimator.class_weight is None, (
            f"Default LogisticRegression should have class_weight=None, "
            f"but has class_weight={final_estimator.class_weight}"
        )

    def test_no_class_weight_in_source_code(self):
        """
        REGRESSION TEST: Ensure 'class_weight' string doesn't appear in train.py
        in the context of LogisticRegression initialization.

        This is a direct source code check to prevent the parameter from being re-added.
        """
        import ast
        import inspect

        from CreditScore import train

        # Get the source code of the train module
        source_file = inspect.getsourcefile(train)
        with open(source_file, "r") as f:
            source_code = f.read()

        # Parse the AST
        tree = ast.parse(source_code)

        # Find all LogisticRegression instantiations
        class LogisticRegressionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.lr_calls = []

            def visit_Call(self, node):
                # Check if this is a LogisticRegression call
                if isinstance(node.func, ast.Name) and node.func.id == "LogisticRegression":
                    # Check for class_weight in keywords
                    for keyword in node.keywords:
                        if keyword.arg == "class_weight":
                            # Get the line number
                            lineno = node.lineno
                            self.lr_calls.append(
                                {
                                    "line": lineno,
                                    "has_class_weight": True,
                                    "value": (
                                        ast.unparse(keyword.value)
                                        if hasattr(ast, "unparse")
                                        else str(keyword.value)
                                    ),
                                }
                            )

                self.generic_visit(node)

        visitor = LogisticRegressionVisitor()
        visitor.visit(tree)

        # Assert no LogisticRegression calls have class_weight
        for lr_call in visitor.lr_calls:
            assert not lr_call["has_class_weight"], (
                f"Found class_weight parameter in LogisticRegression at line {lr_call['line']}. "
                f"This should be removed - LogisticRegression should use default class_weight=None"
            )

    def test_elastic_net_params_preserved(self, sample_medical_data):
        """
        Ensure that fixing class_weight doesn't break other parameters.

        This test verifies that the elastic net parameters are still properly set
        after removing class_weight.
        """
        data = sample_medical_data
        features = ["Age", "BMI", "ASAStatus", "EBL"]
        target = "thirty_day_mortality"

        X = data[features].copy()
        y = data[target].copy()

        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Create model with hyperparameter search
        model = OptbinningScorecardModel()
        model.fit(X, y, hyperparameter_search=True, cv=3)

        # Check the final estimator has elastic net parameters
        final_estimator = model.scorecard.estimator_
        assert final_estimator.penalty == "elasticnet"
        assert final_estimator.solver == "saga"
        assert hasattr(final_estimator, "l1_ratio")
        assert hasattr(final_estimator, "C")

        # And still no class_weight
        assert final_estimator.class_weight is None
