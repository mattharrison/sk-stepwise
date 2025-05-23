import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import sk_stepwise as sw
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from hyperopt import hp


def test_initialization():
    model = None
    rounds = []
    optimizer = sw.StepwiseHyperoptOptimizer(model, rounds)
    assert optimizer is not None


@pytest.mark.xfail(raises=TypeError)
def test_logistic():
    from sklearn import linear_model

    model = linear_model.LinearRegression()
    rounds = []
    opt = sw.StepwiseHyperoptOptimizer(model, rounds)
    X = [[0, 1], [0, 2]]
    y = [1, 0]
    opt.fit(X, y)


@pytest.mark.matt
def test_matt():
    assert "matt" == "matt"


# Mock _Fitable model for testing args and kwargs passing
class MockModel(LinearRegression):
    def fit(self, X, y, sample_weight=None, custom_arg=None, **kwargs):
        self.fit_called_with_args = (sample_weight, custom_arg, kwargs)
        super().fit(X, y, sample_weight=sample_weight)
        return self

def test_fit_args_kwargs_passing():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    mock_model = MockModel()
    param_space_sequence = [
        {"fit_intercept": hp.choice("fit_intercept", [True, False])}
    ]

    optimizer = sw.StepwiseHyperoptOptimizer(
        model=mock_model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=1
    )

    sample_weight = np.random.rand(len(y))
    custom_arg_value = "test_value"
    extra_kwarg = {"verbose": True}

    optimizer.fit(X, y, sample_weight=sample_weight, custom_arg=custom_arg_value, **extra_kwarg)

    # Check if the underlying model's fit method was called with the correct args and kwargs
    assert hasattr(mock_model, 'fit_called_with_args')
    assert mock_model.fit_called_with_args[0] is sample_weight
    assert mock_model.fit_called_with_args[1] == custom_arg_value
    assert mock_model.fit_called_with_args[2] == extra_kwarg

    # Also check if the model was actually fitted
    assert hasattr(mock_model, 'coef_')
    assert mock_model.coef_ is not None
