import sk_stepwise as sw
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
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
        max_evals_per_step=1,
    )

    sample_weight = np.random.rand(len(y))
    custom_arg_value = "test_value"
    extra_kwarg = {"verbose": True}

    optimizer.fit(
        X, y, sample_weight=sample_weight, custom_arg=custom_arg_value, **extra_kwarg
    )

    # Check if the underlying model's fit method was called with the correct args and kwargs
    assert hasattr(mock_model, "fit_called_with_args")
    assert mock_model.fit_called_with_args[0] is sample_weight
    assert mock_model.fit_called_with_args[1] == custom_arg_value
    assert mock_model.fit_called_with_args[2] == extra_kwarg

    # Also check if the model was actually fitted
    assert hasattr(mock_model, "coef_")
    assert mock_model.coef_ is not None


def test_integer_hyperparameter_cleaning():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = HistGradientBoostingRegressor(random_state=42)

    # Define a parameter space where 'max_iter' and 'max_depth' might be sampled as floats
    # hp.quniform samples floats, so we need to ensure they are converted to int
    param_space_sequence = [
        {
            "max_iter": hp.quniform("max_iter", 10, 100, 1),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
        }
    ]

    # Specify which parameters should be treated as integers
    int_params_to_clean = ["max_iter", "max_depth"]

    optimizer = sw.StepwiseHyperoptOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,  # Run a few evaluations to get varied params
        random_state=42,
        int_params=int_params_to_clean,  # Pass the list of integer parameters
    )

    optimizer.fit(X, y)

    # After fitting, check that the best_params_ for 'max_iter' and 'max_depth' are integers
    assert isinstance(optimizer.best_params_["max_iter"], int)
    assert isinstance(optimizer.best_params_["max_depth"], int)

    # Verify that other parameters are not coerced to int
    assert isinstance(optimizer.best_params_["learning_rate"], float)

    # Ensure the model was fitted with the cleaned integer parameters
    assert hasattr(optimizer.model, "n_iter_")
    assert optimizer.model.n_iter_ is not None
