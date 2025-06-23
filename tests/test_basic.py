import sk_stepwise as sw
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import make_regression
from hyperopt import hp
from sklearn.svm import SVC


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


def test_svm_conditional_hyperparameters():
    # Generate a classification dataset
    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
    # Convert regression target to binary classification for SVC
    y = (y > np.median(y)).astype(int)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = SVC(random_state=42, probability=True) # probability=True for cross_val_score with default scoring

    # Define a parameter space with conditional parameters for SVC
    # This demonstrates how hyperopt handles dependencies
    param_space_sequence = [
        {
            "C": hp.loguniform("C", np.log(0.1), np.log(10)),
            "kernel": hp.choice(
                "kernel",
                [
                    ("linear", {}),  # No extra params for linear
                    ("rbf", {"gamma": hp.loguniform("gamma_rbf", np.log(0.01), np.log(10))}),
                    ("poly", {
                        "degree": hp.quniform("degree", 2, 5, 1),
                        "gamma": hp.loguniform("gamma_poly", np.log(0.01), np.log(10)),
                        "coef0": hp.uniform("coef0", 0, 1)
                    }),
                ],
            ),
        }
    ]

    # The objective function will receive a flattened dictionary of parameters.
    # We need to unpack the 'kernel' choice.
    def unpack_kernel_params(params):
        kernel_name, kernel_specific_params = params["kernel"]
        new_params = {**params, "kernel": kernel_name}
        new_params.update(kernel_specific_params)
        return new_params

    # Override the objective function to handle the nested kernel choice
    class SVCOptimizer(sw.StepwiseHyperoptOptimizer):
        def objective(self, params: dict[str, sw.PARAM]) -> float:
            # Unpack the kernel and its specific parameters
            unpacked_params = unpack_kernel_params(params)
            
            # Clean integer parameters if any (e.g., degree for poly kernel)
            unpacked_params = self.clean_int_params(unpacked_params)

            current_params = {**self.best_params_, **unpacked_params}
            self.model.set_params(**current_params)
            
            # Use 'accuracy' for classification
            score = cross_val_score(
                self.model, self.X, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1
            )
            return -np.mean(score)

    # Specify 'degree' as an integer parameter
    int_params_to_clean = ["degree"]

    optimizer = SVCOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=10, # More evals to explore kernel choices
        random_state=42,
        int_params=int_params_to_clean,
        scoring="accuracy" # Set scoring for classification
    )

    optimizer.fit(X, y)

    assert optimizer.best_params_ is not None
    assert "C" in optimizer.best_params_
    assert "kernel" in optimizer.best_params_

    # Check that if 'poly' kernel is chosen, 'degree' is an integer
    if optimizer.best_params_["kernel"] == "poly":
        assert "degree" in optimizer.best_params_
        assert isinstance(optimizer.best_params_["degree"], int)
    
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ > 0 # Score should be positive for accuracy
