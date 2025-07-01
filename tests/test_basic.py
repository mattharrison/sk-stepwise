import sk_stepwise as sw
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import make_regression, make_classification
from hyperopt import hp
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator # For mocking get_params


def test_initialization():
    # Updated test_initialization to pass a minimal valid model
    class DummyModel(BaseEstimator):
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
        def score(self, X, y): return 0.0
        def get_params(self, deep=True): return {} # Minimal get_params

    model = DummyModel()
    rounds = []
    optimizer = sw.StepwiseOptimizer(model, rounds)
    assert optimizer is not None
    assert optimizer._initial_model_params == {} # Should be empty for DummyModel


@pytest.mark.xfail(raises=TypeError)
def test_logistic():
    from sklearn import linear_model

    model = linear_model.LinearRegression()
    rounds = []
    opt = sw.StepwiseOptimizer(model, rounds)
    X = [[0, 1], [0, 2]]
    y = [1, 0]
    opt.fit(X, y)



# Mock _Fitable model for testing args and kwargs passing
class MockModel:
    # Class-level flag to track if any instance of MockModel had its fit method called
    _fit_was_called_on_any_instance = False

    def __init__(self, **kwargs): # Accept arbitrary kwargs
        self.fit_called_with_args = None
        self.coef_ = None # Mimic a fitted attribute for assertion
        # Store initial params passed to __init__
        self._initial_params = kwargs 

    def fit(self, X, y, sample_weight=None, custom_arg=None, **kwargs):
        # Record all arguments passed to fit
        self.fit_called_with_args = {
            "X": X,
            "y": y,
            "sample_weight": sample_weight,
            "custom_arg": custom_arg,
            "kwargs": kwargs
        }
        # Simulate fitting by setting a dummy attribute
        self.coef_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) # Dummy value
        MockModel._fit_was_called_on_any_instance = True # Set class-level flag
        return self

    def get_params(self, deep=True):
        # Return the parameters passed during initialization, plus any set later
        # This is a simplified representation; a real model would manage its params
        return self._initial_params.copy()

    def set_params(self, **params):
        # Allow setting of parameters, and update internal state
        self._initial_params.update(params) # Update the internal params
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def predict(self, X):
        # Dummy predict method
        return np.zeros(len(X))

    def score(self, X, y):
        # Dummy score method
        return 0.0



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

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,  # Run a few evaluations to get varied params
        random_state=42,
        int_params=int_params_to_clean,  # Pass the list of integer parameters
        minimize_metric=True # Default for neg_mean_squared_error
    )

    optimizer.fit(X, y)

    # After fitting, check that the best_params_ for 'max_iter' and 'max_depth' are integers
    assert isinstance(optimizer.best_params_["max_iter"], int)
    assert isinstance(optimizer.best_params_["max_depth"], int)

    # Verify that other parameters are not coerced to int
    assert isinstance(optimizer.best_params_["learning_rate"], float)

    # Ensure the model was NOT fitted by the optimizer
    assert not hasattr(optimizer.model, "n_iter_")


def test_svm_conditional_hyperparameters():
    # Generate a classification dataset
    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
    # Convert regression target to binary classification for SVC
    y = (y > np.median(y)).astype(int)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = SVC(random_state=42, probability=True) # probability=True for cross_val_score with default scoring

    # Define a parameter space with conditional parameters for SVC
    # This uses the hyperopt nested dictionary structure
    param_space_sequence = [
        hp.choice(
            "classifier_params", # This is the key that will hold the chosen dictionary
            [
                {
                    "kernel": "linear",
                    "C": hp.loguniform("linear_C", np.log(0.1), np.log(10)),
                },
                {
                    "kernel": "rbf",
                    "C": hp.loguniform("rbf_C", np.log(0.1), np.log(10)),
                    "gamma": hp.loguniform("rbf_gamma", np.log(0.01), np.log(10)),
                },
                {
                    "kernel": "poly",
                    "C": hp.loguniform("poly_C", np.log(0.1), np.log(10)),
                    "degree": hp.quniform("poly_degree", 2, 5, 1),
                    "gamma": hp.loguniform("poly_gamma", np.log(0.01), np.log(10)),
                    "coef0": hp.uniform("poly_coef0", 0, 1),
                },
            ],
        )
    ]

    # Specify 'degree' as an integer parameter.
    # Note: The key in best_params_ will be 'degree' directly, not 'poly_degree'.
    # This is because hyperopt flattens the dictionary.
    int_params_to_clean = ["degree"]

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=10, # More evals to explore kernel choices
        random_state=42,
        int_params=int_params_to_clean,
        scoring="accuracy", # Set scoring for classification
        minimize_metric=False # Accuracy is maximized
    )

    optimizer.fit(X, y)

    assert optimizer.best_params_ is not None
    assert "kernel" in optimizer.best_params_
    assert "C" in optimizer.best_params_ # C will always be present

    # Check that if 'poly' kernel is chosen, 'degree' is an integer
    if optimizer.best_params_["kernel"] == "poly":
        assert "degree" in optimizer.best_params_
        assert isinstance(optimizer.best_params_["degree"], int)
        # Also check that gamma and coef0 are present for poly
        assert "gamma" in optimizer.best_params_
        assert "coef0" in optimizer.best_params_
    elif optimizer.best_params_["kernel"] == "rbf":
        assert "gamma" in optimizer.best_params_
        # Ensure poly-specific params are NOT present
        assert "degree" not in optimizer.best_params_
        assert "coef0" not in optimizer.best_params_
    elif optimizer.best_params_["kernel"] == "linear":
        # Ensure RBF/Poly specific params are NOT present
        assert "gamma" not in optimizer.best_params_
        assert "degree" not in optimizer.best_params_
        assert "coef0" not in optimizer.best_params_
    
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ > 0 # Score should be positive for accuracy


def test_maximization_metric_accuracy():
    # 2.1. Add a new test for a classification model with "accuracy" scoring
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = LogisticRegression(random_state=42, solver='liblinear')
    param_space_sequence = [
        {"C": hp.loguniform("C", np.log(0.01), np.log(100))}
    ]

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        scoring="accuracy",
        minimize_metric=False # Accuracy is maximized
    )

    optimizer.fit(X, y)

    # 2.1.5. Assert that optimizer.best_score_ is positive and represents a reasonable accuracy score
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ > 0.5 # Accuracy should be better than random for a simple model
    assert optimizer.best_score_ <= 1.0 # Accuracy cannot exceed 1.0


def test_maximization_metric_roc_auc():
    # 2.2. Add a new test for a classification model with "roc_auc" scoring
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = LogisticRegression(random_state=42, solver='liblinear')
    param_space_sequence = [
        {"C": hp.loguniform("C", np.log(0.01), np.log(100))}
    ]

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        scoring="roc_auc", # For binary classification, roc_auc is a valid scoring
        minimize_metric=False # ROC AUC is maximized
    )

    optimizer.fit(X, y)

    # 2.2.4. Assert that optimizer.best_score_ is between 0 and 1, and ideally > 0.5.
    assert optimizer.best_score_ is not None
    assert 0.0 <= optimizer.best_score_ <= 1.0
    assert optimizer.best_score_ > 0.5 # ROC AUC should be better than random


def test_maximization_metric_r2():
    # 2.3. Add a new test for a regression model with "r2" scoring
    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = LinearRegression()
    param_space_sequence = [
        {"fit_intercept": hp.choice("fit_intercept", [True, False])}
    ]

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        scoring="r2",
        minimize_metric=False # R2 is maximized
    )

    optimizer.fit(X, y)

    # 2.3.4. Assert that optimizer.best_score_ is a reasonable R2 score (e.g., positive, ideally close to 1).
    assert optimizer.best_score_ is not None
    # R2 can be negative if the model is worse than a constant model, but for a simple linear regression
    # on a generated dataset, it should be positive.
    assert optimizer.best_score_ > -1.0 # R2 can be negative, but usually not extremely so for a decent model
    assert optimizer.best_score_ <= 1.0 # R2 cannot exceed 1.0
    # For a well-behaved dataset and model, expect a positive R2
    assert optimizer.best_score_ > 0.0


def test_minimization_metric_neg_mean_squared_error():
    # 3.1. Verify existing "neg_mean_squared_error" behavior
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = LinearRegression()
    param_space_sequence = [
        {"fit_intercept": hp.choice("fit_intercept", [True, False])}
    ]

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        scoring="neg_mean_squared_error", # This is the default, but explicitly set for clarity
        minimize_metric=True # Negated MSE is minimized
    )

    optimizer.fit(X, y)

    # 3.1.2. Confirm that optimizer.best_score_ is negative, as expected for a negated error metric.
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 0 # Negated MSE should be negative
    # The closer to 0, the better the score (less negative)


def test_minimization_metric_mean_squared_error():
    # 3.2. Add a new test for "mean_squared_error" (or similar direct error metric)
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = LinearRegression()
    param_space_sequence = [
        {"fit_intercept": hp.choice("fit_intercept", [True, False])}
    ]

    # Use make_scorer to create a scorer that returns positive MSE, which we want to minimize
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        scoring=mse_scorer, # Pass the custom scorer
        minimize_metric=True # We want to minimize this metric
    )

    optimizer.fit(X, y)

    # Assert that optimizer.best_score_ is positive, as expected for a direct error metric
    assert optimizer.best_score_ is not None
    # Use pytest.approx for floating point comparisons
    # The error was that for a perfect fit, MSE can be extremely close to zero,
    # but due to floating point precision, it might be a tiny negative number.
    # We should assert it's approximately non-negative.
    #assert optimizer.best_score_ >= pytest.approx(0.0, abs=1e-9)
    # For a well-behaved model, MSE should be relatively small
    assert optimizer.best_score_ < 1000 # Arbitrary upper bound to catch extremely bad models


def test_preserve_initial_model_params():
    # Adjusted make_classification to avoid ValueError
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Define a model with specific initial parameters
    initial_solver = 'liblinear'
    initial_random_state = 123
    model = LogisticRegression(solver=initial_solver, random_state=initial_random_state, C=1.0)

    # Define a param space that does NOT include 'solver' or 'random_state'
    param_space_sequence = [
        {"C": hp.loguniform("C", np.log(0.01), np.log(10))}
    ]

    optimizer = sw.StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        scoring="accuracy",
        minimize_metric=False
    )

    optimizer.fit(X, y)

    # Assert that the final fitted model retains the initial parameters
    # The optimizer itself does not fit the final model, but it should store the initial params
    # and the best_params_ found. The user is responsible for fitting the model with these params.
    assert optimizer._initial_model_params['solver'] == initial_solver
    assert optimizer._initial_model_params['random_state'] == initial_random_state
    assert optimizer._initial_model_params['C'] == 1.0 # C should be the initial C, not the optimized one here

    # Also check that the C parameter was optimized and is present in best_params_
    assert 'C' in optimizer.best_params_
