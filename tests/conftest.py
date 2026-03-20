import numpy as np
import pandas as pd
import pytest
import sk_stepwise as sw
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class ParamScoreRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.full(len(X), self.alpha + self.beta, dtype=float)

    def score(self, X, y):
        return 100.0 - (self.alpha - 2) ** 2 - (self.beta - 5) ** 2


class ArbitraryParamRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tree_count=1, shrinkage=0.1):
        self.tree_count = tree_count
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.full(len(X), self.tree_count * self.shrinkage, dtype=float)

    def score(self, X, y):
        return float(self.tree_count) + float(self.shrinkage)


class WeightedLinearRegression(LinearRegression):
    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self.fit_call_sample_weights = []

    def fit(self, X, y, sample_weight=None):
        self.fit_call_sample_weights.append(sample_weight)
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.fixture
def readme_regression_frame():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((100, 5)), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(rng.random(100))
    return X, y


@pytest.fixture
def regression_data():
    return make_regression(n_samples=60, n_features=4, random_state=0)


@pytest.fixture
def small_regression_data():
    return make_regression(n_samples=40, n_features=3, random_state=0)


@pytest.fixture
def random_forest_step_space():
    return [
        {"n_estimators": sw.Int(10, 20)},
        {"max_depth": sw.Int(2, 4)},
    ]


@pytest.fixture
def random_forest_small_space():
    return [
        {"n_estimators": sw.Int(10, 20)},
        {"max_depth": sw.Int(2, 4)},
    ]


@pytest.fixture
def readme_random_forest_space():
    return [
        {"n_estimators": sw.Int(10, 20)},
        {"max_depth": sw.Int(3, 6)},
        {"min_samples_split": sw.Float(0.1, 0.9)},
    ]


@pytest.fixture
def random_forest_estimator():
    return RandomForestRegressor(random_state=0)


@pytest.fixture
def param_score_regressor():
    return ParamScoreRegressor()


@pytest.fixture
def arbitrary_param_regressor():
    return ArbitraryParamRegressor()


@pytest.fixture
def weighted_linear_regression():
    return WeightedLinearRegression()


@pytest.fixture
def weighted_regression_data():
    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    sample_weight = np.linspace(1.0, 2.0, num=len(y))
    return X, y, sample_weight
