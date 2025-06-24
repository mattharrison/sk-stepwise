import pytest
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from hyperopt import hp

from src.sk_stepwise import StepwiseHyperoptOptimizer


@pytest.fixture
def catboost_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_catboost_regressor_initialization(catboost_data):
    X_train, _, y_train, _ = catboost_data

    model = CatBoostRegressor(random_state=42, silent=True)
    param_space_sequence = [
        {
            "iterations": hp.quniform("iterations", 10, 50, 10),
            "learning_rate": hp.loguniform("learning_rate", -3, 0),
        }
    ]

    optimizer = StepwiseHyperoptOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
    )

    # This fit is expected to fail initially, demonstrating the "Red" step of TDD
    # The failure might be due to parameter handling, or other CatBoost specifics
    optimizer.fit(X_train, y_train)

    assert optimizer.best_params_ is not None
    assert "iterations" in optimizer.best_params_
    assert "learning_rate" in optimizer.best_params_
    assert optimizer.best_score_ is not None
