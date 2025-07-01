import pytest
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from hyperopt import hp

from src.sk_stepwise import StepwiseOptimizer


@pytest.fixture
def xgboost_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.mark.slow
def test_xgboost_regressor_initialization(xgboost_data):
    X_train, X_test, y_train, y_test = xgboost_data

    model = XGBRegressor(random_state=42, n_jobs=1) # n_jobs=1 for consistent results

    param_space_sequence = [
        {
            "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
        },
        {
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
            "gamma": hp.loguniform("gamma", np.log(0.1), np.log(10)),
        },
        # Test conditional parameters for tree_method
        hp.choice(
            "tree_method_params",
            [
                {
                    "tree_method": "hist",
                    "grow_policy": hp.choice("grow_policy", ["depthwise", "lossguide"]),
                    "max_leaves": hp.quniform("max_leaves", 0, 64, 1), # Only for lossguide
                    "max_bin": hp.quniform("max_bin", 64, 512, 1),
                },
                {
                    "tree_method": "exact",
                },
            ]
        )
    ]

    int_params = [
        "n_estimators", "max_depth", "max_leaves", "max_bin"
    ]

    optimizer = StepwiseOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=3,
        random_state=42,
        int_params=int_params,
        scoring="neg_root_mean_squared_error",
        debug=True,
    )

    optimizer.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    assert optimizer.best_params_ is not None
    assert "n_estimators" in optimizer.best_params_
    assert "learning_rate" in optimizer.best_params_
    assert "max_depth" in optimizer.best_params_
    assert "subsample" in optimizer.best_params_
    assert "colsample_bytree" in optimizer.best_params_
    assert "gamma" in optimizer.best_params_
    assert "tree_method" in optimizer.best_params_

    if optimizer.best_params_["tree_method"] == "hist":
        assert "grow_policy" in optimizer.best_params_
        assert "max_bin" in optimizer.best_params_
        if optimizer.best_params_.get("grow_policy") == "lossguide":
            assert "max_leaves" in optimizer.best_params_
        else:
            assert "max_leaves" not in optimizer.best_params_
    else: # tree_method == "exact"
        assert "grow_policy" not in optimizer.best_params_
        assert "max_leaves" not in optimizer.best_params_
        assert "max_bin" not in optimizer.best_params_

    assert isinstance(optimizer.best_params_["n_estimators"], int)
    assert isinstance(optimizer.best_params_["max_depth"], int)
    if "max_leaves" in optimizer.best_params_:
        assert isinstance(optimizer.best_params_["max_leaves"], int)
    if "max_bin" in optimizer.best_params_:
        assert isinstance(optimizer.best_params_["max_bin"], int)

    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 0 # For neg_root_mean_squared_error, score is negative
