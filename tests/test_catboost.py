import pytest
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from hyperopt import hp
import numpy as np

from src.sk_stepwise import StepwiseHyperoptOptimizer


@pytest.fixture
def catboost_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_catboost_regressor_initialization(catboost_data):
    X_train, _, y_train, _ = catboost_data

    model = CatBoostRegressor(random_state=42, silent=True)

    # Define conditional bootstrap_type and bagging_temperature
    bootstrap_type_and_temp = hp.choice(
        "bootstrap_choice",
        [
            ("Bayesian", {"bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0)}),
            "Bernoulli",
            "MVS"
        ]
    )

    # Define param_space_sequence organized into logical steps
    param_space_sequence = [
        # Combined Step 1 (formerly Step 5 & 1): Boosting Type, Grow Policy, and Core Tree Parameters
        hp.choice(
            "boosting_strategy_and_core_params", # A choice for the boosting strategy sub-space
            [
                # Option 1: Ordered Boosting (grow_policy must be SymmetricTree)
                {
                    "boosting_type": "Ordered",
                    "grow_policy": "SymmetricTree", # Forced to SymmetricTree for Ordered boosting
                    "iterations": hp.quniform("iterations_ordered", 10, 200, 10),
                    "depth": hp.quniform("depth_ordered", 4, 10, 1),
                    # max_leaves is NOT applicable here
                },
                # Option 2: Plain Boosting (grow_policy can be any)
                hp.choice(
                    "plain_boosting_grow_policy_and_core_params",
                    [
                        {
                            "boosting_type": "Plain",
                            "grow_policy": "SymmetricTree",
                            "iterations": hp.quniform("iterations_plain_symmetric", 10, 200, 10),
                            "depth": hp.quniform("depth_plain_symmetric", 4, 10, 1),
                            # max_leaves is NOT applicable here
                        },
                        {
                            "boosting_type": "Plain",
                            "grow_policy": "Depthwise",
                            "iterations": hp.quniform("iterations_plain_depthwise", 10, 200, 10),
                            "depth": hp.quniform("depth_plain_depthwise", 4, 10, 1),
                            # max_leaves is NOT applicable here
                        },
                        {
                            "boosting_type": "Plain",
                            "grow_policy": "Lossguide",
                            "iterations": hp.quniform("iterations_plain_lossguide", 10, 200, 10),
                            "depth": hp.quniform("depth_plain_lossguide", 4, 10, 1),
                            "max_leaves": hp.quniform("max_leaves", 16, 128, 16), # max_leaves only with Lossguide
                        },
                    ]
                ),
            ]
        ),
        # Step 2 (formerly Step 4): Feature Handling
        {
            "one_hot_max_size": hp.quniform("one_hot_max_size", 2, 20, 1),
            "border_count": hp.quniform("border_count", 32, 255, 1),
            "max_ctr_complexity": hp.quniform("max_ctr_complexity", 1, 8, 1),
            "has_time": hp.choice("has_time", [True, False]),
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 1, 30, 1),
        },
        # Step 3 (formerly Step 2): Regularization & Overfitting Prevention
        {
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1), np.log(10)),
            "random_strength": hp.loguniform("random_strength", np.log(0.1), np.log(10)),
            "od_type": hp.choice("od_type", ["IncToDec", "Iter"]),
            "od_pval": hp.loguniform("od_pval", np.log(1e-10), np.log(1.0)),
            "od_wait": hp.quniform("od_wait", 10, 50, 5),
        },
        # Step 4 (formerly Step 3): Learning Process & Data Sampling
        {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.6, 1.0),
            "bootstrap_type": bootstrap_type_and_temp, # Use the defined conditional choice
        },
        # Step 5 (formerly Step 6): Miscellaneous/Advanced
        {
            "use_best_model": hp.choice("use_best_model", [True, False]),
            "eval_metric": hp.choice("eval_metric", ["RMSE", "MAE"]), # Example metrics for regression
            "objective": hp.choice("objective", ["RMSE", "MAE"]), # Objective function
            "used_ram_limit": hp.choice("used_ram_limit", [None, "1GB", "2GB"]), # Example RAM limit
        }
    ]

    # Specify integer parameters for CatBoost.
    # Note: When using nested hp.choice, the keys in best_params_ will be flattened.
    # So, 'iterations_ordered' or 'iterations_plain_symmetric' will become 'iterations'.
    # We need to list the final parameter names that should be integers.
    catboost_int_params = [
        "iterations", "depth", "max_leaves", "od_wait",
        "one_hot_max_size", "border_count", "max_ctr_complexity", "min_data_in_leaf"
    ]

    optimizer = StepwiseHyperoptOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=10,
        random_state=42,
        int_params=catboost_int_params,
        scoring="neg_root_mean_squared_error" # Appropriate scoring for RMSE loss
    )

    optimizer.fit(X_train, y_train)

    assert optimizer.best_params_ is not None
    # Assertions for tuned parameters
    assert "iterations" in optimizer.best_params_
    assert "learning_rate" in optimizer.best_params_
    assert "depth" in optimizer.best_params_
    assert "l2_leaf_reg" in optimizer.best_params_
    assert "random_strength" in optimizer.best_params_
    assert "one_hot_max_size" in optimizer.best_params_
    assert "min_data_in_leaf" in optimizer.best_params_
    assert "boosting_type" in optimizer.best_params_
    assert "grow_policy" in optimizer.best_params_
    assert "subsample" in optimizer.best_params_
    assert "colsample_bylevel" in optimizer.best_params_
    assert "bootstrap_type" in optimizer.best_params_
    
    # Assert bagging_temperature only if bootstrap_type is Bayesian
    if optimizer.best_params_["bootstrap_type"] == "Bayesian":
        assert "bagging_temperature" in optimizer.best_params_
    else:
        assert "bagging_temperature" not in optimizer.best_params_

    assert "use_best_model" in optimizer.best_params_
    assert "eval_metric" in optimizer.best_params_
    assert "od_type" in optimizer.best_params_
    assert "od_pval" in optimizer.best_params_
    assert "od_wait" in optimizer.best_params_
    assert "border_count" in optimizer.best_params_
    assert "has_time" in optimizer.best_params_
    assert "max_ctr_complexity" in optimizer.best_params_
    assert "used_ram_limit" in optimizer.best_params_
    assert "objective" in optimizer.best_params_

    # Assert max_leaves only if grow_policy is Lossguide
    if optimizer.best_params_["grow_policy"] == "Lossguide":
        assert "max_leaves" in optimizer.best_params_
        assert isinstance(optimizer.best_params_["max_leaves"], int)
    else:
        assert "max_leaves" not in optimizer.best_params_


    # Assert that if boosting_type is 'Ordered', grow_policy is 'SymmetricTree'
    if optimizer.best_params_["boosting_type"] == "Ordered":
        assert optimizer.best_params_["grow_policy"] == "SymmetricTree"
    
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 0 # For neg_root_mean_squared_error, score is negative
