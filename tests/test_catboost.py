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

    # Define param_space_sequence based on the "Which Parameters to tune?" table
    # Correctly define conditional grow_policy based on boosting_type
    param_space_sequence = [
        hp.choice(
            "catboost_params", # A single top-level choice for the entire parameter set
            [
                # Option 1: Ordered Boosting (grow_policy must be SymmetricTree)
                {
                    "boosting_type": "Ordered",
                    "grow_policy": "SymmetricTree", # Forced to SymmetricTree for Ordered boosting
                    # Tuned parameters
                    "learning_rate": hp.loguniform("learning_rate_ordered", np.log(0.01), np.log(0.3)),
                    "random_strength": hp.loguniform("random_strength_ordered", np.log(0.1), np.log(10)),
                    "one_hot_max_size": hp.quniform("one_hot_max_size_ordered", 2, 20, 1),
                    "l2_leaf_reg": hp.loguniform("l2_leaf_reg_ordered", np.log(1), np.log(10)),
                    # Conditional bagging_temperature
                    "bootstrap_type": hp.choice(
                        "bootstrap_type_ordered",
                        [
                            ("Bayesian", {"bagging_temperature": hp.uniform("bagging_temperature_ordered_bayesian", 0.0, 1.0)}),
                            "Bernoulli",
                            "MVS"
                        ]
                    ),
                    "iterations": hp.quniform("iterations_ordered", 10, 200, 10),
                    "use_best_model": hp.choice("use_best_model_ordered", [True, False]),
                    "eval_metric": hp.choice("eval_metric_ordered", ["RMSE", "MAE"]), # Example metrics for regression
                    "od_type": hp.choice("od_type_ordered", ["IncToDec", "Iter"]), # Overfitting detector type
                    "od_pval": hp.loguniform("od_pval_ordered", np.log(1e-10), np.log(1.0)), # Overfitting detector threshold
                    "od_wait": hp.quniform("od_wait_ordered", 10, 50, 5), # Number of iterations to wait
                    "depth": hp.quniform("depth_ordered", 4, 10, 1),
                    "border_count": hp.quniform("border_count_ordered", 32, 255, 1), # Number of splits for numeric features
                    "has_time": hp.choice("has_time_ordered", [True, False]),
                    "min_data_in_leaf": hp.quniform("min_data_in_leaf_ordered", 1, 30, 1),
                    "max_leaves": hp.quniform("max_leaves_ordered", 16, 128, 16), # Max leaves in a tree
                    "max_ctr_complexity": hp.quniform("max_ctr_complexity_ordered", 1, 8, 1),
                    "subsample": hp.uniform("subsample_ordered", 0.6, 1.0),
                    "colsample_bylevel": hp.uniform("colsample_bylevel_ordered", 0.6, 1.0),
                    #"used_ram_limit": hp.choice("used_ram_limit_ordered", [None, "1GB", "2GB"]), # Example RAM limit
                    "objective": hp.choice("objective_ordered", ["RMSE", "MAE"]), # Objective function
                },
                # Option 2: Plain Boosting (grow_policy can be any)
                {
                    "boosting_type": "Plain",
                    "grow_policy": hp.choice("grow_policy_plain", ["SymmetricTree", "Depthwise", "Lossguide"]),
                    # Tuned parameters (with different labels to avoid name collisions in hyperopt)
                    "learning_rate": hp.loguniform("learning_rate_plain", np.log(0.01), np.log(0.3)),
                    "random_strength": hp.loguniform("random_strength_plain", np.log(0.1), np.log(10)),
                    "one_hot_max_size": hp.quniform("one_hot_max_size_plain", 2, 20, 1),
                    "l2_leaf_reg": hp.loguniform("l2_leaf_reg_plain", np.log(1), np.log(10)),
                    # Conditional bagging_temperature
                    "bootstrap_type": hp.choice(
                        "bootstrap_type_plain",
                        [
                            ("Bayesian", {"bagging_temperature": hp.uniform("bagging_temperature_plain_bayesian", 0.0, 1.0)}),
                            "Bernoulli",
                            "MVS"
                        ]
                    ),
                    "iterations": hp.quniform("iterations_plain", 10, 200, 10),
                    "use_best_model": hp.choice("use_best_model_plain", [True, False]),
                    "eval_metric": hp.choice("eval_metric_plain", ["RMSE", "MAE"]),
                    "od_type": hp.choice("od_type_plain", ["IncToDec", "Iter"]),
                    "od_pval": hp.loguniform("od_pval_plain", np.log(1e-10), np.log(1.0)),
                    "od_wait": hp.quniform("od_wait_plain", 10, 50, 5),
                    "depth": hp.quniform("depth_plain", 4, 10, 1),
                    "border_count": hp.quniform("border_count_plain", 32, 255, 1),
                    "has_time": hp.choice("has_time_plain", [True, False]),
                    "min_data_in_leaf": hp.quniform("min_data_in_leaf_plain", 1, 30, 1),
                    "max_leaves": hp.quniform("max_leaves_plain", 16, 128, 16),
                    "max_ctr_complexity": hp.quniform("max_ctr_complexity_plain", 1, 8, 1),
                    "subsample": hp.uniform("subsample_plain", 0.6, 1.0),
                    "colsample_bylevel": hp.uniform("colsample_bylevel_plain", 0.6, 1.0),
                    #"used_ram_limit": hp.choice("used_ram_limit_plain", [None, "1GB", "2GB"]),
                    "objective": hp.choice("objective_plain", ["RMSE", "MAE"]),
                },
            ]
        )
    ]

    # Specify integer parameters for CatBoost.
    catboost_int_params = [
        "iterations", "depth", "one_hot_max_size", "od_wait",
        "border_count", "min_data_in_leaf", "max_leaves", "max_ctr_complexity"
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
    assert "max_leaves" in optimizer.best_params_
    assert "max_ctr_complexity" in optimizer.best_params_
    assert "colsample_bylevel" in optimizer.best_params_
    assert "used_ram_limit" in optimizer.best_params_
    assert "objective" in optimizer.best_params_


    # Assert that if boosting_type is 'Ordered', grow_policy is 'SymmetricTree'
    if optimizer.best_params_["boosting_type"] == "Ordered":
        assert optimizer.best_params_["grow_policy"] == "SymmetricTree"
    
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 0 # For neg_root_mean_squared_error, score is negative
