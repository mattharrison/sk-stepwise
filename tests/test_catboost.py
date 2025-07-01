import pytest
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from hyperopt import hp
import numpy as np

from src.sk_stepwise import CatBoostStepwiseOptimizer


@pytest.fixture
def catboost_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.mark.slow
def test_catboost_regressor_initialization(catboost_data):
    X_train, X_test, y_train, y_test = catboost_data

    model = CatBoostRegressor(random_state=42, silent=True)

    # Define conditional bootstrap_type and bagging_temperature
    # These are now directly part of the param_space_sequence structure
    bootstrap_type_options = [
        {"bootstrap_type": "Bayesian", "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0)},
        {"bootstrap_type": "Bernoulli"},
        {"bootstrap_type": "MVS"}
    ]

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
                },
                # Option 2: Plain Boosting (grow_policy can be any)
                {
                    'boosting_type': 'Plain',
                    'grow_policy': hp.choice(
                        'grow_policy_plain',
                        [
                            'SymmetricTree',
                            'Depthwise',
                            'Lossguide'
                        ]
                    ),
                    "iterations": hp.quniform("iterations_plain", 10, 200, 10),
                    "depth": hp.quniform("depth_plain", 4, 10, 1),
                },
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
        },
        # Step 4 (formerly Step 3): Learning Process & Data Sampling
        {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.6, 1.0),
            "bootstrap_params": hp.choice("bootstrap_params", bootstrap_type_options), # Embed the choice directly
        },
        # Step 5 (formerly Step 6): Miscellaneous/Advanced
        {
            "use_best_model": hp.choice("use_best_model", [True, False]),
            "eval_metric": hp.choice("eval_metric", ["RMSE", "MAE"]), # Example metrics for regression
            "objective": hp.choice("objective", ["RMSE", "MAE"]), # Objective function
        }
    ]

    # Specify integer parameters for CatBoost.
    catboost_int_params = [
        "iterations", "depth",
        "one_hot_max_size", "border_count", "max_ctr_complexity", "min_data_in_leaf"
    ]

    optimizer = CatBoostStepwiseOptimizer( # Changed to CatBoostStepwiseOptimizer
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=3,
        random_state=42,
        int_params=catboost_int_params,
        scoring="neg_root_mean_squared_error", # Appropriate scoring for RMSE loss
        debug=True,
    )

    optimizer.fit(X_train, y_train, eval_set=[(X_test, y_test)])

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
    assert "colsample_bylevel" in optimizer.best_params_

    # Assert subsample is present UNLESS bootstrap_type is Bayesian
    if optimizer.best_params_.get("bootstrap_type") == "Bayesian":
        assert "subsample" not in optimizer.best_params_
        assert "bagging_temperature" in optimizer.best_params_
    else:
        assert "subsample" in optimizer.best_params_
        assert "bagging_temperature" not in optimizer.best_params_


    assert "use_best_model" in optimizer.best_params_
    assert "eval_metric" in optimizer.best_params_
    
    # Assert that od_params and od_wait are NOT present
    assert "od_params" not in optimizer.best_params_
    assert "od_wait" not in optimizer.best_params_
    assert "od_type" not in optimizer.best_params_
    assert "od_pval" not in optimizer.best_params_

    assert "border_count" in optimizer.best_params_
    assert "has_time" in optimizer.best_params_
    assert "max_ctr_complexity" in optimizer.best_params_

    assert "objective" in optimizer.best_params_

    # Assert max_leaves is NOT present, as it's removed from the space
    assert "max_leaves" not in optimizer.best_params_

    # Assert that if boosting_type is 'Ordered', grow_policy is 'SymmetricTree'
    if optimizer.best_params_["boosting_type"] == "Ordered":
        assert optimizer.best_params_["grow_policy"] == "SymmetricTree"
    
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 0 # For neg_root_mean_squared_error, score is negative
