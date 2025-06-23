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

    # Define boosting_type as a variable to use in conditional grow_policy
    # Based on the provided tuning table, boosting_type is tuned.
    boosting_type_choice = hp.choice("boosting_type", ["Ordered", "Plain"])

    # Define grow_policy conditionally based on boosting_type
    # If boosting_type is "Ordered", grow_policy must be "SymmetricTree".
    # Otherwise, it can be any of the three.
    grow_policy_choice = hp.choice(
        "grow_policy",
        [
            "SymmetricTree",
            hp.pchoice(("Depthwise", 0.0) if boosting_type_choice._obj == "Ordered" else ("Depthwise", 1.0)),
            hp.pchoice(("Lossguide", 0.0) if boosting_type_choice._obj == "Ordered" else ("Lossguide", 1.0))
        ]
    )

    # Define param_space_sequence based on the "Which Parameters to tune?" table
    # and the provided default/example values.
    param_space_sequence = [
        {
            # Tuned parameters
            "iterations": hp.quniform("iterations", 10, 200, 10), # Range from 10 to 200, step 10
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)), # Common range for learning rate
            "depth": hp.quniform("depth", 4, 10, 1), # Range from 4 to 10, step 1
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1), np.log(10)), # Common range for L2 regularization
            "random_strength": hp.loguniform("random_strength", np.log(0.1), np.log(10)), # Common range for random strength
            "one_hot_max_size": hp.quniform("one_hot_max_size", 2, 20, 1), # Range from 2 to 20, step 1
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 1, 30, 1), # Range from 1 to 30, step 1
            "boosting_type": boosting_type_choice, # Use the defined choice variable
            "grow_policy": grow_policy_choice, # Use the defined conditional choice
            "subsample": hp.uniform("subsample", 0.6, 1.0), # Range from 0.6 to 1.0
            "bootstrap_type": hp.choice("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]), # Common bootstrap types
            # "bagging_temperature" is often used with Bayesian bootstrap, so we'll include it conditionally or generally
            # For simplicity, let's include it generally if bootstrap_type is chosen.
            "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0), # Common range for bagging_temperature

            # Fixed parameters (from the provided list, not marked for tuning)
            "loss_function": "RMSE", # Changed from Logloss as it's a regressor
            "eval_metric": "RMSE", # Changed from Logloss
            "border_count": 254,
            "max_ctr_complexity": 4,
            "feature_border_type": 'GreedyLogSum',
            "combinations_ctr": ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1', 'Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1'],
            "ctr_leaf_count_limit": 18446744073709551615,
            "ctr_target_border_count": 1,
            "model_shrink_rate": 0,
            "model_size_reg": 0.5,
            "leaf_estimation_iterations": 10,
            "leaf_estimation_method": 'Newton',
            "leaf_estimation_backtracking": 'AnyImprovement',
            "auto_class_weights": 'None',
            "eval_fraction": 0,
            "fold_permutation_block": 0,
            "counter_calc_method": 'SkipTest',
            "posterior_sampling": False,
            "score_function": 'Cosine',
            "sampling_frequency": 'PerTree',
            "boost_from_average": False,
            "best_model_min_trees": 1,
            "random_seed": 0, # Using random_state in optimizer, so this can be fixed
            "random_score_type": 'NormalWithModelSizeDecrease',
            "penalties_coefficient": 1,
            "task_type": 'CPU',
            "use_best_model": False,
            "nan_mode": 'Min',
            "has_time": False, # Assuming no time feature in synthetic data
            "diffusion_temperature": hp.loguniform("diffusion_temperature", 0, 4), # Still tuning this as it was in original test
            "allow_const_label": hp.choice("allow_const_label", [True, False]), # Still tuning this as it was in original test
            "posterior_sampling": hp.choice("posterior_sampling", [hp.choice("langevin_inner", [True, False]), False]), # Re-defining for clarity
            "langevin": hp.choice("langevin_outer", [True, False]), # Re-defining for clarity
            "fold_len_multiplier": hp.uniform("fold_len_multiplier", 1.01, 2.0), # Still tuning this
            "approx_on_full_history": hp.choice("approx_on_full_history", [True, False]), # Still tuning this
        }
    ]

    # Specify integer parameters for CatBoost
    catboost_int_params = [
        "iterations", "depth", "min_data_in_leaf", "one_hot_max_size",
        "fold_permutation_block", "leaf_estimation_iterations", "best_model_min_trees",
        "border_count", "max_ctr_complexity", "ctr_target_border_count" # Added from fixed params
    ]

    optimizer = StepwiseHyperoptOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=10, # Increased evals to explore the space better
        random_state=42,
        int_params=catboost_int_params, # Pass CatBoost specific integer parameters
        scoring="neg_root_mean_squared_error" # Appropriate scoring for RMSE loss
    )

    # This fit is expected to pass now due to the conditional hyperparameter space
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
    assert "bagging_temperature" in optimizer.best_params_

    # Assertions for fixed parameters (optional, but good for verification)
    assert optimizer.best_params_["loss_function"] == "RMSE"
    assert optimizer.best_params_["eval_metric"] == "RMSE"
    assert optimizer.best_params_["random_seed"] == 0
    assert optimizer.best_params_["task_type"] == 'CPU'

    # Assert that if boosting_type is 'Ordered', grow_policy is 'SymmetricTree'
    if optimizer.best_params_["boosting_type"] == "Ordered":
        assert optimizer.best_params_["grow_policy"] == "SymmetricTree"
    
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 0 # For neg_root_mean_squared_error, score is negative
