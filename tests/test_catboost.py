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
    # and the provided default/example values.
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
                    "iterations": hp.quniform("iterations_ordered", 10, 200, 10),
                    "learning_rate": hp.loguniform("learning_rate_ordered", np.log(0.01), np.log(0.3)),
                    "depth": hp.quniform("depth_ordered", 4, 10, 1),
                    "l2_leaf_reg": hp.loguniform("l2_leaf_reg_ordered", np.log(1), np.log(10)),
                    "random_strength": hp.loguniform("random_strength_ordered", np.log(0.1), np.log(10)),
                    "one_hot_max_size": hp.quniform("one_hot_max_size_ordered", 2, 20, 1),
                    "min_data_in_leaf": hp.quniform("min_data_in_leaf_ordered", 1, 30, 1),
                    "subsample": hp.uniform("subsample_ordered", 0.6, 1.0),
                    "bootstrap_type": hp.choice("bootstrap_type_ordered", ["Bayesian", "Bernoulli", "MVS"]),
                    "bagging_temperature": hp.uniform("bagging_temperature_ordered", 0.0, 1.0),
                    "diffusion_temperature": hp.loguniform("diffusion_temperature_ordered", 0, 4),
                    "allow_const_label": hp.choice("allow_const_label_ordered", [True, False]),
                    "posterior_sampling": hp.choice("posterior_sampling_ordered", [hp.choice("langevin_inner_ordered", [True, False]), False]),
                    "langevin": hp.choice("langevin_outer_ordered", [True, False]),
                    "fold_len_multiplier": hp.uniform("fold_len_multiplier_ordered", 1.01, 2.0),
                    "approx_on_full_history": hp.choice("approx_on_full_history_ordered", [True, False]),

                    # Fixed parameters (from the provided list, not marked for tuning)
                    "loss_function": "RMSE",
                    "eval_metric": "RMSE",
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
                    "score_function": 'Cosine',
                    "sampling_frequency": 'PerTree',
                    "boost_from_average": False,
                    "best_model_min_trees": 1,
                    "random_seed": 0,
                    "random_score_type": 'NormalWithModelSizeDecrease',
                    "penalties_coefficient": 1,
                    "task_type": 'CPU',
                    "use_best_model": False,
                    "nan_mode": 'Min',
                    "has_time": False,
                },
                # Option 2: Plain Boosting (grow_policy can be any)
                {
                    "boosting_type": "Plain",
                    "grow_policy": hp.choice("grow_policy_plain", ["SymmetricTree", "Depthwise", "Lossguide"]),
                    # Tuned parameters (with different labels to avoid name collisions in hyperopt)
                    "iterations": hp.quniform("iterations_plain", 10, 200, 10),
                    "learning_rate": hp.loguniform("learning_rate_plain", np.log(0.01), np.log(0.3)),
                    "depth": hp.quniform("depth_plain", 4, 10, 1),
                    "l2_leaf_reg": hp.loguniform("l2_leaf_reg_plain", np.log(1), np.log(10)),
                    "random_strength": hp.loguniform("random_strength_plain", np.log(0.1), np.log(10)),
                    "one_hot_max_size": hp.quniform("one_hot_max_size_plain", 2, 20, 1),
                    "min_data_in_leaf": hp.quniform("min_data_in_leaf_plain", 1, 30, 1),
                    "subsample": hp.uniform("subsample_plain", 0.6, 1.0),
                    "bootstrap_type": hp.choice("bootstrap_type_plain", ["Bayesian", "Bernoulli", "MVS"]),
                    "bagging_temperature": hp.uniform("bagging_temperature_plain", 0.0, 1.0),
                    "diffusion_temperature": hp.loguniform("diffusion_temperature_plain", 0, 4),
                    "allow_const_label": hp.choice("allow_const_label_plain", [True, False]),
                    "posterior_sampling": hp.choice("posterior_sampling_plain", [hp.choice("langevin_inner_plain", [True, False]), False]),
                    "langevin": hp.choice("langevin_outer_plain", [True, False]),
                    "fold_len_multiplier": hp.uniform("fold_len_multiplier_plain", 1.01, 2.0),
                    "approx_on_full_history": hp.choice("approx_on_full_history_plain", [True, False]),

                    # Fixed parameters (same as above)
                    "loss_function": "RMSE",
                    "eval_metric": "RMSE",
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
                    "score_function": 'Cosine',
                    "sampling_frequency": 'PerTree',
                    "boost_from_average": False,
                    "best_model_min_trees": 1,
                    "random_seed": 0,
                    "random_score_type": 'NormalWithModelSizeDecrease',
                    "penalties_coefficient": 1,
                    "task_type": 'CPU',
                    "use_best_model": False,
                    "nan_mode": 'Min',
                    "has_time": False,
                },
            ]
        )
    ]

    # Specify integer parameters for CatBoost.
    # Note: When using nested hp.choice, the keys in best_params_ will be flattened.
    # So, 'iterations_ordered' or 'iterations_plain' will become 'iterations' after space_eval.
    # We need to list the final parameter names that should be integers.
    catboost_int_params = [
        "iterations", "depth", "min_data_in_leaf", "one_hot_max_size",
        "fold_permutation_block", "leaf_estimation_iterations", "best_model_min_trees",
        "border_count", "max_ctr_complexity", "ctr_target_border_count"
    ]

    optimizer = StepwiseHyperoptOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=10,
        random_state=42,
        int_params=catboost_int_params,
        scoring="neg_root_mean_squared_error"
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
    assert optimizer.best_score_ < 0
