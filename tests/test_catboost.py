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

    # Define Langevin and Posterior Sampling choices separately
    langevin_choice = hp.choice("langevin", [True, False])

    # Define boosting_type as a variable to use in conditional grow_policy
    boosting_type_choice = hp.choice("boosting_type", ["Ordered", "Plain"])

    param_space_sequence = [
        {
            "iterations": hp.quniform("iterations", 10, 100, 10),
            "learning_rate": hp.loguniform("learning_rate", -3, 0),
            "depth": hp.quniform("depth", 4, 10, 1),
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", -2, 1),
            "bootstrap_type": hp.choice("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "random_strength": hp.loguniform("random_strength", -2, 1),
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 1, 20, 1),
            # Conditional grow_policy: if boosting_type is "Ordered", grow_policy must be "SymmetricTree"
            "grow_policy": hp.choice(
                "grow_policy",
                [
                    "SymmetricTree",
                    hp.pchoice(("Depthwise", 0.0) if boosting_type_choice._obj == "Ordered" else ("Depthwise", 1.0)),
                    hp.pchoice(("Lossguide", 0.0) if boosting_type_choice._obj == "Ordered" else ("Lossguide", 1.0))
                ]
            ),
            "nan_mode": hp.choice("nan_mode", ["Forbidden", "Min", "Max"]),
            "one_hot_max_size": hp.quniform("one_hot_max_size", 2, 10, 1),
            "has_time": hp.choice("has_time", [True, False]),
            "rsm": hp.uniform("rsm", 0.8, 1.0),
            "fold_permutation_block": hp.quniform("fold_permutation_block", 1, 10, 1),
            "leaf_estimation_method": hp.choice("leaf_estimation_method", ["Newton", "Gradient"]),
            "leaf_estimation_iterations": hp.quniform("leaf_estimation_iterations", 1, 10, 1),
            "leaf_estimation_backtracking": hp.choice("leaf_estimation_backtracking", ["No", "AnyImprovement"]),
            "fold_len_multiplier": hp.uniform("fold_len_multiplier", 1.01, 2.0),
            "approx_on_full_history": hp.choice("approx_on_full_history", [True, False]),
            "boosting_type": boosting_type_choice, # Use the defined choice variable
            "boost_from_average": hp.choice("boost_from_average", [True, False]),
            "langevin": langevin_choice, # Use the defined choice
            "diffusion_temperature": hp.loguniform("diffusion_temperature", 0, 4),
            # posterior_sampling is True only if langevin is True
            "posterior_sampling": hp.choice("posterior_sampling", [langevin_choice, False]),
            "allow_const_label": hp.choice("allow_const_label", [True, False]),
            "score_function": hp.choice("score_function", ["Cosine", "L2"]),
            "penalties_coefficient": hp.uniform("penalties_coefficient", 0.1, 10.0),
            "model_shrink_rate": hp.uniform("model_shrink_rate", 0.0, 1.0),
            "model_shrink_mode": hp.choice("model_shrink_mode", ["Constant", "Decreasing"]),
        }
    ]

    # Specify integer parameters for CatBoost
    catboost_int_params = [
        "iterations", "depth", "min_data_in_leaf", "one_hot_max_size",
        "fold_permutation_block", "leaf_estimation_iterations"
    ]

    optimizer = StepwiseHyperoptOptimizer(
        model=model,
        param_space_sequence=param_space_sequence,
        max_evals_per_step=5,
        random_state=42,
        int_params=catboost_int_params, # Pass CatBoost specific integer parameters
    )

    # This fit is expected to pass now due to the conditional hyperparameter space
    optimizer.fit(X_train, y_train)

    assert optimizer.best_params_ is not None
    assert "iterations" in optimizer.best_params_
    assert "learning_rate" in optimizer.best_params_
    assert "depth" in optimizer.best_params_
    assert "l2_leaf_reg" in optimizer.best_params_
    assert "bootstrap_type" in optimizer.best_params_
    assert "subsample" in optimizer.best_params_
    assert "random_strength" in optimizer.best_params_
    assert "min_data_in_leaf" in optimizer.best_params_
    assert "grow_policy" in optimizer.best_params_
    assert "nan_mode" in optimizer.best_params_
    assert "one_hot_max_size" in optimizer.best_params_
    assert "has_time" in optimizer.best_params_
    assert "rsm" in optimizer.best_params_
    assert "fold_permutation_block" in optimizer.best_params_
    assert "leaf_estimation_method" in optimizer.best_params_
    assert "leaf_estimation_iterations" in optimizer.best_params_
    assert "leaf_estimation_backtracking" in optimizer.best_params_
    assert "fold_len_multiplier" in optimizer.best_params_
    assert "approx_on_full_history" in optimizer.best_params_
    assert "boosting_type" in optimizer.best_params_
    assert "boost_from_average" in optimizer.best_params_
    assert "langevin" in optimizer.best_params_
    assert "diffusion_temperature" in optimizer.best_params_
    assert "posterior_sampling" in optimizer.best_params_
    assert "allow_const_label" in optimizer.best_params_
    assert "score_function" in optimizer.best_params_
    assert "penalties_coefficient" in optimizer.best_params_
    assert "model_shrink_rate" in optimizer.best_params_
    assert "model_shrink_mode" in optimizer.best_params_
    assert optimizer.best_score_ is not None

    # Assert that if boosting_type is 'Ordered', grow_policy is 'SymmetricTree'
    if optimizer.best_params_["boosting_type"] == "Ordered":
        assert optimizer.best_params_["grow_policy"] == "SymmetricTree"
