import numpy as np
import pytest
import sk_stepwise as sw
import sk_stepwise.search as sw_search
import warnings
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def test_stepwise_optuna_search_cv_supports_readme_style_flow(
    readme_regression_frame, random_forest_estimator, readme_random_forest_space
):
    X, y = readme_regression_frame
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=random_forest_estimator,
        param_distributions=readme_random_forest_space,
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)
    predictions = optimizer.predict(X)

    assert predictions.shape == (100,)
    assert set(optimizer.best_params_) == {
        "n_estimators",
        "max_depth",
        "min_samples_split",
    }
    assert optimizer.best_estimator_ is not random_forest_estimator


def test_predict_before_fit_raises_not_fitted_error():
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[],
        n_trials_per_step=1,
    )

    X, _ = make_regression(n_samples=20, n_features=3, random_state=0)

    with pytest.raises(NotFittedError):
        optimizer.predict(X)


def test_stepwise_search_carries_forward_best_params_across_steps(
    param_score_regressor,
):
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)

    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=param_score_regressor,
        param_distributions=[
            {"alpha": sw.Int(1, 2)},
            {"beta": sw.Int(4, 5)},
        ],
        n_trials_per_step=2,
        cv=2,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert optimizer.best_params_ == {"alpha": 2, "beta": 5}
    assert optimizer.step_results_[0]["best_params"] == {"alpha": 2}
    assert optimizer.step_results_[1]["best_params"] == {"alpha": 2, "beta": 5}
    assert [result["n_trials"] for result in optimizer.step_results_] == [2, 2]


def test_learned_attributes_are_created_during_fit_only():
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        random_state=0,
    )
    X, y = make_regression(n_samples=40, n_features=4, random_state=0)

    assert not hasattr(optimizer, "best_params_")
    assert not hasattr(optimizer, "best_score_")
    assert not hasattr(optimizer, "best_estimator_")
    assert not hasattr(optimizer, "step_results_")
    assert not hasattr(optimizer, "study_")
    assert not hasattr(optimizer, "studies_")

    optimizer.fit(X, y)

    assert isinstance(optimizer.best_params_, dict)
    assert isinstance(optimizer.best_score_, float)
    assert isinstance(optimizer.step_results_, list)
    assert isinstance(optimizer.studies_, list)
    assert optimizer.study_ is optimizer.studies_[-1]
    assert optimizer.best_estimator_ is not optimizer.estimator


def test_random_forest_integration_fit_and_predict(
    readme_regression_frame, random_forest_estimator, random_forest_small_space
):
    X, y = readme_regression_frame
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=random_forest_estimator,
        param_distributions=random_forest_small_space,
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)
    predictions = optimizer.predict(X.iloc[:5])

    assert predictions.shape == (5,)
    assert isinstance(optimizer.best_estimator_, RandomForestRegressor)


def test_random_state_makes_results_reproducible(
    readme_regression_frame, random_forest_small_space
):
    X, y = readme_regression_frame

    first = sw.StepwiseOptunaSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_distributions=random_forest_small_space,
        n_trials_per_step=2,
        random_state=7,
    )
    second = sw.StepwiseOptunaSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_distributions=random_forest_small_space,
        n_trials_per_step=2,
        random_state=7,
    )

    first.fit(X, y)
    second.fit(X, y)

    assert first.best_params_ == second.best_params_
    assert first.best_score_ == second.best_score_


def test_pipeline_estimator_with_namespaced_params(readme_regression_frame):
    X, y = readme_regression_frame
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regressor", RandomForestRegressor(random_state=0)),
        ]
    )
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=pipeline,
        param_distributions=[
            {"regressor__n_estimators": sw.Int(10, 20)},
            {"regressor__max_depth": sw.Int(2, 4)},
        ],
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert "regressor__n_estimators" in optimizer.best_params_
    assert "regressor__max_depth" in optimizer.best_params_
    assert optimizer.predict(X.iloc[:3]).shape == (3,)


def test_sample_weight_is_forwarded_to_cv_and_final_fit(
    weighted_linear_regression, weighted_regression_data
):
    X, y, sample_weight = weighted_regression_data
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=weighted_linear_regression,
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        cv=2,
        random_state=0,
    )

    optimizer.fit(X, y, sample_weight=sample_weight)

    assert weighted_linear_regression.fit_call_sample_weights == []
    assert len(optimizer.best_estimator_.fit_call_sample_weights) == 1
    np.testing.assert_allclose(
        optimizer.best_estimator_.fit_call_sample_weights[0],
        sample_weight,
    )


def test_xgboost_integration_fit_and_predict(readme_regression_frame):
    X, y = readme_regression_frame
    estimator = XGBRegressor(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=0,
        verbosity=0,
    )
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=estimator,
        param_distributions=[
            {"n_estimators": sw.Int(10, 20)},
            {"max_depth": sw.Int(2, 4)},
        ],
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert optimizer.predict(X.iloc[:4]).shape == (4,)


def test_catboost_integration_fit_and_predict(readme_regression_frame):
    X, y = readme_regression_frame
    estimator = CatBoostRegressor(
        iterations=20,
        depth=4,
        learning_rate=0.1,
        random_seed=0,
        verbose=False,
    )
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=estimator,
        param_distributions=[
            {"depth": sw.Int(3, 5)},
            {"learning_rate": sw.Float(0.05, 0.2)},
        ],
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert optimizer.predict(X.iloc[:4]).shape == (4,)


def test_fit_uses_cloned_estimators_and_does_not_fit_original_estimator(
    readme_regression_frame,
):
    X, y = readme_regression_frame
    estimator = RandomForestRegressor(random_state=0)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=estimator,
        param_distributions=[{"n_estimators": sw.Int(10, 20)}],
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert not hasattr(estimator, "estimators_")
    assert optimizer.best_estimator_ is not estimator


def test_classifier_supports_string_scoring_and_prediction_methods():
    X, y = make_regression(n_samples=60, n_features=4, random_state=0)
    y = (y > np.median(y)).astype(int)
    estimator = LogisticRegression(max_iter=200)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=estimator,
        param_distributions=[{"C": sw.Float(0.1, 1.0)}],
        n_trials_per_step=2,
        cv=3,
        scoring="accuracy",
        random_state=0,
    )

    optimizer.fit(X, y)

    assert optimizer.predict(X[:5]).shape == (5,)


def test_regression_supports_callable_scoring():
    X, y = make_regression(n_samples=50, n_features=4, random_state=0)
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        cv=3,
        scoring=scoring,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert isinstance(optimizer.best_score_, float)


def test_classifier_aware_cv_selection_uses_stratified_folds(monkeypatch):
    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    y = (y > np.median(y)).astype(int)
    recorded = {}
    original_check_cv = sw_search.check_cv

    def recording_check_cv(cv, y=None, classifier=False):
        recorded["classifier"] = classifier
        return original_check_cv(cv=cv, y=y, classifier=classifier)

    monkeypatch.setattr(sw_search, "check_cv", recording_check_cv)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LogisticRegression(max_iter=200),
        param_distributions=[{"C": sw.Float(0.1, 1.0)}],
        n_trials_per_step=1,
        cv=3,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert recorded["classifier"] is True


def test_cv_accepts_splitter_object(readme_regression_frame):
    X, y = readme_regression_frame
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        cv=cv,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert isinstance(optimizer.best_score_, float)


def test_cv_accepts_iterable_of_splits(readme_regression_frame):
    X, y = readme_regression_frame
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    splits = list(splitter.split(X, y))
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        cv=splits,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert isinstance(optimizer.best_score_, float)


def test_predict_proba_delegates_to_best_estimator():
    X, y = make_regression(n_samples=60, n_features=4, random_state=0)
    y = (y > np.median(y)).astype(int)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LogisticRegression(max_iter=200),
        param_distributions=[{"C": sw.Float(0.1, 1.0)}],
        n_trials_per_step=2,
        scoring="accuracy",
        random_state=0,
    )

    optimizer.fit(X, y)

    probabilities = optimizer.predict_proba(X[:5])

    assert probabilities.shape == (5, 2)


def test_predict_proba_before_fit_raises_not_fitted_error():
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LogisticRegression(max_iter=200),
        param_distributions=[{"C": sw.Float(0.1, 1.0)}],
        n_trials_per_step=1,
    )
    X = np.zeros((4, 2))

    with pytest.raises(NotFittedError):
        optimizer.predict_proba(X)


def test_fit_uses_sklearn_cross_val_score(monkeypatch, readme_regression_frame):
    X, y = readme_regression_frame
    calls = []
    original_cross_val_score = sw_search.cross_val_score

    def recording_cross_val_score(*args, **kwargs):
        calls.append((args, kwargs))
        return original_cross_val_score(*args, **kwargs)

    monkeypatch.setattr(sw_search, "cross_val_score", recording_cross_val_score)
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert calls


def test_fit_and_predict_accept_plain_python_lists():
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[{"fit_intercept": sw.Categorical([True, False])}],
        n_trials_per_step=2,
        cv=2,
        random_state=0,
    )

    optimizer.fit(X, y)
    predictions = optimizer.predict([[1.5], [2.5]])

    assert predictions.shape == (2,)


def test_deprecated_hyperopt_class_warns_and_maps_legacy_arguments(
    readme_regression_frame,
):
    X, y = readme_regression_frame

    with pytest.deprecated_call():
        optimizer = sw.StepwiseHyperoptOptimizer(
            model=RandomForestRegressor(random_state=0),
            param_space_sequence=[{"n_estimators": sw.Int(10, 20)}],
            max_evals_per_step=2,
            random_state=0,
        )

    optimizer.fit(X, y)

    assert isinstance(optimizer, sw.StepwiseOptunaSearchCV)
    assert optimizer.best_params_["n_estimators"] in range(10, 21)


def test_deprecated_hyperopt_class_rejects_old_hyperopt_spaces():
    class LegacySpace:
        pass

    with pytest.deprecated_call():
        optimizer = sw.StepwiseHyperoptOptimizer(
            model=LinearRegression(),
            param_space_sequence=[{"fit_intercept": LegacySpace()}],
            max_evals_per_step=1,
        )

    with pytest.raises(TypeError, match="Unsupported search dimension"):
        optimizer.fit([[0.0], [1.0]], [0.0, 1.0])


def test_verbose_logging_reports_progress(capsys, readme_regression_frame):
    X, y = readme_regression_frame
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_distributions=[
            {"n_estimators": sw.Int(10, 20)},
            {"max_depth": sw.Int(2, 4)},
        ],
        n_trials_per_step=2,
        random_state=0,
        verbose=1,
    )

    optimizer.fit(X, y)
    output = capsys.readouterr().out

    assert "Optimizing step 1/2" in output
    assert "Best parameters after step 1:" in output
    assert "Best score after step 1:" in output
    assert "Improvement:" in output


def test_numeric_categorical_warns():
    with pytest.warns(UserWarning, match="Prefer Int or Float"):
        sw.Categorical([1, 2, 3])


def test_boolean_categorical_does_not_warn():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sw.Categorical([True, False])
    assert len(record) == 0


def test_refit_false_skips_best_estimator_fit(readme_regression_frame):
    X, y = readme_regression_frame
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_distributions=[{"n_estimators": sw.Int(10, 20)}],
        n_trials_per_step=2,
        random_state=0,
        refit=False,
    )

    optimizer.fit(X, y)

    assert not hasattr(optimizer, "best_estimator_")
    with pytest.raises(NotFittedError):
        optimizer.predict(X)


def test_refit_true_produces_best_estimator(readme_regression_frame):
    X, y = readme_regression_frame
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_distributions=[{"n_estimators": sw.Int(10, 20)}],
        n_trials_per_step=2,
        random_state=0,
        refit=True,
    )

    optimizer.fit(X, y)

    assert isinstance(optimizer.best_estimator_, RandomForestRegressor)


def test_parameter_types_come_from_dimensions_not_parameter_names(
    arbitrary_param_regressor,
):
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=arbitrary_param_regressor,
        param_distributions=[
            {"tree_count": sw.Int(1, 3)},
            {"shrinkage": sw.Float(0.1, 0.3)},
        ],
        n_trials_per_step=1,
    )
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)

    optimizer.fit(X, y)

    assert isinstance(optimizer.best_params_["tree_count"], int)
    assert isinstance(optimizer.best_params_["shrinkage"], float)


def test_verbose_zero_emits_no_progress_output(capsys, readme_regression_frame):
    X, y = readme_regression_frame
    optimizer = sw.StepwiseOptunaSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_distributions=[{"n_estimators": sw.Int(10, 20)}],
        n_trials_per_step=1,
        random_state=0,
        verbose=0,
    )

    optimizer.fit(X, y)

    assert capsys.readouterr().out == ""


def test_stepwise_optuna_search_cv_uses_estimator_not_model_keyword():
    with pytest.raises(TypeError):
        sw.StepwiseOptunaSearchCV(
            model=LinearRegression(),
            param_distributions=[],
            n_trials_per_step=1,
        )


def test_deprecated_hyperopt_class_still_accepts_model_keyword():
    with pytest.deprecated_call():
        optimizer = sw.StepwiseHyperoptOptimizer(
            model=LinearRegression(),
            param_space_sequence=[],
            max_evals_per_step=1,
        )

    assert isinstance(optimizer, sw.StepwiseOptunaSearchCV)
