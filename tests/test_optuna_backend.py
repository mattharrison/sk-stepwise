from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

import sk_stepwise as sw
import sk_stepwise.search as sw_search
from sk_stepwise.spaces import SearchDimension


def test_search_dimension_alias_accepts_all_dimension_types() -> None:
    dimensions: list[SearchDimension] = [
        sw.Categorical(["a", "b"]),
        sw.Int(1, 3),
        sw.Float(0.1, 0.9),
    ]

    assert [type(dimension).__name__ for dimension in dimensions] == [
        "Categorical",
        "Int",
        "Float",
    ]


def test_suggest_params_maps_dimensions_to_optuna_trial_methods() -> None:
    calls: list[tuple[str, str, object]] = []

    class RecordingTrial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            calls.append(("categorical", name, tuple(choices)))
            return choices[0]

        def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
            calls.append(("int", name, (low, high, log)))
            return low

        def suggest_float(
            self, name: str, low: float, high: float, log: bool = False
        ) -> float:
            calls.append(("float", name, (low, high, log)))
            return low

    search = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[],
        n_trials_per_step=1,
    )

    params = search._suggest_params(
        RecordingTrial(),  # type: ignore[arg-type]
        {
            "criterion": sw.Categorical(["squared_error", "absolute_error"]),
            "n_estimators": sw.Int(10, 20),
            "learning_rate": sw.Float(0.01, 0.1, log=True),
        },
    )

    assert params == {
        "criterion": "squared_error",
        "n_estimators": 10,
        "learning_rate": 0.01,
    }
    assert calls == [
        ("categorical", "criterion", ("squared_error", "absolute_error")),
        ("int", "n_estimators", (10, 20, False)),
        ("float", "learning_rate", (0.01, 0.1, True)),
    ]


def test_stepwise_search_records_stable_trials_with_fixed_seed() -> None:
    X, y = make_regression(n_samples=50, n_features=4, random_state=0)

    kwargs = {
        "estimator": LinearRegression(),
        "param_distributions": [{"fit_intercept": sw.Categorical([True, False])}],
        "n_trials_per_step": 2,
        "random_state": 11,
    }

    first = sw.StepwiseOptunaSearchCV(**kwargs)
    second = sw.StepwiseOptunaSearchCV(**kwargs)

    first.fit(X, y)
    second.fit(X, y)

    first_trials = [trial.params for trial in first.studies_[0].trials]
    second_trials = [trial.params for trial in second.studies_[0].trials]

    assert first_trials == second_trials


def test_stepwise_search_evaluates_merged_params_from_previous_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)
    seen_param_sets: list[dict[str, object]] = []

    original_cross_val_score = sw_search.cross_val_score

    def recording_cross_val_score(estimator, X, y=None, **kwargs):  # type: ignore[no-untyped-def]
        seen_param_sets.append(estimator.get_params(deep=False))
        return original_cross_val_score(estimator, X, y=y, **kwargs)

    monkeypatch.setattr(sw_search, "cross_val_score", recording_cross_val_score)

    search = sw.StepwiseOptunaSearchCV(
        estimator=LinearRegression(),
        param_distributions=[
            {"fit_intercept": sw.Categorical([True, False])},
            {"positive": sw.Categorical([True, False])},
        ],
        n_trials_per_step=1,
        cv=2,
        random_state=0,
    )

    search.fit(X, y)

    assert len(seen_param_sets) == 2
    assert "fit_intercept" in seen_param_sets[0]
    assert "fit_intercept" in seen_param_sets[1]
    assert "positive" in seen_param_sets[1]
