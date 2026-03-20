from __future__ import annotations

import sk_stepwise.search as sw_search


def test_merge_params_prefers_current_step_values() -> None:
    merged = sw_search._merge_params(  # type: ignore[attr-defined]
        {"n_estimators": 50, "max_depth": 3},
        {"max_depth": 5, "min_samples_split": 0.2},
    )

    assert merged == {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 0.2,
    }


def test_build_step_result_tracks_improvement() -> None:
    result = sw_search._build_step_result(  # type: ignore[attr-defined]
        best_params={"n_estimators": 100},
        best_value=0.8,
        n_trials=4,
        previous_best=0.5,
    )

    assert result == {
        "best_params": {"n_estimators": 100},
        "best_value": 0.8,
        "n_trials": 4,
        "improvement": 0.30000000000000004,
    }


def test_build_step_result_uses_baseline_for_first_step() -> None:
    result = sw_search._build_step_result(  # type: ignore[attr-defined]
        best_params={"max_depth": 4},
        best_value=0.7,
        n_trials=2,
        previous_best=None,
    )

    assert result["improvement"] is None
