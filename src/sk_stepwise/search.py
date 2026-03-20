from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast
import warnings

import optuna
from optuna.samplers import TPESampler
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.utils.validation import check_is_fitted

from .spaces import Categorical, Float, Int, SearchDimension

MatrixLike: TypeAlias = Any | spmatrix


def _merge_params(
    previous_best: dict[str, Any], step_params: dict[str, Any]
) -> dict[str, Any]:
    return {**previous_best, **step_params}


def _build_step_result(
    *,
    best_params: dict[str, Any],
    best_value: float,
    n_trials: int,
    previous_best: float | None,
) -> dict[str, Any]:
    improvement = None if previous_best is None else best_value - previous_best
    return {
        "best_params": dict(best_params),
        "best_value": best_value,
        "n_trials": n_trials,
        "improvement": improvement,
    }


class StepwiseOptunaSearchCV(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: list[dict[str, SearchDimension]],
        *,
        n_trials_per_step: int = 50,
        cv: int = 5,
        scoring: str | Callable[..., float] | None = None,
        random_state: int | None = None,
        refit: bool = True,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials_per_step = n_trials_per_step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.refit = refit
        self.verbose = verbose

    def _suggest_params(
        self, trial: optuna.Trial, step_distribution: dict[str, SearchDimension]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, dimension in step_distribution.items():
            if isinstance(dimension, Categorical):
                params[name] = trial.suggest_categorical(name, list(dimension.choices))
            elif isinstance(dimension, Int):
                params[name] = trial.suggest_int(
                    name, dimension.low, dimension.high, log=dimension.log
                )
            elif isinstance(dimension, Float):
                params[name] = trial.suggest_float(
                    name, dimension.low, dimension.high, log=dimension.log
                )
            else:
                raise TypeError(
                    f"Unsupported search dimension for {name!r}: {dimension!r}"
                )
        return params

    def fit(self, X: MatrixLike, y: Any, **fit_params: Any) -> "StepwiseOptunaSearchCV":
        self.best_params_: dict[str, Any] = {}
        self.step_results_: list[dict[str, Any]] = []
        self.studies_: list[optuna.study.Study] = []

        cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))
        direction = "maximize"
        previous_best: float | None = None

        for step_index, step_distribution in enumerate(
            self.param_distributions, start=1
        ):
            if self.verbose:
                print(f"Optimizing step {step_index}/{len(self.param_distributions)}")
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(direction=direction, sampler=sampler)

            def objective(trial: optuna.Trial) -> float:
                step_params = self._suggest_params(trial, step_distribution)
                all_params = _merge_params(self.best_params_, step_params)
                estimator = clone(self.estimator).set_params(**all_params)
                scores = cross_val_score(
                    estimator,
                    X,
                    y,
                    cv=cv,
                    scoring=self.scoring,
                    params=fit_params if fit_params else None,
                )
                return float(scores.mean())

            study.optimize(objective, n_trials=self.n_trials_per_step)
            self.studies_.append(study)
            self.best_params_.update(study.best_params)
            step_result = _build_step_result(
                best_params=self.best_params_,
                best_value=study.best_value,
                n_trials=len(study.trials),
                previous_best=previous_best,
            )
            self.step_results_.append(step_result)
            if self.verbose:
                print(f"Best parameters after step {step_index}: {self.best_params_}")
                print(f"Best score after step {step_index}: {study.best_value}")
                if step_result["improvement"] is None:
                    print("Improvement: baseline")
                else:
                    print(f"Improvement: {step_result['improvement']:+.6f}")
            previous_best = study.best_value

        self.best_score_ = (
            self.step_results_[-1]["best_value"] if self.step_results_ else None
        )
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)
        self.study_ = self.studies_[-1] if self.studies_ else None
        return self

    def predict(self, X: MatrixLike) -> Any:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def score(self, X: MatrixLike, y: Any) -> float:
        check_is_fitted(self, "best_estimator_")
        return cast(float, self.best_estimator_.score(X, y))

    def predict_proba(self, X: MatrixLike) -> Any:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X: MatrixLike) -> Any:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.decision_function(X)

    def transform(self, X: MatrixLike) -> Any:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.transform(X)


class StepwiseHyperoptOptimizer(StepwiseOptunaSearchCV):
    def __init__(
        self,
        model: BaseEstimator,
        param_space_sequence: list[dict[str, SearchDimension]],
        *,
        max_evals_per_step: int = 50,
        cv: int = 5,
        scoring: str | Callable[..., float] | None = None,
        random_state: int | None = None,
        refit: bool = True,
        verbose: int = 0,
        **_: Any,
    ) -> None:
        warnings.warn(
            "StepwiseHyperoptOptimizer is deprecated; use StepwiseOptunaSearchCV instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            estimator=model,
            param_distributions=param_space_sequence,
            n_trials_per_step=max_evals_per_step,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            refit=refit,
            verbose=verbose,
        )
