# sk-stepwise

`sk-stepwise` is a small Python library for staged hyperparameter optimization of scikit-learn compatible estimators.

The main API is `StepwiseOptunaSearchCV`, which runs Optuna search one step at a time. Each step optimizes a subset of parameters while carrying forward the best settings found in earlier steps.

## Why stepwise search

A flat search space is often larger than it needs to be. Many workflows are easier to reason about in stages:

- tune structural parameters first
- tune regularization or sampling parameters next
- tune learning-rate style parameters later

That is the model this library supports.

## Installation

```sh
uv add sk-stepwise
```

For development:

```sh
uv sync
uv run pytest
uv run pytest -q tests/test_readme_doctest.py
```

## Quickstart

```python
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sk_stepwise import Float, Int, StepwiseOptunaSearchCV
>>>
>>> rng = np.random.default_rng(42)
>>> X = pd.DataFrame(rng.random((100, 5)), columns=[f"feature_{i}" for i in range(5)])
>>> y = pd.Series(rng.random(100))
>>>
>>> estimator = RandomForestRegressor(random_state=0)
>>> param_distributions = [
...     {"n_estimators": Int(50, 150)},
...     {"max_depth": Int(3, 10)},
...     {"min_samples_split": Float(0.1, 1.0)},
... ]
>>>
>>> search = StepwiseOptunaSearchCV(
...     estimator=estimator,
...     param_distributions=param_distributions,
...     n_trials_per_step=2,
...     random_state=0,
... )
>>> search.fit(X, y)  # doctest: +ELLIPSIS
StepwiseOptunaSearchCV(...)
>>> predictions = search.predict(X)
>>> len(predictions)
100
>>> sorted(search.best_params_.keys())
['max_depth', 'min_samples_split', 'n_estimators']
>>> isinstance(search.best_score_, float)
True

```

## Build a real model from the search results

You can use `best_params_` directly with a fresh estimator instance.

```python
>>> from sklearn.ensemble import RandomForestRegressor
>>>
>>> best_params = search.best_params_
>>> sorted(best_params)
['max_depth', 'min_samples_split', 'n_estimators']
>>> final_model = RandomForestRegressor(random_state=0, **best_params)
>>> final_model.fit(X, y)
RandomForestRegressor(...)
>>> isinstance(final_model.get_params()["n_estimators"], int)
True
>>> tuned_predictions = final_model.predict(X)
>>> len(tuned_predictions)
100

```

## Search-space types

Use the backend-neutral dimension helpers:

- `Int(low, high, log=False)` for ordered integer values like `n_estimators`, `max_depth`, `depth`, `min_samples_leaf`
- `Float(low, high, log=False)` for continuous values like `learning_rate`, `subsample`, regularization strengths
- `Categorical(choices)` for unordered values like `criterion`, `solver`, `bootstrap`

Examples:

```python
>>> from sk_stepwise import Categorical, Float, Int
>>>
>>> space = [
...     {"n_estimators": Int(50, 300)},
...     {"max_depth": Int(2, 12)},
...     {"learning_rate": Float(1e-3, 1e-1, log=True)},
...     {"criterion": Categorical(["squared_error", "absolute_error"])},
... ]
>>> len(space)
4

```

### Numeric categorical warning

If you write `Categorical([10, 20, 30])`, the library now emits a warning. For ordered numeric values, `Int(...)` or `Float(...)` is usually a better fit because the optimizer can use the numeric ordering.

## Progress logging

Set `verbose=1` to print step-by-step progress:

- `Optimizing step 1/3`
- `Best parameters after step 1: ...`
- `Best score after step 1: ...`
- `Improvement: ...`

This is intentionally opt-in.

## scikit-learn behavior

`StepwiseOptunaSearchCV` is designed to behave like a sklearn-style search estimator:

- supports `fit`, `predict`, and `score`
- exposes `best_params_`, `best_score_`, `best_estimator_`, `study_`, `studies_`, and `step_results_`
- works with pipelines and namespaced params like `regressor__max_depth`
- supports scorer strings and scorer callables
- supports `cv` as an `int`, splitter object, or iterable of splits
- passes fit metadata such as `sample_weight` through sklearn evaluation

Optional methods are delegated when supported by the fitted best estimator:

- `predict_proba`
- `decision_function`
- `transform`

## Migration from Hyperopt

The old `StepwiseHyperoptOptimizer` name is deprecated.

Current behavior:

- `StepwiseHyperoptOptimizer(...)` still works as a compatibility shim
- it emits `DeprecationWarning`
- it maps old constructor names onto `StepwiseOptunaSearchCV`

Example migration:

```python
>>> import warnings
>>> from sk_stepwise import StepwiseHyperoptOptimizer, StepwiseOptunaSearchCV
>>> warnings.simplefilter("ignore", DeprecationWarning)
>>> # old
>>> search = StepwiseHyperoptOptimizer(
...     model=estimator,
...     param_space_sequence=space,
...     max_evals_per_step=20,
... )
>>> # new
>>> search = StepwiseOptunaSearchCV(
...     estimator=estimator,
...     param_distributions=space,
...     n_trials_per_step=20,
... )

```

Important:

- backend-neutral dimensions such as `Int`, `Float`, and `Categorical` are the supported path
- old Hyperopt space objects are not part of the new mainline API

## Example: pipeline usage

```python
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from sk_stepwise import Int, StepwiseOptunaSearchCV
>>>
>>> pipeline = Pipeline(
...     [
...         ("scale", StandardScaler()),
...         ("regressor", RandomForestRegressor(random_state=0)),
...     ]
... )
>>> space = [
...     {"regressor__n_estimators": Int(50, 150)},
...     {"regressor__max_depth": Int(2, 8)},
... ]
>>> search = StepwiseOptunaSearchCV(
...     estimator=pipeline,
...     param_distributions=space,
...     n_trials_per_step=2,
...     random_state=0,
... )

```

## Example: sample weights

```python
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from sk_stepwise import Categorical, StepwiseOptunaSearchCV
>>>
>>> sample_weight = np.linspace(1.0, 2.0, len(y))
>>> search = StepwiseOptunaSearchCV(
...     estimator=LinearRegression(),
...     param_distributions=[{"fit_intercept": Categorical([True, False])}],
...     n_trials_per_step=2,
...     random_state=0,
... )
>>> search.fit(X, y, sample_weight=sample_weight)  # doctest: +ELLIPSIS
StepwiseOptunaSearchCV(...)

```

## Status

The core Optuna path is implemented and covered by tests for:

- NumPy, pandas, and plain list inputs
- regression and classification
- sklearn pipelines
- XGBoost and CatBoost integration
- deprecated Hyperopt shim behavior

## License

MIT
