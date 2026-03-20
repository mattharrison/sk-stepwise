# Product Requirements Document: sk-stepwise

## Product name

`sk-stepwise`

## Product summary

`sk-stepwise` is a Python library for stepwise hyperparameter optimization of scikit-learn compatible estimators. It is intended to let users define hyperparameter tuning as a sequence of search stages rather than as one large flat search space.

The product should feel native to the scikit-learn ecosystem:
- sklearn-compatible estimator API
- clean integration with pipelines and model-selection workflows
- predictable behavior across NumPy arrays, pandas objects, and sparse inputs

## Problem statement

Standard hyperparameter search workflows often require users to define a large search space up front. That can be inefficient, hard to reason about, and awkward when some parameters should only be tuned after earlier high-impact choices are fixed.

Users need a search tool that:
- narrows the space in stages
- keeps earlier best settings while optimizing later stages
- works like an sklearn search estimator instead of a one-off tuning script

## Target users

Primary users:
- Python ML practitioners using scikit-learn
- data scientists tuning tree models, linear models, and boosting models
- users who prefer structured, staged optimization over one-shot random or grid search

Secondary users:
- teams building reusable sklearn pipelines
- notebook users doing exploratory model development
- library authors who want a lightweight stepwise search component

## Goals

### Primary goals

- Provide a reliable stepwise hyperparameter search abstraction for sklearn estimators.
- Support Optuna as the primary optimization backend.
- Expose a public API that is idiomatic for scikit-learn users.
- Make staged optimization easier to define, inspect, reproduce, and test.

### Secondary goals

- Offer a migration path from the current Hyperopt-based implementation.
- Support common estimator fit metadata such as `sample_weight`.
- Provide useful optimization artifacts such as best params, best estimator, and per-step results.

## Non-goals

- Compete with full distributed HPO platforms.
- Support every tuning backend at launch.
- Reproduce every feature of `GridSearchCV` and `RandomizedSearchCV` immediately.
- Build a GUI or hosted optimization service.

## Product principles

- sklearn-first: API and behavior should match sklearn expectations where practical.
- backend-light: backend choice should not dominate the public contract.
- deterministic when seeded: repeated runs should be reproducible when possible.
- transparent: users should be able to inspect what happened at each step.
- minimal magic: parameter typing and behavior should come from explicit search-space definitions, not name-based heuristics.

## Core use cases

### 1. Stepwise tuning of a single estimator

A user wants to tune a `RandomForestRegressor` in stages:
- stage 1: structural parameters such as `n_estimators`
- stage 2: tree shape parameters such as `max_depth`
- stage 3: sample split thresholds

### 2. Tuning inside an sklearn pipeline

A user wants to tune `clf__max_depth` and `clf__learning_rate` within a `Pipeline`.

### 3. Reproducible notebook workflow

A user wants to run staged tuning in a notebook and retrieve:
- `best_params_`
- `best_score_`
- `best_estimator_`
- per-step search results

### 4. Model training with fit metadata

A user needs to pass `sample_weight` through tuning and final refit.

## User stories

- As a sklearn user, I want to call `fit`, `predict`, and `score` on the search object so it behaves like a familiar estimator.
- As a model developer, I want to tune parameters in steps so I can reduce search complexity.
- As a pipeline user, I want parameter names like `clf__max_depth` to work without special handling.
- As a reproducibility-focused user, I want `random_state` to produce stable search behavior.
- As a debugging user, I want access to per-step best parameters and scores.
- As a migrating user, I want a clear path from the current Hyperopt-based API to the new Optuna-based API.

## Functional requirements

### Search API

- The library must provide a primary sklearn-style search estimator.
- The search estimator must accept:
  - `estimator`
  - stepwise parameter search definitions
  - `cv`
  - `scoring`
  - `random_state`
  - trials-per-step configuration
  - `refit`
- The estimator must support `fit(X, y, **fit_params)`.

### Stepwise optimization behavior

- The search process must optimize one stage at a time.
- Best parameters from earlier stages must carry forward into later stages.
- Each stage must produce a best parameter set and best score.
- The final result must reflect the cumulative best parameters across all stages.

### Backend behavior

- Optuna must be the primary optimization backend.
- The public search-space API should be backend-neutral.
- Backend-specific internals should be isolated from the sklearn-facing estimator.

### sklearn compatibility

- The primary class must inherit from sklearn estimator base classes as appropriate.
- The estimator must clone the wrapped estimator during evaluation.
- The estimator must expose learned attributes only after `fit`.
- The estimator must support standard sklearn scorer semantics.
- The estimator must accept sklearn-compatible `cv` inputs.
- The estimator should expose `best_estimator_`, `best_params_`, and `best_score_`.
- The estimator should support pipeline parameter names.

### Input support

- The library must support:
  - NumPy arrays
  - pandas DataFrames and Series
  - scipy sparse matrices where the wrapped estimator supports them

### Reporting and inspection

- The library should expose per-step results in a structured form.
- The library should expose the Optuna study or a summarized step-study object for debugging.
- The library should offer configurable verbosity instead of unconditional printing.

## Non-functional requirements

### Reliability

- The estimator must avoid leaking model state between CV folds or trials.
- The library must fail clearly when called before fitting.

### Performance

- The implementation should avoid unnecessary data copies.
- The stepwise search overhead should remain small relative to estimator training time.

### Maintainability

- The codebase should separate public estimator logic, search-space definitions, and backend integration.
- The package should include a meaningful test suite for search semantics and sklearn compatibility.

### Compatibility

- The package must declare and test supported Python versions explicitly.
- Dependencies should resolve cleanly in supported environments.

## Proposed public API direction

Suggested primary class:

```python
StepwiseOptunaSearchCV(
    estimator,
    param_distributions,
    n_trials_per_step=50,
    scoring=None,
    cv=None,
    refit=True,
    random_state=None,
    n_jobs=None,
    verbose=0,
)
```

Suggested learned attributes:
- `best_estimator_`
- `best_params_`
- `best_score_`
- `step_results_`
- `study_` or `studies_`

## Migration requirements

- The project should provide a documented migration path from `StepwiseHyperoptOptimizer`.
- The project should either:
  - provide a deprecation window for the old class, or
  - make a deliberate major-version API break with clear release notes

## Documentation requirements

- README must show a complete quickstart example using the new Optuna-based API.
- Documentation must explain stepwise search semantics.
- Documentation must describe how to define search spaces.
- Documentation must include at least one pipeline example.
- Documentation must explain migration from Hyperopt usage.

## Testing requirements

The project should include tests for:
- basic fit/predict flow
- step carry-forward across multiple stages
- reproducibility with fixed seed
- NumPy and pandas inputs
- pipeline compatibility
- `sample_weight` passthrough
- fitted-state checks
- classifier and regressor examples
- backend integration behavior

## Success criteria

The product will be successful if:
- users can perform staged tuning with an sklearn-native API
- Optuna fully replaces Hyperopt in the mainline implementation
- the package behaves predictably in notebooks, scripts, and sklearn pipelines
- the test suite covers the main semantics of staged search
- installation and CI are stable across declared Python versions

## Open questions

- Should the public product name remain `sk-stepwise` if the primary class is renamed?
- Should Hyperopt compatibility be maintained temporarily or removed immediately?
- Is `cv_results_` required for first release, or can `step_results_` be the primary reporting artifact?
- Should the first Optuna release support pruning, or should that wait until iterative-estimator support is better defined?

## Milestones

### Milestone 1: foundation

- stabilize packaging and Python support
- improve tests
- define backend-neutral search-space objects

### Milestone 2: core Optuna product

- implement sklearn-style Optuna-backed stepwise search estimator
- expose best-estimator and per-step results
- publish updated docs

### Milestone 3: migration and polish

- deprecate or remove Hyperopt path
- expand sklearn compatibility
- tighten CI and release process
