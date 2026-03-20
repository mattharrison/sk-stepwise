# sk-stepwise Audit: hyperopt -> optuna migration, sklearn fit, and cleanup

## Scope

Audit target:
- current `hyperopt` coupling
- migration plan to `optuna`
- ideas to make the library fit better in the scikit-learn ecosystem
- Python/package/test cleanup opportunities

Repository shape today:
- the package is effectively one public module: `src/sk_stepwise/__init__.py`
- the core API is one estimator-like wrapper: `StepwiseHyperoptOptimizer`
- tests are minimal and mostly smoke-level: `tests/test_basic.py`

## Executive summary

The migration is feasible because almost all `hyperopt` coupling lives in one class, but a straight package swap would preserve several deeper design problems:

1. The public API exposes backend-specific search-space objects instead of a backend-neutral stepwise search description.
2. The estimator does not follow several scikit-learn conventions closely enough to behave robustly in pipelines, model selection, or estimator checks.
3. Validation is fragile because the package/dependency setup is not stable on the current Python runtime.

The best path is not "replace `hyperopt.fmin` with `optuna.study.optimize` in place". The better path is:

1. Introduce a backend-neutral stepwise search API.
2. Rebuild the optimizer around sklearn-native evaluation primitives.
3. Add an Optuna backend first.
4. Keep a temporary Hyperopt compatibility layer only if you need a deprecation window.

## Highest-priority findings

### 1. Public API is hard-coupled to Hyperopt search objects

Evidence:
- `src/sk_stepwise/__init__.py:4`
- `src/sk_stepwise/__init__.py:21`
- `src/sk_stepwise/__init__.py:79`
- `README.md:50`
- `README.md:61`
- `tests/test_basic.py:7`
- `tests/test_basic.py:49`

Why it matters:
- `param_space_sequence` currently accepts `hyperopt.pyll` objects (`SymbolTable`) directly.
- That makes `hyperopt` part of the user-facing contract, not just an implementation detail.
- An Optuna migration will otherwise become a breaking API rewrite unless you add another abstraction layer.

What to change:
- Replace `param_space_sequence: list[dict[str, PARAM | SymbolTable]]` with a backend-neutral representation.
- Recommended shape:

```python
Step = dict[str, SearchDimension]
param_distributions: list[Step]
```

- Example dimension types:
  - `Categorical(["gini", "entropy"])`
  - `Int(low=2, high=20, log=False)`
  - `Float(low=1e-4, high=1e-1, log=True)`

- Then compile those dimensions into Optuna calls inside the backend:
  - `trial.suggest_categorical`
  - `trial.suggest_int`
  - `trial.suggest_float`

Recommended migration strategy:
- Add a new estimator such as `StepwiseOptunaSearchCV`.
- Mark `StepwiseHyperoptOptimizer` deprecated.
- If you want a shorter path, keep the public class name but change the internals and accept both old and new search-space inputs during one deprecation cycle.

Risk:
- This is the main breaking-change surface.

### 2. The estimator mutates and reuses the same model instance across CV folds and trials

Evidence:
- `src/sk_stepwise/__init__.py:96`
- `src/sk_stepwise/__init__.py:97`
- `src/sk_stepwise/__init__.py:130`

Why it matters:
- sklearn model selection clones estimators per fold/trial for isolation.
- The current code repeatedly calls `self.model.set_params(...)` and `fit(...)` on the same underlying estimator object.
- This can leak learned state across folds for estimators with warm-start behavior, incremental behavior, caches, or mutable fit-time internals.
- It also makes correctness depend on estimator implementation details rather than sklearn guarantees.

What to change:
- Store the base estimator as `estimator`, not `model`.
- Use `sklearn.base.clone(self.estimator)` inside each trial evaluation and before the final fit.
- Do not mutate the original estimator instance during search.

Recommended implementation direction:
- Objective function:
  - clone base estimator
  - set suggested params
  - evaluate with sklearn CV helpers
- Final fit:
  - `best_estimator_ = clone(self.estimator).set_params(**best_params_)`
  - fit `best_estimator_`

### 3. Cross-validation path is custom, pandas-specific, and ignores sklearn scoring conventions

Evidence:
- `src/sk_stepwise/__init__.py:48`
- `src/sk_stepwise/__init__.py:65`
- `src/sk_stepwise/__init__.py:69`
- `src/sk_stepwise/__init__.py:70`

Why it matters:
- `_cross_val_score_with_fit_params` assumes `X.iloc` and `y.iloc`, so plain NumPy arrays and many array-likes will fail.
- The `scoring` argument is accepted but never actually used. The code calls `estimator.score(...)` instead.
- Classification/regression splitter choice is fixed to `KFold`, which is wrong for many classifiers where `StratifiedKFold` is expected.
- The CV random seed is hard-coded to `42` rather than using `self.random_state`.

What to change:
- Replace the custom scorer with sklearn-native evaluation:
  - `cross_val_score`
  - or `cross_validate`
  - and `sklearn.utils._safe_indexing` only if custom splitting remains necessary
- Use `check_scoring` / sklearn scoring semantics rather than bypassing them.
- Accept `cv` as `None | int | CV splitter | iterable`, as sklearn APIs do.
- Use `check_cv(cv, y, classifier=is_classifier(estimator))`.

Notes:
- If fit-time metadata like `sample_weight` must be supported, align with sklearn’s newer metadata-routing direction instead of maintaining a separate partial reimplementation of `cross_val_score`.
- If you keep a custom loop temporarily, use `_safe_indexing` or indexable conversion rather than `.iloc`.

### 4. Learned attributes and fit state do not follow sklearn conventions consistently

Evidence:
- `src/sk_stepwise/__init__.py:84`
- `src/sk_stepwise/__init__.py:85`
- `src/sk_stepwise/__init__.py:102`
- `src/sk_stepwise/__init__.py:135`
- `src/sk_stepwise/__init__.py:138`

Why it matters:
- `best_params_` and `best_score_` are dataclass fields, so they exist before fitting. In sklearn, learned attributes typically appear only after `fit`.
- `predict` and `score` do not check fitted state.
- `X` and `y` are persisted on `self`, which is unnecessary and makes the estimator hold training data references longer than needed.

What to change:
- Remove learned attributes from constructor fields.
- Set learned attributes only during `fit`, for example:
  - `best_params_`
  - `best_score_`
  - `best_estimator_`
  - `study_` if Optuna is exposed
  - `n_trials_` / `cv_results_` if you add reporting
- Use `check_is_fitted(self, "best_estimator_")` in `predict` and `score`.
- Delegate `predict`, `score`, and optionally `predict_proba`, `decision_function`, `transform` when supported by the wrapped estimator.

### 5. Parameter coercion is ad hoc and model-specific

Evidence:
- `src/sk_stepwise/__init__.py:87`
- `src/sk_stepwise/__init__.py:88`

Why it matters:
- `clean_int_params` hard-codes `["max_depth", "reg_alpha"]`.
- `reg_alpha` is usually float-valued in common estimators such as XGBoost, so this can silently change semantics.
- Parameter typing should come from the search-space definition, not string name heuristics.

What to change:
- Encode integer/float/categorical behavior in the search dimension abstraction.
- Remove all parameter-name-based coercion.

## Optuna migration plan

### Recommended target API

Use an sklearn-flavored class name and contract. Suggested primary class:

```python
class StepwiseOptunaSearchCV(BaseEstimator, MetaEstimatorMixin):
    estimator: BaseEstimator
    param_distributions: list[dict[str, SearchDimension]]
    n_trials_per_step: int = 50
    scoring: str | Callable | None = None
    cv: int | CVSplitter | None = None
    refit: bool = True
    random_state: int | None = None
    n_jobs: int | None = None
    study_sampler: optuna.samplers.BaseSampler | None = None
    direction: Literal["maximize", "minimize"] | None = None
```

Important sklearn-facing attributes:
- `best_params_`
- `best_score_`
- `best_estimator_`
- `study_` or `studies_` if one study per step
- `cv_results_` if you want parity with sklearn search estimators
- `n_splits_`
- `multimetric_` if multi-metric scoring is added later

### Backend design

Recommended internal split:
- `search_space.py`
  - dimension dataclasses or typed specs
- `optuna_backend.py`
  - compiles a step spec into trial suggestions
- `estimator.py`
  - sklearn-facing estimator/search class

This reduces future coupling if you ever want:
- random search backend
- sklearn `ParameterSampler` backend
- Ray Tune or another orchestrator

### Stepwise search semantics

Current behavior:
- each step optimizes only the new params for that stage
- prior best params are carried forward

That semantics is fine and maps cleanly to Optuna:
- for each step, create a new study
- objective samples only that step’s dimensions
- merge step params with cumulative best params from prior steps
- evaluate merged params

Open design choice:
- one study per step is simplest and matches current behavior
- one global study with conditional spaces is possible, but not necessary here

Recommendation:
- use one study per step first
- persist `step_results_` with best params and best score per step for transparency

### Sampler/pruner suggestions

Good defaults:
- sampler: `TPESampler(seed=random_state)`
- pruner: none initially

Add pruners only if you support estimators with iterative partial progress reporting:
- XGBoost can benefit
- most plain sklearn estimators cannot report intermediate values cleanly

### Compatibility bridge options

Option A: clean break
- remove `hyperopt`
- rename class/API
- simplest long-term design

Option B: transitional
- keep `StepwiseHyperoptOptimizer`
- introduce `StepwiseOptunaSearchCV`
- deprecate old class and old search-space inputs

Recommendation:
- Option B if this library already has users
- Option A if usage is still very limited and you want to fix the API properly now

## scikit-learn ecosystem improvement ideas

### 1. Align naming with sklearn search estimators

Current name:
- `StepwiseHyperoptOptimizer`

Better names:
- `StepwiseOptunaSearchCV`
- `StepwiseSearchCV`

Why:
- sklearn users expect `*SearchCV` for CV-based hyperparameter search wrappers
- the current name emphasizes backend instead of sklearn behavior

### 2. Prefer `estimator` over `model`

Evidence:
- `src/sk_stepwise/__init__.py:78`
- `README.md:84`

Why:
- `estimator` is the standard sklearn term
- makes cloning, delegation, and docs more idiomatic

### 3. Expose `best_estimator_` and delegate estimator interface methods

Useful delegated methods when available:
- `predict`
- `predict_proba`
- `predict_log_proba`
- `decision_function`
- `transform`
- `inverse_transform`
- `score`

Why:
- this matches behavior users expect from `GridSearchCV` and `RandomizedSearchCV`

### 4. Support sklearn-compatible `cv` and `scoring`

Current behavior is narrower than sklearn.

Recommended support:
- `cv=None` default with `check_cv`
- scorer strings and scorer callables
- possibly multi-metric scoring later

### 5. Consider `refit`

Add:
- `refit: bool | str | Callable = True`

Why:
- standard sklearn behavior
- supports workflows where users only want the search results without a final fitted estimator

### 6. Consider metadata routing for fit params

The current custom `fit(*fit_args, **fit_kwargs)` support is useful, but the implementation is not sklearn-native.

Suggested direction:
- near term: accept `fit_params` carefully and pass them through the CV evaluation path
- longer term: align with sklearn metadata routing patterns where feasible

### 7. Add estimator tags or at least estimator checks to CI

Use:
- `check_estimator`
- targeted estimator checks for meta-estimators

Even if full estimator-check compliance is not immediate, running them will expose gaps early.

### 8. Support arrays, sparse matrices, and DataFrames uniformly

Current code advertises broad input types but implementation depends on pandas indexing.

Use sklearn validation utilities:
- `indexable`
- `check_X_y` where appropriate
- `validate_data` if you move to a more standard estimator implementation

## Test and validation audit

### Current test coverage is too thin for a backend migration

Evidence:
- `tests/test_basic.py`
- `tests/conftest.py`

Observations:
- one test only checks construction
- one `xfail` test appears to encode a failure instead of intended behavior
- one test is a placeholder (`test_matt`)
- no test asserts scoring correctness
- no test asserts fold isolation via estimator cloning
- no test asserts `best_params_`, `best_score_`, or `predict` behavior rigorously
- no tests cover classifier use, sparse inputs, NumPy inputs, or pipeline composition

Recommended tests before/during migration:
- fit on NumPy arrays
- fit on pandas inputs
- classifier path uses classification scoring successfully
- `best_estimator_` is cloned and fitted, not the original estimator reused
- stepwise carry-forward behavior across two or more search steps
- deterministic results with fixed `random_state`
- passthrough of `sample_weight`
- pipeline compatibility with parameter names like `clf__max_depth`
- `predict`/`score` raise before fit
- Optuna study records expected trial count per step

### Baseline test run is blocked by packaging/runtime issues

Observed on 2026-03-20 in this environment:
- `uv run pytest -q` failed before test execution while building `numpy==2.1.1`
- runtime was `CPython 3.14.2`

Interpretation:
- current dependency constraints are not validated against the active Python version
- CI/runtime support is unclear

Recommendation:
- explicitly define supported Python versions
- pin versions that actually have wheels in CI
- add a CI matrix instead of relying on local ad hoc installs

## Packaging and Python cleanup ideas

### 1. Package metadata needs cleanup

Evidence:
- `pyproject.toml:4`
- `pyproject.toml:8`
- `pyproject.toml:18`

Issues:
- placeholder description
- likely accidental dependency on `disutils>=1.4.32.post2`
- direct dependency on `setuptools` probably unnecessary at runtime
- deprecated `tool.uv.dev-dependencies`

Recommended changes:
- set a real project description
- remove `disutils` unless there is a proven runtime need
- remove `setuptools` from runtime dependencies unless imported at runtime
- migrate to `[dependency-groups]` for dev deps

### 2. Split `__init__.py`

Current issue:
- implementation lives in package root

Why it matters:
- harder to evolve APIs cleanly
- harder to add more search backends or helper modules

Recommended structure:
- `src/sk_stepwise/__init__.py`
- `src/sk_stepwise/search.py`
- `src/sk_stepwise/spaces.py`
- `src/sk_stepwise/_optuna.py`
- `src/sk_stepwise/_validation.py`

Keep `__init__.py` as a light export surface.

### 3. Remove unnecessary dataclass use for estimator state

Current issue:
- dataclass fields mix constructor params and learned attrs

Recommendation:
- either keep dataclass but mark learned attrs with `init=False` and set in `fit`
- or switch to a plain sklearn-style class with an explicit `__init__`

For sklearn compatibility, a plain explicit `__init__` is often clearer.

### 4. Tighten type hints around estimators and scorers

Issues:
- `_Fitable` and `_FitableWithArgs` are loose
- `PARAM = int | float | str | bool` excludes common valid parameter values such as `None`, tuples, lists, callables, nested estimators

Recommendation:
- do not overconstrain estimator param value types
- use broader mappings such as `dict[str, Any]` internally for params
- keep strict typing around your own search-space objects instead

### 5. Remove dead or misleading code

Evidence:
- `src/sk_stepwise/__init__.py:3` imports `cross_val_score` but does not use it
- several commented `set_params` signatures remain in source

Recommendation:
- delete commented-out experimental typing
- delete unused imports

### 6. Remove print-based progress from library code

Evidence:
- `src/sk_stepwise/__init__.py:107`
- `src/sk_stepwise/__init__.py:126`
- `src/sk_stepwise/__init__.py:127`

Why:
- prints are noisy inside notebooks, pipelines, and larger training systems

Recommended options:
- add `verbose: int = 0`
- use Optuna logging controls
- or use Python logging with quiet default behavior

### 7. Add CI and quality gates

Recommended baseline:
- `ruff`
- `mypy`
- `pytest`
- targeted sklearn estimator compatibility tests
- CI matrix across supported Python versions

### 8. Remove generated egg-info from source tree if it is not intentionally committed

Evidence:
- `src/sk_stepwise.egg-info/`

Recommendation:
- generate build metadata during packaging, not as source-controlled implementation clutter

## Suggested phased plan

### Phase 1: stabilization

- clean package metadata and dependency groups
- define supported Python versions
- make tests run reliably in CI
- add a meaningful regression test suite around current stepwise semantics

### Phase 2: API redesign

- introduce backend-neutral search dimensions
- rename primary class toward sklearn conventions
- add `best_estimator_`, fitted-state checks, clone-based evaluation

### Phase 3: Optuna implementation

- implement one-study-per-step search
- map dimensions to `trial.suggest_*`
- support deterministic sampling via `TPESampler(seed=...)`

### Phase 4: compatibility and deprecation

- optionally support old Hyperopt spaces temporarily
- document deprecation path
- remove `hyperopt` dependency after transition

## Concrete short-list of first changes

If you want the smallest high-value sequence, do these first:

1. Replace the custom CV loop with sklearn-native evaluation and estimator cloning.
2. Introduce a backend-neutral search-space abstraction.
3. Add an Optuna-backed `StepwiseOptunaSearchCV`.
4. Add tests for step carry-forward, NumPy inputs, `sample_weight`, and pipeline compatibility.
5. Clean `pyproject.toml` so the project installs reproducibly on supported Python versions.

## Bottom line

Optuna is a good fit for this library, but the main work is API and sklearn alignment, not the sampler swap itself. If you fix the estimator contract and search-space abstraction first, the backend migration becomes straightforward and the library will fit much better into normal sklearn workflows.
