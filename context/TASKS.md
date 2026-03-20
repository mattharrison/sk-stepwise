# sk-stepwise Migration Task List

Execution rules for every task in this file:
- Write a failing pytest first.
- Implement the smallest change needed to make the test pass.
- Run the narrowest relevant test with `uv run pytest ...` before moving on.
- Do not merge behavior changes without test coverage.
- Prefer incremental commits/PRs grouped by task clusters.

Suggested workflow per task:
1. Add or update pytest coverage for the target behavior.
2. Run the new test and confirm it fails for the expected reason.
3. Implement the change.
4. Run the focused pytest target.
5. Run the broader relevant test slice.

## Foundation

- [x] `SKW-001` Add a supported Python version policy to the project and encode it in `pyproject.toml`; write a failing test or CI-style validation check first that asserts unsupported runtimes are rejected or unsupported versions are not in the matrix.
- [x] `SKW-002` Clean `pyproject.toml` runtime dependencies by removing accidental or unnecessary deps such as `disutils` and likely `setuptools`; write a failing packaging-oriented test or validation first.
- [x] `SKW-003` Migrate deprecated `tool.uv.dev-dependencies` to modern `uv` dependency group configuration; write a failing validation check first.
- [x] `SKW-004` Add a reliable `uv`-based contributor workflow to docs and test commands; write a failing documentation or smoke-validation test first if you add executable examples.
- [x] `SKW-005` Remove generated/source-noise packaging artifacts from the tracked source layout where appropriate and add ignore rules if needed; write a failing repo hygiene check first.

## Test Harness

- [x] `SKW-006` Replace placeholder tests in `tests/test_basic.py` with behavior-focused tests; write the replacement tests first and remove `test_matt` only after equivalent or better coverage exists.
- [ ] `SKW-007` Add a focused regression test file for current stepwise semantics so migration work preserves intended behavior; start with a failing end-to-end regression test.
- [ ] `SKW-008` Add tests that cover NumPy array inputs to `fit`, `predict`, and scoring; write failing tests first.
- [ ] `SKW-009` Add tests that cover pandas `DataFrame`/`Series` inputs; write failing tests first.
- [ ] `SKW-010` Add tests that cover sparse matrix inputs where the wrapped estimator supports them; write failing tests first.
- [x] `SKW-011` Add tests for `predict` and `score` before fit, asserting proper sklearn-style fitted-state errors; write failing tests first.
- [x] `SKW-012` Add tests for deterministic behavior under fixed `random_state`; write failing tests first.
- [x] `SKW-013` Add tests for classifier usage, including classification-appropriate scoring and CV behavior; write failing tests first.
- [x] `SKW-014` Add tests for regressor usage with explicit `scoring`; write failing tests first.
- [x] `SKW-014A` Add an executable end-to-end regression test based on the README-style example using `numpy`, `pandas`, `RandomForestRegressor`, staged parameter spaces, `fit`, and `predict`, but rewritten for the Optuna-based API instead of Hyperopt; write the failing pytest first and use it as a release smoke test.
- [x] `SKW-014B` Add an sklearn `RandomForestRegressor` integration test that runs a real multi-step Optuna search and verifies `best_params_`, `best_estimator_`, and `predict` behavior; write the failing pytest first.
- [x] `SKW-014C` Add an XGBoost integration test for both estimator compatibility and staged search behavior, including realistic parameter types such as integer depth and float learning-rate style params; write the failing pytest first.
- [x] `SKW-014D` Add a CatBoost integration test covering sklearn-compatible fit/predict behavior, staged search, and any required fit metadata or categorical-handling edge cases; write the failing pytest first.
- [x] `SKW-015` Add tests for pipeline compatibility with namespaced parameters like `clf__max_depth`; write failing tests first.
- [x] `SKW-016` Add tests for fit metadata passthrough such as `sample_weight`; write failing tests first.
- [x] `SKW-017` Add tests that prove fold/trial isolation by asserting cloned estimators are used instead of mutating the same estimator instance; write failing tests first.
- [x] `SKW-018` Add tests for per-step carry-forward behavior across two or more search stages; write failing tests first.
- [x] `SKW-019` Add tests for learned attributes after fit only, including `best_params_`, `best_score_`, and `best_estimator_`; write failing tests first.
- [ ] `SKW-020` Add sklearn estimator compatibility checks or targeted `check_estimator`-style tests for the meta-estimator behavior; start with a failing targeted check first.

## API Redesign

- [x] `SKW-021` Introduce a backend-neutral search-space abstraction for categorical, integer, and float dimensions; begin with failing tests that define and validate these dimension objects.
- [x] `SKW-022` Add tests that prove parameter types come from search-space definitions rather than parameter-name coercion; write failing tests first.
- [x] `SKW-023` Remove the ad hoc `clean_int_params` behavior and replace it with typed dimension handling; write failing tests first.
- [x] `SKW-024` Rename the wrapped model parameter from `model` to `estimator` in the new API while preserving or deprecating the old path intentionally; write failing API tests first.
- [x] `SKW-025` Split implementation out of `src/sk_stepwise/__init__.py` into dedicated modules for estimator logic, search-space definitions, and backend integration; write failing import/public-API tests first.
- [x] `SKW-026` Replace constructor-time learned dataclass fields with sklearn-style learned attributes set only during `fit`; write failing tests first.
- [x] `SKW-027` Add `best_estimator_` as a first-class learned attribute and test that predictions delegate to it; write failing tests first.
- [x] `SKW-028` Add configurable verbosity/logging behavior and remove unconditional `print` calls; write failing tests first.
- [x] `SKW-029` Add `refit` support and test both `refit=True` and `refit=False` behavior; write failing tests first.
- [x] `SKW-030` Add structured per-step reporting such as `step_results_`; write failing tests first.

## sklearn Alignment

- [x] `SKW-031` Replace the custom pandas-only CV loop with sklearn-native evaluation using estimator cloning and proper scorer handling; start with a failing test that exposes current scoring/CV defects.
- [x] `SKW-032` Support sklearn-compatible `cv` values (`None`, `int`, splitter objects, iterable splits); write failing tests first.
- [x] `SKW-033` Use classification-aware CV selection via sklearn helpers when appropriate; write failing tests first.
- [x] `SKW-034` Ensure scorer strings and scorer callables are both honored correctly; write failing tests first.
- [x] `SKW-035` Add fitted-state checks via sklearn utilities in delegated methods; write failing tests first.
- [x] `SKW-036` Delegate optional estimator methods such as `predict_proba`, `decision_function`, and `transform` when the best estimator supports them; write failing tests first.
- [x] `SKW-037` Validate inputs using sklearn-friendly indexing/validation utilities instead of `.iloc` assumptions; write failing tests first.
- [x] `SKW-038` Ensure the original base estimator is never left in a mutated fitted state after search unless explicitly documented; write failing tests first.

## Optuna Backend

- [x] `SKW-039` Add `optuna` as the primary optimization dependency and remove direct runtime dependence on `hyperopt` in the new code path; write a failing import/backend smoke test first.
- [x] `SKW-040` Create Optuna backend helpers that map backend-neutral dimensions to `trial.suggest_categorical`, `trial.suggest_int`, and `trial.suggest_float`; write failing unit tests first.
- [x] `SKW-041` Implement one-study-per-step optimization semantics in Optuna; write a failing multi-step behavior test first.
- [x] `SKW-042` Seed Optuna sampling via `TPESampler(seed=...)` and verify reproducibility; write failing tests first.
- [x] `SKW-043` Store study artifacts on the fitted object as `study_` or `studies_`; write failing tests first.
- [x] `SKW-044` Record trial counts per step and test that `n_trials_per_step` is respected; write failing tests first.
- [x] `SKW-045` Add a primary Optuna-backed sklearn-style class such as `StepwiseOptunaSearchCV`; write failing API and fit/predict tests first.
- [x] `SKW-046` Validate that stepwise merged params are what Optuna actually evaluates at each stage; write failing tests first.

## Hyperopt Migration

- [x] `SKW-047` Decide and implement the migration strategy for `StepwiseHyperoptOptimizer`: compatibility shim or removal; write failing migration tests first.
- [x] `SKW-048` If keeping compatibility temporarily, add deprecation warnings for Hyperopt-specific API usage; write failing warning tests first.
- [ ] `SKW-049` If supporting both old and new search-space inputs during transition, add tests for both paths and then implement the adapter; write failing tests first.
- [x] `SKW-050` Remove Hyperopt-specific imports and public docs from the mainline API once the migration path is in place; write failing docs/import tests first.
- [x] `SKW-051` Remove Hyperopt from runtime dependencies after compatibility needs are resolved; write failing packaging validation first.

## Python Cleanup

- [ ] `SKW-052` Remove unused imports, dead commented-out typing attempts, and misleading code paths; write failing lint or targeted tests first.
- [ ] `SKW-053` Revisit internal type aliases and protocols so parameter values are not artificially restricted to `int | float | str | bool`; write failing typing-oriented tests first.
- [x] `SKW-054` Add narrow internal helper tests for parameter merging and step-result aggregation; write failing unit tests first.
- [x] `SKW-055` Refactor internal modules to isolate pure functions from estimator side effects; write failing unit tests first for the extracted logic.
- [ ] `SKW-056` Add logging hooks or structured diagnostics suitable for notebooks and scripts; write failing tests first.

## Documentation

- [x] `SKW-057` Rewrite `README.md` around the Optuna-based product direction; write failing doctest or smoke-example tests first.
- [x] `SKW-058` Add a quickstart example for the new sklearn-style API; write a failing executable example test first, and make it the same `numpy` + `pandas` + `RandomForestRegressor` staged-tuning scenario captured in `SKW-014A`.
- [x] `SKW-059` Add a migration guide from Hyperopt spaces to the new backend-neutral search-space API; write a failing docs coverage or example validation first.
- [x] `SKW-060` Add a pipeline example and a `sample_weight` example to docs; write failing executable example tests first.

## CI and Quality Gates

- [x] `SKW-061` Add a `uv`-based CI test command matrix for supported Python versions; write the CI config after defining failing validation expectations.
- [x] `SKW-062` Add `ruff` checks and make them part of the default validation path; write failing lint validation first.
- [x] `SKW-063` Add `mypy` checks for the supported code paths and tighten typing around the new modules; write failing type-check validation first.
- [x] `SKW-064` Add a release-validation task that installs the package in a clean environment and runs focused smoke tests with `uv`; write failing smoke-validation first.

## Suggested implementation order

Recommended sequence:
1. `SKW-001` to `SKW-020`
2. `SKW-021` to `SKW-038`
3. `SKW-039` to `SKW-046`
4. `SKW-047` to `SKW-051`
5. `SKW-052` to `SKW-064`

## Definition of done

A task cluster is done only when:
- the new pytest starts red and ends green
- focused `uv run pytest ...` commands pass
- broader relevant tests pass
- docs/examples are updated if the user-facing API changed
- no Hyperopt coupling remains in the intended mainline Optuna path
