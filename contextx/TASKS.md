1. [x] Add `catboost` to `pyproject.toml` as an optional dependency.
2. [x] Create a failing test in `test/tests_catboost.py` that attempts to use `StepwiseHyperoptOptimizer` with a `CatBoostRegressor` model.
3. [x] Implement the necessary changes in `src/sk_stepwise/__init__.py` to make the `CatBoostRegressor` test pass. This may involve:
    a. [x] Ensuring `CatBoostRegressor` can be passed as the `model` argument.
    b. [x] Handling any `CatBoostRegressor`-specific parameters or behaviors (e.g., `random_state`, `verbose`).
    c. [x] Verifying that `clean_int_params` correctly handles `CatBoostRegressor` integer parameters.
4. [ ] Refactor the code in `src/sk_stepwise/__init__.py` for clarity and maintainability, ensuring all tests still pass.
5. [ ] Add a test to `test/tests_catboost.py` that verifies `StepwiseHyperoptOptimizer` correctly passes `CatBoostRegressor` specific `fit` arguments (e.g., `cat_features`).
