# Tasks for Verifying Metric Optimization Logic

This document outlines tasks to verify that the `StepwiseHyperoptOptimizer` correctly handles both maximization and minimization metrics. The `hyperopt` `fmin` function inherently performs minimization, so metrics that should be maximized (e.g., accuracy, R2, ROC AUC) must be negated before being passed to `fmin`.

## 1. Understand Current Logic

-   [x] **1.1. Review `StepwiseHyperoptOptimizer.objective`**:
    -   [x] 1.1.1. Confirm that the `objective` function returns `-np.mean(score)`. This negates the score, effectively turning a maximization problem into a minimization problem for `hyperopt`.
    -   [x] 1.1.2. Verify that `self.best_score_` is set to `-min(trials.losses())` after `fmin` completes. This should convert the minimized (negated) loss back to the original scale of the metric.

-   [x] **1.2. Review `_custom_cross_val_score`**:
    -   [x] 1.2.1. Confirm that `check_scoring` correctly retrieves the scorer based on `self.scoring`.
    -   [x] 1.2.2. Understand how `scorer(fold_estimator, X_val, y_val)` behaves for different `scoring` strings (e.g., "accuracy", "neg_mean_squared_error").

## 2. Test Cases for Maximization Metrics

-   [x] **2.1. Add a new test for a classification model with "accuracy" scoring**:
    -   [x] 2.1.1. Create a simple classification dataset (e.g., using `make_classification`).
    -   [x] 2.1.2. Initialize `StepwiseHyperoptOptimizer` with a classification model (e.g., `LogisticRegression`, `SVC`) and `scoring="accuracy"`.
    -   [x] 2.1.3. Define a simple `param_space_sequence`.
    -   [x] 2.1.4. Run `optimizer.fit(X, y)`.
    -   [x] 2.1.5. Assert that `optimizer.best_score_` is positive and represents a reasonable accuracy score (e.g., > 0.5 for a binary classification).
    -   [x] 2.1.6. (Optional) Manually calculate the accuracy for `optimizer.best_params_` to cross-verify.

## 3. Test Cases for Minimization Metrics

-   [x] **3.1. Verify existing "neg_mean_squared_error" behavior**:
    -   [x] 3.1.1. Review `test_integer_hyperparameter_cleaning` and `test_fit_args_kwargs_passing` to ensure they implicitly test `neg_mean_squared_error` (the default).
    -   [x] 3.1.2. Confirm that `optimizer.best_score_` is negative, as expected for a negated error metric.

-   [x] **3.2. Add a new test for "mean_squared_error" (or similar direct error metric)**:
    -   [x] 3.2.1. Create a regression dataset.
    -   [x] 3.2.2. Initialize `StepwiseHyperoptOptimizer` with `scoring="neg_mean_squared_error"`. (Note: `sklearn`'s `mean_squared_error` is a scorer that needs to be minimized, but `check_scoring` will return a *negated* version if you pass `scoring="mean_squared_error"` directly. To truly test a non-negated metric, you'd need to pass a custom callable scorer that returns a positive value for error).
    -   [x] 3.2.3. Alternatively, if we want to test a metric that is *not* negated by `check_scoring` (e.g., a custom loss function), we would need to pass a callable to `scoring` that returns a value to be minimized.
    -   [x] 3.2.4. For now, focus on `neg_mean_squared_error` and ensure `best_score_` is negative.

## 4. Code Review and Refinement

-   [x] **4.1. Confirm `check_scoring` behavior**:
    -   [x] 4.1.1. Double-check `sklearn.metrics.check_scoring` documentation to ensure it consistently returns a scorer that produces *higher* values for *better* performance, which is then negated by `objective`. This is generally true for `sklearn`'s built-in scorers.
-   [x] **4.2. Edge Cases**:
    -   [x] 4.2.1. What happens if `scoring` is a custom callable that already returns a value to be minimized (e.g., a custom loss function)? The current `objective` would negate it, leading to maximization. If such a use case is desired, the `objective` function might need a flag or more sophisticated logic to determine whether to negate the score.
        -   [x] **Decision**: For now, assume `scoring` will always be an `sklearn` string or a callable that returns a value to be maximized (i.e., higher is better). If a custom loss function (lower is better) is needed, the user should pass `lambda estimator, X, y: -my_custom_loss(estimator, X, y)` as the `scoring` callable.
