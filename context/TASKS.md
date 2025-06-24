# Tasks to Fix Failing Tests

This document outlines the tasks required to address the failing tests identified in the `pytest` run.

## 1. Test Failures

-   [ ] **1.1. `test_fit_args_kwargs_passing` in `tests/test_basic.py`**:
    -   [ ] **Problem**: `ValueError: sample_weight.shape == (100,), expected (80,)!`
    -   [ ] **Root Cause**: The `sample_weight` array passed to the `fit` method has a different number of samples (100) than the `X` and `y` data (80 samples after cross-validation split). This indicates that `sample_weight` is not being correctly split or passed through the cross-validation process within `_custom_cross_val_score`.
    -   [ ] **Action**:
        -   [ ] 1.1.1. Investigate how `fit_params` (which includes `sample_weight`, `custom_arg`, and `extra_kwarg`) are handled within `_custom_cross_val_score`.
        -   [ ] 1.1.2. Ensure that `sample_weight` is correctly split and passed to the `fit` method of the estimator for each fold during cross-validation.
        -   [ ] 1.1.3. Verify that `custom_arg` and `extra_kwarg` are also correctly passed to the `fit` method.

## 2. Warnings

-   [ ] **2.1. `DeprecationWarning: pkg_resources is deprecated`**:
    -   [ ] **Problem**: `hyperopt/atpe.py` is using `pkg_resources`, which is deprecated.
    -   [ ] **Action**: This is likely an upstream issue with `hyperopt`. Note it, but it might not be directly fixable without updating `hyperopt` or its dependencies.
-   [ ] **2.2. `PytestUnknownMarkWarning: Unknown pytest.mark.matt`**:
    -   [ ] **Problem**: The `pytest.mark.matt` custom mark is not registered.
    -   [ ] **Action**: Register the custom mark in `pyproject.toml` or `conftest.py` to suppress the warning.
-   [ ] **2.3. `DeprecationWarning: datetime.datetime.utcnow() is deprecated`**:
    -   [ ] **Problem**: `hyperopt/utils.py` is using `datetime.datetime.utcnow()`, which is deprecated.
    -   [ ] **Action**: Similar to the `pkg_resources` warning, this is likely an upstream issue with `hyperopt`. Note it, but it might not be directly fixable without updating `hyperopt` or its dependencies.

## 3. General Improvements

-   [ ] 3.1. Review the `_custom_cross_val_score` function to ensure it correctly handles and propagates all `fit_params` to the underlying estimator's `fit` method during cross-validation.
-   [ ] 3.2. Ensure that the `StepwiseHyperoptOptimizer` correctly passes `fit_params` from its `fit` method to the `objective` function and subsequently to `_custom_cross_val_score`.
