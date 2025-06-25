import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import check_scoring
from hyperopt import fmin, tpe, space_eval, Trials


# From typing
import pandas as pd
from typing import Self


from typing import TypeAlias
from scipy.sparse import spmatrix
import numpy.typing


from typing import Protocol


from collections.abc import Callable
from hyperopt.pyll.base import SymbolTable

from dataclasses import dataclass, field

PARAM = int | float | str | bool
MatrixLike: TypeAlias = np.ndarray | pd.DataFrame | spmatrix
ArrayLike: TypeAlias = numpy.typing.ArrayLike


class _Fitable(Protocol):
    def fit(self, X: MatrixLike, y: ArrayLike, *args, **kwargs) -> Self: ...
    def predict(self, X: MatrixLike) -> ArrayLike: ...
    def set_params(self, **params: PARAM) -> Self: ...
    def score(self, X: MatrixLike, y: ArrayLike) -> float: ...


def _custom_cross_val_score(estimator, X, y, cv, scoring, fit_params):
    """
    An alternative to sklearn.model_selection.cross_val_score that allows
    passing fit_params, including eval_set, where eval_set is dynamically
    created from the validation fold.
    """
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    scorer = check_scoring(estimator, scoring=scoring)

    # Ensure X and y are numpy arrays for consistent indexing, if they are pandas objects
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y

    # Extract sample_weight if present in fit_params, as it needs special handling
    original_sample_weight = fit_params.pop('sample_weight', None)

    for train_idx, val_idx in cv_splitter.split(X_array, y_array):
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]

        current_fit_params = fit_params.copy() # Copy the remaining fit_params
        
        # Handle sample_weight for the current fold
        fold_sample_weight = None
        if original_sample_weight is not None:
            fold_sample_weight = original_sample_weight[train_idx]

        if 'eval_set' in current_fit_params:
            # Assuming eval_set is expected as a list of (X, y) tuples
            # This replaces the placeholder eval_set with the actual validation set
            current_fit_params['eval_set'] = [(X_val, y_val)]

        # Create a new estimator instance for each fold to avoid data leakage
        # and ensure parameters are reset.
        fold_estimator = estimator.__class__(**estimator.get_params())
        
        # Pass sample_weight explicitly if it exists, otherwise pass other fit_params
        if fold_sample_weight is not None:
            fold_estimator.fit(X_train, y_train, sample_weight=fold_sample_weight, **current_fit_params)
        else:
            fold_estimator.fit(X_train, y_train, **current_fit_params)
            
        score = scorer(fold_estimator, X_val, y_val)
        scores.append(score)
    return np.array(scores)


@dataclass
class StepwiseHyperoptOptimizer(BaseEstimator, MetaEstimatorMixin):
    model: _Fitable
    param_space_sequence: list[dict[str, PARAM | SymbolTable]]
    max_evals_per_step: int = 100
    cv: int = 5
    scoring: str | Callable[[ArrayLike, ArrayLike], float] = "neg_mean_squared_error"
    random_state: int = field(default=42, repr=False) # Make random_state not appear in __repr__
    best_params_: dict[str, PARAM] = field(default_factory=dict)
    best_score_: float = None
    # New field to specify which parameters should be integers
    int_params: list[str] = field(default_factory=list)
    debug: bool = False
    _fit_params: dict = field(default_factory=dict) # To store fit_params passed to .fit()

    def _flatten_params(self, params: dict) -> dict:
        """
        Flattens a nested dictionary of parameters, handling cases where
        hp.choice selects a dictionary.
        """
        flattened = {}
        for key, value in params.items():
            if isinstance(value, dict):
                # If the value is a dictionary (e.g., from hp.choice selecting a dict)
                # then merge its contents into the flattened dictionary.
                flattened.update(self._flatten_params(value))
            else:
                flattened[key] = value
        return flattened

    def _filter_catboost_params(self, params: dict) -> dict:
        """
        Filters CatBoost-specific parameters based on conditional logic.
        This function assumes params is already flattened.
        """
        #filtered_params = params.copy()

        conflicting_keys = [
            # (key,value): remove_key
            ('grow_policy', 'Lossguide', 'max_leaves'),  # max_leaves is only valid for Lossguide
            ('od_type', 'IncToDec', 'od_pval'),  # od_pval is only valid for IncToDec
            ('bootstrap_type', 'Bayesian', 'bagging_temperature'),  # bagging_temperature is only valid for Bayesian bootstrap
            ('bootstrap_type', 'Subsample', 'subsample'),  # subsample is not valid for Bayesian bootstrap
            ('bootstrap_type', 'Bayesian', 'subsample'),  # subsample is not valid for Bayesian bootstrap
            ('bootstrap_type', 'Bayesian', 'bagging_temperature')  # bagging_temperature is only valid for Bayesian bootstrap
        ]

        for k, v, remove in conflicting_keys:
            if params.get(k) == v and remove in params:
                if self.debug:
                    print(f'debug: Removing {remove} because {k} is {v}')
                # If the key is in conflicting_params and exists in params, remove it
                del params[remove]
        if self.debug:
            print(f'debug cb: {params=}')
        return params

    def clean_int_params(self, params: dict[str, PARAM]) -> dict[str, PARAM]:
        # Use the instance's int_params list
        return {k: int(v) if k in self.int_params else v for k, v in params.items()}

    def objective(self, params: dict[str, PARAM]) -> float:
        # Flatten the parameters first
        flattened_params = self._flatten_params(params)
        
        # Combine best_params_ with current trial's flattened params
        # Apply filtering to the combined set of parameters
        combined_params = {**self.best_params_, **flattened_params}
        
        # Filter CatBoost-specific conditional parameters
        filtered_params = self._filter_catboost_params(combined_params)

        # Clean integer parameters
        cleaned_params = self.clean_int_params(filtered_params)
        
        if self.debug:
            print(f'debug: {cleaned_params=}')

        # clear out previous parameters - otherwise models like CatBoost will
        # complain when we set a conflicting parameter
        self.model = self.model.__class__()
        self.model.set_params(**cleaned_params)
        
        # Use the custom cross_val_score that handles fit_params
        score = _custom_cross_val_score(
            self.model, self.X, self.y, cv=self.cv, scoring=self.scoring, fit_params=self._fit_params.copy() # Pass a copy to avoid modifying original
        )
        return -np.mean(score)

    def fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs) -> Self:
        self.X = X
        self.y = y
        # Store fit_params for use in the objective function
        # Convert args to kwargs if necessary, though typically fit_params are kwargs
        self._fit_params = kwargs 

        for step, param_space in enumerate(self.param_space_sequence):
            print(f"Optimizing step {step + 1}/{len(self.param_space_sequence)}")
            trials = Trials()
            best = fmin(
                fn=self.objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=self.max_evals_per_step,
                trials=trials,
                rstate=np.random.default_rng(self.random_state) # Use default_rng for modern numpy random state
            )

            step_best_params = space_eval(param_space, best)
            
            # Flatten the step_best_params
            flattened_step_best_params = self._flatten_params(step_best_params)
            
            # Filter CatBoost-specific conditional parameters for the best_params_
            # This filtering is crucial before updating self.best_params_
            filtered_step_best_params = self._filter_catboost_params(flattened_step_best_params)

            # Clean integer parameters
            cleaned_step_best_params = self.clean_int_params(filtered_step_best_params)
            
            self.best_params_.update(cleaned_step_best_params)
            self.best_score_ = -min(trials.losses())

            print(f"Best parameters after step {step + 1}: {self.best_params_}")
            print(f"Best score after step {step + 1}: {self.best_score_}")

        if self.debug:
            print(f'{kwargs=}')
        # Fit the model with the best parameters on the full dataset
        # Ensure final best_params_ are also filtered before setting them on the model
        final_params_for_model = self._filter_catboost_params(self.best_params_)
        self.model.set_params(**final_params_for_model)
        self.model.fit(X, y, *args, **kwargs) # Pass original args/kwargs for final fit

        return self

    def predict(self, X: pd.DataFrame) -> ArrayLike:
        return self.model.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.model.score(X, y)

