def hello() -> str:
    return "Hello from sk-stepwise!"

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, space_eval, Trials


# From typing
import pandas as pd
from typing import Self

PARAM = int | float | str | bool
from typing import TypeAlias, Any, Unpack
from scipy.sparse import spmatrix
import numpy.typing
from numpy import floating
MatrixLike: TypeAlias = np.ndarray | pd.DataFrame | spmatrix
ArrayLike: TypeAlias = numpy.typing.ArrayLike

from typing import Protocol
class _Fitable(Protocol):
    def fit(self, X: MatrixLike, y: ArrayLike) -> Self: 
        ...
    def predict(self, X: MatrixLike) -> ArrayLike:
        ...
    #def set_params(self, **params: dict[str, PARAM]) -> Self:
    #def set_params(self, **params: Unpack[dict[str, PARAM]]) -> Self:
    #def set_params(self, **params: Unpack[PARAM]) -> Self:
    #def set_params(self, **params: **dict) -> Self:    
    def set_params(self, **params: PARAM) -> Self:
        ...
    def score(self, X: MatrixLike, y: ArrayLike) -> float:
        ...

from hyperopt import hp
from collections.abc import Callable
from hyperopt.pyll.base import SymbolTable

class StepwiseHyperoptOptimizer(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, model: _Fitable, 
                 param_space_sequence: list[dict[str, PARAM | SymbolTable]],
                 max_evals_per_step:int=100, 
                 cv:int=5, 
                 scoring:str|Callable[[ArrayLike, ArrayLike], float]='neg_mean_squared_error', 
                 random_state:int=42) -> None:
        self.model = model
        
        self.param_space_sequence = param_space_sequence
        self.max_evals_per_step = max_evals_per_step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_:dict[str, PARAM] = {}
        self.best_score_ = None

    def clean_int_params(self, params:dict[str, PARAM]) -> dict[str, PARAM]:
        int_vals = ['max_depth', 'reg_alpha']
        return {k: int(v) if k in int_vals else v for k, v in params.items()}

    def objective(self, params:dict[str, PARAM]) -> float:
        # I added this
        params = self.clean_int_params(params)
        # END 
        current_params = {**self.best_params_, **params}
        self.model.set_params(**current_params)
        score = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        return -np.mean(score)

    def fit(self, X:pd.DataFrame, y:pd.Series) -> Self:
        self.X = X
        self.y = y
        for step, param_space in enumerate(self.param_space_sequence):
            print(f"Optimizing step {step + 1}/{len(self.param_space_sequence)}")
            
            trials = Trials()
            best = fmin(
                fn=self.objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=self.max_evals_per_step,
                trials=trials,
                #rstate=np.random.RandomState(self.random_state)
            )

            step_best_params = space_eval(param_space, best)
            # I added this
            step_best_params = self.clean_int_params(step_best_params)
            # END 
            self.best_params_.update(step_best_params)
            self.best_score_ = -min(trials.losses())
            
            print(f"Best parameters after step {step + 1}: {self.best_params_}")
            print(f"Best score after step {step + 1}: {self.best_score_}")

        # Fit the model with the best parameters
        self.model.set_params(**self.best_params_)
        self.model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> ArrayLike:
        return self.model.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.model.score(X, y)
