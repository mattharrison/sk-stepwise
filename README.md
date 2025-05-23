# sk-stepwise

## Overview

`StepwiseHyperoptOptimizer` is a custom Python class that combines the power of the Hyperopt optimization library with a stepwise optimization strategy for hyperparameter tuning of machine learning models. It extends the capabilities of scikit-learn's `BaseEstimator` and `MetaEstimatorMixin`, making it easy to integrate into existing machine learning workflows.

This class enables you to optimize a model's hyperparameters in a sequential manner, following a predefined series of hyperparameter spaces. Each step in the sequence focuses on refining a specific set of parameters, allowing for a more targeted and efficient optimization process. The hyperparameter optimization uses Tree of Parzen Estimators (TPE) through the Hyperopt library.

## Features

- **Stepwise Hyperparameter Tuning**: Break down the optimization process into multiple steps, each refining a specific set of hyperparameters.
- **Hyperopt Integration**: Utilize Hyperopt's TPE algorithm to find the optimal parameters efficiently.
- **Scikit-learn Compatibility**: `StepwiseHyperoptOptimizer` is compatible with the scikit-learn ecosystem, making it easy to use in scikit-learn pipelines and workflows.
- **Flexible Scoring**: Supports both default scikit-learn scoring metrics and custom scoring functions.

## Installation

```sh
pip install sk-stepwise
```

If you are planning on developing this package, you should install
the precommit hooks and other development dependencies.

```sh
pip install -r requirements-dev.txt
```

You should also install the package in editable mode:

```sh
uv pip install -e .
```

Run the tests with:

```sh
uv run pytest
```

## Usage

Here's an example of how to use `StepwiseHyperoptOptimizer` to optimize a scikit-learn model:

```python
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sk_stepwise import StepwiseHyperoptOptimizer
>>> import hyperopt

>>> # Sample data
>>> X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
>>> y = pd.Series(np.random.rand(100))

>>> # Define the model
>>> model = RandomForestRegressor()

>>> # Define the parameter space sequence for stepwise optimization
>>> param_space_sequence = [
...     {"n_estimators": hyperopt.hp.choice("n_estimators", [50, 100, 150])},
...     {"max_depth": hyperopt.hp.quniform("max_depth", 3, 10, 1)},
...     {"min_samples_split": hyperopt.hp.uniform("min_samples_split", 0.1, 1.0)},
... ]

>>> # Create the optimizer
>>> optimizer = StepwiseHyperoptOptimizer(model=model, param_space_sequence=param_space_sequence, max_evals_per_step=50)

>>> # Fit the optimizer
>>> optimizer.fit(X, y)

>>> # Make predictions
>>> predictions = optimizer.predict(X)
```

## Key Methods

- `fit(X, y)`: Fits the optimizer to the data, performing stepwise hyperparameter optimization.
- `predict(X)`: Uses the optimized model to make predictions.
- `score(X, y)`: Evaluates the optimized model on a test set.

## Parameters

- **model** (`_Fitable`): A scikit-learn compatible model that implements `fit`, `predict`, and `set_params` methods.
- **param_space_sequence** (`list[dict]`): A list of dictionaries representing the hyperparameter spaces for each optimization step.
- **max_evals_per_step** (`int`): The maximum number of evaluations to perform for each step of the optimization.
- **cv** (`int`): Number of cross-validation folds.
- **scoring** (`str` or `Callable`): The scoring metric to use for evaluation. Default is "neg_mean_squared_error".
- **random_state** (`int`): Random seed for reproducibility.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for new features, bug fixes, or documentation improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hyperopt](https://github.com/hyperopt/hyperopt) for hyperparameter optimization.
- [scikit-learn](https://scikit-learn.org) for model implementation and evaluation utilities.
