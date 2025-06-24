# Python Best Practices for Professional Development

This document outlines best practices for Python development, focusing on tools and techniques that enhance code quality, maintainability, and collaboration.

## 1. Dependency Management with `uv`

`uv` is a fast and modern Python package installer and resolver. It's recommended for managing project dependencies efficiently.

-   **Use `uv` for all dependency operations**:
    -   Install dependencies: `uv pip install -r requirements.txt`
    -   Add new dependencies: `uv pip install <package_name>`
    -   Upgrade dependencies: `uv pip install --upgrade <package_name>`
    -   Sync dependencies with `pyproject.toml`: `uv sync`
-   **Pin dependencies**: Always use exact versions for production dependencies to ensure reproducibility. `uv` helps with this by default when using `uv pip compile`.
-   **Separate development dependencies**: Use a `requirements-dev.txt` or specify `[tool.uv.dev-dependencies]` in `pyproject.toml` for tools like `pytest`, `ruff`, `mypy`, etc.

## 2. Type Hinting

Type hints improve code readability, enable static analysis, and reduce runtime errors.

-   **Annotate all function signatures**: Include type hints for function arguments and return values.
    ```python
    def calculate_area(length: float, width: float) -> float:
        return length * width
    ```
-   **Use `TypeAlias` for complex types**: For clarity and reusability.
    ```python
    from typing import TypeAlias

    Vector: TypeAlias = list[float]
    Matrix: TypeAlias = list[Vector]

    def dot_product(v1: Vector, v2: Vector) -> float:
        return sum(x * y for x, y in zip(v1, v2))
    ```
-   **Leverage `typing` module**: Utilize types like `Optional`, `Union`, `Any`, `Callable`, `Protocol`, etc., as needed.
-   **Run a static type checker**: Integrate `mypy` or `pyright` into your CI/CD pipeline or pre-commit hooks to enforce type correctness.

## 3. Testing with `pytest`

`pytest` is a powerful and flexible testing framework that encourages writing simple, readable tests.

-   **Write comprehensive tests**: Aim for high test coverage, including unit, integration, and (where applicable) end-to-end tests.
-   **Organize tests**: Place tests in a `tests/` directory, mirroring your source code structure.
-   **Use fixtures**: Leverage `pytest` fixtures for setting up test environments, data, or mock objects.
    ```python
    import pytest

    @pytest.fixture
    def sample_data():
        return [1, 2, 3, 4, 5]

    def test_sum_data(sample_data):
        assert sum(sample_data) == 15
    ```
-   **Parametrize tests**: Use `@pytest.mark.parametrize` to test multiple inputs with a single test function.
    ```python
    import pytest

    @pytest.mark.parametrize("input, expected", [
        (1, 2),
        (2, 3),
        (3, 4),
    ])
    def test_increment(input, expected):
        assert input + 1 == expected
    ```
-   **Mock external dependencies**: Use `unittest.mock` or `pytest-mock` to isolate units of code under test from external systems (databases, APIs, etc.).
-   **Integrate with CI/CD**: Ensure tests run automatically on every push or pull request.

## 4. Logging

Effective logging is crucial for debugging, monitoring, and understanding application behavior in production.

-   **Use the `logging` module**: Avoid `print()` statements for anything beyond simple debugging during development.
-   **Configure logging centrally**: Set up a root logger or specific loggers for different modules.
    ```python
    import logging

    # Basic configuration (for development/simple scripts)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # For more complex applications, configure handlers and formatters
    # logger = logging.getLogger(__name__)
    # handler = logging.FileHandler('app.log')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.setLevel(logging.INFO)
    ```
-   **Use appropriate log levels**:
    -   `DEBUG`: Detailed information, typically only of interest when diagnosing problems.
    -   `INFO`: Confirmation that things are working as expected.
    -   `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., ‘disk space low’). The software is still working as expected.
    -   `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
    -   `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running.
-   **Include context in log messages**: Provide enough information (e.g., user ID, request ID, relevant variable values) to understand the context of the log entry.
-   **Avoid logging sensitive information**: Be careful not to log passwords, API keys, or other sensitive data.
-   **Consider structured logging**: For easier parsing and analysis by log management systems (e.g., JSON format).

By adhering to these best practices, you can significantly improve the quality, reliability, and maintainability of your Python projects.
