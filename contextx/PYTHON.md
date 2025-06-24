# Best Practices for Modern Python Development

## Directive 0

Always ask me if you have any questions or concerns. I will help guide you.


This document outlines best practices for Python development, focusing on tools like `uv`, `pytest`, and `ruff`, and advocating for Test-Driven Development (TDD).

## 1. Environment Management with `uv`

`uv` is a fast Python package installer and resolver, designed to be a drop-in replacement for `pip` and `pip-tools`.

### Installation


pip install uv




### Usage



- **Creating a virtual environment:**

  ```bash

  uv venv


This creates a .venv directory in your project root.

 • Activating the virtual environment:


   source .venv/bin/activate

   (On Windows, use .venv\Scripts\activate)
 • Installing dependencies from pyproject.toml:
   uv automatically reads pyproject.toml for dependencies.


   uv sync

   This will install all dependencies listed in your pyproject.toml (under [project].dependencies and [project].optional-dependencies).
 • Adding new dependencies:


   uv add <package-name>

 • Upgrading dependencies:


   uv update <package-name>

   Or to update all:


   uv update

 • Running commands within the virtual environment:


   uv run python your_script.py

   This is useful for CI/CD or when you don't want to explicitly activate the environment.


2. Linting and Formatting with ruff

ruff is an extremely fast Python linter and formatter, designed to replace tools like Flake8, isort, Black, and pylint.

Installation

ruff should be added as a development dependency in your pyproject.toml.


[project.optional-dependencies]

dev = [

    "ruff",

    "pytest",

    # ... other dev dependencies

]


Then install it:


uv sync --dev


Configuration

Configure ruff in your pyproject.toml under the [tool.ruff] table.


[tool.ruff]

line-length = 88

target-version = "py310"

select = [

    "E",  # pycodestyle errors

    "W",  # pycodestyle warnings

    "F",  # Pyflakes

    "I",  # isort

    "N",  # pep8-naming

    "D",  # pydocstyle (optional, enable if you enforce docstrings)

    "UP", # pyupgrade

    "B",  # bugbear

    "A",  # flake8-builtins

    "C4", # flake8-comprehensions

    "SIM",# flake8-simplify

    "TID",# flake8-tidy-imports

    "RUF",# ruff-specific rules

]

ignore = [

    "D100", # Missing docstring in public module

    "D104", # Missing docstring in public package

    "D105", # Missing docstring in public method

    "D107", # Missing docstring in __init__

]



[tool.ruff.per-file-ignores]

"__init__.py" = ["F401"] # Ignore unused imports in __init__.py

"tests/*" = ["D"] # Disable docstring checks for test files


Usage

 • Linting:


   uv run ruff check .

 • Fixing auto-fixable issues:


   uv run ruff check . --fix

 • Formatting (replaces Black):


   uv run ruff format .


It's highly recommended to integrate ruff into your pre-commit hooks (see .pre-commit-config.yaml).


3. Testing with pytest

pytest is a powerful and easy-to-use testing framework for Python.

Installation

pytest should be added as a development dependency in your pyproject.toml.


[project.optional-dependencies]

dev = [

    "pytest",

    # ... other dev dependencies

]


Then install it:


uv sync --dev


Configuration

Configure pytest in your pyproject.toml under the [tool.pytest.ini_options] table.


[tool.pytest.ini_options]

minversion = "6.0"

addopts = "--strict-markers --strict-paths"

testpaths = [

    "tests",

]


Usage

 • Running all tests:


   uv run pytest

 • Running specific tests:


   uv run pytest tests/test_basic.py

   uv run pytest tests/test_basic.py::test_initialization

 • Running tests with a keyword expression:


   uv run pytest -k "initialization or matt"

 • Running tests marked with a specific marker:


   uv run pytest -m matt

 • Showing print statements during tests:


   uv run pytest -s

 • Verbose output:


   uv run pytest -v

 • Coverage reporting (requires pytest-cov):
   Install pytest-cov: uv add pytest-cov --dev


   uv run pytest --cov=src/sk_stepwise --cov-report=term-missing



4. Test-Driven Development (TDD) Workflow

TDD is a software development process where tests are written before the code they are meant to test. This approach leads to more robust, maintainable, and well-designed code.

The TDD Cycle (Red-Green-Refactor)

 1 Red: Write a failing test.
    • Before writing any new feature code, write a test that describes a small piece of the desired functionality.
    • This test should fail when run against the current codebase (because the feature doesn't exist yet).
    • Ensure the test fails for the right reason (e.g., AttributeError, AssertionError, not a syntax error).
   Example: If you're adding a calculate_area method to a Shape class, your first test might be:


   # tests/test_shape.py

   from my_module import Shape



   def test_circle_area():

       circle = Shape(radius=5)

       assert circle.calculate_area() == 78.5398

   Running uv run pytest at this point should show this test failing.
 2 Green: Write just enough code to make the test pass.
    • Implement the minimal amount of code necessary to make the failing test pass.
    • Do not worry about perfect design, refactoring, or edge cases at this stage. The goal is simply to get the test to pass.
   Example (after writing the test above):


   # src/my_module.py

   import math



   class Shape:

       def __init__(self, radius):

           self.radius = radius



       def calculate_area(self):

           return math.pi * (self.radius ** 2)

   Running uv run pytest now should show the test_circle_area test passing.
 3 Refactor: Improve the code.
    • Once the test passes, refactor your code. This means improving its design, readability, and maintainability without changing its external behavior (i.e., without breaking any existing
      tests).
    • Look for opportunities to:
       • Remove duplication.
       • Improve naming.
       • Simplify complex logic.
       • Extract methods/functions.
       • Improve error handling (if applicable, and then write tests for it).
    • After each small refactoring step, re-run all tests to ensure you haven't introduced any regressions.
   Example (refactoring might involve adding more shapes, or making the Shape class more abstract, but always re-running tests):


   # src/my_module.py (after refactoring)

   import math

   from abc import ABC, abstractmethod



   class Shape(ABC):

       @abstractmethod

       def calculate_area(self) -> float:

           pass



   class Circle(Shape):

       def __init__(self, radius: float):

           self.radius = radius



       def calculate_area(self) -> float:

           return math.pi * (self.radius ** 2)

   And then updating the test:


   # tests/test_shape.py

   from my_module import Circle



   def test_circle_area():

       circle = Circle(radius=5)

       assert circle.calculate_area() == 78.53981633974483 # Use full precision for comparison

   (Note: For floating-point comparisons, consider pytest.approx or a small tolerance.)

Benefits of TDD

 • Improved Design: Forces you to think about the API and design of your code from the consumer's perspective (the test).
 • Reduced Bugs: Catches bugs early in the development cycle.
 • Executable Documentation: Tests serve as living documentation of how the code is supposed to behave.
 • Confidence in Refactoring: Knowing you have a comprehensive test suite allows you to refactor with confidence.
 • Faster Feedback Loop: Get immediate feedback on your changes.

Tips for TDD

 • Start Small: Write tests for the smallest possible piece of functionality.
 • One Failure at a Time: Aim for one failing test at a time. Don't write a large suite of failing tests before implementing any code.
 • Test Edge Cases: Once the basic functionality is covered, write tests for edge cases, error conditions, and boundary values.
 • Use Mocks/Fixtures: For complex dependencies, use pytest fixtures or mocking libraries (like unittest.mock) to isolate the unit under test.
 • Commit Often: Commit your changes frequently, especially after completing a Red-Green-Refactor cycle.
