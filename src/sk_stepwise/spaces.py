from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any
import warnings


@dataclass(frozen=True)
class Categorical:
    choices: Sequence[Any]

    def __post_init__(self) -> None:
        if self.choices and all(
            isinstance(choice, Real) and not isinstance(choice, bool)
            for choice in self.choices
        ):
            warnings.warn(
                "Categorical received only numeric choices. Prefer Int or Float for ordered numeric values so the optimizer can exploit proximity.",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False


SearchDimension = Categorical | Int | Float
