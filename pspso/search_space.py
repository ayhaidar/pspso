"""Search-space primitives used by the modern PSPSO optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Choice:
    """A categorical hyperparameter."""

    values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]):
        if not values:
            raise ValueError("Choice requires at least one value.")
        object.__setattr__(self, "values", tuple(values))

    @property
    def bounds(self) -> tuple[float, float]:
        return 0.0, float(len(self.values) - 1)

    def decode(self, value: float) -> Any:
        index = int(round(float(value)))
        index = min(max(index, 0), len(self.values) - 1)
        return self.values[index]

    def encode(self, value: Any) -> float:
        try:
            return float(self.values.index(value))
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a valid choice: {self.values!r}") from exc

    def grid_values(self) -> list[float]:
        return [float(i) for i in range(len(self.values))]


@dataclass(frozen=True)
class FloatRange:
    """A continuous numeric hyperparameter rounded to a fixed precision."""

    low: float
    high: float
    precision: int = 2

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("FloatRange high must be greater than or equal to low.")
        if self.precision < 0:
            raise ValueError("FloatRange precision must be non-negative.")

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.low), float(self.high)

    def decode(self, value: float) -> float:
        value = min(max(float(value), self.low), self.high)
        return round(value, self.precision)

    def encode(self, value: float) -> float:
        return self.decode(float(value))

    def grid_values(self) -> list[float]:
        step = 10 ** (-self.precision)
        values = np.arange(self.low, self.high + step / 2, step)
        return [round(float(v), self.precision) for v in values]


@dataclass(frozen=True)
class IntRange:
    """An integer hyperparameter."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("IntRange high must be greater than or equal to low.")

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.low), float(self.high)

    def decode(self, value: float) -> int:
        value = int(round(float(value)))
        return min(max(value, self.low), self.high)

    def encode(self, value: int) -> float:
        return float(self.decode(value))

    def grid_values(self) -> list[float]:
        return [float(v) for v in range(self.low, self.high + 1)]


ParameterSpec = Choice | FloatRange | IntRange


class SearchSpace:
    """Ordered collection of tunable hyperparameters."""

    def __init__(self, params: Mapping[str, ParameterSpec]):
        if not params:
            raise ValueError("SearchSpace requires at least one parameter.")
        self.params = dict(params)
        self.names = list(self.params)

    @classmethod
    def from_legacy(cls, params: Mapping[str, Sequence[Any]]) -> "SearchSpace":
        """Build a search space from the original pspso dict format."""

        converted: dict[str, ParameterSpec] = {}
        for name, values in params.items():
            value_list = list(values)
            if not value_list:
                raise ValueError(f"Parameter {name!r} has an empty value list.")
            if all(isinstance(value, str) for value in value_list):
                converted[name] = Choice(value_list)
                continue
            if len(value_list) != 3:
                raise ValueError(
                    f"Numeric parameter {name!r} must be [low, high, precision]."
                )
            low, high, precision = value_list
            precision = int(precision)
            if precision == 0:
                converted[name] = IntRange(int(round(float(low))), int(round(float(high))))
            else:
                converted[name] = FloatRange(float(low), float(high), precision)
        return cls(converted)

    @property
    def dimensions(self) -> int:
        return len(self.names)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows, highs = zip(*(self.params[name].bounds for name in self.names), strict=True)
        return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)

    def decode(self, particle: Sequence[float]) -> dict[str, Any]:
        if len(particle) != self.dimensions:
            raise ValueError(
                f"Particle has {len(particle)} dimensions, expected {self.dimensions}."
            )
        return {
            name: self.params[name].decode(value)
            for name, value in zip(self.names, particle, strict=True)
        }

    def encode(self, params: Mapping[str, Any]) -> list[float]:
        return [self.params[name].encode(params[name]) for name in self.names]

    def iter_grid(self) -> Iterator[dict[str, Any]]:
        for encoded in self.iter_grid_encoded():
            yield self.decode(encoded)

    def iter_grid_encoded(self) -> Iterator[list[float]]:
        values = [self.params[name].grid_values() for name in self.names]
        for combo in product(*values):
            yield [float(value) for value in combo]

    def random_encoded(self, rng: np.random.Generator) -> list[float]:
        values: list[float] = []
        for name in self.names:
            spec = self.params[name]
            low, high = spec.bounds
            if isinstance(spec, Choice):
                values.append(float(rng.integers(int(low), int(high) + 1)))
            elif isinstance(spec, IntRange):
                values.append(float(rng.integers(spec.low, spec.high + 1)))
            else:
                values.append(float(rng.uniform(low, high)))
        return values

    def random_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.decode(self.random_encoded(rng))

    def to_legacy(self) -> dict[str, list[Any]]:
        legacy: dict[str, list[Any]] = {}
        for name, spec in self.params.items():
            if isinstance(spec, Choice):
                legacy[name] = list(spec.values)
            elif isinstance(spec, IntRange):
                legacy[name] = [spec.low, spec.high, 0]
            else:
                legacy[name] = [spec.low, spec.high, spec.precision]
        return legacy


def coerce_search_space(value: SearchSpace | Mapping[str, Any]) -> SearchSpace:
    """Accept modern or legacy search-space declarations."""

    if isinstance(value, SearchSpace):
        return value
    if all(isinstance(spec, (Choice, FloatRange, IntRange)) for spec in value.values()):
        return SearchSpace(value)  # type: ignore[arg-type]
    return SearchSpace.from_legacy(value)  # type: ignore[arg-type]


def count_grid(space: SearchSpace, limit: int = 1_000_000) -> int:
    """Count grid combinations with a guard for very large spaces."""

    total = 1
    for name in space.names:
        total *= len(space.params[name].grid_values())
        if total > limit:
            return total
    return total


__all__ = [
    "Choice",
    "FloatRange",
    "IntRange",
    "ParameterSpec",
    "SearchSpace",
    "coerce_search_space",
    "count_grid",
]
