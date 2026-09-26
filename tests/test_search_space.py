import pytest

from pspso import Choice, FloatRange, IntRange, LogFloatRange, SearchSpace


def test_typed_search_space_decodes_encodes_and_serializes():
    space = SearchSpace(
        {
            "kernel": Choice(["linear", "rbf"]),
            "C": FloatRange(0.1, 1.0, 1),
            "degree": IntRange(1, 3),
        }
    )

    decoded = space.decode([1.2, 0.26, 2.8])

    assert decoded == {"kernel": "rbf", "C": 0.3, "degree": 3}
    assert space.encode(decoded) == [1.0, 0.3, 3.0]
    assert space.to_schema()["degree"] == {"type": "int", "low": 1, "high": 3}


def test_search_space_grid_values_are_ordered():
    space = SearchSpace(
        {
            "a": Choice(["x", "y"]),
            "b": IntRange(1, 2),
            "c": FloatRange(0.1, 0.2, precision=1),
        }
    )

    grid = list(space.iter_grid())

    assert len(grid) == 8
    assert space.grid_size == 8
    assert grid[0] == {"a": "x", "b": 1, "c": 0.1}
    assert grid[-1] == {"a": "y", "b": 2, "c": 0.2}


def test_search_space_rejects_untyped_values():
    with pytest.raises(TypeError, match="SearchSpace values must be"):
        SearchSpace({"kernel": ["linear", "rbf"]})  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        Choice([])


def test_log_float_range_encodes_in_log_space():
    parameter = LogFloatRange(0.001, 1.0, precision=4)
    assert parameter.decode(parameter.encode(0.01)) == 0.01
