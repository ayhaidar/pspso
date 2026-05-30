import pytest

from pspso import Choice, FloatRange, IntRange, SearchSpace


def test_legacy_search_space_decodes_and_encodes():
    space = SearchSpace.from_legacy(
        {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 1],
            "degree": [1, 3, 0],
        }
    )

    decoded = space.decode([1.2, 0.26, 2.8])

    assert decoded == {"kernel": "rbf", "C": 0.3, "degree": 3}
    assert space.encode(decoded) == [1.0, 0.3, 3.0]


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
    assert grid[0] == {"a": "x", "b": 1, "c": 0.1}
    assert grid[-1] == {"a": "y", "b": 2, "c": 0.2}


def test_invalid_search_space_rejects_empty_choices():
    with pytest.raises(ValueError):
        SearchSpace.from_legacy({"kernel": []})
