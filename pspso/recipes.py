"""Registration API for trusted, locally installed model recipes."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .search_space import ParameterSpec, SearchSpace


@dataclass(frozen=True)
class Recipe:
    name: str
    factory: Callable[..., Any]
    tasks: tuple[str, ...]
    description: str
    fixed_params: Mapping[str, Mapping[str, Any]]
    search_spaces: Mapping[str, SearchSpace]


_RECIPES: dict[str, Recipe] = {}
_PLUGINS_LOADED = False


def register_recipe(
    name: str,
    factory: Callable[..., Any],
    *,
    tasks: list[str],
    description: str,
    fixed_params: Mapping[str, Mapping[str, Any]] | None = None,
    search_spaces: Mapping[str, SearchSpace | Mapping[str, ParameterSpec]] | None = None,
) -> None:
    """Register a trusted recipe with typed task-specific search spaces."""

    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "Recipe names may contain only letters, numbers, hyphens, and underscores."
        )
    _RECIPES[name.lower()] = Recipe(
        name=name.lower(),
        factory=factory,
        tasks=tuple(tasks),
        description=description,
        fixed_params=fixed_params or {},
        search_spaces={
            task: space if isinstance(space, SearchSpace) else SearchSpace(space)
            for task, space in (search_spaces or {}).items()
        },
    )


def get_recipe(name: str) -> Recipe | None:
    load_recipe_plugins()
    return _RECIPES.get(name.lower())


def recipe_metadata() -> dict[str, dict[str, Any]]:
    load_recipe_plugins()
    return {
        recipe.name: {
            "label": recipe.name.replace("_", " ").title(),
            "tasks": list(recipe.tasks),
            "optional_dependency": None,
            "description": recipe.description,
            "plugin": True,
        }
        for recipe in _RECIPES.values()
    }


def load_recipe_plugins() -> None:
    """Load explicitly configured local plugins once; never accept browser code."""

    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for module_name in filter(None, os.environ.get("PSPSO_RECIPE_PLUGINS", "").split(",")):
        importlib.import_module(module_name.strip())
