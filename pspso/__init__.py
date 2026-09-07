"""Modern public package interface for PSPSO 1.0."""

from importlib.metadata import version

from .api import TrackingConfig, optimize
from .config import EstimatorConfig, OptimizationConfig, OptimizationResult, TrialResult
from .ensemble import CrossValidationEnsemble
from .estimators import get_estimator_info, list_estimators
from .optimizer import PSPSOOptimizer
from .recipes import register_recipe
from .search_space import Choice, FloatRange, IntRange, LogFloatRange, SearchSpace

__version__ = version("pspso")

__all__ = [
    "Choice",
    "CrossValidationEnsemble",
    "EstimatorConfig",
    "FloatRange",
    "IntRange",
    "LogFloatRange",
    "OptimizationConfig",
    "OptimizationResult",
    "PSPSOOptimizer",
    "SearchSpace",
    "TrialResult",
    "TrackingConfig",
    "get_estimator_info",
    "list_estimators",
    "optimize",
    "register_recipe",
    "__version__",
]
