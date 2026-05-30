# Tests and Results

This page answers two practical questions:

1. What has been checked in the codebase?
2. What passed in the latest verification run?

## What We Check

The current quality checks cover four layers of the application:

| Layer | What is checked |
| --- | --- |
| Core optimizer | Search-space decoding, optimizer progress events, generic estimator support, and legacy compatibility. |
| Dashboard API | Validation behavior, run creation, persisted history, experiment tracking, artifacts export, and run lifecycle status. |
| Option matrix | Built-in estimator and task combinations validate correctly when their optional dependency is installed. |
| Frontend build | TypeScript compilation and production bundling succeed for the dashboard. |

## Current Automated Test Files

| Test file | Purpose |
| --- | --- |
| `tests/test_search_space.py` | Search-space validation and encoding behavior. |
| `tests/test_optimizer.py` | Optimizer execution and progress event emission. |
| `tests/test_legacy.py` | Legacy `from pspso import pspso` compatibility. |
| `tests/test_api.py` | FastAPI validation, persisted run history, experiment tracking, and failure handling. |
| `tests/test_scenarios.py` | Scenario JSON validation and a lightweight end-to-end dashboard run. |
| `tests/test_option_matrix.py` | Estimator/task compatibility sweep across built-in presets. |

## Latest Verification Results

Latest verification run recorded during the current documentation update:

| Command | Result |
| --- | --- |
| `python -m pytest` | `18 passed, 1 skipped` |
| `python -m pytest tests/test_api.py tests/test_optimizer.py` | `10 passed, 1 skipped` |
| `npm run build` in `frontend/` | Passed |
| `uv run --isolated --with mkdocs-material --with mkdocs-mermaid2-plugin --with "mkdocstrings[python]" mkdocs build` | Passed |

The single skipped case is the XGBoost-specific validation test when the optional
`xgboost` package is not available in the environment. In this working setup,
XGBoost is installed and available to the dashboard.

## Option Coverage Notes

The codebase now performs a compatibility sweep over built-in estimator presets:

- `svm`
- `random_forest`
- `mlp`
- `logistic_regression`
- `linear_regression`
- `xgboost` when installed
- `gbdt` when installed

For each supported task, the API validation layer is checked against a realistic
example dataset:

- `breast_cancer` for binary classification
- `diabetes` for regression

This does not prove that every possible hyperparameter combination is safe, but
it does prove that the shipped defaults, validation rules, and dashboard
metadata are internally consistent.

## What The Results Mean

When these checks are green, we know the following:

- the dashboard can validate and start runs;
- persisted run history can be saved and reloaded;
- the optimizer can emit the events that the frontend expects;
- the built-in estimator presets are not obviously mismatched to their tasks;
- the frontend still compiles after backend and state-flow changes.

## Remaining Risk

The most important remaining risk is not wiring, but model-specific runtime
behavior. Optional backends such as XGBoost and LightGBM can still fail for
data-dependent reasons if the chosen objective or parameter values are not valid
for the dataset. The dashboard now catches common mistakes earlier, but some
estimator-specific failures remain part of normal experiment work.
