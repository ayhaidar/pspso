# pspso

`pspso` is a Python package for hyperparameter optimization with particle
swarm optimization, grid search, and random search. The project now has a
modern optimizer API, a backward-compatible legacy API, and a FastAPI + React
dashboard for configuring and monitoring runs.

The original citation remains:

> Haidar A, Field M, Sykes J, Carolan M, Holloway L. PSPSO: A package for
> parameters selection using particle swarm optimization. SoftwareX. 2021;
> 15:100706.

## Install

The recommended development workflow uses `uv`:

```bash
uv sync --extra api --extra docs --group dev
uv run pytest
uv run mkdocs serve
```

Run the backend dashboard API:

```bash
uv run pspso-dashboard
```

Run the React dashboard during frontend development:

```bash
cd frontend
npm install
npm run dev
```

Pip remains supported:

```bash
pip install -e .
pip install -e ".[api,docs]"
```

Optional estimator backends are installed only when needed:

```bash
uv sync --extra xgboost
uv sync --extra lightgbm
uv sync --extra tensorflow
```

## Modern Python API

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pspso import Choice, FloatRange, OptimizationConfig, PSPSOOptimizer, SearchSpace

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

space = SearchSpace(
    {
        "kernel": Choice(["linear", "rbf"]),
        "C": FloatRange(0.1, 2.0, precision=1),
        "gamma": FloatRange(0.1, 1.0, precision=1),
    }
)

optimizer = PSPSOOptimizer(
    estimator="svm",
    search_space=space,
    config=OptimizationConfig(
        task="binary classification",
        metric="roc_auc",
        strategy="random",
        max_trials=6,
        random_state=42,
    ),
)

result = optimizer.optimize(X_train, y_train, X_val, y_val)
print(result.best_params)
print(result.best_metric)
```

## Legacy API

Existing code using `from pspso import pspso` continues to work:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pspso import pspso

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

params = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1.0, 1],
    "gamma": [0.1, 1.0, 1],
}

p = pspso(estimator="svm", params=params, task="regression", score="rmse")
pos, cost, duration, model, combinations, results = p.fitpsgrid(
    X_train, y_train, X_val, y_val
)
p.print_results()
```

The wrapper delegates to the modern optimizer internally, while preserving old
method names and tuple return shapes.

## Dashboard API

Start the API:

```bash
uv run pspso-dashboard --host 127.0.0.1 --port 8000
```

Important endpoints:

- `GET /api/datasets/examples`
- `POST /api/datasets/preview`
- `GET /api/estimators`
- `POST /api/runs/validate`
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- `GET /api/runs/{run_id}/result`

The event stream emits:

- `run_started`
- `trial_started`
- `trial_completed`
- `trial_failed`
- `iteration_completed`
- `best_updated`
- `run_completed`
- `run_failed`

Each trial event includes the evaluated parameters, training metric,
validation metric, duration, status, and any error message.

## MkDocs Material Documentation

The documentation lives in `docs/` as Markdown and is built with MkDocs
Material:

```bash
uv run mkdocs serve
uv run mkdocs build
```

Important pages:

- `docs/dashboard/workflow.md`
- `docs/dashboard/frontend-backend-link.md`
- `docs/dashboard/live-monitoring.md`
- `docs/scenarios.md`
- `docs/troubleshooting.md`

## Dashboard Configuration

The dashboard lets users configure:

- Built-in breast cancer or diabetes datasets, or CSV text upload.
- Target column, task type, metric, split ratio, and random seed.
- Numeric scaling, categorical encoding, and ignored columns.
- Estimator preset, fixed training params, and tunable search-space params.
- PSO, grid, or random strategy.
- PSO particles, iterations, and coefficients.
- Runtime max trials and verbosity.
- Validation before training starts, using estimator/task/search-space metadata
  from `GET /api/estimators`.

Fixed training params stay separate from tunable hyperparameters, so users can
choose exactly what remains constant and what PSO optimizes.

## Supported Estimator Presets

- `svm`
- `random_forest`
- `mlp`
- `xgboost` with the `xgboost` extra
- `gbdt` with the `lightgbm` extra
- `logistic_regression`
- `linear_regression`

The XGBoost defaults use modern objectives such as `reg:squarederror`; deprecated
examples such as `load_boston` and `reg:linear` are no longer used.

## Tests

```bash
uv run pytest
```

The default test suite uses sklearn-only scenarios so it can run without
XGBoost, LightGBM, or TensorFlow installed. Optional backend smoke tests can be
added in CI jobs that install those extras.
