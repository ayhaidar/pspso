# PSPSO 1.0

![PSPSO](https://raw.githubusercontent.com/ayhaidar/pspso/main/docs/assets/pspso-logo.svg)

PSPSO is a modern local framework for reproducible tabular machine-learning
experiments and hyperparameter optimization. Particle swarm optimization is the
primary search engine, with random and grid search available as comparison
baselines.

> [!CAUTION]
> **Major update in progress**
>
> PSPSO 1.0 is a substantial update with a new dashboard, CLI, experiment
> service, data preparation workflow, and reproducible result tracking. The
> version on this repository is still under active development. APIs, defaults,
> dashboard behaviour, and saved experiment formats may change before release.
>
> **Use this development version at your own risk.** Verify important results
> independently and keep copies of valuable datasets and experiment artifacts.
> The updated package will be published to PyPI after final validation; until
> then, the current PyPI package remains the published release.

Version 1.0 provides one typed contract across:

- a notebook-friendly Python API;
- a React dashboard backed by FastAPI;
- a local experiment and artifact store;
- a managed subprocess worker;
- a CLI for datasets, experiments, and runs.

> Haidar A, Field M, Sykes J, Carolan M, Holloway L. PSPSO: A package for
> parameters selection using particle swarm optimization. SoftwareX. 2021;
> 15:100706.

## See PSPSO in Action

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/ayhaidar/pspso@main/docs/assets/media/pspso-overview.mp4">
    <img src="https://raw.githubusercontent.com/ayhaidar/pspso/main/docs/assets/images/pspso-overview.jpg" alt="Watch the PSPSO dashboard, CLI, and model-search overview" width="960">
  </a>
</p>

<p align="center">
  <strong><a href="https://cdn.jsdelivr.net/gh/ayhaidar/pspso@main/docs/assets/media/pspso-overview.mp4">Play the 25-second PSPSO overview</a></strong>
</p>

The overview follows a Breast Cancer classification experiment from local
installation and data setup through five-fold evaluation, model search, and the
saved result. It uses the current development interface; details may change as
the major update progresses.

## Dashboard Snapshots

### Overview

![PSPSO dashboard overview showing the major-update notice and six-stage workflow](https://raw.githubusercontent.com/ayhaidar/pspso/main/docs/assets/screenshots/dashboard-overview.png)

### Data Setup and Evaluation

![PSPSO data setup showing an inspected binary-classification dataset and evaluation controls](https://raw.githubusercontent.com/ayhaidar/pspso/main/docs/assets/screenshots/dashboard-data-setup.png)

### Experiment Results

![PSPSO results showing the saved predictor, selected winner, evaluation split, and analysis tools](https://raw.githubusercontent.com/ayhaidar/pspso/main/docs/assets/screenshots/dashboard-results.png)

## Dashboard Workflow

The dashboard opens with an overview of its capabilities. Data Setup contains
three sections for dataset selection, profile/evaluation, and feature engineering.
Choose cross-validation or train/validation/test, keep chronological order for
time series, and configure missing values, outlier clipping and categorical encoding.

The interface is organized into six stages:

1. **Data Setup** inspects values, types, missingness, duplicates, target
   distribution, descriptive statistics, and split composition.
2. **Model & Parameters** presents compatible models and separates fixed
   training settings from tunable domains.
3. **Search Engine** configures PSO, random, or grid search and calculates the
   expected model-fit budget.
4. **Live Experiments** shows worker activity, particles or candidates, metrics,
   failures, logs, and the persisted event timeline.
5. **Results** provides task-specific diagnostics, predictions, dataset
   summaries, and feature importance where available.
6. **History** combines dashboard, CLI, and tracked notebook runs.

Each stage includes a quick guide, checklist, glossary, and contextual field
help.

## Installation

Python 3.10 through 3.12 is supported. Install PSPSO from PyPI to get the dashboard,
CLI and Python API together. Node.js is only needed for frontend development.

With **uv**, install the dashboard and CLI as a tool:

```bash
uv tool install pspso
```

This installs PSPSO from PyPI and makes `pspso` and `pspso-dashboard`
available as commands. To use the Python API in an existing uv project, add
PSPSO as a project dependency:

```bash
uv add pspso
```

With **pip**, use your current Python environment:

```bash
python -m pip install --upgrade pspso
```

Optional model engines can be included when installing the uv tool. Choose the
extras you need in one command:

```bash
uv tool install "pspso[xgboost]"
uv tool install "pspso[xgboost,lightgbm,torch]"
```

For an existing uv project, use `uv add "pspso[xgboost]"` or combine the
extras in the same way.

With pip:

```bash
python -m pip install --upgrade "pspso[xgboost]"
python -m pip install --upgrade "pspso[lightgbm]"
python -m pip install --upgrade "pspso[torch]"
```

Contributors working from the repository should follow the
[source setup](https://ayhaidar.github.io/pspso/development/setup/).

## Start the Dashboard

After installing with either uv or pip:

```bash
pspso-dashboard
```

Open [the dashboard](http://127.0.0.1:8000). The installed package serves the
interface directly. API documentation is available at
[API v1 docs](http://127.0.0.1:8000/api/v1/docs).
The browser submits managed jobs; it does not open a machine terminal or execute
arbitrary Python.

Dashboard defaults use five-fold CV, per-fold preprocessing, a final refit on all
development rows and one untouched test partition. Exact datasets, partition and
fold indices, models, environments and seeds are saved. Candidate concurrency
is bounded, and progress reports actual fit totals and worker slots.
If the additional refit is disabled, PSPSO saves the winning candidate's fold
models as an ensemble and averages them for predictions.

The long-lived service owns the durable queue. CLI start, cancel and retry use
that same service. Queued work survives a restart; interrupted attempts remain
historical. For foreground execution without HTTP, use
`pspso run start SPEC --standalone --wait`. The compatible `api` extra remains
accepted, but service dependencies are included in the base package.

See [evaluation](https://ayhaidar.github.io/pspso/concepts/evaluation/),
[CLI](https://ayhaidar.github.io/pspso/guides/cli/),
[architecture](https://ayhaidar.github.io/pspso/development/architecture/) and the
[release completion record](https://github.com/ayhaidar/pspso/blob/main/RELEASE_CHECKLIST.md)
for details and verification.

## Notebook API

Use `optimize()` for concise, in-process work:

```python
from sklearn.datasets import load_diabetes
from pspso import IntRange, OptimizationConfig, SearchSpace, optimize

X, y = load_diabetes(return_X_y=True)
space = SearchSpace({
    "n_estimators": IntRange(20, 80),
    "max_depth": IntRange(2, 10),
})
config = OptimizationConfig(
    task="regression",
    metric="rmse",
    strategy="pso",
    n_particles=4,
    n_iterations=3,
    random_state=42,
)
result = optimize(
    X,
    y,
    estimator="random_forest",
    search_space=space,
    config=config,
)

print(result.summary())
display(result.trials_frame())
predictions = result.predict(X[:5])
```

`PSPSOOptimizer` remains available when the optimizer instance itself is needed.
`OptimizationResult` provides the best model and parameters, trials, failures,
duration, prediction, probability, evaluation, tabular, and notebook-display
helpers.

## Optional Notebook Tracking

Notebook runs do not write to disk by default. Supply `TrackingConfig` to record
one in the same experiment history used by the dashboard and CLI:

```python
from pspso import TrackingConfig

result = optimize(
    X,
    y,
    estimator="random_forest",
    search_space=space,
    config=config,
    tracking=TrackingConfig(
        experiment_name="Diabetes study",
        run_name="Notebook PSO",
        tags=("notebook",),
        snapshot_data=True,
    ),
)
```

Tracked notebook calls do not require FastAPI.

## Typed Search Spaces

Version 1.0 accepts explicit domains only:

```python
from pspso import Choice, FloatRange, IntRange, LogFloatRange, SearchSpace

space = SearchSpace({
    "kernel": Choice(["linear", "rbf"]),
    "max_depth": IntRange(2, 12),
    "subsample": FloatRange(0.7, 1.0, precision=2),
    "learning_rate": LogFloatRange(0.001, 0.3, precision=5),
})
```

Use `space.decode()`, `space.encode()`, `space.iter_grid()`, `space.grid_size`,
and `space.to_schema()` for optimizer and API representations.

## Models and Metrics

Built-in canonical model IDs are:

- `linear_regression`, `logistic_regression`, and `elastic_net`;
- `random_forest`, `extra_trees`, and `hist_gradient_boosting`;
- `svm` and `sklearn_mlp`;
- optional `xgboost`, `lightgbm`, and `pytorch_mlp`.

Use `list_estimators()` and `get_estimator_info()` to inspect task support,
dependencies, capabilities, defaults, and typed search spaces.

Canonical tasks are `regression`, `binary_classification`, and
`multiclass_classification`. Supported metrics include RMSE, MAE, R2, accuracy,
ROC AUC, PR AUC, log loss, and macro F1. Classification diagnostics include
sensitivity, specificity, confusion matrices, and threshold analysis.

## Data Preparation

Built-in examples include Breast Cancer, Diabetes, Wine Recognition, Banknote
Authentication, Auto MPG, and Palmer Penguins. Each shows its source, license,
standard task, target, and default metric; selecting it applies those defaults.
The external examples are packaged for offline use. CSV files can also be
inspected directly or saved as fingerprinted local dataset versions.

Numeric and categorical imputation, missingness indicators, categorical
encoding, scaling, ignored columns, stratification, and deterministic split
seeds are configurable. Transformers are fitted on training rows only to avoid
validation and test leakage.

## CLI

The CLI consumes the same `ExperimentSpec` as the dashboard. Save the complete
example from the [CLI guide](https://ayhaidar.github.io/pspso/guides/cli/) as
`experiment.json`, then run
these commands in a second terminal while the dashboard service is running.
If PSPSO is a dependency of an existing uv project instead of an installed tool,
prefix commands with `uv run`.

```bash
pspso --help
pspso run validate experiment.json
pspso run start experiment.json --wait
pspso run list
pspso run show <run_id>
pspso run cancel <run_id>
pspso run retry <run_id>
```

## Experiments and Storage

Version 1.0 uses an isolated workspace:

```text
.pspso/v1/
  tracking.sqlite3
  datasets/
  artifacts/runs/<run_id>/
```

Set `PSPSO_HOME` to relocate the workspace root. Pre-1.0 files under `.pspso/`
are not read, displayed, migrated, or deleted.

SQLite stores experiments, runs, attempts, summaries, and ordered events.
Artifact directories store specifications, results, analysis, environment
details, and serializable models and preprocessors.

## REST API and Monitoring

The supported API is versioned under `/api/v1`. Important endpoints include:

```text
GET  /api/v1/estimators
GET  /api/v1/datasets
POST /api/v1/datasets/inspect
POST /api/v1/workflow/validate
POST /api/v1/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/events
GET  /api/v1/runs/{run_id}/history
GET  /api/v1/runs/{run_id}/analysis
GET  /api/v1/runs/{run_id}/predictions
POST /api/v1/runs/{run_id}/cancel
POST /api/v1/runs/{run_id}/retry
```

OpenAPI is served at `/api/v1/docs` and `/api/v1/openapi.json`. Live progress
uses Server-Sent Events with polling and persisted-history recovery.

## Security

The dashboard is a local application without built-in user accounts or API
authentication. It binds to `127.0.0.1` by default for normal single-user use.
Other bind addresses, including `0.0.0.0`, remain available and produce a clear
warning. Keep the workspace and exported model files trusted: model artifacts
use Python's ML serialization formats and should not be loaded after modification
by an untrusted party. See [SECURITY.md](SECURITY.md) for supported versions and
private vulnerability reporting.

## Tests and Documentation

```bash
uv run pytest
cd frontend && npm run build
uv run mkdocs build --strict
uv build
```

These checks run from a [development checkout](https://ayhaidar.github.io/pspso/development/setup/).
Run the documentation alongside the dashboard with
`uv run --no-sync mkdocs serve --dev-addr 127.0.0.1:8001`. The detailed guides
cover notebooks, the CLI, dashboard workflow, data preparation, search
strategies, model recipes, REST/SSE contracts, tracking, architecture, and the
[0.2 to 1.0 migration](https://ayhaidar.github.io/pspso/migration/1.0/).
