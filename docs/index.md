# pspso

![PSPSO](assets/pspso-logo.svg){ width="420" }

PSPSO 1.0 is a local hyperparameter-optimization and experiment framework for
tabular machine learning. It provides a typed notebook API, PSO, grid and random
search, durable experiment tracking, a CLI, and a FastAPI + React dashboard.

!!! warning "Major update in progress"
    PSPSO 1.0 adds a new dashboard, CLI, experiment service, data preparation
    workflow, and reproducible result tracking. It remains under active
    development until the updated package is published to PyPI.

The current dashboard is designed for local/private experimentation:

- choose a built-in, imported, or versioned CSV dataset;
- select a task, metric, estimator, fixed training parameters, and tunable
  hyperparameters;
- validate the configuration before a run starts;
- stream live training and validation progress from the backend to the
  frontend;
- inspect trials, failures, best parameters, and final results.

Version 1.0 has one modern contract across Python, CLI, REST, workers, and the
dashboard. Start with the notebook helper for direct exploration or use tracked
runs when history and artifacts matter.

## First Commands

Install and start PSPSO from PyPI with:

=== "uv"

    ```bash
    uv tool install pspso
    pspso-dashboard
    ```

=== "pip"

    ```bash
    python -m pip install --upgrade pspso
    pspso-dashboard
    ```

The uv command installs PSPSO as a PyPI tool; it does not create a new project.
Use `uv add pspso` only when importing the Python API from an existing uv
project. Open [the dashboard](http://127.0.0.1:8000), or run `pspso --help`.
The package includes the interface. Contributors can use
[source setup](development/setup.md) to work from the repository.

## Where To Go Next

- [Getting started](getting-started.md) for a full local setup.
- [CLI workflow](guides/cli.md) for a complete command-line experiment.
- [Notebook workflow](guides/notebooks.md) for direct and tracked Python runs.
- [Dashboard workflow](dashboard/workflow.md) for every field in the run form.
- [Frontend backend link](dashboard/frontend-backend-link.md) for the request
  and event flow.
- [Scenarios](scenarios.md) for ready-to-run examples.
- [Troubleshooting](troubleshooting.md) when training fails.
- [Migration to 1.0](migration/1.0.md) for removed 0.2 interfaces.
