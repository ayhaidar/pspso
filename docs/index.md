# pspso

`pspso` is a local hyperparameter-optimization toolkit for tabular machine
learning. It supports particle swarm optimization, grid search, and random
search through a Python API, a compatibility wrapper for the original package,
and a FastAPI + React dashboard.

The current dashboard is designed for local/private experimentation:

- choose a built-in dataset or paste CSV data;
- select a task, metric, estimator, fixed training parameters, and tunable
  hyperparameters;
- validate the configuration before a run starts;
- stream live training and validation progress from the backend to the
  frontend;
- inspect trials, failures, best parameters, and final results.

## First Commands

```bash
uv sync --extra api --extra docs --group dev
uv run pytest
uv run mkdocs serve
```

Run the backend:

```bash
uv run pspso-dashboard
```

Run the frontend during development:

```bash
cd frontend
npm install
npm run dev
```

The frontend uses relative `/api/...` URLs. Vite proxies those requests to the
FastAPI backend at `http://127.0.0.1:8000`.

## Where To Go Next

- [Getting started](getting-started.md) for a full local setup.
- [Dashboard workflow](dashboard/workflow.md) for every field in the run form.
- [Frontend backend link](dashboard/frontend-backend-link.md) for the request
  and event flow.
- [Scenarios](scenarios.md) for ready-to-run examples.
- [Troubleshooting](troubleshooting.md) when training fails.
