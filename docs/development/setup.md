# Source setup

This page is for contributors working from the PSPSO repository. Package users
install PSPSO from PyPI as described in [Getting started](../getting-started.md).

## Install development dependencies

From a source checkout, use [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group dev --extra docs
```

Add an optional model engine with `--extra xgboost`, `--extra lightgbm`, or
`--extra torch`. Use `--all-extras` when running the complete model test matrix.
FastAPI and Uvicorn are included in the base package; the `api` extra remains
accepted for compatibility.

## Build and start the dashboard

Node.js is required only to develop or rebuild the frontend. From the repository
root, build the interface first:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
uv run --no-sync pspso-dashboard
```

Open [the dashboard](http://127.0.0.1:8000). For live frontend development, run
`npm --prefix frontend run dev` in a second terminal. Vite serves port 5173 and
forwards API calls to the backend on port 8000.

## Browse and build MkDocs

Use a separate port so the documentation can run alongside the dashboard:

```bash
uv run --no-sync mkdocs serve --dev-addr 127.0.0.1:8001
```

Open [the documentation](http://127.0.0.1:8001/pspso/). To check all documentation pages:

```bash
uv run --no-sync mkdocs build --strict
```

Generated output goes into the ignored `site/` directory.

## Verification

```bash
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy pspso
npm --prefix frontend test
npm --prefix frontend run test:e2e
```

The repository's verification workflow also checks coverage, optional model
engines, dependency audits and installed-package behavior. See
[Releasing to PyPI](releasing.md) for the publication process.
