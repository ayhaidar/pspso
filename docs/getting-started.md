# Getting Started

This guide starts from a clean checkout and ends with a successful dashboard
run.

## Install With uv

```bash
uv sync --extra api --extra docs --group dev
```

The `api` extra installs FastAPI and Uvicorn. The `docs` extra installs MkDocs
Material and API-reference tooling.

Pip remains available for users who are not using uv:

```bash
pip install -e ".[api,docs]"
```

## Start The Backend

```bash
uv run pspso-dashboard --host 127.0.0.1 --port 8000
```

FastAPI exposes:

- `GET /api/datasets/examples`
- `POST /api/datasets/preview`
- `GET /api/estimators`
- `POST /api/runs/validate`
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- `GET /api/runs/{run_id}/result`

## Start The Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

## First Successful Run

Use these choices:

- Dataset: `Breast cancer`
- Task: `Binary classification`
- Metric: `ROC AUC`
- Estimator: `SVM`
- Strategy: `Random`
- Max trials: `6`

Press validation first. When validation succeeds, start the run. The live panel
should show trial events, training metric, validation metric, and best
parameters.

## Build The Docs

```bash
uv run mkdocs build
```

MkDocs writes generated output to `site/`, which is ignored by git.
