# REST API v1

FastAPI serves the stable local contract under `/api/v1`. Interactive OpenAPI
documentation is available at `/api/v1/docs`, with the machine-readable schema
at `/api/v1/openapi.json`.

## Main Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/estimators` | Tasks, metrics, models, capabilities, and defaults. |
| GET/POST | `/api/v1/datasets` | List or import versioned datasets. |
| POST | `/api/v1/datasets/inspect` | Profile data and preview the requested split. |
| GET/POST | `/api/v1/experiments` | List or create experiments. |
| POST | `/api/v1/workflow/validate` | Validate data, model, search, or the full specification. |
| POST | `/api/v1/runs/validate` | Validate a complete `ExperimentSpec`. |
| GET/POST | `/api/v1/runs` | List or start runs. |
| GET | `/api/v1/runs/{run_id}` | Read status and summary. |
| GET | `/api/v1/runs/{run_id}/events` | Stream SSE progress. |
| GET | `/api/v1/runs/{run_id}/history` | Replay persisted events. |
| GET | `/api/v1/runs/{run_id}/analysis` | Read task-specific diagnostics. |
| GET | `/api/v1/runs/{run_id}/predictions` | Inspect train, validation, or test predictions. |
| POST | `/api/v1/runs/{run_id}/cancel` | Request cancellation. |
| POST | `/api/v1/runs/{run_id}/retry` | Create a linked retry run. |

## ExperimentSpec

Every run requires `schema_version: 1`, a dataset source and target, canonical
task/model/metric IDs, fixed parameters, typed search-space schemas, strategy,
split, preprocessing, PSO, evaluation and runtime settings. Unknown fields are rejected.

Validation returns errors grouped by `dataset`, `task`, `estimator`,
`fixed_params`, `search_space`, and `strategy`. Invalid requests are not
persisted.

The API is intended for local/private use and currently has no authentication.
Do not expose it directly to an untrusted network.

## Durable runs and downloads

Run responses include attempts, service and worker identity, heartbeat, queue
position, cancellation intent and parent linkage. Status is `queued`, `running`,
`completed`, `failed`, `cancelled` or `interrupted`.

`POST /api/v1/experiments/{experiment_id}/tournament` accepts at least two model
specifications under one frozen dataset, preprocessing, evaluation and budget.
All are validated before any is submitted.

`GET /api/v1/runs/{run_id}/exports/{kind}` downloads `spec`, `result`, `analysis`,
`events`, `predictions`, `metrics`, `manifest`, `model`, `selection_model`,
`preprocessor`, `environment`, `split_indices` or `log`. Prediction CSV includes
all rows, raw features and class probabilities. The `split` query chooses `train`,
`validation` or `test`; defaults favor the untouched test set.

SSE messages carry an `id` matching the persisted sequence number. Reconnect with
`Last-Event-ID`, or supply `?after=N`; the larger cursor wins. Idle streams send
heartbeat comments. `/history?after=N` returns only later events. Local CORS
origins are configured through comma-separated `PSPSO_CORS_ORIGINS`.
