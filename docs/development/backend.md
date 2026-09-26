# Backend Development

The backend lives in `pspso/dashboard/`.

## Important Files

| File | Role |
| --- | --- |
| `pspso/dashboard/app.py` | FastAPI app, request models, REST endpoints, SSE stream. |
| `pspso/dashboard/tracking.py` | SQLite persistence for experiments, runs, and ordered run events. |
| `pspso/dashboard/data.py` | Built-in datasets, CSV preview, preprocessing. |
| `pspso/dashboard/manager.py` | Local subprocess queue, cancellation, timeout, and worker supervision. |
| `pspso/dashboard/worker.py` | Child-process execution, result/artifact persistence, final analysis. |
| `pspso/dashboard/runtime.py` | Shared data/optimizer construction used by the API, CLI, and worker without importing FastAPI. |
| `pspso/dashboard/artifacts.py` | Persisted run specification, result, model, preprocessor, and analysis. |
| `pspso/dashboard/cli.py` | Console entrypoints. |
| `pspso/optimizer.py` | Optimization loop and progress events. |
| `pspso/estimators.py` | Estimator presets, factories, metadata, dependency checks. |

## Run Lifecycle

1. Validate request.
2. Validate data and freeze evaluation settings; preprocessing is fitted inside folds.
3. Build `SearchSpace`.
4. Build `OptimizationConfig` and `EstimatorConfig`.
5. Persist the queued run and experiment records in SQLite.
6. Start a managed local worker subprocess from the lightweight runtime module; FastAPI does not train models itself.
7. Store every worker and optimizer event in the persistent event store before SSE replay.
8. Stream ordered events to the frontend over SSE.
9. Store final results, model/preprocessor artifacts, diagnostics, and feature importance for historical replay.

## Validation Responsibilities

Validation should happen before a run starts whenever possible:

- dataset exists;
- target column exists;
- task and metric are compatible;
- estimator supports the selected task;
- optional dependency is installed;
- fixed params and tunable params are accepted by the estimator;
- search-space specs are valid.

Trial-level failures are still possible, but validation should catch the common
dashboard configuration mistakes.

## Persistence Responsibilities

The tracking repository is responsible for:

- creating named experiments and ad hoc experiments;
- storing the full submitted run request as JSON;
- storing ordered step events with `sequence_number`;
- deriving run summary fields such as `best_metric`, `n_trials`, and `n_failures`;
- returning historical run state after a restart or page refresh.

`GET /api/v1/runs/{run_id}/analysis` reads the saved diagnostics for a successful
run. `GET /api/v1/runs/{run_id}/predictions` uses the persisted model and
preprocessor and frozen dataset. Missing serialized inputs produce an explicit
error; reading results never retrains a replacement model.
