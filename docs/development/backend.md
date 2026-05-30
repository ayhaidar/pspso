# Backend Development

The backend lives in `pspso/dashboard/`.

## Important Files

| File | Role |
| --- | --- |
| `pspso/dashboard/app.py` | FastAPI app, request models, run lifecycle, SSE stream. |
| `pspso/dashboard/tracking.py` | SQLite persistence for experiments, runs, and ordered run events. |
| `pspso/dashboard/data.py` | Built-in datasets, CSV preview, preprocessing. |
| `pspso/dashboard/cli.py` | Console entrypoints. |
| `pspso/optimizer.py` | Optimization loop and progress events. |
| `pspso/estimators.py` | Estimator presets, factories, metadata, dependency checks. |

## Run Lifecycle

1. Validate request.
2. Load and preprocess dataset.
3. Build `SearchSpace`.
4. Build `OptimizationConfig` and `EstimatorConfig`.
5. Start optimizer in a background thread.
6. Persist the run and experiment records in SQLite.
7. Store each progress event in both the live `RunState` and the persistent event store.
8. Stream events to the frontend over SSE.
9. Store the final result for retrieval and historical replay.

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
