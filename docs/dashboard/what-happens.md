# What Happens During a Run

This page explains the real run lifecycle in `pspso`, from dashboard form
submission to saved experiment history.

## High-Level Flow

1. The React dashboard builds a `RunRequest`.
2. The backend validates the request before any model training starts.
3. A run record is written to SQLite.
4. A background thread prepares the dataset and starts the optimizer.
5. The optimizer emits progress events for each major step.
6. The backend saves each event immediately and streams it to the frontend.
7. The final result is stored on the run record.
8. The run can be reloaded later from history, even after a page refresh or API restart.

## Step-by-Step Lifecycle

### 1. Configuration and validation

The dashboard sends the configuration to:

```text
POST /api/runs/validate
```

The backend checks:

- dataset source and target column;
- task and metric compatibility;
- estimator availability;
- optional dependency presence;
- fixed parameter validity;
- tunable search-space validity;
- preprocessing and dataset preparation feasibility.

If validation passes, the dashboard can start the run.

### 2. Run and experiment creation

When the dashboard calls:

```text
POST /api/runs
```

the backend:

- attaches the run to a selected experiment, or
- creates an ad hoc experiment automatically if none was selected.

The run configuration is stored exactly as submitted so it can be inspected
later.

### 3. Persisted execution tracking

As soon as the run is accepted, the backend starts recording step events in the
`run_events` table. Each event gets:

- `run_id`
- `sequence_number`
- `event_type`
- `timestamp`
- `payload_json`

This means the execution timeline is durable, ordered, and replayable.

### 4. Dataset preparation

Before optimization starts, the backend loads the dataset, applies
preprocessing, and splits the data. A `dataset_prepared` event is recorded with
summary information such as row counts and feature counts.

### 5. Optimizer execution

The optimizer emits the main runtime events:

- `run_started`
- `trial_started`
- `trial_completed`
- `trial_failed`
- `iteration_completed`
- `best_updated`
- `run_completed`
- `run_failed`

These events are used twice:

- for live frontend updates;
- for historical replay from SQLite.

### 6. Final result storage

When the optimizer finishes, the backend stores:

- final status;
- best params;
- best metric and cost;
- number of trials;
- number of failures;
- full result JSON;
- terminal error information if the run failed.

### 7. History and replay

The frontend can later fetch:

```text
GET /api/runs/{run_id}
GET /api/runs/{run_id}/history
GET /api/runs/{run_id}/result
GET /api/runs/{run_id}/artifacts
```

That is why the run still appears after a refresh and why the timeline can be
reconstructed exactly.

## Why Failed Trials Still Matter

A failed trial is not wasted information. It tells us:

- which parameter set was attempted;
- which estimator/backend rejected it;
- whether validation should become stricter in future;
- whether the search space is too permissive.

This is why failed trials are preserved in the event history instead of being
discarded.
