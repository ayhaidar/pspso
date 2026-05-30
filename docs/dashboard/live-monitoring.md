# Live Monitoring

The backend emits structured events while training and validation happen. The
frontend listens with `EventSource`.

The same event stream is also persisted in SQLite, so the live timeline and the
saved history are driven by the same underlying records.

## Event Types

| Event | Meaning |
| --- | --- |
| `validation_passed` | Configuration was accepted and a persisted run was created. |
| `dataset_prepared` | Dataset loading, preprocessing, and splitting completed. |
| `validation_failed` | Configuration was rejected before training. |
| `run_started` | The optimizer has started. |
| `trial_started` | A parameter set is about to be fitted. |
| `trial_completed` | Training and validation finished for one trial. |
| `trial_failed` | The trial raised an estimator or metric error. |
| `iteration_completed` | A PSO iteration finished. |
| `best_updated` | A new best validation score was found. |
| `run_warning` | The backend found a non-fatal issue. |
| `run_completed` | The run completed with a best model. |
| `run_failed` | The run ended without a valid best model. |

## Trial Metrics

Each completed trial includes:

- trial number;
- evaluated params;
- training metric;
- validation metric;
- cost used for minimization;
- duration.

For classification metrics such as ROC AUC and accuracy, higher is better. The
optimizer converts those to cost with `1 - metric` internally.

## Polling Fallback

If the SSE connection drops, the frontend should poll:

```text
GET /api/runs/{run_id}
```

Polling returns the run status, event count, best params so far, error summary,
and final result when available.

For a full replay, the frontend can also load:

```text
GET /api/runs/{run_id}/history
```
