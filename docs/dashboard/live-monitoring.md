# Live Monitoring

The backend emits structured events while training and validation happen. The
frontend listens with `EventSource`.

The same event stream is also persisted in SQLite, so the live timeline and the
saved history are driven by the same underlying records.

## PSO Scheduling And Progress

A PSO run plans `particles x iterations` candidates. Each uses the configured
fold count; a final refit adds one model fit when enabled. Up to
`runtime.trial_workers` candidates train concurrently. Queued candidates wait
for an available slot. Actual fit counts come from fit start and finish events.

The `run_started` event records `planned_trials` and `execution_mode`. Both
`trial_completed` and `trial_failed` advance the finished-candidate count, while only
a successful `trial_completed` event has a valid cost to add to the convergence
chart. The chart therefore has one point per successful candidate, not one point
per fold fit. It plots the best cost found so far in candidate-completion order;
with parallel workers, that order can differ from the candidate IDs. After every
particle in an iteration has been handled, the optimizer
emits `iteration_completed`. At the end of a successful three-particle,
two-iteration run, the expected totals are six trial events, two iteration
events, and six chart points.
With five-fold CV and final refit, that example plans 31 fits. Fit counters use
`model_fit_completed` and `model_fit_failed`; fold events expose candidate,
iteration, fold number and worker slot. Early stopping can leave unused budget.
Queue position, PID, heartbeat, attempt count and cancellation intent appear
separately from active training.

The Cancel button changes to **Cancelling…** as soon as the service accepts the
request. Cooperative estimators stop at their next progress checkpoint; other
worker processes are forcibly contained after the short cleanup grace period.
The complete worker log remains available as an artifact, while log lines are
persisted to the timeline in a single database transaction so noisy estimators
cannot hold the worker queue open during cancellation.

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
| `fold_started` / `fold_completed` | A candidate's fold and metric. |
| `model_fit_started` / `model_fit_completed` / `model_fit_failed` | Actual fit work and cumulative totals. |
| `final_refit_started` / `final_refit_completed` | The selected pipeline fits all development rows. |
| `training_epoch` | A PyTorch tabular model completed an epoch; includes training loss and, when configured, validation loss and validation accuracy. |
| `iteration_completed` | A PSO iteration finished. |
| `best_updated` | A new best validation score was found. |
| `run_warning` | The backend found a non-fatal issue. |
| `run_cancel_requested` | The service accepted a user cancellation request and cleanup is in progress. |
| `run_cancelled` | The worker and its contained child processes were stopped. |
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
- the complete selected-task metric set in `train_metrics` and
  `validation_metrics` (for example RMSE/MAE/R2 for regression, or accuracy,
  macro F1, log loss, and binary diagnostics where applicable).

For a binary classifier, the saved completed-run analysis also provides ROC AUC
(the area under the ROC curve), PR AUC, sensitivity/recall, specificity, a
confusion matrix, and ROC curve coordinates. Multiclass analysis provides a
confusion matrix and per-class precision, sensitivity, specificity and F1.
When probabilities are available, it also includes one-vs-rest ROC curves and
macro ROC AUC.

## Completed-Run Analysis

After a successful run, the worker saves `analysis.json` beside the model and
preprocessor artifacts. The dashboard reads it through:

```text
GET /api/v1/runs/{run_id}/analysis
```

The analysis includes train, validation, and optional untouched test-split
metrics. Where an estimator exposes native explanations, it also lists feature
importance from `feature_importances_` (tree recipes) or absolute coefficients
(linear recipes). Other recipes are marked unavailable rather than receiving a
made-up importance ranking.

For classification metrics such as ROC AUC and accuracy, higher is better. The
optimizer converts those to cost with `1 - metric` internally.

## Polling Fallback

The browser reconciles snapshots and incremental history periodically, including
when SSE disconnects. SSE carries event IDs, idle heartbeats and replay using
`Last-Event-ID` or `?after=N`. The client deduplicates by sequence number. Read a
snapshot through:

```text
GET /api/v1/runs/{run_id}
```

Polling returns the run status, event count, best params so far, error summary,
and final result when available.

For incremental replay, the frontend loads:

```text
GET /api/v1/runs/{run_id}/history?after=N
```
